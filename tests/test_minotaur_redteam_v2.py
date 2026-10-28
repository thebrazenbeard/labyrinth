"""Hostile boundary tests; all evidence is synthetic, never an independent exam."""

from copy import deepcopy
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.graduation_gate import GateState, decide_graduation
from labyrinth.graduation_state import decide_campaign, initialize_campaign
from labyrinth.processbench_adapter import evaluate_processbench, load_processbench_jsonl
from tests.test_graduation_gate import KEY, POLICY, evaluated_receipt, independently_signed


STATE_KEY = b"redteam-separate-state-custody-secret-2026"
DOMAIN = b"labyrinth.minotaur.graduation.v1\x00"


class MinotaurRedteamV2(unittest.TestCase):
    def test_signed_duplicate_json_and_nonfinite_number_fail_closed(self):
        for raw in (
            b'{"attempt":1,"attempt":2}',
            b'{"attempt":NaN}',
            b'{"attempt":Infinity}',
            b'{"attempt":1e999}',
            b'[]',
        ):
            with self.subTest(raw=raw):
                signature = hmac.new(KEY, DOMAIN + raw, hashlib.sha256).hexdigest()
                result = decide_graduation(POLICY, GateState(), raw, signature,
                                           independent_evaluator_key=KEY)
                self.assertEqual(result.disposition, "BLOCKED")
                self.assertEqual(result.state, GateState())

    def test_borrowed_evaluator_key_fails_state_custody_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custody.sqlite3"
            initialize_campaign(path, POLICY, state_key=KEY)
            raw, signature = independently_signed(evaluated_receipt())
            with self.assertRaisesRegex(ValueError, "different secrets"):
                decide_campaign(path, POLICY, raw, signature,
                                independent_evaluator_key=KEY, state_key=KEY)

    def test_scoring_swap_same_response_is_blocked(self):
        receipt = evaluated_receipt()
        trial = receipt["slices"]["heldout"][0]
        trial["baseline_response_sha256"] = trial["candidate_response_sha256"]
        raw, signature = independently_signed(receipt)
        result = decide_graduation(POLICY, GateState(), raw, signature,
                                   independent_evaluator_key=KEY)
        self.assertEqual(result.disposition, "BLOCKED")

    def test_relabelled_replay_cases_and_prompts_fail_closed(self):
        first = evaluated_receipt(candidate_correct=False)
        raw, signature = independently_signed(first)
        state = decide_graduation(POLICY, GateState(), raw, signature,
                                  independent_evaluator_key=KEY).state
        second = evaluated_receipt(attempt=2, run_id="new-run-after-failure")
        for name in POLICY.required_slices:
            for old, new in zip(first["slices"][name], second["slices"][name]):
                new["prompt_commitment"] = old["prompt_commitment"]
        raw, signature = independently_signed(second)
        denied = decide_graduation(POLICY, state, raw, signature,
                                   independent_evaluator_key=KEY)
        self.assertEqual(denied.disposition, "BLOCKED")
        self.assertEqual(denied.state, state)

    def test_distinct_hashes_do_not_prove_independent_equivalent_questions(self):
        # This is a documented custody risk, NOT a private exam or verified
        # isomorphism. A signer can rename paraphrases and supply new hashes.
        receipt = evaluated_receipt()
        for name in POLICY.required_slices:
            for index, trial in enumerate(receipt["slices"][name]):
                trial["case_id"] = f"variant-{name}-{index}"
                trial["prompt_commitment"] = hashlib.sha256(
                    f"paraphrase {index} of the same underlying task".encode()).hexdigest()
        # Keep the two slices' commitments distinct, as the gate requires.
        for index, trial in enumerate(receipt["slices"]["transfer"]):
            trial["prompt_commitment"] = hashlib.sha256(
                f"alternate wording {index} of the same task".encode()).hexdigest()
        raw, signature = independently_signed(receipt)
        result = decide_graduation(POLICY, GateState(), raw, signature,
                                   independent_evaluator_key=KEY)
        self.assertEqual(result.disposition, "GRADUATED")
        # The result reflects only syntactically valid *signed claims*.
        # No question text or oracle is available for semantic deduplication.

    def test_signed_claims_can_masquerade_as_training_effect(self):
        receipt = evaluated_receipt()
        # No subprocess, checkpoint, training run, or oracle is invoked here.
        # Possession of the evaluator key suffices to attest fabricated wins.
        raw, signature = independently_signed(receipt)
        self.assertEqual(decide_graduation(
            POLICY, GateState(), raw, signature,
            independent_evaluator_key=KEY).disposition, "GRADUATED")
        modified = deepcopy(receipt)
        modified["arm_training_steps"]["candidate"] = 0
        raw, signature = independently_signed(modified)
        self.assertEqual(decide_graduation(
            POLICY, GateState(), raw, signature,
            independent_evaluator_key=KEY).disposition, "BLOCKED")

    def test_processbench_perfect_score_on_fabricated_labels_is_unverified(self):
        row = {"id": "invented-1", "generator": "invented", "problem": "1 + 1?",
               "steps": ["1 + 1 = 3"], "final_answer_correct": True, "label": -1}
        cases = load_processbench_jsonl((json.dumps(row) + "\n").encode(), split="math")
        score = evaluate_processbench(cases, lambda _problem, _steps: "ALL_CORRECT")
        self.assertEqual(score["accuracy"], 1.0)
        self.assertEqual(score["reasoning_qualification"], "NOT_ESTABLISHED")
        self.assertEqual(score["evidence_status"], "UNVERIFIED_CALLER_SUPPLIED_DATA")


if __name__ == "__main__":
    unittest.main()
