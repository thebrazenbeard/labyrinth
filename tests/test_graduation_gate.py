"""Minotaur qualification behavior: no authenticated capability, no exit."""

from dataclasses import replace
import hashlib
import hmac
import json
import unittest

from labyrinth.graduation_gate import (
    GateState, GraduationPolicy, SCHEMA, decide_graduation,
)


KEY = b"independent-test-custodian-secret-32-bytes-not-for-production"
MODEL = "a" * 64
BASE = "f" * 64
POLICY = GraduationPolicy(
    objective_id="register-program-reasoning-v1", trainee_sha256=MODEL, base_sha256=BASE,
    session_nonce="0123456789abcdef" * 2, max_attempts=2,
    expected_evaluator_sha256="b" * 64,
    expected_conditions_sha256="c" * 64,
)


def evaluated_receipt(*, attempt=1, run_id="fresh-run-001", candidate_correct=True,
                      controls_correct=False):
    """Synthetic trusted-evaluator fixture; NOT a genuine neural experiment."""
    return {
        "schema": SCHEMA,
        "objective_id": POLICY.objective_id,
        "trainee_sha256": MODEL,
        "base_sha256": BASE,
        "session_nonce": POLICY.session_nonce,
        "run_id": run_id,
        "attempt": attempt,
        "evaluator_sha256": "b" * 64,
        "conditions_sha256": "c" * 64,
        "suite_sha256": hashlib.sha256(run_id.encode()).hexdigest(),
        "fresh_process_reload": True,
        "independent_private_oracle": True,
        "training_case_ids": ["train-001", "train-002"],
        "training_prompt_commitments": [hashlib.sha256(f"train:{i}".encode()).hexdigest()
                                        for i in range(2)],
        "arm_checkpoint_sha256s": {
            "candidate": MODEL, "baseline": BASE,
            "zero_update": "d" * 64, "corrupted": "e" * 64,
        },
        "arm_training_steps": {
            "candidate": 32, "baseline": 0, "zero_update": 32, "corrupted": 32,
        },
        "slices": {
            slice_name: [
                {
                    "case_id": f"{slice_name}-{index:03d}",
                    "prompt_commitment": hashlib.sha256(
                        f"prompt:{slice_name}:{index}".encode()).hexdigest(),
                    "candidate_correct": candidate_correct,
                    "candidate_format": True,
                    "candidate_semantic_correct": candidate_correct,
                    "baseline_correct": controls_correct,
                    "baseline_semantic_correct": controls_correct,
                    "baseline_format": True,
                    "zero_update_correct": controls_correct,
                    "zero_update_semantic_correct": controls_correct,
                    "zero_update_format": True,
                    "corrupted_correct": controls_correct,
                    "corrupted_semantic_correct": controls_correct,
                    "corrupted_format": True,
                    "candidate_response_sha256": hashlib.sha256(
                        f"{run_id}:{slice_name}:{index}:candidate".encode()).hexdigest(),
                    "baseline_response_sha256": hashlib.sha256(
                        f"{run_id}:{slice_name}:{index}:baseline".encode()).hexdigest(),
                    "zero_update_response_sha256": hashlib.sha256(
                        f"{run_id}:{slice_name}:{index}:zero".encode()).hexdigest(),
                    "corrupted_response_sha256": hashlib.sha256(
                        f"{run_id}:{slice_name}:{index}:corrupt".encode()).hexdigest(),
                }
                for index in range(16)
            ]
            for slice_name in POLICY.required_slices
        },
    }


def independently_signed(receipt):
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(KEY, b"labyrinth.minotaur.graduation.v1\x00" + raw,
                         hashlib.sha256).hexdigest()
    return raw, signature


def decision(receipt, state=GateState(), *, key=KEY, signature=None):
    raw, generated = independently_signed(receipt)
    return decide_graduation(POLICY, state, raw, generated if signature is None else signature,
                             independent_evaluator_key=key)


class MinotaurGraduationTests(unittest.TestCase):
    def test_valid_separately_attested_capability_graduates_without_deployment(self):
        result = decision(evaluated_receipt())
        self.assertEqual(result.disposition, "GRADUATED")
        self.assertEqual(result.state.next_attempt, 2)
        self.assertEqual(result.state.consumed_run_ids, frozenset({"fresh-run-001"}))
        self.assertTrue(result.state.graduated)
        self.assertFalse(hasattr(result, "metrics"))

    def test_missing_or_self_reported_evidence_cannot_qualify(self):
        valid = evaluated_receipt()
        raw, sig = independently_signed(valid)
        for receipt, signature, key in (
            (None, None, KEY),
            (raw, None, KEY),
            (raw, "0" * 64, KEY),
            (raw, sig, None),
            (raw, sig, b"learner-controlled-key-is-not-independent"),
            (b'{"accuracy":1,"status":"PASS"}', sig, KEY),
        ):
            with self.subTest(receipt=receipt, key=key):
                result = decide_graduation(POLICY, GateState(), receipt, signature,
                                           independent_evaluator_key=key)
                self.assertEqual(result.disposition, "BLOCKED")
                self.assertEqual(result.state, GateState())

    def test_tamper_binding_subject_session_run_and_attempt(self):
        for field, value in (
            ("objective_id", "other"), ("trainee_sha256", "d" * 64),
            ("base_sha256", "d" * 64),
            ("session_nonce", "f" * 32), ("attempt", 2),
            ("fresh_process_reload", False), ("independent_private_oracle", False),
            ("conditions_sha256", "d" * 64), ("evaluator_sha256", "e" * 64),
        ):
            with self.subTest(field=field):
                receipt = evaluated_receipt()
                receipt[field] = value
                self.assertEqual(decision(receipt).disposition, "BLOCKED")

        raw, signature = independently_signed(evaluated_receipt())
        modified = raw.replace(b"fresh-run-001", b"fresh-run-002")
        self.assertEqual(decide_graduation(POLICY, GateState(), modified, signature,
                         independent_evaluator_key=KEY).disposition, "BLOCKED")

    def test_unmet_objective_stays_then_finite_budget_exhausts(self):
        first = decision(evaluated_receipt(candidate_correct=False))
        self.assertEqual(first.disposition, "STAY")
        self.assertEqual(first.reasons, ("objective not demonstrated",))
        self.assertFalse(first.state.graduated)
        second = decision(evaluated_receipt(attempt=2, run_id="fresh-run-002",
                                             candidate_correct=False), first.state)
        self.assertEqual(second.disposition, "EXHAUSTED")
        self.assertEqual(second.state.next_attempt, 3)
        self.assertEqual(decision(evaluated_receipt(attempt=3), second.state).disposition,
                         "EXHAUSTED")

    def test_control_tie_and_format_only_does_not_count_as_reasoning_gain(self):
        equal = decision(evaluated_receipt(controls_correct=True))
        self.assertEqual(equal.disposition, "STAY")
        self.assertEqual(equal.reasons, ("objective not demonstrated",))
        formatted = evaluated_receipt(candidate_correct=False)
        self.assertTrue(all(t["candidate_format"] for g in formatted["slices"].values() for t in g))
        self.assertEqual(decision(formatted).disposition, "STAY")
        # This candidate only fixes output syntax. Every control knew the
        # substantive answer, but lacked the prescribed final-line format.
        format_only = evaluated_receipt(candidate_correct=True, controls_correct=False)
        for group in format_only["slices"].values():
            for trial in group:
                for arm in ("baseline", "zero_update", "corrupted"):
                    trial[f"{arm}_semantic_correct"] = True
                    trial[f"{arm}_format"] = False
                    trial[f"{arm}_correct"] = False
        self.assertEqual(decision(format_only).disposition, "STAY")

    def test_arm_checkpoints_and_equal_optimizer_budget_required(self):
        for path, value in (
            (("arm_checkpoint_sha256s", "candidate"), BASE),
            (("arm_checkpoint_sha256s", "baseline"), "9" * 64),
            (("arm_checkpoint_sha256s", "zero_update"), MODEL),
            (("arm_checkpoint_sha256s", "corrupted"), MODEL),
            (("arm_training_steps", "candidate"), 0),
            (("arm_training_steps", "zero_update"), 31),
            (("arm_training_steps", "corrupted"), 0),
            (("arm_training_steps", "baseline"), 1),
        ):
            with self.subTest(path=path):
                evidence = evaluated_receipt()
                evidence[path[0]][path[1]] = value
                self.assertEqual(decision(evidence).disposition, "BLOCKED")

    def test_transfer_missing_duplicate_cases_and_train_leakage_block(self):
        missing = evaluated_receipt()
        missing["slices"].pop("transfer")
        self.assertEqual(decision(missing).disposition, "BLOCKED")
        overlap = evaluated_receipt()
        overlap["slices"]["transfer"][0]["case_id"] = "heldout-000"
        self.assertEqual(decision(overlap).disposition, "BLOCKED")
        train_leak = evaluated_receipt()
        train_leak["slices"]["heldout"][0]["case_id"] = "train-001"
        self.assertEqual(decision(train_leak).disposition, "BLOCKED")
        prompt_leak = evaluated_receipt()
        prompt_leak["slices"]["heldout"][0]["prompt_commitment"] = (
            prompt_leak["training_prompt_commitments"][0])
        self.assertEqual(decision(prompt_leak).disposition, "BLOCKED")
        tiny = evaluated_receipt()
        tiny["slices"]["heldout"] = tiny["slices"]["heldout"][:15]
        self.assertEqual(decision(tiny).disposition, "BLOCKED")

    def test_inconsistent_accuracy_formats_or_nonboolean_scores_block(self):
        inconsistent = evaluated_receipt()
        inconsistent["slices"]["heldout"][0]["candidate_format"] = False
        self.assertEqual(decision(inconsistent).disposition, "BLOCKED")
        wrong_semantic = evaluated_receipt()
        wrong_semantic["slices"]["heldout"][0]["candidate_semantic_correct"] = False
        self.assertEqual(decision(wrong_semantic).disposition, "BLOCKED")
        not_boolean = evaluated_receipt()
        not_boolean["slices"]["heldout"][0]["candidate_correct"] = 1
        self.assertEqual(decision(not_boolean).disposition, "BLOCKED")
        missing_control = evaluated_receipt()
        del missing_control["slices"]["heldout"][0]["zero_update_correct"]
        self.assertEqual(decision(missing_control).disposition, "BLOCKED")
        contradictory = evaluated_receipt()
        item = contradictory["slices"]["heldout"][0]
        item["baseline_response_sha256"] = item["candidate_response_sha256"]
        self.assertEqual(decision(contradictory).disposition, "BLOCKED")

    def test_replay_attempt_is_bound_to_trusted_state(self):
        first = decision(evaluated_receipt(candidate_correct=False))
        replay = decision(evaluated_receipt(candidate_correct=False), first.state)
        self.assertEqual(replay.disposition, "BLOCKED")
        self.assertEqual(replay.state, first.state)
        repeated_id = decision(evaluated_receipt(attempt=2, candidate_correct=False), first.state)
        self.assertEqual(repeated_id.disposition, "BLOCKED")
        reused_suite = evaluated_receipt(attempt=2, run_id="fresh-run-002", candidate_correct=False)
        reused_suite["suite_sha256"] = hashlib.sha256(b"fresh-run-001").hexdigest()
        self.assertEqual(decision(reused_suite, first.state).disposition, "BLOCKED")

    def test_graduated_state_cannot_be_reused_for_new_subject_or_weaker_policy(self):
        qualified = decision(evaluated_receipt())
        self.assertTrue(qualified.state.graduated)
        # A persisted success must be *bound*; no optional receipt can qualify
        # a different trainee, objective, session, or easier criterion.
        for changed in (
            replace(POLICY, trainee_sha256="e" * 64),
            replace(POLICY, objective_id="other-objective"),
            replace(POLICY, base_sha256="e" * 64),
            replace(POLICY, session_nonce="f" * 32),
            replace(POLICY, min_accuracy=0.25),
            replace(POLICY, expected_evaluator_sha256="d" * 64),
            replace(POLICY, expected_conditions_sha256="e" * 64),
        ):
            with self.subTest(changed=changed):
                denied = decide_graduation(changed, qualified.state, None, None,
                                           independent_evaluator_key=None)
                self.assertEqual(denied.disposition, "BLOCKED")
        self.assertEqual(decide_graduation(POLICY, qualified.state, None, None,
                         independent_evaluator_key=None).disposition, "GRADUATED")
        self.assertEqual(decide_graduation(POLICY, GateState(graduated=True), None, None,
                         independent_evaluator_key=None).disposition, "BLOCKED")

    def test_signed_duplicate_json_keys_fail_closed(self):
        raw = b'{"schema":"duplicate","schema":"'+SCHEMA.encode()+b'"}'
        sig = hmac.new(KEY, b"labyrinth.minotaur.graduation.v1\x00"+raw,
                       hashlib.sha256).hexdigest()
        self.assertEqual(decide_graduation(POLICY, GateState(), raw, sig,
                         independent_evaluator_key=KEY).disposition, "BLOCKED")

    def test_invalid_policies_and_state_are_rejected(self):
        for kwargs in (
            {"required_slices": ("heldout", "heldout")},
            {"max_attempts": 0},
            {"min_accuracy": float("nan")},
            {"objective_id": "bad name"},
            {"session_nonce": "predictable"},
            {"expected_evaluator_sha256": "bad"},
            {"expected_conditions_sha256": "bad"},
            {"base_sha256": "bad"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GraduationPolicy(**({"objective_id": "objective-v1", "trainee_sha256": MODEL,
                                     "base_sha256": BASE,
                                     "session_nonce": "a" * 32,
                                     "expected_evaluator_sha256": "b" * 64,
                                     "expected_conditions_sha256": "c" * 64} | kwargs))
        invalid = GateState(next_attempt=0)
        self.assertEqual(decision(evaluated_receipt(), invalid).disposition, "BLOCKED")


if __name__ == "__main__":
    unittest.main()
