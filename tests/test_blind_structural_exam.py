"""Fresh, answer-withheld transition systems: controls must produce actual plans.

A source-public generator is not a held-out neural examiner. Tests use
deterministic RNG solely to inspect bounded behavior and witness verification.
"""
from __future__ import annotations

from copy import deepcopy
import base64
from contextlib import redirect_stdout
from io import StringIO
from hashlib import sha256
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from labyrinth.blind_structural_exam import prepare_exam, score_exam, main
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


class BlindStructuralExamTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.private = self.root / "private.json"
        self.public = self.root / "public.json"
        self.commit = self.root / "precommit.json"
        self.responses = self.root / "responses.json"
        self.out = self.root / "score.json"

    def prepare(self, *, count=8, seed=37, worlds=None):
        prepare_exam(self.private, self.public, self.commit,
                     count=count, rng=random.Random(seed), worlds=worlds)
        private = json.loads(self.private.read_text(encoding="utf-8"))
        public = json.loads(self.public.read_text(encoding="utf-8"))
        plan = json.loads(self.commit.read_text(encoding="utf-8"))
        return private, public, plan

    def responses_for(self, private, *, use_oracle):
        items = []
        for trial in private["trials"]:
            if use_oracle:
                response = ("PLAN: " + " ".join(trial["oracle_witness"])
                            if trial["oracle_witness"] is not None else "UNSAT")
            else:
                response = "UNSAT"
            items.append({"trial_id": trial["trial_id"], "response": response})
        self.responses.write_text(json.dumps({
            "schema": "labyrinth.blind-mechanics-responses.v1",
            "exam_id": private["exam_id"], "responses": items,
        }), encoding="utf-8")

    def score(self, plan):
        return score_exam(self.private, self.public, self.commit, self.responses,
                          self.out, expected_precommit_sha256=plan["receipt_sha256"])

    def test_public_prompt_omits_witness_and_private_material(self):
        private, public, plan = self.prepare()
        self.assertEqual(len(public["trials"]), 8)
        self.assertEqual(len({case["trial_id"] for case in public["trials"]}), 8)
        for source, advertised in zip(private["trials"], public["trials"]):
            self.assertEqual(source["trial_id"], advertised["trial_id"])
            self.assertEqual(source["world"], advertised["world"])
            self.assertNotIn("oracle_witness", advertised)
            self.assertNotIn("satisfiable", advertised)
            self.assertNotIn("ANSWER:", advertised["prompt"])
            self.assertIn("PLAN:", advertised["prompt"])
            self.assertIn("phase", advertised["prompt"].lower())
            self.assertIn("action_tables", advertised["prompt"])
        self.assertNotIn("oracle_witness", json.dumps(public))
        self.assertNotIn("satisfiable", json.dumps(public))
        self.assertFalse(plan["independent_custody_verified"])

    def test_exact_oracle_action_plans_score_but_do_not_prove_model_reasoning(self):
        private, public, plan = self.prepare()
        self.responses_for(private, use_oracle=True)
        result = self.score(plan)
        self.assertEqual(result["correct"], 8)
        self.assertEqual(result["case_count"], 8)
        self.assertTrue(result["dual_oracles_agree"])
        self.assertFalse(result["neural_learning_verified"])
        self.assertFalse(result["independent_custody_verified"])
        self.assertEqual(result["score_evidence"], "SOURCE_ONLY_LOCAL_REPLAY")
        self.assertEqual(json.loads(self.out.read_text())["receipt_sha256"],
                         result["receipt_sha256"])

    def test_zero_update_all_unsat_control_fails_half_the_new_mechanics(self):
        private, _, plan = self.prepare()
        self.responses_for(private, use_oracle=False)
        result = self.score(plan)
        self.assertEqual(sum(t["oracle_witness"] is not None
                             for t in private["trials"]), 4)
        self.assertEqual(sum(t["oracle_witness"] is None
                             for t in private["trials"]), 4)
        self.assertEqual(result["correct"], 4)
        self.assertEqual(result["verified_plans"], 0)
        self.assertEqual(result["verified_unsat"], 4)

    def test_mutated_public_rule_table_fails_original_source_commit(self):
        private, public, plan = self.prepare()
        self.responses_for(private, use_oracle=True)
        public["trials"][0]["world"]["action_tables"]["A"][0][0] = -1
        self.public.write_text(json.dumps(public), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "public|commit|source|mechanic"):
            self.score(plan)
        self.assertFalse(self.out.exists())

    def test_mutated_private_oracle_witness_fails_even_if_source_unaltered(self):
        private, public, plan = self.prepare()
        self.responses_for(private, use_oracle=True)
        private["trials"][0]["oracle_witness"] = ["A", "A", "A"]
        self.private.write_text(json.dumps(private), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "private|commit|source"):
            self.score(plan)

    def test_wrong_original_pin_rejects_full_consistent_reissued_exam(self):
        _, _, original_plan = self.prepare()
        second = self.root / "reissued"
        second.mkdir()
        p2, u2, c2 = second / "private.json", second / "public.json", second / "commit.json"
        prepare_exam(p2, u2, c2, count=8, rng=random.Random(999))
        replacement = json.loads(c2.read_text())
        resp2 = second / "responses.json"
        private2 = json.loads(p2.read_text())
        resp2.write_text(json.dumps({
            "schema": "labyrinth.blind-mechanics-responses.v1",
            "exam_id": private2["exam_id"],
            "responses": [{"trial_id": t["trial_id"], "response": "UNSAT"}
                          for t in private2["trials"]],
        }), encoding="utf-8")
        self.assertNotEqual(replacement["receipt_sha256"], original_plan["receipt_sha256"])
        with self.assertRaisesRegex(ValueError, "precommit|pin|expected"):
            score_exam(p2, u2, c2, resp2, self.out,
                       expected_precommit_sha256=original_plan["receipt_sha256"])

    def test_duplicate_response_identity_rejected_without_output(self):
        private, _, plan = self.prepare()
        self.responses_for(private, use_oracle=True)
        content = json.loads(self.responses.read_text())
        content["responses"][1]["trial_id"] = content["responses"][0]["trial_id"]
        self.responses.write_text(json.dumps(content), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "response|roster|duplicate"):
            self.score(plan)
        self.assertFalse(self.out.exists())

    def test_wrong_plan_does_not_become_success_due_to_declared_verdict(self):
        private, _, plan = self.prepare()
        self.responses_for(private, use_oracle=False)
        data = json.loads(self.responses.read_text())
        winning_index = next(i for i, t in enumerate(private["trials"])
                             if t["oracle_witness"] is not None)
        data["responses"][winning_index]["response"] = "PLAN: NEVER_DEFINED"
        self.responses.write_text(json.dumps(data), encoding="utf-8")
        result = self.score(plan)
        self.assertEqual(result["correct"], 4)
        self.assertEqual(result["verified_plans"], 0)

    def test_signed_precommit_authenticates_key_not_independent_case_custody(self):
        private, _, plan = self.prepare()
        self.responses_for(private, use_oracle=True)
        signer = Ed25519PrivateKey.generate()
        public = signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        key_file = self.root / "signer.pub"
        key_file.write_bytes(public)
        key_hash = sha256(public).hexdigest()
        payload = {
            "schema": "labyrinth.examiner-signed-precommit-payload.v1",
            "exam_id": private["exam_id"],
            "precommit_sha256": plan["receipt_sha256"],
            "examiner_key_sha256": key_hash,
            "scope": "EXAMINER_AUTHORSHIP_OF_PRECOMMIT_ONLY",
        }
        signature = base64.b64encode(signer.sign(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8"))).decode("ascii")
        envelope = self.root / "signed.json"
        envelope.write_text(json.dumps({
            "schema": "labyrinth.examiner-signed-precommit.v1",
            "payload": payload, "signature_ed25519_b64": signature,
        }), encoding="utf-8")
        result = score_exam(
            self.private, self.public, self.commit, self.responses, self.out,
            expected_precommit_sha256=plan["receipt_sha256"],
            examiner_attestation_path=envelope, examiner_public_key_path=key_file,
            expected_examiner_key_sha256=key_hash, examiner_exam_id=private["exam_id"])
        self.assertEqual(result["examiner_signature_evidence"],
                         "ED25519_VALID_CALLER_PINNED_KEY")
        self.assertFalse(result["independent_custody_verified"])
        self.assertEqual(result["correct"], 8)

    def test_public_cli_prepares_and_scores_caller_supplied_answers(self):
        output = StringIO()
        with redirect_stdout(output):
            main(["prepare", "--out-private", str(self.private),
                  "--out-public", str(self.public),
                  "--out-precommit", str(self.commit), "--count", "4"])
        plan = json.loads(self.commit.read_text(encoding="utf-8"))
        private = json.loads(self.private.read_text(encoding="utf-8"))
        self.assertIn(plan["receipt_sha256"], output.getvalue())
        self.responses_for(private, use_oracle=True)
        output = StringIO()
        with redirect_stdout(output):
            main(["score", "--private", str(self.private),
                  "--public", str(self.public), "--precommit", str(self.commit),
                  "--responses", str(self.responses),
                  "--expected-precommit-sha256", plan["receipt_sha256"],
                  "--out", str(self.out)])
        score = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(score["correct"], 4)
        self.assertFalse(score["neural_learning_verified"])
        self.assertIn(score["receipt_sha256"], output.getvalue())

    def test_independently_authored_mechanics_table_can_be_admitted(self):
        # An externally authored table is accepted without hardcoding its
        # transitions in source. Its authorship is NOT authenticated here.
        n = 6
        table = {action: [[-1] * n, [-1] * n] for action in ("A", "B", "C")}
        table["C"][0][0] = 5
        sat = {"state_count": n, "start": 0, "target": 5, "max_moves": 2,
               "action_tables": table}
        unsat = deepcopy(sat)
        unsat["action_tables"]["C"][0][0] = -1
        private, _, plan = self.prepare(count=2, worlds=[sat, unsat])
        self.assertIsNotNone(private["trials"][0]["oracle_witness"])
        self.assertIsNone(private["trials"][1]["oracle_witness"])
        self.responses_for(private, use_oracle=True)
        self.assertEqual(self.score(plan)["correct"], 2)


if __name__ == "__main__":
    unittest.main()
