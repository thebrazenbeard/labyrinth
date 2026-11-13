"""Falsify source-replayed claims when the precommit is self-rehashed or replaced.

Public synthetic fixtures only. A literal expected SHA pin is useful only
when preserved by a separate custodian; caller input is not attestation.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.live_model_probe import commit_cases, compare_reports
from test_live_model_probe_receipt_integrity import _observation, _receipt


class PrecommitCustodyHostileTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cases = self.root / "cases.json"
        self.precommit_path = self.root / "precommit.json"
        self.before = self.root / "before.json"
        self.after = self.root / "after.json"
        self.out = self.root / "paired.json"
        self.prompt = "Compute 2+2. Output ANSWER: followed by the integer only."
        self.cases.write_text(json.dumps({
            "schema": "labyrinth.live-probe-cases.v1",
            "cases": [{"case_id": "trial-1", "prompt": self.prompt, "answer": "4"}],
        }), encoding="utf-8")
        self.plan = commit_cases(self.cases, self.precommit_path)

    def _two_reports(self, *, plan=None, text="ANSWER: 4", correct=True):
        plan = self.plan if plan is None else plan
        value = _observation()
        value["cases_file_sha256"] = plan["cases_file_sha256"]
        value["commitment_sha256"] = plan["receipt_sha256"]
        value["trials"][0]["prompt_sha256"] = sha256(
            self.prompt.encode("utf-8")).hexdigest()
        value["trials"][0]["response"] = text
        value["trials"][0]["response_sha256"] = sha256(
            text.encode("utf-8")).hexdigest()
        value["trials"][0]["correct"] = correct
        value["correct"] = int(correct)
        self.before.write_text(json.dumps(_receipt(value)), encoding="utf-8")
        self.after.write_text(json.dumps(_receipt(value)), encoding="utf-8")

    def _rewrite_precommit(self, change):
        data = json.loads(self.precommit_path.read_text(encoding="utf-8"))
        data = change(data)
        data = _receipt(data)
        self.precommit_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def test_self_rehashed_independent_custody_claim_not_admitted(self):
        forged = self._rewrite_precommit(
            lambda p: {**p, "custody_class": "INDEPENDENT_EVALUATOR_VERIFIED"})
        self._two_reports(plan=forged)
        with self.assertRaisesRegex(ValueError, "precommit|custody|commitment"):
            compare_reports(self.before, self.after, self.out,
                            cases_path=self.cases, commitment_path=self.precommit_path)
        self.assertFalse(self.out.exists())

    def test_self_rehashed_independent_custody_field_not_admitted(self):
        forged = self._rewrite_precommit(
            lambda p: {**p, "independent_custody_verified": True})
        self._two_reports(plan=forged)
        with self.assertRaisesRegex(ValueError, "precommit|custody|commitment"):
            compare_reports(self.before, self.after, self.out,
                            cases_path=self.cases, commitment_path=self.precommit_path)

    def test_original_external_precommit_digest_catches_full_case_reissue(self):
        separately_saved_original_sha = self.plan["receipt_sha256"]
        self.cases.write_text(json.dumps({
            "schema": "labyrinth.live-probe-cases.v1",
            "cases": [{"case_id": "trial-1", "prompt": self.prompt, "answer": "9"}],
        }), encoding="utf-8")
        replacement_path = self.root / "replacement-precommit.json"
        replacement = commit_cases(self.cases, replacement_path)
        self._two_reports(plan=replacement, text="ANSWER: 9")
        # All source hashes, observation receipts and the wrong new answer
        # are now internally consistent. Only the separately retained old pin
        # can detect the replaced original benchmark.
        with self.assertRaisesRegex(ValueError, "pin|expected|commitment"):
            compare_reports(self.before, self.after, self.out,
                            cases_path=self.cases, commitment_path=replacement_path,
                            expected_commitment_sha256=separately_saved_original_sha)
        self.assertFalse(self.out.exists())

    def test_correct_pin_allows_only_caller_source_replay(self):
        self._two_reports()
        compared = compare_reports(
            self.before, self.after, self.out,
            cases_path=self.cases, commitment_path=self.precommit_path,
            expected_commitment_sha256=self.plan["receipt_sha256"])
        self.assertTrue(compared["precommit_pin_match"])
        self.assertEqual(compared["precommit_pin_evidence"], "CALLER_SUPPLIED_SHA256")
        self.assertEqual(compared["score_evidence"], "REPLAYED_CALLER_SUPPLIED_CASES")
        self.assertFalse(compared["independent_custody_verified"])

    def test_pin_without_source_or_invalid_digest_fails_closed(self):
        self._two_reports()
        with self.assertRaisesRegex(ValueError, "pin|cases|commitment"):
            compare_reports(self.before, self.after, self.out,
                            expected_commitment_sha256=self.plan["receipt_sha256"])
        with self.assertRaisesRegex(ValueError, "pin|digest"):
            compare_reports(self.before, self.after, self.out,
                            cases_path=self.cases, commitment_path=self.precommit_path,
                            expected_commitment_sha256="this-is-not-a-sha256")
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
