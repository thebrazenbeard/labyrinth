"""The comparator must not turn self-signed correctness into verified paired wins.

All data here is synthetic. Source bytes + unkeyed commitments can validate local
consistency, but they do not establish independent evaluator custody or that a
model ever produced the declared responses.
"""
from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.live_model_probe import commit_cases, compare_reports
from test_live_model_probe_receipt_integrity import _observation, _receipt


class CaseBoundComparisonTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cases = self.root / "held-by-local-evaluator.json"
        self.precommit = self.root / "cases-precommit.json"
        self.before = self.root / "before.json"
        self.after = self.root / "after.json"
        self.output = self.root / "comparison.json"
        # A question with no answer leaked into the prompt.
        self.prompt = "What is two plus two? Return only ANSWER: followed by the integer."
        self.cases.write_text(json.dumps({
            "schema": "labyrinth.live-probe-cases.v1",
            "cases": [{"case_id": "trial-1", "prompt": self.prompt, "answer": "4"}],
        }), encoding="utf-8")
        self.precommitted = commit_cases(self.cases, self.precommit)

    def report(self, answer, correct):
        report = copy.deepcopy(_observation())
        report["commitment_sha256"] = self.precommitted["receipt_sha256"]
        report["cases_file_sha256"] = self.precommitted["cases_file_sha256"]
        report["trials"][0]["prompt_sha256"] = sha256(self.prompt.encode("utf-8")).hexdigest()
        report["trials"][0]["response"] = "ANSWER: " + answer
        report["trials"][0]["response_sha256"] = sha256(
            report["trials"][0]["response"].encode("utf-8")).hexdigest()
        report["trials"][0]["correct"] = correct
        report["correct"] = int(correct)
        return _receipt(report)

    def store(self, before, after):
        self.before.write_text(json.dumps(before), encoding="utf-8")
        self.after.write_text(json.dumps(after), encoding="utf-8")

    def test_self_rehashed_forged_win_not_promoted_without_case_source(self):
        self.store(self.report("9", False), self.report("999", True))
        compared = compare_reports(self.before, self.after, self.output)
        # The old implementation announces a candidate-only win here, despite
        # never checking the underlying answer; this must remain unverified.
        self.assertIsNone(compared["paired_candidate_only_wins"])
        self.assertIsNone(compared["paired_candidate_only_losses"])
        self.assertEqual(compared["score_evidence"], "SELF_REPORTED_ONLY")
        self.assertFalse(compared["independent_custody_verified"])

    def test_source_bound_replay_rejects_self_consistent_false_correctness(self):
        self.store(self.report("9", False), self.report("999", True))
        with self.assertRaisesRegex(ValueError, "correct|answer|score"):
            compare_reports(self.before, self.after, self.output,
                            cases_path=self.cases, commitment_path=self.precommit)
        self.assertFalse(self.output.exists())

    def test_source_bound_replay_accepts_real_local_answer_but_not_training(self):
        self.store(self.report("9", False), self.report("4", True))
        compared = compare_reports(self.before, self.after, self.output,
                                   cases_path=self.cases, commitment_path=self.precommit)
        self.assertEqual(compared["score_evidence"], "REPLAYED_CALLER_SUPPLIED_CASES")
        self.assertEqual(compared["paired_candidate_only_wins"], 1)
        self.assertEqual(compared["paired_candidate_only_losses"], 0)
        self.assertFalse(compared["independent_custody_verified"])
        self.assertFalse(compared["neural_learning_verified"])

    def test_missing_half_of_source_binding_fails_closed(self):
        self.store(self.report("9", False), self.report("4", True))
        with self.assertRaisesRegex(ValueError, "cases|commitment"):
            compare_reports(self.before, self.after, self.output, cases_path=self.cases)
        with self.assertRaisesRegex(ValueError, "cases|commitment"):
            compare_reports(self.before, self.after, self.output, commitment_path=self.precommit)

    def test_rehashed_different_source_case_answer_cannot_replace_precommitted_answer(self):
        self.store(self.report("9", False), self.report("4", True))
        altered = json.loads(self.cases.read_text(encoding="utf-8"))
        altered["cases"][0]["answer"] = "9"
        self.cases.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source|commitment|case"):
            compare_reports(self.before, self.after, self.output,
                            cases_path=self.cases, commitment_path=self.precommit)

    def test_both_receipts_same_forged_prompt_digest_rejected_by_original_case_bytes(self):
        before = self.report("9", False)
        after = self.report("4", True)
        before["trials"][0]["prompt_sha256"] = "a" * 64
        after["trials"][0]["prompt_sha256"] = "a" * 64
        self.store(_receipt(before), _receipt(after))
        with self.assertRaisesRegex(ValueError, "prompt|source|case"):
            compare_reports(self.before, self.after, self.output,
                            cases_path=self.cases, commitment_path=self.precommit)


if __name__ == "__main__":
    unittest.main()
