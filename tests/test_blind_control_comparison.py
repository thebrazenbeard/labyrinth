"""Matched controls must use the candidate's exact original exam and case order."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.blind_structural_exam import prepare_exam
from labyrinth.blind_control_comparison import compare_with_controls, _mcnemar_exact_two_sided


def world(steps, *, n=6):
    table = {action: [[-1] * n, [-1] * n] for action in "ABC"}
    for action, phase, state, next_state in steps:
        table[action][phase][state] = next_state
    return {"state_count": n, "start": 0, "target": n - 1,
            "max_moves": 3, "action_tables": table}


class SameExamControlTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.private = self.root / "exam-private.json"
        self.public = self.root / "exam-public.json"
        self.commit = self.root / "exam-precommit.json"
        self.answers = self.root / "candidate.json"
        self.out = self.root / "paired.json"
        sat1 = world([("C", 0, 0, 5)])
        unsat1 = world([("C", 0, 0, 2)])
        sat2 = world([("A", 0, 0, 3), ("B", 1, 3, 5)])
        unsat2 = world([("A", 0, 0, 3)])
        self.precommit = prepare_exam(
            self.private, self.public, self.commit,
            worlds=[sat1, unsat1, sat2, unsat2])
        self.trials = json.loads(self.private.read_text(encoding="utf-8"))["trials"]

    def record(self, replies):
        assert len(replies) == len(self.trials)
        exam_id = json.loads(self.public.read_text(encoding="utf-8"))["exam_id"]
        self.answers.write_text(json.dumps({
            "schema": "labyrinth.blind-mechanics-responses.v1",
            "exam_id": exam_id,
            "responses": [{"trial_id": row["trial_id"], "response": reply}
                          for row, reply in zip(self.trials, replies)],
        }), encoding="utf-8")

    def compare(self, **kwargs):
        return compare_with_controls(
            self.private, self.public, self.commit, self.answers, self.out,
            expected_precommit_sha256=self.precommit["receipt_sha256"], **kwargs)

    def test_mixed_candidate_is_paired_against_same_exact_four_worlds(self):
        self.record(["PLAN: A", "UNSAT", "PLAN: A B", "PLAN: A"])
        report = self.compare()
        self.assertEqual(report["candidate_correct"], 2)
        self.assertEqual(report["all_unsat_correct"], 2)
        self.assertEqual(report["fixed_abc_correct"], 0)
        self.assertEqual(report["all_unsat_paired"]["candidate_only"], 1)
        self.assertEqual(report["all_unsat_paired"]["control_only"], 1)
        self.assertEqual(report["all_unsat_paired"]["both_correct"], 1)
        self.assertEqual(report["all_unsat_paired"]["both_wrong"], 1)
        self.assertEqual(report["all_unsat_paired"]["mcnemar_exact_two_sided_p"], 1.0)
        self.assertEqual(report["case_count"], 4)
        self.assertEqual(report["invocation_evidence"], "NOT_SUPPLIED")
        self.assertFalse(report["neural_learning_verified"])
        self.assertFalse(report["independent_custody_verified"])

    def test_perfect_symbolic_candidate_is_not_proclaimed_generalization(self):
        self.record(["PLAN: C", "UNSAT", "PLAN: A B", "UNSAT"])
        report = self.compare()
        self.assertEqual(report["candidate_correct"], 4)
        self.assertEqual(report["all_unsat_correct"], 2)
        self.assertEqual(report["all_unsat_paired"]["candidate_only"], 2)
        self.assertEqual(report["all_unsat_paired"]["control_only"], 0)
        self.assertEqual(report["all_unsat_paired"]["mcnemar_exact_two_sided_p"], 0.5)
        self.assertEqual(report["claim_ceiling"], "NO_NEURAL_OR_GENERALIZATION_CERTIFICATION")
        self.assertFalse(report["neural_learning_verified"])

    def test_all_unsat_candidate_ties_all_unsat_control_exactly(self):
        self.record(["UNSAT"] * 4)
        report = self.compare()
        self.assertEqual(report["candidate_correct"], report["all_unsat_correct"])
        self.assertEqual(report["all_unsat_paired"]["candidate_only"], 0)
        self.assertEqual(report["all_unsat_paired"]["control_only"], 0)
        self.assertEqual(report["all_unsat_paired"]["mcnemar_exact_two_sided_p"], 1.0)

    def test_exact_mcnemar_keeps_small_sample_from_being_claimed_significant(self):
        self.assertEqual(_mcnemar_exact_two_sided(1, 6), 0.125)
        self.assertEqual(_mcnemar_exact_two_sided(0, 2), 0.5)
        self.assertEqual(_mcnemar_exact_two_sided(0, 0), 1.0)
        self.assertEqual(_mcnemar_exact_two_sided(1, 1), 1.0)

    def test_malformed_or_reordered_candidate_roster_cannot_be_compared(self):
        self.record(["PLAN: C", "UNSAT", "PLAN: A B", "UNSAT"])
        data = json.loads(self.answers.read_text(encoding="utf-8"))
        data["responses"][1]["trial_id"] = data["responses"][0]["trial_id"]
        self.answers.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "response|roster|trial"):
            self.compare()
        self.assertFalse(self.out.exists())

    def test_mismatched_original_case_pin_blocks_comparison(self):
        self.record(["PLAN: C", "UNSAT", "PLAN: A B", "UNSAT"])
        with self.assertRaisesRegex(ValueError, "pin|precommit"):
            compare_with_controls(
                self.private, self.public, self.commit, self.answers, self.out,
                expected_precommit_sha256="d" * 64)
        self.assertFalse(self.out.exists())

    def test_http_observation_arguments_must_be_complete(self):
        self.record(["PLAN: C", "UNSAT", "PLAN: A B", "UNSAT"])
        with self.assertRaisesRegex(ValueError, "observation|pin"):
            self.compare(observation_path=self.root / "witness.json")
        with self.assertRaisesRegex(ValueError, "observation|pin"):
            self.compare(expected_observation_sha256="a" * 64)
        self.assertFalse(self.out.exists())

    def test_write_once_prevents_replacing_original_paired_result(self):
        self.record(["PLAN: C", "UNSAT", "PLAN: A B", "UNSAT"])
        self.compare()
        with self.assertRaises(FileExistsError):
            self.compare()


if __name__ == "__main__":
    unittest.main()
