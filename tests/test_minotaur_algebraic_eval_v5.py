"""Tests for the standalone V5 algebraic evaluator; fixtures contain no generator oracle."""
import unittest

from labyrinth.minotaur_algebraic_eval_v5 import (
    AtomicEvaluationCase,
    ProgramRolloutCase,
    evaluate_atomic_predictions,
    evaluate_program_rollouts,
)


class AlgebraicEvalV5Tests(unittest.TestCase):
    def test_atomic_scores_recomputed_labels_subgroups_and_controls(self):
        cases = [
            AtomicEvaluationCase("add-zero", "ADD", 0, 4, 4, "g1"),
            AtomicEvaluationCase("sub-diagonal", "SUB", 7, 7, 0, "g1"),
            AtomicEvaluationCase("mul-pair", "MUL", 3, 5, 15, "g2"),
        ]
        report = evaluate_atomic_predictions(
            cases, {"add-zero": 4, "sub-diagonal": None, "mul-pair": 14},
            training_atoms=(("ADD", 1, 1), ("MUL", 3, 5)),
            training_group_ids=("train-group",),
        )
        self.assertEqual(report["disposition"], "SCORED")
        self.assertEqual(report["metrics"]["exact_correct"], 1)
        self.assertEqual(report["per_operation"]["ADD"]["exact_accuracy"], 1.0)
        self.assertEqual(report["subgroups"]["diagonal"]["abstentions"], 1)
        self.assertEqual(report["subgroups"]["unseen_pair"]["total"], 2)
        self.assertEqual(report["structural_overlap_audit"]["exact_training_pair_overlap"], [["MUL", 3, 5]])
        self.assertIn("always_constant", report["baselines"])
        self.assertIn("copy_x", report["baselines"])
        self.assertEqual(report["confidence_intervals"], "not_computed_grouped_or_paired_dependence")

    def test_atomic_refuses_mislabels_and_prediction_id_mismatches(self):
        mislabeled = AtomicEvaluationCase("a", "ADD", 2, 2, 5)
        refused = evaluate_atomic_predictions([mislabeled], {"a": 4})
        self.assertEqual(refused["disposition"], "REFUSED")
        self.assertEqual(refused["audit"]["failures"][0]["reason"], "claimed_output_disagrees_with_recomputed_arithmetic")
        good = AtomicEvaluationCase("a", "ADD", 2, 2, 4)
        self.assertEqual(evaluate_atomic_predictions([good], {"wrong": 4})["disposition"], "REFUSED")

    def test_rollout_independently_scores_clean_corrupted_and_abstention(self):
        clean = ProgramRolloutCase(
            "clean", (2, 3), (("ADD", "A", "B"), ("MUL", "B", "A")),
            ((5, 3), (5, 15)), None, "pair-1",
        )
        corrupted = ProgramRolloutCase(
            "corrupt", (2, 3), (("ADD", "A", "B"), ("MUL", "B", "A")),
            ((5, 3), (5, 16)), 2, "pair-1",
        )
        def model(op, x, y):
            return {"ADD": (x + y) % 97, "SUB": (x - y) % 97, "MUL": (x * y) % 97}[op]
        report = evaluate_program_rollouts([clean, corrupted], model)
        self.assertEqual(report["disposition"], "SCORED")
        self.assertEqual(report["metrics"]["exact_first_error_correct"], 2)
        self.assertEqual(report["trace_composition"], {"clean": 1, "corrupted": 1})
        self.assertEqual(report["metrics"]["false_approvals"], 0)
        abstained = evaluate_program_rollouts([clean, corrupted], lambda op, x, y: None)
        self.assertEqual(abstained["metrics"]["abstentions"], 2)

    def test_rollout_refuses_bad_claim_or_callback_output(self):
        bad = ProgramRolloutCase("bad", (1, 1), (("ADD", "A", 1),), ((2, 1),), 1)
        self.assertEqual(evaluate_program_rollouts([bad], lambda op, x, y: (x + y) % 97)["disposition"], "REFUSED")
        good = ProgramRolloutCase("good", (1, 1), (("ADD", "A", 1),), ((2, 1),), None)
        report = evaluate_program_rollouts([good], lambda op, x, y: 97)
        self.assertEqual(report["disposition"], "REFUSED")
        self.assertEqual(report["audit"]["failures"][0]["reason"], "invalid_model_atomic_output")


if __name__ == "__main__":
    unittest.main()
