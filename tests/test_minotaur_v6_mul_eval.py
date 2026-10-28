"""Independent V6 multiplication evaluation fixtures (no generator oracle)."""
import unittest

from labyrinth.minotaur_v6_mul_eval import (
    MulEvaluationCase,
    MulProgramRolloutCase,
    evaluate_mul_predictions,
    evaluate_mul_program_rollouts,
)


class MinotaurV6MulEvalTests(unittest.TestCase):
    def test_scores_by_withheld_group_and_audits_reversed_aliases(self):
        cases = [
            MulEvaluationCase("held-a", 3, 5, 15, "value-3"),
            MulEvaluationCase("held-b", 0, 7, 0, "zero"),
        ]
        report = evaluate_mul_predictions(
            cases, {"held-a": 15, "held-b": None},
            training_pairs=((5, 3),), training_group_ids=("train",),
        )
        self.assertEqual(report["disposition"], "SCORED")
        self.assertEqual(report["metrics"]["exact_accuracy"], 0.5)
        self.assertEqual(report["withheld_groups"]["value-3"]["exact_accuracy"], 1.0)
        self.assertEqual(report["withheld_groups"]["zero"]["abstentions"], 1)
        audit = report["overlap_audit"]
        self.assertEqual(audit["exact_training_pair_overlap"], [])
        self.assertEqual(audit["reversed_pair_alias_overlap"], [[3, 5]])
        self.assertEqual(audit["withheld_group_overlap"], [])
        self.assertIn("always_constant", report["negative_baselines"])
        self.assertIn("copy_left", report["negative_baselines"])
        self.assertIn("copy_right", report["negative_baselines"])

    def test_refuses_forged_labels_and_all_malformed_prediction_forms(self):
        forged = MulEvaluationCase("forged", 4, 4, 17, "g")
        self.assertEqual(evaluate_mul_predictions([forged], {"forged": 16})["disposition"], "REFUSED")
        good = MulEvaluationCase("good", 4, 4, 16, "g")
        for predictions in ({"good": "16"}, {"good": 97}, {}, {"good": 16, "extra": 0}):
            self.assertEqual(evaluate_mul_predictions([good], predictions)["disposition"], "REFUSED")

    def test_rollout_callback_receives_only_current_multiplication_operands(self):
        clean = MulProgramRolloutCase(
            "clean", 2, (3, 4), (6, 24), None, "composition-a",
        )
        corrupted = MulProgramRolloutCase(
            "corrupt", 2, (3, 4), (6, 23), 2, "composition-b",
        )
        calls = []

        def callback(left, right):
            calls.append((left, right))
            return (left * right) % 97

        report = evaluate_mul_program_rollouts([clean, corrupted], callback)
        self.assertEqual(report["disposition"], "SCORED")
        self.assertEqual(calls, [(2, 3), (6, 4), (2, 3), (6, 4)])
        self.assertEqual(report["metrics"]["exact_first_error_accuracy"], 1.0)
        self.assertEqual(report["trace_composition"], {"clean": 1, "corrupted": 1})

    def test_rollout_refuses_bad_claims_and_bad_callback_outputs(self):
        bad_claim = MulProgramRolloutCase("bad", 2, (3,), (6,), 1)
        self.assertEqual(evaluate_mul_program_rollouts([bad_claim], lambda x, y: x * y)["disposition"], "REFUSED")
        good = MulProgramRolloutCase("good", 2, (3,), (6,), None)
        report = evaluate_mul_program_rollouts([good], lambda x, y: True)
        self.assertEqual(report["disposition"], "REFUSED")
        self.assertEqual(report["audit"]["failures"][0]["reason"], "invalid_model_multiplication_output")


if __name__ == "__main__":
    unittest.main()
