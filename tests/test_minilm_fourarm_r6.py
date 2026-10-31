"""Source-contract tests for MiniLM's truthful four-arm development pilot."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_minilm_cpu.py"
SPEC = importlib.util.spec_from_file_location("qualify_minilm_fourarm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MiniLMFourArmTests(unittest.TestCase):
    def test_four_distinct_physical_arms(self):
        self.assertEqual(MODULE.PHYSICAL_ARMS, (
            "baseline", "zero_update", "correct_labels", "corrupted_labels"))

    def test_baseline_has_no_training_or_optimizer_steps(self):
        baseline = MODULE.arm_contract("baseline", 0.002)
        self.assertFalse(baseline["trained"])
        self.assertEqual(baseline["learning_rate"], 0.0)
        self.assertFalse(baseline["optimizer_step"])

    def test_training_arms_have_matched_work_and_distinct_update_modes(self):
        controls = {arm: MODULE.arm_contract(arm, 0.002)
                    for arm in MODULE.PHYSICAL_ARMS[1:]}
        self.assertTrue(all(item["trained"] and item["optimizer_step"]
                            for item in controls.values()))
        self.assertEqual(controls["zero_update"]["learning_rate"], 0.0)
        self.assertEqual(controls["correct_labels"]["learning_rate"], 0.002)
        self.assertEqual(controls["corrupted_labels"]["learning_rate"], 0.002)
        self.assertEqual(controls["correct_labels"]["invert_labels"], False)
        self.assertEqual(controls["corrupted_labels"]["invert_labels"], True)

    def test_batch_manifest_counts_example_visits_separately_from_optimizer_steps(self):
        schedule = MODULE.training_batches(0, steps=5, batch_size=4)
        self.assertEqual(len(schedule), 5)
        self.assertEqual(sum(len(item["examples"]) for item in schedule), 20)
        self.assertEqual([item["step"] for item in schedule], list(range(5)))
        self.assertTrue(all(len(item["examples"]) == 4 for item in schedule))
        self.assertEqual(schedule[0]["examples"][0]["case_id"], "train-c0-r0")
        self.assertEqual(schedule[3]["examples"][0]["case_id"], "train-c0-r0")
        self.assertEqual(schedule[0]["examples"][0]["structure_key"],
                         "public-sentiment-cycle-0")
        self.assertNotIn("private_exam", schedule[0]["examples"][0])

    def test_corruption_changes_targets_not_exposure_identity_or_compute(self):
        correct = MODULE.training_batches(1, steps=8)
        corrupt = MODULE.training_batches(1, steps=8, invert_labels=True)
        self.assertEqual(len(correct), len(corrupt))
        for left, right in zip(correct, corrupt):
            self.assertEqual(left["step"], right["step"])
            for first, second in zip(left["examples"], right["examples"]):
                for field in ("case_id", "prompt_sha256", "structure_key"):
                    self.assertEqual(first[field], second[field])
                self.assertEqual(first["target_label"] ^ 1, second["target_label"])

    def test_unknown_arm_and_unsafe_schedule_fail_closed(self):
        with self.assertRaises(ValueError):
            MODULE.arm_contract("inverted", .002)
        with self.assertRaises(ValueError):
            MODULE.training_batches(0, steps=0)
        with self.assertRaises(ValueError):
            MODULE.training_batches(0, steps=33)

    def test_fresh_public_predictions_require_binary_output_and_all_rows(self):
        rows = MODULE.HELDOUT
        report = MODULE.prediction_records("heldout", rows, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(len(report), len(rows))
        self.assertEqual(report[0]["case_id"], "heldout-r0")
        self.assertEqual(report[0]["predicted_label"], 1)
        self.assertEqual(report[0]["expected_label"], 1)
        with self.assertRaises(ValueError):
            MODULE.prediction_records("heldout", rows, [1])
        with self.assertRaises(ValueError):
            MODULE.prediction_records("heldout", rows, [2] * len(rows))

    def test_comparison_uses_all_four_and_marks_public_baseline(self):
        result = MODULE.physical_comparison({"baseline": .45, "zero_update": .45,
                                             "correct_labels": .85,
                                             "corrupted_labels": .15})
        self.assertAlmostEqual(result["correct_minus_baseline"], .4)
        self.assertAlmostEqual(result["correct_minus_zero"], .4)
        self.assertAlmostEqual(result["correct_minus_corrupted"], .7)
        self.assertTrue(result["correct_outperforms_all_controls"])
        with self.assertRaises(ValueError):
            MODULE.physical_comparison({"baseline": .5})


if __name__ == "__main__":
    unittest.main()
