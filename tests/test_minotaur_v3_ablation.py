"""Contracts for the bounded Minotaur V3 scratch ablation sweep."""

import unittest

from labyrinth.minotaur_v3_ablation import (
    AblationConfig,
    VARIANTS,
    build_ablation_splits,
    choose_promotion,
    run_ablation,
)


class MinotaurV3AblationTests(unittest.TestCase):
    def test_split_identity_is_reproducible_and_holdouts_are_not_augmented(self):
        config = AblationConfig(seeds=(101, 202), train_programs=8,
                                eval_programs=4, epochs=1)
        first = build_ablation_splits(config)
        second = build_ablation_splits(config)
        self.assertEqual(first, second)
        train_ids = {case.case_id for case in first["train"]}
        for name in ("dev", "composition_transfer", "longer_transfer"):
            self.assertFalse(train_ids & {case.case_id for case in first[name]})
            self.assertEqual(len(first[name]), 8)
        self.assertGreater(len(first["train_augmented"]), len(first["train"]))
        self.assertEqual({case.case_id for case in first["dev"]},
                         {case.case_id for case in first["dev_evaluation_copy"]})

    def test_variants_cover_distinct_loss_and_data_ablation_families(self):
        self.assertEqual(set(VARIANTS), {
            "v2_joint", "weighted_correctness", "joint_label", "augmented_joint_label"
        })

    def test_promotion_requires_replicated_structural_transfer(self):
        weak = {"seed_results": [
            {"evaluations": {"longer_transfer": {"balanced_accuracy": .7}}},
            {"evaluations": {"longer_transfer": {"balanced_accuracy": .5}}},
        ]}
        self.assertEqual(choose_promotion({"candidate": weak}), "DO_NOT_PROMOTE")
        strong = {"seed_results": [
            {"evaluations": {"longer_transfer": {"balanced_accuracy": .7}}},
            {"evaluations": {"longer_transfer": {"balanced_accuracy": .65}}},
        ]}
        self.assertEqual(choose_promotion({"candidate": strong}), "STRUCTURAL_SIGNAL_ONLY")

    def test_tiny_run_performs_real_gradients_and_reports_controls(self):
        report = run_ablation(AblationConfig(
            seeds=(11, 12), train_programs=4, eval_programs=4,
            epochs=1, width=16, heads=2, batch_size=8,
            max_optimizer_seconds=10.0,
        ), variants=("weighted_correctness",))
        self.assertEqual(report["source"]["requested_base_commit"],
                         "d0263dbf6e431f0b8f3140b29ee72c140318265e")
        self.assertEqual(set(report["controls"]), {
            "always_approve", "always_reject_step_1", "frozen_random"
        })
        results = report["variants"]["weighted_correctness"]["seed_results"]
        self.assertEqual(len(results), 2)
        for result in results:
            training = result["training"]
            self.assertGreater(training["optimizer_steps"], 0)
            self.assertNotEqual(training["initial_weight_sha256"],
                                training["trained_weight_sha256"])
            self.assertGreater(training["gradient_norm_max"], 0.0)
            self.assertLessEqual(training["optimizer_seconds"], 10.0)
            self.assertEqual(set(result["evaluations"]), {
                "train", "dev", "composition_transfer", "longer_transfer"
            })

    def test_invalid_budget_and_duplicate_seeds_are_rejected(self):
        with self.assertRaises(ValueError):
            AblationConfig(seeds=(1, 1)).validate()
        with self.assertRaises(ValueError):
            AblationConfig(max_optimizer_seconds=90.1).validate()


if __name__ == "__main__":
    unittest.main()
