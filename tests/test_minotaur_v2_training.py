"""Contracts for the bounded, synthetic Minotaur V2 CPU training experiment."""

import unittest

from labyrinth.minotaur_v2_training import (
    ExperimentConfig,
    build_splits,
    dataset_fingerprint,
    score_predictions,
)


class MinotaurV2DatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(train_programs=12, eval_programs=8, epochs=1)
        self.splits = build_splits(self.config)

    def test_splits_are_reproducible_balanced_disjoint_and_fingerprinted(self):
        again = build_splits(self.config)
        self.assertEqual(self.splits, again)
        fingerprints = {name: dataset_fingerprint(cases)
                        for name, cases in self.splits.items()}
        self.assertEqual(fingerprints, {name: dataset_fingerprint(again[name])
                                        for name in again})
        all_ids = set()
        for name, cases in self.splits.items():
            self.assertEqual(sum(case.first_error_step is None for case in cases),
                             len(cases) // 2)
            ids = {case.case_id for case in cases}
            self.assertFalse(all_ids & ids)
            all_ids |= ids
            self.assertEqual(len({case.structure_fingerprint for case in cases}), 4)
            self.assertGreaterEqual(len({case.corruption_family for case in cases
                                         if case.first_error_step is not None}), 3)

    def test_longer_transfer_is_structurally_reserved(self):
        train_structures = {case.structure_fingerprint for case in self.splits["train"]}
        longer = self.splits["longer_transfer"]
        self.assertFalse(train_structures & {case.structure_fingerprint for case in longer})
        self.assertTrue(all(len(case.steps) >= 5 for case in longer))
        self.assertTrue(all(len(case.steps) <= 4 for case in self.splits["train"]))


class MinotaurV2MetricTests(unittest.TestCase):
    def test_confusion_and_false_approval_are_explicit(self):
        cases = build_splits(ExperimentConfig(
            train_programs=4, eval_programs=4, epochs=1))["dev"]
        labels = [case.first_error_step or 0 for case in cases]
        perfect = score_predictions(cases, labels)
        self.assertEqual(perfect["accuracy"], 1.0)
        self.assertEqual(perfect["false_approval_rate"], 0.0)
        self.assertEqual(perfect["confusion"], {
            "actual_correct_predicted_correct": 4,
            "actual_correct_predicted_incorrect": 0,
            "actual_incorrect_predicted_correct": 0,
            "actual_incorrect_predicted_incorrect": 4,
        })
        approve = score_predictions(cases, [0] * len(cases))
        self.assertEqual(approve["false_approval_rate"], 1.0)
        self.assertEqual(approve["balanced_accuracy"], 0.5)
        reject = score_predictions(cases, [1] * len(cases))
        self.assertEqual(reject["false_rejection_rate"], 1.0)
        with self.assertRaises(ValueError):
            score_predictions(cases, [7] * len(cases))


if __name__ == "__main__":
    unittest.main()
