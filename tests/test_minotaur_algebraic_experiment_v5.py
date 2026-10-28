"""Source-first V5 incomplete-atom experiment contracts."""
import unittest
from labyrinth.minotaur_algebraic_experiment_v5 import (
    ExperimentConfig, checked_split_fingerprints,
)

class AlgebraicExperimentV5Tests(unittest.TestCase):
    def test_bounded_configuration(self):
        ExperimentConfig(seed=20261028).validate()
        for config in (ExperimentConfig(seed=True),
                       ExperimentConfig(train_fraction=.99),
                       ExperimentConfig(steps=0),
                       ExperimentConfig(batch_size=0)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_split_membership_and_op_mapping_are_stable(self):
        from labyrinth.minotaur_algebraic_data_v5 import atomic_split
        split=atomic_split(seed=11,train_fraction=.8)
        report=checked_split_fingerprints(split)
        self.assertEqual(report["atom_count"],28227)
        self.assertEqual(report["train_test_overlap"],0)
        self.assertEqual(report["dev_test_overlap"],0)
        self.assertEqual(set(report["sha256"]),{"train","val","holdout"})
