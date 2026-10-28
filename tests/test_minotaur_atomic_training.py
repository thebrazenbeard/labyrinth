"""Finite-domain learned-atom qualification checks (not private reasoning evidence)."""
import unittest

from labyrinth.minotaur_atomic_training import (
    AtomicTrainingConfig, atomic_supervision, atomic_train_holdout,
)


class AtomicTrainingTests(unittest.TestCase):
    def test_atomic_supervision_is_complete_and_grounded_in_labels(self):
        entries = atomic_supervision()
        self.assertEqual(len(entries), 3 * 97 * 97)
        self.assertEqual(len({key for key, target in entries}), len(entries))
        self.assertIn((0, 0), entries)
        add = 1 * 97 + 2
        self.assertIn((add, 3), entries)
        sub = 97 * 97 + 1 * 97 + 2
        self.assertIn((sub, 96), entries)
        mul = 2 * 97 * 97 + 8 * 97 + 12
        self.assertIn((mul, 96), entries)

    def test_train_holdout_is_disjoint_and_deterministic(self):
        train, heldout = atomic_train_holdout(20261027)
        train2, heldout2 = atomic_train_holdout(20261027)
        self.assertEqual((train, heldout), (train2, heldout2))
        self.assertFalse({x[0] for x in train} & {x[0] for x in heldout})
        self.assertEqual(len(train) + len(heldout), 3 * 97 * 97)
        self.assertGreater(len(heldout), 2000)

    def test_configs_are_bounded(self):
        AtomicTrainingConfig(seed=2).validate()
        with self.assertRaises(ValueError):
            AtomicTrainingConfig(seed=True).validate()
        with self.assertRaises(ValueError):
            AtomicTrainingConfig(lr=.0).validate()

if __name__ == "__main__":
    unittest.main()
