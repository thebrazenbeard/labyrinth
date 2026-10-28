"""Contracts for bounded combinatorial Minotaur V2 training."""
from dataclasses import replace
import math
import unittest

from labyrinth.minotaur_combinatorial_curriculum import (
    CurriculumConfig,
    build_combinatorial_splits,
)
from labyrinth.minotaur_combinatorial_training import (
    CombinatorialTrainingConfig,
    build_training_splits,
    mask_invalid_step_logits,
    visible_tokens,
)
from labyrinth.minotaur_independent_eval import independently_label


class CombinatorialTrainingDataTests(unittest.TestCase):
    def test_new_curriculum_is_independently_replayable(self):
        splits = build_combinatorial_splits(CurriculumConfig(
            seed=91, train_cases=12, dev_cases=10, transfer_cases=10))
        for split in splits.values():
            for case in split.cases:
                replay = independently_label(case.question, case.steps)
                self.assertEqual(replay["reason"], "independently_replayed")
                self.assertEqual(replay["first_error_step"], case.first_error_step)

    def test_training_uses_balanced_disjoint_variable_length_curriculum(self):
        config = CombinatorialTrainingConfig(
            seeds=(7, 11), train_cases=128, dev_cases=64, transfer_cases=64,
            epochs=1)
        splits = build_training_splits(config)
        occupied = set()
        for name, split in splits.items():
            self.assertEqual(sum(c.first_error_step is None for c in split.cases),
                             len(split.cases) // 2)
            structures = {c.canonical_sequence for c in split.cases}
            self.assertTrue(occupied.isdisjoint(structures), name)
            occupied.update(structures)
        self.assertEqual({len(c.steps) for c in splits["train"].cases}, set(range(1, 7)))

    def test_labels_are_not_part_of_visible_model_tokens(self):
        case = build_training_splits(CombinatorialTrainingConfig(
            seeds=(7, 11), train_cases=128, dev_cases=64,
            transfer_cases=64, epochs=1))["train"].cases[1]
        changed = replace(case, first_error_step=None if case.first_error_step else 1)
        self.assertEqual(visible_tokens(case), visible_tokens(changed))


class CombinatorialTrainingMaskTests(unittest.TestCase):
    def test_every_nonexistent_step_logit_is_negative_infinity(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional CPU PyTorch is absent from this interpreter")

        logits = torch.zeros((3, 6))
        masked = mask_invalid_step_logits(logits, torch.tensor([1, 3, 6]), torch)
        for row, length in enumerate((1, 3, 6)):
            self.assertTrue(all(math.isfinite(value) for value in masked[row, :length].tolist()))
            self.assertTrue(all(value == -math.inf for value in masked[row, length:].tolist()))
        self.assertTrue(torch.equal(logits, torch.zeros((3, 6))))

    def test_protocol_bounds_are_enforced(self):
        invalid = (
            {"seeds": (1,)}, {"seeds": (1, 1)}, {"train_cases": 126},
            {"dev_cases": 62}, {"transfer_cases": 130}, {"epochs": 31},
            {"max_optimizer_seconds": 90.1},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CombinatorialTrainingConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
