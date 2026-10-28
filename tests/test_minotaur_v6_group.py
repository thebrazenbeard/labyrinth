"""Contracts for the constrained V6 group-composition experiment."""
import inspect
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch belongs to neural CI")
class MinotaurV6GroupTests(unittest.TestCase):
    def _module(self):
        import labyrinth.minotaur_v6_group as group
        return group

    def test_grouped_split_is_reversal_safe_and_deterministic(self):
        group = self._module()
        first = group.build_grouped_split(seed=91)
        second = group.build_grouped_split(seed=91)
        for name in ("train", "holdout"):
            self.assertTrue(torch.equal(first[name].x, second[name].x))
        train = set(zip(first["train"].x.tolist(), first["train"].y.tolist()))
        held = set(zip(first["holdout"].x.tolist(), first["holdout"].y.tolist()))
        self.assertFalse(train & held)
        self.assertEqual(len(train) + len(held), group.MODULUS ** 2)
        self.assertTrue(all((y, x) in train for x, y in train))
        self.assertTrue(all((y, x) in held for x, y in held))

    def test_challenges_fully_remove_value_or_label_53_from_training(self):
        group = self._module()
        withheld = group.build_withheld_value_split()
        self.assertFalse(bool(((withheld["train"].x == 53) | (withheld["train"].y == 53)).any()))
        self.assertTrue(bool(((withheld["withheld_value"].x == 53) | (withheld["withheld_value"].y == 53)).all()))
        absent = group.build_absent_output_split()
        self.assertFalse(bool((absent["train"].target == 53).any()))
        self.assertTrue(bool((absent["absent_output"].target == 53).all()))

    def test_forward_is_compact_commutative_and_oracle_free(self):
        group = self._module()
        config = group.GroupConfig(dimension=24, epochs=1)
        model = group.build_group_model(config, 7)
        self.assertLess(sum(p.numel() for p in model.parameters()), 10_000)
        self.assertNotIn("target", inspect.signature(model.forward).parameters)
        x, y = torch.tensor([1, 53], dtype=torch.long), torch.tensor([2, 91], dtype=torch.long)
        with patch.object(group, "multiplication_targets", side_effect=RuntimeError("oracle used")):
            self.assertTrue(torch.equal(model(x, y), model(y, x)))

    def test_two_seed_bounded_experiment_reports_controls_and_negative_challenges(self):
        group = self._module()
        report = group.run_experiment(group.GroupConfig(dimension=8, epochs=2, max_optimizer_seconds=10.0))
        self.assertEqual(report["model"]["learned_group_axiom"], False)
        self.assertEqual(len(report["studies"]["grouped_pair_holdout"]), 2)
        self.assertIsInstance(report["negative_challenges"], list)
        for runs in report["studies"].values():
            for run in runs:
                self.assertEqual(set(run["controls"]), {"constant", "frozen_random"})
                self.assertGreater(run["training"]["optimizer_steps"], 0)
                self.assertNotEqual(run["training"]["initial_weight_sha256"], run["training"]["trained_weight_sha256"])

    def test_short_runs_are_reproducible_by_seed(self):
        group = self._module()
        config = group.GroupConfig(dimension=6, epochs=1, max_optimizer_seconds=10.0)
        first, second = group.run_experiment(config), group.run_experiment(config)
        for study in first["studies"]:
            self.assertEqual(
                [run["training"]["trained_weight_sha256"] for run in first["studies"][study]],
                [run["training"]["trained_weight_sha256"] for run in second["studies"][study]],
            )


if __name__ == "__main__":
    unittest.main()
