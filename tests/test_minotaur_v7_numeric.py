"""TDD contracts for the V7 numeric residue experiment."""

import inspect
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch belongs to neural CI")
class MinotaurV7NumericTests(unittest.TestCase):
    def _module(self):
        import labyrinth.minotaur_v7_numeric as numeric
        return numeric

    def test_split_removes_target_and_operand_53_with_reversal_groups(self):
        numeric = self._module()
        split = numeric.build_scientific_split(modulus=97, seed=17, holdout_fraction=0.10)
        train = set(zip(split.train.x.tolist(), split.train.y.tolist()))
        self.assertEqual(len(train), len(split.train))
        self.assertTrue(all(x != 53 and y != 53 for x, y in train))
        self.assertTrue(all(t != 53 for t in split.train.target.tolist()))
        for x, y in train:
            self.assertIn((y, x), train)
        self.assertGreater(len(split.target_53), 0)
        self.assertTrue(all(t == 53 for t in split.target_53.target.tolist()))
        self.assertGreater(len(split.input_53), 0)
        self.assertTrue(all(x == 53 or y == 53
                            for x, y in zip(split.input_53.x.tolist(), split.input_53.y.tolist())))
        independent = set(zip(split.independent_holdout.x.tolist(),
                              split.independent_holdout.y.tolist()))
        self.assertFalse(train & independent)
        for x, y in independent:
            self.assertIn((y, x), independent)

    def test_numeric_model_is_shared_scalar_and_does_not_call_label_teacher(self):
        numeric = self._module()
        model = numeric.build_numeric_model(numeric.NumericConfig(seed=4, width=24, depth=2))
        self.assertNotIn("target", inspect.signature(model.forward).parameters)
        self.assertLess(sum(p.numel() for p in model.parameters()), 20_000)
        x = torch.tensor([0, 1, 53, 96], dtype=torch.long)
        y = torch.tensor([96, 33, 1, 4], dtype=torch.long)
        modulus = torch.full_like(x, 97)
        with patch.object(numeric, "multiplication_targets", side_effect=RuntimeError):
            prediction = model(x, y, modulus)
        self.assertEqual(prediction.shape, (4,))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.equal(prediction, model(y, x, modulus)))

    def test_two_seed_study_includes_controls_and_ood_101_without_claiming_success(self):
        numeric = self._module()
        report = numeric.run_study(numeric.NumericConfig(
            seeds=(5, 6), width=20, depth=2, epochs=1, batch_size=1024,
            max_optimizer_seconds=8.0, holdout_fraction=0.05,
        ))
        self.assertEqual(report["schema"], "labyrinth.minotaur.v7-numeric.v1")
        self.assertEqual(len(report["runs"]), 2)
        for run in report["runs"]:
            self.assertEqual(set(run["controls"]), {"constant", "frozen_random"})
            self.assertEqual(run["split"]["target_53_train_rows"], 0)
            self.assertEqual(run["split"]["input_53_train_rows"], 0)
            self.assertIn("target_53", run["evaluation"])
            self.assertIn("modulus_101", run["evaluation"])
            self.assertGreater(run["training"]["optimizer_steps"], 0)
            self.assertLessEqual(run["training"]["optimizer_seconds"], 8.0)
            self.assertIn(run["conclusion"], {
                "target_53_improves_over_controls", "no_target_53_improvement_over_controls",
            })


if __name__ == "__main__":
    unittest.main()
