"""Contracts for the bounded modulo-97 spectral multiplication study."""

import inspect
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - exercised in neural CI
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch belongs to neural CI")
class MinotaurV6SpectralTests(unittest.TestCase):
    def _module(self):
        import labyrinth.minotaur_v6_spectral as spectral

        return spectral

    def test_grouped_split_is_deterministic_incomplete_and_reversal_safe(self):
        spectral = self._module()
        first = spectral.build_grouped_split(seed=20261028, holdout_fraction=0.20)
        second = spectral.build_grouped_split(seed=20261028, holdout_fraction=0.20)
        for name in ("train", "holdout"):
            self.assertTrue(torch.equal(first[name].x, second[name].x))
            self.assertTrue(torch.equal(first[name].y, second[name].y))
            self.assertTrue(torch.equal(first[name].target, second[name].target))

        train_pairs = set(zip(first["train"].x.tolist(), first["train"].y.tolist()))
        held_pairs = set(zip(first["holdout"].x.tolist(), first["holdout"].y.tolist()))
        self.assertEqual(len(train_pairs) + len(held_pairs), 97 * 97)
        self.assertFalse(train_pairs & held_pairs)
        self.assertTrue(0.19 <= len(held_pairs) / (97 * 97) <= 0.21)
        for x, y in train_pairs:
            self.assertIn((y, x), train_pairs)
        for x, y in held_pairs:
            self.assertIn((y, x), held_pairs)

    def test_model_is_compact_shared_commutative_and_oracle_free(self):
        spectral = self._module()
        model = spectral.build_spectral_model(
            spectral.SpectralConfig(seed=7, rank=128, epochs=1)
        )
        self.assertLess(sum(p.numel() for p in model.parameters()), 150_000)
        self.assertIs(model.left_embedding, model.right_embedding)
        self.assertNotIn("target", inspect.signature(model.forward).parameters)
        for name, parameter in model.named_parameters():
            self.assertNotIn(97 * 97, parameter.shape, name)

        x = torch.tensor([0, 1, 13, 96], dtype=torch.long)
        y = torch.tensor([96, 72, 65, 4], dtype=torch.long)
        with patch.object(spectral, "multiplication_targets", side_effect=RuntimeError):
            direct = model(x, y)
            reversed_logits = model(y, x)
        self.assertEqual(direct.shape, (4, 97))
        self.assertTrue(torch.equal(direct, reversed_logits))

    def test_tiny_experiment_trains_and_reports_frozen_and_constant_controls(self):
        spectral = self._module()
        report = spectral.run_experiment(spectral.SpectralConfig(
            seed=11,
            rank=24,
            epochs=2,
            learning_rate=0.02,
            max_optimizer_seconds=10.0,
        ))
        self.assertEqual(report["split"]["method"], "grouped_unordered_pair_80_20")
        self.assertEqual(set(report["controls"]), {"constant", "frozen_random"})
        self.assertGreater(report["training"]["optimizer_steps"], 0)
        self.assertGreater(report["training"]["gradient_norm_max"], 0.0)
        self.assertNotEqual(
            report["training"]["initial_weight_sha256"],
            report["training"]["trained_weight_sha256"],
        )
        self.assertLessEqual(report["training"]["optimizer_seconds"], 10.0)
        for arm in (report["candidate"], *report["controls"].values()):
            self.assertEqual(set(arm), {"train", "holdout"})
            for metrics in arm.values():
                self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
                self.assertEqual(metrics["count"], int(metrics["count"]))

    def test_config_rejects_unbounded_or_invalid_runs(self):
        spectral = self._module()
        for config in (
            spectral.SpectralConfig(rank=0),
            spectral.SpectralConfig(epochs=0),
            spectral.SpectralConfig(learning_rate=0.0),
            spectral.SpectralConfig(max_optimizer_seconds=90.1),
            spectral.SpectralConfig(rank=800),
        ):
            with self.assertRaises((TypeError, ValueError)):
                config.validate()


if __name__ == "__main__":
    unittest.main()
