"""Bounded public probes for the polynomial-bias neural interaction."""

from __future__ import annotations

import unittest

from labyrinth.minotaur_transfer_next import (
    QuadraticInteraction, _score_shifted, experiment, train_quadratic,
)
from labyrinth.minotaur_neural_interaction import _np


class QuadraticTransferResearchTests(unittest.TestCase):
    def test_raw_operand_architecture_and_zero_update_control(self):
        np = _np()
        model = QuadraticInteraction(seed=41)
        self.assertEqual(model.parameters["w"].shape, (2, 8))
        self.assertEqual(len(model.weights()), 33)
        raw, hidden = model.forward(np.array([[0.25, 0.5]], dtype=np.float64))
        self.assertEqual(raw.shape, (1, 1))
        self.assertEqual(hidden[0].shape, (1, 2))
        before = model.weights()
        train_quadratic(model, [(0, 0), (2, 2)], [0, 4],
                        steps=15, learning_rate=0.0)
        np.testing.assert_array_equal(before, model.weights())

    def test_two_seed_transfer_and_representation_limit(self):
        for seed in (41, 42):
            with self.subTest(seed=seed):
                report = experiment(seed=seed, steps=3000)
                scores = report["scores"]
                trained = scores["trained"]
                coefficients = report["trained_posthoc_polynomial_coefficients"]
                self.assertEqual(report["heldout_operand_overlap"], 0)
                self.assertAlmostEqual(coefficients["xy"], 1.0, delta=1e-3)
                for name in ("x_squared", "y_squared", "x", "y", "constant"):
                    self.assertAlmostEqual(coefficients[name], 0.0, delta=1e-3)
                self.assertGreaterEqual(trained["heldout_odd"]["correct"], 95)
                self.assertGreaterEqual(trained["outside_21_32"]["correct"], 130)
                self.assertGreaterEqual(trained["negative_minus12_minus1"]["correct"], 130)
                # A separately supplied modulo projection changes the target
                # without proving the neural model learned modular arithmetic.
                self.assertGreaterEqual(
                    trained["outside_modulus_101_external_projection"]["correct"], 130)
                # A changed coordinate encoding without adaptation fails; the
                # model has no robust learned representation-invariance claim.
                self.assertLessEqual(trained["odd_coordinate_units_doubled"]["correct"], 10)
                for arm in ("frozen", "zero_lr_sham", "corrupted_labels"):
                    self.assertLessEqual(scores[arm]["heldout_odd"]["correct"], 5)
                    self.assertLessEqual(scores[arm]["outside_21_32"]["correct"], 10)
                self.assertFalse(report["minotaur_graduated"])
                self.assertFalse(report["deployed"])

    def test_shift_validates_bounds(self):
        model = QuadraticInteraction(seed=41)
        for kw in ({"coordinate_scale": 0.0}, {"coordinate_scale": 10.0},
                   {"modulus": 1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                _score_shifted(model, [(1, 3)], **kw)


if __name__ == "__main__":
    unittest.main()
