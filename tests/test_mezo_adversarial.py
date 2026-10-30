"""Independent numerical and failure-injection checks for the MeZO step.

These exercise the public optimizer seam with a small deterministic tensor
facade; they are not a substitute for real Torch/HF execution.
"""

import random
import unittest

from labyrinth.mezo import mezo_step
from tests.test_mezo import FakeTorch, Parameter


class TwoParameterModel:
    training = False

    def __init__(self):
        self.first = Parameter([2.0])
        self.second = Parameter([-3.0])
        self.frozen = Parameter([11.0], trainable=False)

    def named_parameters(self):
        return (("first", self.first), ("second", self.second), ("frozen", self.frozen))


class TorchWithEquality(FakeTorch):
    @staticmethod
    def equal(left, right):
        return left.values == right.values


class OptimizerAdversarialTests(unittest.TestCase):
    def test_multitensor_finite_difference_matches_independent_analytic_direction(self):
        model = TwoParameterModel()
        z = random.Random(137)
        direction_a, direction_b = z.gauss(0.0, 1.0), z.gauss(0.0, 1.0)
        # The independent gradient is (2a, 6b) for f(a,b) = a² + 3b².
        expected_projection = 4 * direction_a - 18 * direction_b
        before_frozen = model.frozen.values[:]
        result = mezo_step(
            model, FakeTorch,
            lambda: model.first.values[0] ** 2 + 3 * model.second.values[0] ** 2,
            epsilon=0.001, learning_rate=0.02, seed=137,
        )
        self.assertAlmostEqual(result["projected_grad"], expected_projection, places=8)
        self.assertAlmostEqual(model.first.values[0], 2.0 - 0.02 * expected_projection * direction_a, places=8)
        self.assertAlmostEqual(model.second.values[0], -3.0 - 0.02 * expected_projection * direction_b, places=8)
        self.assertEqual(model.frozen.values, before_frozen)

    def test_mid_update_exception_rolls_back_all_tensors(self):
        model = TwoParameterModel()
        original_first, original_second = model.first.values[:], model.second.values[:]
        original_add = model.second.add_
        calls = 0

        def fail_on_update(other, *, alpha=1.0):
            nonlocal calls
            calls += 1
            # Calls one and two are the positive and negative perturbations.
            if calls == 3:
                raise RuntimeError("injected optimizer write failure")
            return original_add(other, alpha=alpha)

        model.second.add_ = fail_on_update
        with self.assertRaisesRegex(RuntimeError, "optimizer write failure"):
            mezo_step(
                model, FakeTorch,
                lambda: model.first.values[0] ** 2 + model.second.values[0] ** 2,
                epsilon=0.01, learning_rate=0.03, seed=9,
            )
        self.assertEqual(model.first.values, original_first)
        self.assertEqual(model.second.values, original_second)

    def test_unrepresentable_perturbation_rejected_before_forward(self):
        model = TwoParameterModel()
        model.first.values = [1e20]
        model.second.values = [1e20]
        calls = 0

        def loss():
            nonlocal calls
            calls += 1
            return 1.0

        with self.assertRaisesRegex(RuntimeError, "no representable perturbation"):
            mezo_step(model, TorchWithEquality, loss,
                      epsilon=1e-20, learning_rate=0.01, seed=2)
        self.assertEqual(calls, 0)
        self.assertEqual(model.first.values, [1e20])
        self.assertEqual(model.second.values, [1e20])

    def test_half_precision_trainable_tensor_is_rejected(self):
        model = TwoParameterModel()
        model.second.dtype = "float16"
        with self.assertRaisesRegex(ValueError, "fp32/fp64"):
            mezo_step(model, FakeTorch, lambda: self.fail("forward unexpectedly called"),
                      epsilon=0.01, learning_rate=0.01, seed=5)
        self.assertEqual(model.first.values, [2.0])
        self.assertEqual(model.second.values, [-3.0])


if __name__ == "__main__":
    unittest.main()
