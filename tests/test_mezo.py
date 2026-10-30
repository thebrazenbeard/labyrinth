"""Small, dependency-free optimizer tests plus optional real-Torch checks."""

from contextlib import nullcontext
import math
import random
import unittest

from labyrinth.mezo import mezo_step


class Tensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
        self.dtype = "float32"
        self.device = "cpu"
        self.requires_grad = False

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values)

    def copy_(self, other):
        self.values = list(other.values)
        return self

    def add_(self, other, *, alpha=1.0):
        self.values = [a + alpha * b for a, b in zip(self.values, other.values)]
        return self


class Parameter(Tensor):
    def __init__(self, values, trainable=True):
        super().__init__(values)
        self.requires_grad = trainable


class Model:
    training = False

    def __init__(self, *, trainable=2.0, frozen=9.0):
        self.parameter = Parameter([trainable])
        self.frozen = Parameter([frozen], trainable=False)

    def named_parameters(self):
        return [("weight", self.parameter), ("frozen", self.frozen)]


class Generator:
    def __init__(self, *, device):
        self.rng = random.Random()

    def manual_seed(self, seed):
        self.rng.seed(seed)
        return self


class FakeTorch:
    Generator = Generator
    float32 = "float32"
    float64 = "float64"

    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def randn(shape, *, device, dtype, generator):
        return Tensor([generator.rng.gauss(0.0, 1.0) for _ in range(shape[0])])

    @staticmethod
    def isfinite(tensor):
        class Predicate:
            def all(self):
                return all(math.isfinite(v) for v in tensor.values)

        return Predicate()


class MezoContractTests(unittest.TestCase):
    def test_quadratic_step_matches_analytic_two_sided_estimator(self):
        model = Model()
        before = model.parameter.values[0]
        frozen = model.frozen.values[:]
        z = random.Random(42).gauss(0.0, 1.0)
        output = mezo_step(model, FakeTorch, lambda: model.parameter.values[0] ** 2,
                           epsilon=0.01, learning_rate=0.05, seed=42)
        self.assertAlmostEqual(output["projected_grad"], 2 * before * z, places=10)
        self.assertAlmostEqual(model.parameter.values[0], before - 0.05 * 2 * before * z * z, places=10)
        self.assertEqual(model.frozen.values, frozen)
        self.assertEqual(output["forward_evaluations"], 2)

    def test_exact_zero_learning_rate_is_compute_matched_but_no_update(self):
        model = Model()
        calls = 0

        def loss():
            nonlocal calls
            calls += 1
            return model.parameter.values[0] ** 2

        before = model.parameter.values[:]
        output = mezo_step(model, FakeTorch, loss, epsilon=0.01,
                           learning_rate=0.0, seed=42)
        self.assertEqual(calls, 2)
        self.assertEqual(output["forward_evaluations"], 2)
        self.assertEqual(model.parameter.values, before)

    def test_identical_seed_is_reproducible_and_distinct_seed_differs(self):
        results = []
        for seed in (12, 12, 13):
            model = Model()
            mezo_step(model, FakeTorch, lambda: model.parameter.values[0] ** 2,
                      epsilon=0.01, learning_rate=0.05, seed=seed)
            results.append(model.parameter.values[:])
        self.assertEqual(results[0], results[1])
        self.assertNotEqual(results[0], results[2])

    def test_failure_in_either_forward_restores_original_exactly(self):
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at):
                model = Model()
                original = model.parameter.values[:]
                calls = 0

                def loss():
                    nonlocal calls
                    calls += 1
                    if calls == fail_at:
                        raise RuntimeError("forward exploded")
                    return model.parameter.values[0] ** 2

                with self.assertRaisesRegex(RuntimeError, "forward exploded"):
                    mezo_step(model, FakeTorch, loss, epsilon=0.01,
                              learning_rate=0.05, seed=0)
                self.assertEqual(model.parameter.values, original)

    def test_nonfinite_losses_and_updates_fail_without_changes(self):
        for bad in (math.nan, math.inf, -math.inf):
            model = Model()
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                mezo_step(model, FakeTorch, lambda: bad, epsilon=0.01,
                          learning_rate=0.05, seed=3)
            self.assertEqual(model.parameter.values, [2.0])

        model = Model()
        # Each loss is individually finite; their difference still overflows.
        def catastrophic_loss():
            return 1e308 if model.parameter.values[0] > 2.0 else -1e308

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            mezo_step(model, FakeTorch, catastrophic_loss, epsilon=0.01,
                      learning_rate=0.05, seed=3)
        self.assertEqual(model.parameter.values, [2.0])

    def test_invalid_configuration_rejected_before_loss_invocation(self):
        bad_arguments = [
            {"epsilon": 0.0}, {"epsilon": -0.01}, {"epsilon": math.nan},
            {"epsilon": math.inf}, {"learning_rate": -1.0},
            {"learning_rate": math.nan}, {"learning_rate": math.inf},
            {"seed": -1}, {"seed": True},
        ]
        for override in bad_arguments:
            with self.subTest(override=override):
                model = Model()
                kwargs = {"epsilon": 0.01, "learning_rate": 0.05, "seed": 1}
                kwargs.update(override)
                with self.assertRaises(ValueError):
                    mezo_step(model, FakeTorch, lambda: self.fail("loss called"), **kwargs)
                self.assertEqual(model.parameter.values, [2.0])

    def test_no_trainable_parameters_fails(self):
        model = Model()
        model.parameter.requires_grad = False
        with self.assertRaisesRegex(ValueError, "trainable"):
            mezo_step(model, FakeTorch, lambda: 1.0, epsilon=0.01,
                      learning_rate=0.05, seed=5)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch not installed")
class RealTorchMezoTests(unittest.TestCase):
    def test_no_backprop_or_gradient_tensors(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
        model.eval()
        before = model.weight.detach().clone()
        output = mezo_step(model, torch, lambda: model.weight.square().sum(),
                           epsilon=0.01, learning_rate=0.05, seed=31)
        self.assertEqual(output["forward_evaluations"], 2)
        self.assertIsNone(model.weight.grad)
        self.assertFalse(torch.equal(model.weight, before))


if __name__ == "__main__":
    unittest.main()
