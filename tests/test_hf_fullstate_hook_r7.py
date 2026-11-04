"""Regression evidence for the current adapter-only HF continuation boundary.

These dependency-free fakes exercise the real _train_steps control flow; they
do not pretend to be physical Torch/PEFT model training.  A full-state
continuation must change the optimizer and RNG lifecycle tested here.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.hf_pilot import PilotConfig, _train_steps
from labyrinth.maze import Challenge


class _Parameter:
    requires_grad = True


class _Loss:
    def backward(self):
        pass

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return 1.0


class _Model:
    def __init__(self):
        self.parameter = _Parameter()

    def parameters(self):
        return iter((self.parameter,))

    def train(self):
        pass

    def __call__(self, *, input_ids, labels):
        return SimpleNamespace(loss=_Loss())


class _AdamW:
    instances = []

    def __init__(self, parameters, *, lr):
        self.parameters = list(parameters)
        self.lr = lr
        self.step_counter = 0
        self.initial_step_counter = self.step_counter
        self.instances.append(self)

    def zero_grad(self, *, set_to_none):
        pass

    def step(self):
        self.step_counter += 1


class _Torch:
    long = "long"

    def __init__(self):
        self.seeds = []
        self.optim = SimpleNamespace(AdamW=_AdamW)
        self.nn = SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a, **k: None))

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def tensor(self, data, *, dtype, device):
        return data

    def isfinite(self, value):
        return SimpleNamespace(all=lambda: True)


class HFOptimizerContinuityBoundary(unittest.TestCase):
    def test_new_train_call_reseeds_and_recreates_adamw_without_moment_handoff(self):
        """The current resume cycle cannot be called a full-state resume."""
        _AdamW.instances.clear()
        fake_torch = _Torch()
        cases = [
            Challenge("train-t1-0", "arithmetic", 1, "train", "Question 0", "5"),
            Challenge("train-t1-1", "arithmetic", 1, "train", "Question 1", "6"),
        ]
        config = PilotConfig("unused", "unused", seed=17, steps=2, device="cpu")
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            first = _train_steps(_Model(), object(), fake_torch, cases, ["5", "6"], config)
            second = _train_steps(_Model(), object(), fake_torch, cases, ["5", "6"], config)

        self.assertEqual((first["steps"], second["steps"]), (2, 2))
        self.assertEqual(fake_torch.seeds, [17, 17])
        self.assertEqual(len(_AdamW.instances), 2)
        self.assertIsNot(_AdamW.instances[0], _AdamW.instances[1])
        self.assertEqual([item.initial_step_counter for item in _AdamW.instances], [0, 0])
        self.assertEqual([item.step_counter for item in _AdamW.instances], [2, 2])


if __name__ == "__main__":
    unittest.main()
