"""Physical CPU checkpoint-continuity gate, not a pretrained-model result.

This runs only with real Torch, safetensors and NumPy available. It checks an
interrupted AdamW journey against an uninterrupted journey through the public
full-state checkpoint API; a skipped run establishes no physical continuity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

from labyrinth.neural_resume_checkpoint import (
    FullCheckpointContext,
    load_full_checkpoint,
    save_full_checkpoint,
)


HAS_CPU_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "safetensors", "numpy")
)


@unittest.skipUnless(HAS_CPU_STACK, "real CPU Torch+safetensors+NumPy unavailable")
class PhysicalFullStateContinuityGate(unittest.TestCase):
    def test_interrupted_vs_uninterrupted_adamw_rng_and_exposure_cursor(self):
        import numpy as np
        import torch

        previous_torch = torch.get_rng_state().clone()
        previous_python = random.getstate()
        previous_numpy = np.random.get_state()
        self.addCleanup(torch.set_rng_state, previous_torch)
        self.addCleanup(random.setstate, previous_python)
        self.addCleanup(np.random.set_state, previous_numpy)

        def new_learner():
            model = torch.nn.Sequential(
                torch.nn.Linear(3, 4),
                torch.nn.Tanh(),
                torch.nn.Linear(4, 2),
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01,
                                          foreach=False)
            return model, optimizer

        def step(model, optimizer):
            # All three independent random generators affect the next batch.
            torch_draw = torch.randn((2, 3), dtype=torch.float32)
            python_draw = random.random()
            numpy_draw = float(np.random.normal())
            batch = torch_draw + python_draw * 0.1 + numpy_draw * 0.1
            target = torch.tensor([[0.25, -0.5], [-0.2, 0.4]])
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            torch.square(output - target).mean().backward()
            optimizer.step()
            return output.detach().clone(), (python_draw, numpy_draw)

        context = FullCheckpointContext(
            run_id="physical-gate-r8",
            objective_id="program-reasoning",
            base_checkpoint_sha256="a" * 64,
            prior_checkpoint_sha256="b" * 64,
            fence=7,
            cursor=1,
            exposures=(("case-1", "c" * 64), ("case-2", "d" * 64)),
        )

        torch.manual_seed(2044)
        random.seed(2045)
        np.random.seed(2046)
        uninterrupted, uninterrupted_adamw = new_learner()
        step(uninterrupted, uninterrupted_adamw)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "cycle-0"
            saved = save_full_checkpoint(
                checkpoint, model=uninterrupted,
                optimizer=uninterrupted_adamw, context=context,
            )
            self.assertEqual(saved.context.cursor, 1)
            self.assertEqual(saved.context.exposures[1][0], "case-2")

            expected_output, expected_non_torch_draws = step(
                uninterrupted, uninterrupted_adamw,
            )
            expected_parameters = {
                key: value.detach().clone()
                for key, value in uninterrupted.named_parameters()
            }
            expected_adamw = uninterrupted_adamw.state_dict()
            expected_torch_rng = torch.get_rng_state().clone()
            expected_python_rng = random.getstate()
            expected_numpy_rng = np.random.get_state()

            # Recreate the learner in a fresh object with intentionally wrong
            # initialization and RNG before the trusted full-state restore.
            torch.manual_seed(9999)
            random.seed(9998)
            np.random.seed(9997)
            resumed, resumed_adamw = new_learner()
            loaded = load_full_checkpoint(
                checkpoint, model=resumed, optimizer=resumed_adamw,
                expected_sha256=saved.sha256, expected_context=context,
            )
            self.assertEqual(loaded.sha256, saved.sha256)
            self.assertEqual(loaded.context.cursor, 1)
            self.assertEqual(loaded.context.exposures, context.exposures)

            actual_output, actual_non_torch_draws = step(resumed, resumed_adamw)
            self.assertTrue(torch.equal(actual_output, expected_output))
            self.assertEqual(actual_non_torch_draws, expected_non_torch_draws)
            for name, parameter in resumed.named_parameters():
                self.assertTrue(torch.equal(parameter.detach(), expected_parameters[name]),
                                f"continued trainable tensor diverged: {name}")
            actual_adamw = resumed_adamw.state_dict()
            self.assertEqual(actual_adamw["param_groups"],
                             expected_adamw["param_groups"])
            self.assertEqual(set(actual_adamw["state"]),
                             set(expected_adamw["state"]))
            for parameter_id, expected_values in expected_adamw["state"].items():
                actual_values = actual_adamw["state"][parameter_id]
                self.assertEqual(set(actual_values), set(expected_values))
                for key, expected_value in expected_values.items():
                    actual_value = actual_values[key]
                    if isinstance(expected_value, torch.Tensor):
                        self.assertTrue(torch.equal(actual_value, expected_value),
                                        f"AdamW {parameter_id}:{key} diverged")
                    else:
                        self.assertEqual(actual_value, expected_value)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_rng))
            self.assertEqual(random.getstate(), expected_python_rng)
            actual_numpy_rng = np.random.get_state()
            self.assertEqual(actual_numpy_rng[0], expected_numpy_rng[0])
            np.testing.assert_array_equal(actual_numpy_rng[1], expected_numpy_rng[1])
            self.assertEqual(actual_numpy_rng[2:], expected_numpy_rng[2:])


if __name__ == "__main__":
    unittest.main()
