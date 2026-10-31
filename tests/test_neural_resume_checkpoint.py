"""Fail-closed, pickle-free neural optimizer checkpoint tests."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import random
import stat
import tempfile
import unittest
from unittest.mock import patch

import labyrinth.neural_resume_checkpoint as checkpoint
from labyrinth.neural_resume_checkpoint import (
    FullCheckpointContext,
    FullCheckpointError,
    load_full_checkpoint,
    save_full_checkpoint,
)


def _context() -> FullCheckpointContext:
    return FullCheckpointContext(
        run_id="bounded-neural-run-1",
        objective_id="program-reasoning",
        base_checkpoint_sha256="a" * 64,
        prior_checkpoint_sha256="b" * 64,
        fence=2,
        cursor=1,
        exposures=(("case-0", "c" * 64), ("case-1", "d" * 64)),
    )


class CheckpointContractTests(unittest.TestCase):
    def test_publication_fsyncs_tensor_manifest_and_parent_directory(self):
        # Isolate the filesystem publication path from optional neural libraries.
        # A green manifest fsync cannot substitute for an unsynced tensor file.
        class FakeParameter:
            def detach(self):
                return self

        class FakeOptimizer:
            def state_dict(self):
                return {}

        class FakeTorch:
            @staticmethod
            def get_rng_state():
                return "fake-rng"

        def save_tensor(_tensors, destination):
            Path(destination).write_bytes(b"fake-safetensors-for-filesystem-test")

        original_fsync = os.fsync
        synced: set[tuple[int, int]] = set()

        def recording_fsync(fd):
            info = os.fstat(fd)
            synced.add((stat.S_IFMT(info.st_mode), info.st_ino))
            return original_fsync(fd)

        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "checkpoint"
            with (patch.object(checkpoint, "_dependencies",
                               return_value=(FakeTorch, None, save_tensor)),
                  patch.object(checkpoint, "_trainable_and_groups",
                               return_value=({"weight": FakeParameter()}, [["weight"]])),
                  patch.object(checkpoint, "_encode_tree",
                               return_value={"type": "dict", "items": []}),
                  patch.object(checkpoint, "_numpy_rng_snapshot", return_value=None),
                  patch.object(checkpoint.os, "fsync", side_effect=recording_fsync)):
                save_full_checkpoint(destination, model=object(),
                                     optimizer=FakeOptimizer(), context=_context())

            for name in ("state.safetensors", "manifest.json"):
                self.assertIn((stat.S_IFREG, (destination / name).stat().st_ino), synced,
                              f"published {name} lacked a file fsync")
            if os.name == "posix":
                self.assertIn((stat.S_IFDIR, destination.stat().st_ino), synced,
                              "staged checkpoint file entries lacked directory fsync")
                self.assertIn((stat.S_IFDIR, destination.parent.stat().st_ino), synced,
                              "published checkpoint directory entry lacked parent fsync")

    def test_reject_invalid_cursor_and_malformed_exposure(self):
        with self.assertRaisesRegex(ValueError, "cursor"):
            replace(_context(), cursor=3).validate()
        with self.assertRaisesRegex(ValueError, "exposure"):
            replace(_context(), exposures=(("case-0", "bad"),)).validate()
        with self.assertRaisesRegex(ValueError, "fence"):
            replace(_context(), fence=-1).validate()

    def test_reject_duplicate_exposure_occurrence_ids(self):
        # The occurrence token identifies one consumed training visit, not
        # the underlying case. Reusing it must never count as new exposure.
        for repeated in (("case-0", "c" * 64), ("case-0", "d" * 64)):
            with self.subTest(second_digest=repeated[1]):
                context = replace(_context(), cursor=2,
                                  exposures=(("case-0", "c" * 64), repeated))
                with self.assertRaisesRegex(ValueError, "duplicate.*exposure"):
                    context.validate()

    def test_repeated_training_case_under_distinct_occurrence_ids_is_valid(self):
        context = replace(_context(), cursor=2,
                          exposures=(("visit-0", "c" * 64),
                                     ("visit-1", "c" * 64)))
        context.validate()

    def test_missing_and_symlink_checkpoint_rejected_before_torch_import(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "missing"
            with self.assertRaises(FullCheckpointError):
                load_full_checkpoint(
                    target, model=None, optimizer=None, expected_sha256="a" * 64,
                    expected_context=_context(),
                )
            alias = Path(temp) / "alias"
            try:
                alias.symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("host does not allow symlink creation")
            with self.assertRaisesRegex(FullCheckpointError, "symlink"):
                load_full_checkpoint(
                    alias, model=None, optimizer=None, expected_sha256="a" * 64,
                    expected_context=_context(),
                )


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
    "optional torch+safetensors physical CPU dependencies unavailable",
)
class PhysicalTorchCheckpointTests(unittest.TestCase):
    @staticmethod
    def _model_and_optimizer():
        import torch

        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        return model, optimizer

    @staticmethod
    def _step(model, optimizer):
        import torch

        optimizer.zero_grad(set_to_none=True)
        prediction = model(torch.tensor([[0.25, -0.4, 0.7]], dtype=torch.float32))
        loss = torch.square(prediction - torch.tensor([[0.5, -0.25]])).mean()
        loss.backward()
        optimizer.step()

    def test_full_optimizer_rng_and_parameter_resume(self):
        import torch

        torch.manual_seed(51)
        random.seed(1234)
        model, optimizer = self._model_and_optimizer()
        self._step(model, optimizer)
        saved_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        saved_moments = [
            (state["exp_avg"].detach().clone(), state["exp_avg_sq"].detach().clone())
            for state in optimizer.state.values()
        ]
        saved_torch_rng = torch.get_rng_state().clone()
        saved_python_rng = random.getstate()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cycle-0"
            receipt = save_full_checkpoint(path, model=model, optimizer=optimizer,
                                           context=_context())
            self.assertRegex(receipt.sha256, r"^[0-9a-f]{64}$")
            self.assertTrue((path / "state.safetensors").is_file())
            self.assertTrue((path / "manifest.json").is_file())
            with self.assertRaisesRegex(FullCheckpointError, "exists"):
                save_full_checkpoint(path, model=model, optimizer=optimizer,
                                     context=_context())

            self._step(model, optimizer)
            torch.manual_seed(987)
            random.seed(987)
            restored = load_full_checkpoint(
                path, model=model, optimizer=optimizer,
                expected_sha256=receipt.sha256, expected_context=_context(),
            )
            self.assertEqual(restored.sha256, receipt.sha256)
            for name, p in model.named_parameters():
                self.assertTrue(torch.equal(p.detach(), saved_parameters[name]))
            for (avg, sq), state in zip(saved_moments, optimizer.state.values()):
                self.assertTrue(torch.equal(state["exp_avg"], avg))
                self.assertTrue(torch.equal(state["exp_avg_sq"], sq))
            self.assertTrue(torch.equal(torch.get_rng_state(), saved_torch_rng))
            self.assertEqual(random.getstate(), saved_python_rng)

    def test_digest_and_fence_mismatch_fail_without_mutating_model(self):
        import torch

        torch.manual_seed(19)
        model, optimizer = self._model_and_optimizer()
        self._step(model, optimizer)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cycle-0"
            receipt = save_full_checkpoint(path, model=model, optimizer=optimizer,
                                           context=_context())
            untouched = {name: p.detach().clone() for name, p in model.named_parameters()}
            with self.assertRaisesRegex(FullCheckpointError, "digest"):
                load_full_checkpoint(path, model=model, optimizer=optimizer,
                                     expected_sha256="f" * 64,
                                     expected_context=_context())
            with self.assertRaisesRegex(FullCheckpointError, "context"):
                load_full_checkpoint(path, model=model, optimizer=optimizer,
                                     expected_sha256=receipt.sha256,
                                     expected_context=replace(_context(), fence=3))
            with self.assertRaisesRegex(ValueError, "duplicate.*exposure"):
                # Even the caller's nominally trusted resume context must
                # not contain a replayed training occurrence identifier.
                load_full_checkpoint(
                    path, model=model, optimizer=optimizer,
                    expected_sha256=receipt.sha256,
                    expected_context=replace(
                        _context(), cursor=2,
                        exposures=(("visit", "c" * 64),
                                   ("visit", "d" * 64))),
                )
            for name, p in model.named_parameters():
                self.assertTrue(torch.equal(p.detach(), untouched[name]))
            state_path = path / "state.safetensors"
            with state_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(FullCheckpointError, "digest"):
                load_full_checkpoint(path, model=model, optimizer=optimizer,
                                     expected_sha256=receipt.sha256,
                                     expected_context=_context())

    def test_interrupted_adamw_matches_uninterrupted_second_step(self):
        import torch

        torch.manual_seed(45)
        original, original_optimizer = self._model_and_optimizer()
        self._step(original, original_optimizer)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recovery"
            receipt = save_full_checkpoint(path, model=original,
                                           optimizer=original_optimizer,
                                           context=_context())
            self._step(original, original_optimizer)
            expected = {name: parameter.detach().clone()
                        for name, parameter in original.named_parameters()}

            recovered, recovered_optimizer = self._model_and_optimizer()
            load_full_checkpoint(path, model=recovered,
                                 optimizer=recovered_optimizer,
                                 expected_sha256=receipt.sha256,
                                 expected_context=_context())
            self._step(recovered, recovered_optimizer)
            for name, parameter in recovered.named_parameters():
                self.assertTrue(torch.equal(parameter.detach(), expected[name]))

    def test_unexpected_checkpoint_entry_rejected(self):
        import torch

        torch.manual_seed(12)
        model, optimizer = self._model_and_optimizer()
        self._step(model, optimizer)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint"
            receipt = save_full_checkpoint(path, model=model, optimizer=optimizer,
                                           context=_context())
            (path / "extra.bin").write_bytes(b"not part of checkpoint")
            with self.assertRaisesRegex(FullCheckpointError, "unexpected"):
                load_full_checkpoint(path, model=model, optimizer=optimizer,
                                     expected_sha256=receipt.sha256,
                                     expected_context=_context())


if __name__ == "__main__":
    unittest.main()
