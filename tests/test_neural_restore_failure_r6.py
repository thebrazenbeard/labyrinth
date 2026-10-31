"""Fault-injection checks for the untrusted full-state restore boundary."""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from labyrinth import neural_resume_checkpoint as checkpoint


class _FakeTensor:
    def __init__(self, value: int, dtype: str = "float"):
        self.value = value
        self.dtype = dtype
        self.shape = (1,)
        self.device = type("Device", (), {"type": "cpu"})()

    def clone(self):
        return _FakeTensor(self.value, self.dtype)


class _FakeParameter(_FakeTensor):
    def detach(self):
        return self

    def copy_(self, source):
        self.value = source.value


class _FakeTorch:
    Tensor = _FakeTensor
    uint8 = "uint8"

    @staticmethod
    def get_rng_state():
        return _FakeTensor(1, "uint8")

    @staticmethod
    def no_grad():
        return nullcontext()


class _FailingOptimizer:
    def load_state_dict(self, _state):
        # A storage or custom extension failure is not necessarily RuntimeError.
        raise OSError("injected optimizer backend write error")


class NeuralRestoreFailureTests(unittest.TestCase):
    def test_repeat_publication_cannot_replace_a_pinned_checkpoint(self):
        class _FakeOptimizer:
            @staticmethod
            def state_dict():
                return {}

        def _save(_tensors, destination):
            Path(destination).write_bytes(b"stable-fake-tensors")

        context = checkpoint.FullCheckpointContext(
            run_id="recovery-r6", objective_id="arithmetic",
            base_checkpoint_sha256="a" * 64,
            prior_checkpoint_sha256="b" * 64,
            fence=9, cursor=0, exposures=(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "published"
            with (
                patch.object(checkpoint, "_dependencies", return_value=(
                    _FakeTorch, None, _save,
                )),
                patch.object(checkpoint, "_trainable_and_groups", return_value=(
                    {"weight": _FakeParameter(1)}, [["weight"]],
                )),
                patch.object(checkpoint, "_encode_tree", return_value={}),
                patch.object(checkpoint, "_numpy_rng_snapshot", return_value=None),
            ):
                first = checkpoint.save_full_checkpoint(
                    root, model=object(), optimizer=_FakeOptimizer(), context=context,
                )
                before = {name: (root / name).read_bytes()
                          for name in ("manifest.json", "state.safetensors")}
                with self.assertRaisesRegex(checkpoint.FullCheckpointError,
                                            "already exists"):
                    checkpoint.save_full_checkpoint(
                        root, model=object(), optimizer=_FakeOptimizer(), context=context,
                    )
                self.assertEqual(before, {name: (root / name).read_bytes()
                                          for name in before})
                self.assertRegex(first.sha256, r"^[0-9a-f]{64}$")

    def test_nonstandard_optimizer_failure_after_parameter_mutation_requires_discard(self):
        original = _FakeParameter(1)
        replacement = _FakeTensor(2)
        context = checkpoint.FullCheckpointContext(
            run_id="recovery-r6", objective_id="arithmetic",
            base_checkpoint_sha256="a" * 64,
            prior_checkpoint_sha256="b" * 64,
            fence=9, cursor=0, exposures=(),
        )
        manifest = {
            "schema": checkpoint.SCHEMA,
            "context": checkpoint._context_json(context),
            "optimizer": "AdamW",
            "optimizer_groups": [["weight"]],
            "trainable_names": ["weight"],
            "model_tree": {}, "optimizer_tree": {}, "rng_tree": {},
            "numpy_rng": None,
            "tensor_sha256": sha256(b"fake-tensors").hexdigest(),
        }
        manifest_bytes = checkpoint._json_bytes(manifest)
        expected_sha = checkpoint._manifest_digest(manifest_bytes)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            root.mkdir()
            (root / "manifest.json").write_bytes(manifest_bytes)
            (root / "state.safetensors").write_bytes(b"fake-tensors")
            with (
                patch.object(checkpoint, "_dependencies", return_value=(
                    _FakeTorch, lambda *_args, **_kwargs: {}, None,
                )),
                patch.object(checkpoint, "_trainable_and_groups", return_value=(
                    {"weight": original}, [["weight"]],
                )),
                patch.object(checkpoint, "_decode_tree", side_effect=[
                    {"weight": replacement},
                    {"state": {}, "param_groups": []},
                    {"torch": _FakeTensor(1, "uint8"), "python": random.getstate()},
                ]),
            ):
                with self.assertRaisesRegex(checkpoint.FullCheckpointError,
                                            "discard this process"):
                    checkpoint.load_full_checkpoint(
                        root, model=object(), optimizer=_FailingOptimizer(),
                        expected_sha256=expected_sha, expected_context=context,
                    )
        # The first write happened, so the caller cannot safely retry using
        # this already-partially-restored process, regardless of error class.
        self.assertEqual(original.value, 2)


if __name__ == "__main__":
    unittest.main()
