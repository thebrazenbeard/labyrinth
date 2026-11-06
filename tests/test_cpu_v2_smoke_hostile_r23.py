"""Independent source-level challenges for the optional R22 toy CPU smoke.

These tests do not require Torch and cannot establish physical execution. The
full-state fingerprint challenge is intentionally adversarial: an AdamW
optimizer's hyperparameters affect subsequent updates and belong to the state
whose continuity is being claimed, even when tensor moments match.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from unittest import TestCase, mock

from scripts import qualify_v2_checkpoint_cpu_r22 as smoke


class _Array:
    def __init__(self, encoded: bytes):
        self.encoded = encoded

    def tobytes(self) -> bytes:
        return self.encoded


class _Tensor:
    def __init__(self, encoded: bytes):
        self.encoded = encoded

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return _Array(self.encoded)


class _Module:
    def __init__(self, encoded: bytes):
        self.value = _Tensor(encoded)

    def state_dict(self):
        return {"weight": self.value}


class _Optimizer:
    def __init__(self, *, learning_rate: float, weight_decay: float):
        parameter = object()
        self.param_groups = [{
            "params": [parameter],
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }]
        self.state = {parameter: {
            "step": _Tensor(b"\x04"),
            "exp_avg": _Tensor(b"\x01\x02"),
            "exp_avg_sq": _Tensor(b"\x03\x04"),
        }}


class CpuV2SmokeHostileTests(TestCase):
    def test_full_optimizer_state_fingerprint_must_bind_learning_rate(self):
        """A changed AdamW LR changes future updates even if moments match."""
        encoder, head = _Module(b"encoder"), _Module(b"head")
        reference = smoke._state_fingerprint(
            None, encoder, head,
            _Optimizer(learning_rate=0.002, weight_decay=0.01))
        changed = smoke._state_fingerprint(
            None, encoder, head,
            _Optimizer(learning_rate=0.2, weight_decay=0.01))
        self.assertNotEqual(
            reference, changed,
            "full optimizer continuity digest omitted AdamW learning rate")

    def test_full_optimizer_state_fingerprint_must_bind_weight_decay(self):
        """A changed AdamW decay changes the next update with matching moments."""
        encoder, head = _Module(b"encoder"), _Module(b"head")
        reference = smoke._state_fingerprint(
            None, encoder, head,
            _Optimizer(learning_rate=0.002, weight_decay=0.01))
        changed = smoke._state_fingerprint(
            None, encoder, head,
            _Optimizer(learning_rate=0.002, weight_decay=0.3))
        self.assertNotEqual(
            reference, changed,
            "full optimizer continuity digest omitted AdamW weight decay")

    def test_preflight_never_constructs_a_training_plan(self):
        output = io.StringIO()
        with mock.patch.object(smoke, "_plan", side_effect=AssertionError(
                "preflight must not create a Maze training plan")):
            with redirect_stdout(output):
                code = smoke.main([])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"],
                         "PREFLIGHT_ONLY_NO_EFFECT")

    def test_dependency_skip_never_constructs_a_training_plan(self):
        output = io.StringIO()
        with mock.patch.object(smoke, "missing_dependencies",
                               return_value=("torch",)):
            with mock.patch.object(smoke, "_plan", side_effect=AssertionError(
                    "skip path must not enter training")):
                with redirect_stdout(output):
                    code = smoke.main(["--execute"])
        self.assertEqual(code, 3)
        record = json.loads(output.getvalue())
        self.assertEqual(record["status"], "SKIPPED_NO_PHYSICAL_EXECUTION")
        self.assertFalse(record["graduation_authorized"])


if __name__ == "__main__":
    import unittest
    unittest.main()
