"""Offline MiniLM pilot preflight and reporting invariants (no ML dependency)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_minilm_cpu.py"
SPEC = importlib.util.spec_from_file_location("qualify_minilm_cpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MiniLMCPUProbeTests(unittest.TestCase):
    def test_frozen_pretrained_base_fingerprint_and_lora_trainability(self):
        class FakeTensor:
            def __init__(self, payload: bytes, requires_grad: bool):
                self.payload = payload
                self.requires_grad = requires_grad
                self.dtype = "float32"
                self.shape = (len(payload),)

            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return self
            def tobytes(self): return self.payload

        class FakeModel:
            def __init__(self, base: FakeTensor, lora: FakeTensor,
                         *, lora_b: FakeTensor | None = None,
                         buffer: FakeTensor | None = None,
                         lora_named_buffer: FakeTensor | None = None):
                self.params = (
                    ("base_model.model.encoder.query.base_layer.weight", base),
                    ("base_model.model.encoder.query.lora_A.default.weight", lora),
                    ("base_model.model.encoder.query.lora_B.default.weight",
                     lora if lora_b is None else lora_b),
                )
                self.buffers = tuple(
                    (name, tensor) for name, tensor in (
                        ("base_model.model.encoder.position_ids", buffer),
                        ("base_model.model.encoder.query.lora_A.default.registered_state",
                         lora_named_buffer),
                    ) if tensor is not None
                )

            def named_parameters(self): return iter(self.params)
            def named_buffers(self): return iter(self.buffers)

        base = FakeTensor(b"actual-frozen-base-bytes", False)
        lora = FakeTensor(b"trainable-adapter", True)
        fingerprint = MODULE.frozen_base_digest(FakeModel(base, lora))
        self.assertEqual(fingerprint, MODULE.frozen_base_digest(FakeModel(base, lora)))
        self.assertEqual(fingerprint, MODULE.frozen_base_digest(FakeModel(
            base, FakeTensor(b"updated-adapter-weights", True))))
        self.assertNotEqual(fingerprint, MODULE.frozen_base_digest(
            FakeModel(FakeTensor(b"changed-base-weights", False), lora)))
        self.assertNotEqual(fingerprint, MODULE.frozen_base_digest(
            FakeModel(base, lora, buffer=FakeTensor(b"buffer", False))))
        self.assertNotEqual(
            MODULE.frozen_base_digest(FakeModel(base, lora,
                                                buffer=FakeTensor(b"buffer-a", False))),
            MODULE.frozen_base_digest(FakeModel(base, lora,
                                                buffer=FakeTensor(b"buffer-b", False))),
        )
        # The frozen-state contract cannot exempt arbitrary registered buffers
        # merely because their names resemble adapter parameters.
        self.assertNotEqual(
            MODULE.frozen_base_digest(FakeModel(
                base, lora, lora_named_buffer=FakeTensor(b"before", False))),
            MODULE.frozen_base_digest(FakeModel(
                base, lora, lora_named_buffer=FakeTensor(b"after", False))),
        )
        with self.assertRaisesRegex(RuntimeError, "non-LoRA base parameter is trainable"):
            MODULE.frozen_base_digest(FakeModel(FakeTensor(b"unsafe", True), lora))
        with self.assertRaisesRegex(RuntimeError, "trainable LoRA"):
            MODULE.frozen_base_digest(FakeModel(base, FakeTensor(b"frozen-adapter", False)))
        with self.assertRaisesRegex(RuntimeError, "trainable LoRA"):
            MODULE.frozen_base_digest(FakeModel(
                base, lora, lora_b=FakeTensor(b"untracked-frozen-B", False)))

    def test_frozen_public_fixture_has_nonoverlapping_balanced_splits(self):
        sizes = MODULE.validate_fixtures()
        self.assertEqual(sizes, {"cycle0": 12, "cycle1": 12,
                                 "heldout": 10, "transfer": 10})
        self.assertEqual(set(MODULE.ARMS), {"correct", "zero_update", "inverted"})

    def test_output_run_preflight_never_overwrites_existing_run(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            candidate = MODULE.plan_output(parent, "minilm-cycle-01")
            self.assertEqual(candidate.parent, parent.resolve())
            self.assertFalse(candidate.exists())
            candidate.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.plan_output(parent, "minilm-cycle-01")
            for name in ("", "../escape", "bad/path", ".hidden", "a" * 70):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    MODULE.plan_output(parent, name)

    def test_control_comparison_only_reports_observed_bounded_values(self):
        result = MODULE.comparison({"correct": .9, "zero_update": .5,
                                    "inverted": .1})
        self.assertTrue(result["correct_outperforms_both"])
        self.assertAlmostEqual(result["correct_minus_zero"], .4)
        self.assertAlmostEqual(result["correct_minus_inverted"], .8)
        for bad in ({"correct": 1},
                    {"correct": 1, "zero_update": 0, "inverted": -1},
                    {"correct": 1, "zero_update": 0, "inverted": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MODULE.comparison(bad)

    def test_read_only_plan_does_not_load_or_initialize_ml(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("builtins.print") as printed:
                self.assertEqual(MODULE.main(["--output-parent", temp,
                                              "--run-name", "probe"]), 0)
            payload = json.loads(printed.call_args.args[0])
            self.assertEqual(payload["schema"], MODULE.SCHEMA)
            self.assertFalse(payload["execute"])
            self.assertFalse(payload["graduation_authorized"])
            self.assertFalse((Path(temp) / "probe").exists())

    def test_zero_update_sham_runs_optimizer_at_zero_learning_rate(self):
        self.assertEqual(MODULE.optimizer_learning_rate("zero_update", .002), 0.0)
        self.assertEqual(MODULE.optimizer_learning_rate("correct", .002), .002)
        self.assertEqual(MODULE.optimizer_learning_rate("inverted", .002), .002)
        with self.assertRaises(ValueError):
            MODULE.optimizer_learning_rate("unknown", .002)

    def test_reload_checks_full_heldout_logits_not_just_parameter_hashes(self):
        original = [[.2, .8], [.9, .1]]
        self.assertEqual(MODULE.reload_logit_delta(original, original), 0.0)
        self.assertAlmostEqual(
            MODULE.reload_logit_delta(original, [[.200001, .799999], [.9, .1]]),
            1e-6,
        )
        with self.assertRaises(RuntimeError):
            MODULE.reload_logit_delta(original, [[.2, .81], [.9, .1]])
        with self.assertRaises(RuntimeError):
            MODULE.reload_logit_delta(original, [[.2, .8]])
        with self.assertRaises(RuntimeError):
            MODULE.reload_logit_delta(original, [[float("nan"), .8], [.9, .1]])


if __name__ == "__main__":
    unittest.main()
