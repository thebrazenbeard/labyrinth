"""Read-only physical GGUF/HF dual-track entrance and local-only route contract."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from labyrinth.dual_track_w0_entrance_v1 import (
    DualTrackEntranceError, describe_dual_track_w0,
)
from labyrinth.gguf_entrance_fingerprint_v1 import capture_gguf_entrance
from labyrinth.local_full_state_probe_v1 import LocalFullStateProbeError, LocalProbeTarget
from tests.test_gguf_entrance_fingerprint_v1 import _gguf
from tests.test_weight_space_spark_v1 import _capture, _tensor


class DualTrackEntranceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.gguf = self.root / "mock_model_q5_k_s.gguf"
        self.lora = self.root / "mock_vera_lora.gguf"
        self.gguf.write_bytes(_gguf([
            ("general.architecture", 8, "qwen3"),
            ("general.file_type", 4, 16),
        ]))
        self.lora.write_bytes(_gguf([
            ("general.type", 8, "adapter"),
            ("general.architecture", 8, "qwen3"),
            ("adapter.type", 8, "lora"),
        ], tensor_name="blk.0.attn_q.lora_a"))
        w0 = self.root / "w0"
        for role in ("base", "optimizer", "scheduler", "rng"):
            (w0 / role).mkdir(parents=True)
        (w0 / "base" / "config.json").write_text('{"model_type":"toy"}')
        (w0 / "base" / "tokenizer.json").write_text('{"tokens":["A"]}')
        (w0 / "base" / "model.safetensors").write_bytes(_tensor(
            (1., 2., 3., 4., 5., 6., 7., 8.)))
        for role in ("optimizer", "scheduler", "rng"):
            (w0 / role / "state.bin").write_bytes(role.encode())
        (w0 / "cursor.json").write_text('{"step":0}')
        self.hf, self.hf_manifest = _capture(w0)
        self.runtime = capture_gguf_entrance(base_path=self.gguf,
                                             lora_adapter_path=self.lora)

    def test_runtime_gguf_and_hf_full_state_are_distinct_even_with_consistent_pins(self):
        route = LocalProbeTarget("parent", "http://127.0.0.1:5011/api/v1/generate",
                                 "koboldcpp", "claimed_local_model")
        original_gguf = self.gguf.read_bytes()
        original_hf = (self.root / "w0" / "base" / "model.safetensors").read_bytes()
        result = describe_dual_track_w0(
            runtime_gguf_path=self.gguf,
            runtime_lora_gguf_path=self.lora,
            trainable_hf_manifest_path=self.hf_manifest,
            expected_gguf_manifest_sha256=self.runtime["manifest_sha256"],
            expected_hf_state_sha256=self.hf["state_sha256"],
            local_target=route,
        )
        self.assertEqual(result["schema"], "labyrinth.dual-track-w0-entrance.v1")
        self.assertEqual(result["gguf"]["base_sha256"],
                         self.runtime["base"]["sha256"])
        self.assertEqual(result["gguf"]["file_type_claim"], 16)
        self.assertEqual(result["gguf"]["pairing_status"],
                         "DECLARED_ARCHITECTURE_MATCH_ONLY")
        self.assertTrue(result["gguf"]["caller_supplied_gguf_pin_matches"])
        self.assertEqual(result["hf_trainable_w0"]["state_sha256"],
                         self.hf["state_sha256"])
        self.assertEqual(result["hf_trainable_w0"]["mutation_kind"], "full_weight_edit")
        self.assertTrue(result["hf_trainable_w0"]["hf_state_pin_matches"])
        self.assertFalse(result["hf_trainable_w0"]["optimizer_rng_cursor_restoration_proven"])
        self.assertFalse(result["runtime_equals_hf_weights_proven"])
        self.assertFalse(result["gguf_quantization_transform_proven"])
        self.assertFalse(result["gguf_adapter_applied_proven"])
        self.assertFalse(result["runtime_genuine_publisher_identity_proven"])
        self.assertFalse(result["graduation_authorized"])
        self.assertFalse(result["local_only_probe_route"]["model_service_contacted"])
        self.assertFalse(result["local_only_probe_route"]["loaded_gguf_sha256_proven"])
        self.assertEqual(self.gguf.read_bytes(), original_gguf)
        self.assertEqual((self.root / "w0" / "base" / "model.safetensors").read_bytes(),
                         original_hf)

    def test_inference_only_gguf_and_misleading_filename_do_not_invent_trainable_w0(self):
        result = describe_dual_track_w0(runtime_gguf_path=self.gguf)
        self.assertIsNone(result["hf_trainable_w0"])
        self.assertEqual(result["gguf"]["file_type_claim"], 16)
        self.assertFalse(result["runtime_equals_hf_weights_proven"])
        self.assertIsNone(result["gguf"]["lora_adapter_sha256"])
        self.assertFalse(result["gguf"]["caller_supplied_gguf_pin_matches"])

    def test_bad_expected_pins_and_remote_url_fail(self):
        with self.assertRaisesRegex(ValueError, "commitment"):
            describe_dual_track_w0(
                runtime_gguf_path=self.gguf,
                expected_gguf_manifest_sha256="0"*64,
            )
        with self.assertRaisesRegex(DualTrackEntranceError, "differs"):
            describe_dual_track_w0(
                runtime_gguf_path=self.gguf,
                trainable_hf_manifest_path=self.hf_manifest,
                expected_hf_state_sha256="e"*64,
            )
        with self.assertRaises(LocalFullStateProbeError):
            describe_dual_track_w0(
                runtime_gguf_path=self.gguf,
                local_target=LocalProbeTarget("parent", "https://remote.example.com/v1/completions",
                                              "openai_local"),
            )
        with self.assertRaises(DualTrackEntranceError):
            describe_dual_track_w0(
                runtime_gguf_path=self.gguf,
                local_target=LocalProbeTarget("candidate", "http://127.0.0.1:5011/completion",
                                              "llamacpp"),
            )

    def test_modified_hf_state_refuses_runtime_training_relation(self):
        (self.root / "w0" / "rng" / "state.bin").write_bytes(b"tampered rng")
        with self.assertRaises(ValueError):
            describe_dual_track_w0(runtime_gguf_path=self.gguf,
                                    trainable_hf_manifest_path=self.hf_manifest)

    def test_cli_prints_read_only_source_descriptor_without_running_inference(self):
        command = [sys.executable, "-m", "labyrinth.dual_track_w0_entrance_v1",
                   "--runtime-gguf", str(self.gguf),
                   "--lora-gguf", str(self.lora),
                   "--hf-full-state-manifest", str(self.hf_manifest),
                   "--loopback-url", "http://127.0.0.1:5011/completion",
                   "--backend", "llamacpp"]
        result = subprocess.run(command, check=True, capture_output=True, text=True,
                                timeout=60)
        record = json.loads(result.stdout)
        self.assertFalse(record["local_only_probe_route"]["model_service_contacted"])
        self.assertEqual(record["hf_trainable_w0"]["state_sha256"], self.hf["state_sha256"])
        self.assertFalse(record["runtime_equals_hf_weights_proven"])
        self.assertEqual(record["gguf"]["lora_adapter_sha256"],
                         self.runtime["lora_adapter"]["sha256"])


if __name__ == "__main__":
    unittest.main()
