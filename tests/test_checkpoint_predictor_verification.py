"""Fresh-process, actual-byte-dependent public developmental predictor checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.entrance_state_v1 import (
    _inventory, _state_digest, build_entrance_state, save_entrance_state_once,
)
from labyrinth.checkpoint_predictor_verification import (
    compare_fresh_process_checkpoints, validate_local_predictor_endpoint,
)


class FreshCheckpointPredictorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.w0 = self.root / "w0"
        self.wt = self.root / "wt"
        for directory in (self.base, self.w0, self.wt):
            directory.mkdir()
        (self.base / "model-00001-of-00002.safetensors").write_bytes(b"fake shard one")
        (self.base / "model-00002-of-00002.safetensors").write_bytes(b"fake shard two")
        (self.base / "model.safetensors.index.json").write_text(
            '{"weight_map":{"layer.0":"model-00001-of-00002.safetensors",'
            '"layer.1":"model-00002-of-00002.safetensors"}}')
        (self.base / "config.json").write_text('{"model_type":"toy"}')
        (self.base / "tokenizer.json").write_text('{"type":"toy"}')
        for directory, payload in ((self.w0, b"TOY_WRONG_RESPONSE"),
                                   (self.wt, b"TOY_CORRECT_RESPONSE")):
            (directory / "adapter_model.safetensors").write_bytes(payload)
            (directory / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        self.receipt = self.root / "receipt.json"
        self.receipt.write_text('{"event":"dev-only"}')
        self.entrance = self.root / "entrance.json"
        baseline_prompt = "Calculate 3 + 4."
        snapshot = build_entrance_state(
            base_dir=self.base, adapter_dir=self.w0,
            scope_sha256="a" * 64, source_tree_sha256="b" * 64,
            source_receipt_path=self.receipt,
            source_receipt_sha256=hashlib.sha256(self.receipt.read_bytes()).hexdigest(),
            baseline_observations=[{
                "case_id": "devbaseline-0", "skill": "arithmetic", "prompt": baseline_prompt,
                "prompt_sha256": hashlib.sha256(baseline_prompt.encode()).hexdigest(),
                "score": 0, "max_score": 1, "scorer": "local-arithmetic-oracle",
            }],
        )
        save_entrance_state_once(snapshot, self.entrance)
        self.successor_sha = _state_digest(
            _inventory(self.base, kind="base"), _inventory(self.wt, kind="adapter"),
        )
        self.probes = [
            {"case_id": "devprobe-0", "skill": "arithmetic", "prompt": "Calculate 3 + 4.",
             "expected": "7"},
            {"case_id": "devprobe-1", "skill": "arithmetic", "prompt": "Calculate 9 - 2.",
             "expected": "7"},
        ]

    def _compare(self, **kwargs):
        params = {"entrance_path": self.entrance, "successor_adapter_dir": self.wt,
                  "public_probes": self.probes, "predictor": "mock-local-file-arithmetic",
                  "expected_successor_sha256": self.successor_sha}
        params.update(kwargs)
        return compare_fresh_process_checkpoints(**params)

    def test_actual_bytes_affect_predictions_in_distinct_fresh_processes(self):
        report = self._compare()
        self.assertEqual(report["development_status"], "PUBLIC_PROBE_GAIN_OBSERVED")
        self.assertEqual(report["w0"]["correct"], 0)
        self.assertEqual(report["wt"]["correct"], 2)
        self.assertTrue(report["checkpoint_bytes_differ"])
        self.assertNotEqual(report["w0"]["pid"], report["wt"]["pid"])
        self.assertNotEqual(report["w0"]["loaded_state_sha256"], report["wt"]["loaded_state_sha256"])
        self.assertEqual(report["evidence_kind"], "LOCAL_TEST_PREDICTOR_NOT_REAL_LLM")
        self.assertFalse(report["causal_training_effect_established"])
        self.assertFalse(report["graduation_authorized"])

    def test_changed_bytes_without_predictive_skill_is_not_learning(self):
        report = compare_fresh_process_checkpoints(
            entrance_path=self.entrance, successor_adapter_dir=self.wt,
            public_probes=self.probes, predictor="mock-bytes-digest-only",
            expected_successor_sha256=self.successor_sha,
        )
        self.assertTrue(report["checkpoint_bytes_differ"])
        self.assertEqual(report["w0"]["correct"], 0)
        self.assertEqual(report["wt"]["correct"], 0)
        self.assertEqual(report["development_status"], "NO_PUBLIC_PROBE_GAIN")

    def test_forged_or_stale_checkpoint_identity_refused(self):
        with self.assertRaisesRegex(ValueError, "successor .*digest"):
            self._compare(expected_successor_sha256="f" * 64)
        (self.w0 / "adapter_model.safetensors").write_bytes(b"tampered W0")
        with self.assertRaisesRegex(ValueError, "adapter bytes changed"):
            self._compare()

    def test_complete_base_shard_and_tokenizer_binding_not_single_file_hash(self):
        (self.base / "model-00002-of-00002.safetensors").write_bytes(b"swapped second shard")
        with self.assertRaisesRegex(ValueError, "base model/tokenizer bytes changed"):
            self._compare()

    def test_forged_worker_nonce_and_stale_pid_refused(self):
        from labyrinth import checkpoint_predictor_verification as verifier
        genuine = verifier._run_worker

        def spoofed(*args, **kwargs):
            response = genuine(*args, **kwargs)
            response["nonce"] = "0" * 64
            return response

        with patch.object(verifier, "_run_worker", side_effect=spoofed):
            with self.assertRaisesRegex(ValueError, "nonce"):
                self._compare()

        def stale_source(*args, **kwargs):
            response = genuine(*args, **kwargs)
            response["worker_source_sha256"] = "f" * 64
            return response

        with patch.object(verifier, "_run_worker", side_effect=stale_source):
            with self.assertRaisesRegex(ValueError, "source"):
                self._compare()

        def stale_pid(*args, **kwargs):
            response = genuine(*args, **kwargs)
            response["pid"] = __import__("os").getpid()
            return response

        with patch.object(verifier, "_run_worker", side_effect=stale_pid):
            with self.assertRaisesRegex(ValueError, "independent fresh process"):
                self._compare()

    def test_no_proxy_local_only_inputs_and_train_safe_unique_probes(self):
        for endpoint in ("https://127.0.0.1:8000", "http://example.com:8000",
                         "http://localhost.evil.com:8000", "file:///etc/passwd"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_local_predictor_endpoint(endpoint)
        self.assertEqual(validate_local_predictor_endpoint("http://127.0.0.1:8000"),
                         "http://127.0.0.1:8000")
        with self.assertRaisesRegex(ValueError, "development-only"):
            self._compare(public_probes=[{**self.probes[0], "case_id": "private-final-0"}])
        with self.assertRaisesRegex(ValueError, "repeated"):
            self._compare(public_probes=[self.probes[0], self.probes[0]])


if __name__ == "__main__":
    unittest.main()
