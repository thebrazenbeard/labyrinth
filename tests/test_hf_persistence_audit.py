"""Fail-closed paired LoRA persistence checks without importing Torch."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.test_hf_report_audit import pilot_fixture
from labyrinth.hf_pilot import EVAL_SCHEMA


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class HfPersistenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.run = self.root / "run"
        self.base.mkdir()
        self.run.mkdir()
        (self.base / "config.json").write_bytes(b"{}")
        (self.base / "model.safetensors").write_bytes(b"base")
        adapter = self.run / "candidate_adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        self.training = pilot_fixture()
        self.training["base_weights_sha256"] = {"model.safetensors": _sha(b"base")}
        self.training["model_metadata_sha256"] = {"config.json": _sha(b"{}")}
        self.training["candidate_adapter_sha256"] = {"adapter_model.safetensors": _sha(b"adapter")}
        (self.run / "pilot_report.json").write_text(json.dumps(self.training), encoding="utf-8")
        self.reload = {
            "schema": EVAL_SCHEMA,
            "training_report_sha256": _sha((self.run / "pilot_report.json").read_bytes()),
            "adapter_sha256": self.training["candidate_adapter_sha256"],
            "process_id": 4321,
            "reloaded_lora_tensor_value_sha256": self.training["candidate_lora_tensor_value_sha256"],
            "evaluation": self.training["evaluations"]["candidate"],
            "claimed_fresh_process": None,
            "process_id_differs": True,
            "matches_original_candidate_trials": True,
        }
        self.reload_path = self.root / "reload.json"
        self._write_reload()

    def _write_reload(self):
        self.reload_path.write_text(json.dumps(self.reload), encoding="utf-8")

    def _audit(self):
        from labyrinth.hf_persistence_audit import audit_hf_persistence
        return audit_hf_persistence(self.run, self.reload_path, self.base)

    def test_bytes_and_matched_receipts_pass_internal_gate_not_attestation(self):
        result = self._audit()
        self.assertEqual(result["status"], "PASS", result)
        self.assertTrue(result["base_bytes_verified"])
        self.assertTrue(result["adapter_bytes_verified"])
        self.assertFalse(result["independent_execution_attested"])
        self.assertFalse(result["reasoning_transfer_demonstrated"])

    def test_same_pid_bypasses_self_reported_true_flag_never(self):
        self.reload["process_id"] = self.training["training_process_id"]
        self._write_reload()
        self.assertEqual(self._audit()["status"], "FAIL")

    def test_reported_pid_bool_and_false_flag_never_count_as_reload(self):
        self.reload["process_id"] = True
        self._write_reload()
        self.assertEqual(self._audit()["status"], "FAIL")
        self.reload["process_id"] = 4321
        self.reload["process_id_differs"] = False
        self._write_reload()
        self.assertEqual(self._audit()["status"], "FAIL")

    def test_forged_trial_match_flag_does_not_override_actual_mismatch(self):
        self.reload["evaluation"]["trials"][0]["response_preview"] = "ANSWER: 9"
        self._write_reload()
        self.assertEqual(self._audit()["status"], "FAIL")

    def test_changed_training_report_bytes_fail_even_if_semantics_unchanged(self):
        (self.run / "pilot_report.json").write_text(json.dumps(self.training, indent=2), encoding="utf-8")
        self.assertEqual(self._audit()["status"], "FAIL")

    def test_adapter_checkpoint_mutation_is_detected(self):
        (self.run / "candidate_adapter" / "adapter_model.safetensors").write_bytes(b"changed")
        self.assertEqual(self._audit()["status"], "FAIL")

    def test_base_checkpoint_mutation_is_detected(self):
        (self.base / "model.safetensors").write_bytes(b"changed")
        self.assertEqual(self._audit()["status"], "FAIL")

    def test_reported_adapter_manifest_swap_is_detected(self):
        self.reload["adapter_sha256"] = {"adapter_model.safetensors": "0" * 64}
        self._write_reload()
        self.assertEqual(self._audit()["status"], "FAIL")

    def test_reload_validation_and_parsing_use_one_verified_snapshot(self):
        # The initial regular reload receipt is inconsistent. Replacing its
        # path after the bounded nonsymlink read must not replace the bytes
        # being evaluated with a valid, symlink-backed second read.
        from labyrinth import hf_persistence_audit as auditor

        clean = self.root / "other_reload.json"
        clean.write_text(json.dumps(self.reload), encoding="utf-8")
        self.reload["training_report_sha256"] = "0" * 64
        self._write_reload()

        original_read = auditor._read_regular_bounded

        def swap_after_verified_read(path):
            raw = original_read(path)
            if Path(path) == self.reload_path:
                self.reload_path.unlink()
                self.reload_path.symlink_to(clean)
            return raw

        with patch.object(auditor, "_read_regular_bounded",
                          side_effect=swap_after_verified_read):
            result = self._audit()
        self.assertEqual(result["status"], "FAIL", result)
        self.assertTrue(any("training-report bytes" in reason
                            for reason in result["failures"]), result)

    def test_receipt_replaced_with_symlink_between_check_and_open_fails(self):
        # The initial reload path is an inconsistent regular file. Replace
        # that pathname with a symlink to clean bytes at the actual open.
        # The handle must be linked back to the checked file's identity.
        clean = self.root / "clean_reload.json"
        clean.write_text(json.dumps(self.reload), encoding="utf-8")
        self.reload["training_report_sha256"] = "0" * 64
        self._write_reload()
        original_open = Path.open
        swaps = []

        def replace_before_open(path, *args, **kwargs):
            if path == self.reload_path and not swaps:
                path.unlink()
                path.symlink_to(clean)
                swaps.append("swapped")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", replace_before_open):
            result = self._audit()
        self.assertEqual(swaps, ["swapped"])
        self.assertEqual(result["status"], "FAIL", result)
        self.assertTrue(any("receipt identity changed during open" in reason
                            for reason in result["failures"]), result)

    def test_inplace_reload_overwrite_during_read_cannot_pass(self):
        # Source file stays the same inode throughout. A competing writer
        # changes an initially invalid reload receipt to valid bytes while
        # this auditor is reading it. Identity-only checks miss the mutation.
        clean = self.reload_path.read_bytes()
        self.reload["training_report_sha256"] = "0" * 64
        self._write_reload()
        self.assertEqual(len(self.reload_path.read_bytes()), len(clean))
        original_open = Path.open
        mutations = []

        class MutatingReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def fileno(self):
                return self.stream.fileno()

            def read(self, *args):
                with original_open(self_path, "r+b") as writer:
                    writer.seek(0)
                    writer.write(clean)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.utime(self_path, ns=(initial_stat.st_atime_ns,
                                       initial_stat.st_mtime_ns + 2_000_000_000))
                mutations.append("overwrote-in-place")
                return self.stream.read(*args)

        self_path = self.reload_path
        initial_stat = self_path.stat()

        def replace_reader(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if path == self_path and args and args[0] == "rb":
                return MutatingReader(stream)
            return stream

        with patch.object(Path, "open", replace_reader):
            result = self._audit()
        self.assertEqual(mutations, ["overwrote-in-place"])
        self.assertEqual(result["status"], "FAIL", result)
        self.assertTrue(any("receipt changed during read" in reason
                            for reason in result["failures"]), result)

    def test_duplicate_key_reload_json_is_rejected(self):
        self.reload_path.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        self.assertEqual(self._audit()["status"], "FAIL")

    def test_symlink_adapter_is_refused(self):
        adapter = self.run / "candidate_adapter" / "adapter_model.safetensors"
        other = self.root / "other.bin"
        other.write_bytes(b"adapter")
        adapter.unlink()
        adapter.symlink_to(other)
        self.assertEqual(self._audit()["status"], "FAIL")

    def test_current_source_mismatch_is_incomplete_not_pass(self):
        self.training["source"]["source_tree_sha256"] = "a" * 64
        (self.run / "pilot_report.json").write_text(json.dumps(self.training), encoding="utf-8")
        self.reload["training_report_sha256"] = _sha((self.run / "pilot_report.json").read_bytes())
        self._write_reload()
        self.assertEqual(self._audit()["status"], "INCOMPLETE")

    def test_cli_gate_exits_nonzero_if_unqualified(self):
        cmd = [sys.executable, "-m", "labyrinth", "audit-hf-persistence", "--run-dir",
               str(self.run), "--reload-report", str(self.reload_path),
               "--base-model-dir", str(self.base)]
        completed = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "PASS")
        self.reload["process_id"] = 1234
        self._write_reload()
        completed = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
