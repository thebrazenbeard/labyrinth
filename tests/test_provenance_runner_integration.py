"""Source-level integration tests for the local-base-byte MiniLM wrapper.

Synthetic subprocesses intentionally do not prove physical pretrained learning.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from labyrinth.pretrained_model_provenance import capture_pretrained_model_manifest
from scripts.qualify_model_bytes_provenance import (
    ModelByteWindowError, _RUNNER, main, run_bound_pilot,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProvenanceRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.model = self.base / "base"
        self.model.mkdir()
        (self.model / "config.json").write_bytes(b'{"model_type":"bert"}\n')
        (self.model / "tokenizer.json").write_bytes(b'{"mock":"vocab"}\n')
        (self.model / "model.safetensors").write_bytes(b"physically irrelevant test bytes")
        self.manifest = capture_pretrained_model_manifest(
            self.model, model_id_claim="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.manifest_path = self.base / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.run_parent = self.base / "runs"
        self.run_parent.mkdir()
        self.calls = []

    def _args(self, **overrides):
        values = dict(model_root=self.model, manifest_json=self.manifest_path,
                      trusted_manifest_sha256=self.manifest["manifest_sha256"],
                      output_parent=self.run_parent, run_name="one", seed=42,
                      steps_per_cycle=4, learning_rate=.002)
        values.update(overrides)
        return values

    def _fake_neural_runner(self, args, **kwargs):
        self.calls.append((args, kwargs))
        self.assertEqual(args[:3][1:], ["-m", "scripts.qualify_minilm_cpu"])
        self.assertEqual(args[args.index("--model-id") + 1], str(self.model.absolute()))
        self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(kwargs["env"]["TRANSFORMERS_OFFLINE"], "1")
        result = self.run_parent / "one"
        result.mkdir()
        payload = {
            "schema": "labyrinth.minilm-cpu-development.v1",
            "model_id": str(self.model.absolute()),
            "harness_sha256": _sha(_RUNNER),
            "seed": 42, "steps_per_cycle": 4, "learning_rate": .002,
            "qualification": "PUBLIC_DEVELOPMENT_ONLY",
            "private_final_exam": "NOT_RUN",
            "graduation_authorized": False,
            "physical_model_byte_custody": "SELF_REPORTED",
            "learning_qualification": "NOT_ESTABLISHED_UNTIL_INDEPENDENT_EVALUATION",
            "arms": {name: {} for name in ("correct", "zero_update", "inverted")},
        }
        (result / "report.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def test_no_run_on_dry_run_and_cli_no_side_effects(self):
        result = run_bound_pilot(**self._args())
        self.assertFalse(result["execute"])
        self.assertFalse(result["actual_loader_file_reads_attested"])
        self.assertEqual(list(self.run_parent.iterdir()), [])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["--model-root", str(self.model),
                         "--manifest-json", str(self.manifest_path),
                         "--trusted-manifest-sha256", self.manifest["manifest_sha256"],
                         "--output-parent", str(self.run_parent), "--run-name", "cli"])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(output.getvalue())["execute"])
        self.assertEqual(list(self.run_parent.iterdir()), [])

    def test_receipt_binds_pre_post_verified_model_and_exact_report_bytes(self):
        result = run_bound_pilot(**self._args(), execute=True,
                                 _invoke=self._fake_neural_runner)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(result["preflight_manifest_verified"])
        self.assertTrue(result["postflight_manifest_verified"])
        self.assertEqual(result["model_manifest_sha256"], self.manifest["manifest_sha256"])
        self.assertEqual(result["runner_sha256"], _sha(_RUNNER))
        self.assertEqual(result["pilot_report_sha256"],
                         _sha(self.run_parent / "one" / "report.json"))
        self.assertFalse(result["actual_loader_file_reads_attested"])
        self.assertFalse(result["publisher_origin_attested"])
        self.assertFalse(result["training_worker_identity_attested"])
        self.assertFalse(result["graduation_authorized"])
        self.assertEqual(json.loads((self.run_parent / "one" /
                                     "provenance_binding.json").read_text()), result)
        with self.assertRaises(ModelByteWindowError):
            run_bound_pilot(**self._args(), execute=True,
                            _invoke=self._fake_neural_runner)
        self.assertEqual(len(self.calls), 1, "existing run must fail before subprocess")

    def test_wrong_trusted_digest_rejected_before_launch_or_output(self):
        with self.assertRaisesRegex(ModelByteWindowError, "trusted manifest"):
            run_bound_pilot(**self._args(trusted_manifest_sha256="0" * 64),
                            execute=True, _invoke=self._fake_neural_runner)
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.run_parent.iterdir()), [])

    def test_mutated_model_during_child_execution_prevents_custody_receipt(self):
        def mutated(*args, **kwargs):
            result = self._fake_neural_runner(*args, **kwargs)
            (self.model / "model.safetensors").write_bytes(b"mutated during training")
            return result

        with self.assertRaisesRegex(ModelByteWindowError, "differ"):
            run_bound_pilot(**self._args(), execute=True, _invoke=mutated)
        self.assertFalse((self.run_parent / "one" / "provenance_binding.json").exists())

    def test_forged_report_model_root_or_graduation_rejected(self):
        def forged(*args, **kwargs):
            result = self._fake_neural_runner(*args, **kwargs)
            path = self.run_parent / "one" / "report.json"
            value = json.loads(path.read_text())
            value["model_id"] = "sentence-transformers/all-MiniLM-L6-v2"
            value["graduation_authorized"] = True
            path.write_text(json.dumps(value))
            return result

        with self.assertRaisesRegex(ModelByteWindowError, "binding"):
            run_bound_pilot(**self._args(), execute=True, _invoke=forged)
        self.assertFalse((self.run_parent / "one" / "provenance_binding.json").exists())

    def test_missing_report_or_nonzero_worker_fails_closed(self):
        def no_report(args, **kwargs):
            (self.run_parent / "one").mkdir()
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with self.assertRaises(ModelByteWindowError):
            run_bound_pilot(**self._args(), execute=True, _invoke=no_report)
        self.assertFalse((self.run_parent / "one" / "provenance_binding.json").exists())

    def test_nonzero_worker_does_not_receive_receipt(self):
        def failed(args, **kwargs):
            (self.run_parent / "one").mkdir()
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="failure")

        with self.assertRaisesRegex(ModelByteWindowError, "subprocess failed"):
            run_bound_pilot(**self._args(), execute=True, _invoke=failed)
        self.assertFalse((self.run_parent / "one" / "provenance_binding.json").exists())

    def test_linked_output_directory_rejected_before_receipt_write(self):
        outside = self.base / "outside"
        outside.mkdir()

        def linked(args, **kwargs):
            try:
                (self.run_parent / "one").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable on this platform")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with self.assertRaisesRegex(ModelByteWindowError, "unsafe runner output"):
            run_bound_pilot(**self._args(), execute=True, _invoke=linked)
        self.assertFalse((outside / "provenance_binding.json").exists())

    def test_manifest_json_self_edit_does_not_override_external_trust(self):
        (self.model / "model.safetensors").write_bytes(b"another untrusted model")
        amended = capture_pretrained_model_manifest(self.model, model_id_claim=self.manifest["model_id_claim"])
        self.manifest_path.write_text(json.dumps(amended), encoding="utf-8")
        with self.assertRaisesRegex(ModelByteWindowError, "trusted manifest"):
            run_bound_pilot(**self._args(), execute=True,
                            _invoke=self._fake_neural_runner)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
