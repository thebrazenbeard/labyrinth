"""Dependency-free tests for the Minotaur model's provenance and gate boundary."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.minotaur_model import MODEL_SCHEMA, inference, package_model


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MinotaurArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.run = root / "training"
        self.adapter = self.run / "candidate_adapter"
        self.adapter.mkdir(parents=True)
        (self.adapter / "adapter_model.safetensors").write_bytes(b"trained-weights-fixture")
        (self.adapter / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": "/private/machine/base",
            "r": 4, "lora_alpha": 8,
        }), encoding="utf-8")
        (self.adapter / "README.md").write_text("original-run-metadata")
        self.reload = self.run / "reload_report.json"
        self.package = root / "minotaur"
        adapter_hashes = {p.name: _sha(p) for p in self.adapter.iterdir()}
        self.train = {
            "schema": "labyrinth.hf-lora-pilot.v2",
            "source": {"commit": "f" * 40, "source_tree_sha256": "e" * 64},
            "config": {"seed": 3, "steps": 2, "train_count": 2, "eval_count": 2,
                       "tiers": [1], "max_tokens": 256, "max_new_tokens": 16,
                       "learning_rate": .0002, "lora_rank": 4, "lora_alpha": 8,
                       "device": "cpu", "model_path": "/private/machine/base",
                       "output_dir": str(self.run)},
            "base_weights_sha256": {"model.safetensors": "a" * 64},
            "model_metadata_sha256": {"config.json": "b" * 64},
            "initial_lora_parameter_sha256": "0" * 64,
            "candidate_lora_parameter_sha256": "1" * 64,
            "candidate_lora_tensor_value_sha256": "1" * 64,
            "candidate_adapter_sha256": adapter_hashes,
            "training": {"candidate": {"steps": 2}},
            "evaluations": {name: {"correct": 0, "total": 2, "format_valid": 1}
                            for name in ("baseline", "sham", "candidate", "corrupted")},
        }
        self._write_receipts()

    def _write_receipts(self):
        report = self.run / "pilot_report.json"
        report.write_text(json.dumps(self.train), encoding="utf-8")
        self.reload.write_text(json.dumps({
            "schema": "labyrinth.hf-lora-reload-eval.v2",
            "training_report_sha256": _sha(report),
            "adapter_sha256": self.train["candidate_adapter_sha256"],
            "reloaded_lora_tensor_value_sha256": "1" * 64,
            "matches_original_candidate_trials": True,
        }), encoding="utf-8")

    def _package(self):
        return package_model(
            self.run, self.reload, self.package,
            base_repository="HuggingFaceTB/SmolLM2-135M-Instruct",
            base_revision="c" * 40,
        )

    def test_packages_distinct_advisory_model_without_private_paths(self):
        result = self._package()
        self.assertEqual(result["schema"], MODEL_SCHEMA)
        self.assertEqual(result["model_name"], "Minotaur")
        self.assertEqual(result["graduation_authority"], "none")
        self.assertEqual(result["qualification"], "NOT_ESTABLISHED")
        self.assertIn("not trainee verification", result["trained_behavior"])
        self.assertEqual(result["adaptation"]["completed_steps"], 2)
        adapter_config = json.loads(
            (self.package / "adapter" / "adapter_config.json").read_text())
        self.assertEqual(adapter_config["base_model_name_or_path"],
                         "HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertNotIn("/private/", (self.package / "minotaur_manifest.json").read_text())
        self.assertNotIn("/private/", (self.package / "README.md").read_text())
        self.assertEqual(result["adapter_files_sha256"]["adapter_model.safetensors"],
                         _sha(self.adapter / "adapter_model.safetensors"))
        self.assertFalse((self.package / "adapter" / "README.md").exists())
        with self.assertRaises(FileExistsError):
            self._package()

    def test_rejects_forged_reload_and_unmodified_adapter(self):
        self.train["candidate_lora_parameter_sha256"] = "0" * 64
        self._write_receipts()
        with self.assertRaisesRegex(ValueError, "changed LoRA"):
            self._package()
        self.assertFalse(self.package.exists())

        self.train["candidate_lora_parameter_sha256"] = "1" * 64
        self._write_receipts()
        reload_data = json.loads(self.reload.read_text())
        reload_data["matches_original_candidate_trials"] = False
        self.reload.write_text(json.dumps(reload_data))
        with self.assertRaisesRegex(ValueError, "reproduce"):
            self._package()
        self.assertFalse(self.package.exists())

    def test_rejects_adapter_tamper_before_packaging(self):
        (self.adapter / "adapter_model.safetensors").write_bytes(b"swapped")
        with self.assertRaisesRegex(ValueError, "adapter files"):
            self._package()
        self.assertFalse(self.package.exists())

    def test_invalid_peft_config_does_not_leave_partial_package(self):
        config = self.adapter / "adapter_config.json"
        config.write_text("[]", encoding="utf-8")
        self.train["candidate_adapter_sha256"][config.name] = _sha(config)
        self._write_receipts()
        with self.assertRaisesRegex(ValueError, "PEFT adapter configuration"):
            self._package()
        self.assertFalse(self.package.exists())

    def test_refuses_any_graduation_authority_before_inference(self):
        self._package()
        manifest_file = self.package / "minotaur_manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["graduation_authority"] = "automatic"
        manifest_file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "advisory-only"):
            inference(self.package, self.run, "Should a trainee graduate?")


if __name__ == "__main__":
    unittest.main()
