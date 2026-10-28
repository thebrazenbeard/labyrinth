"""Dependency-free contract checks for the hosted pretrained-model experiment."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci_pilot_verdict import verdict
from scripts.stage_smol_checkpoint import sha256, stage_snapshot


class CheckpointStagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "cache"
        self.source.mkdir()
        (self.source / "model.safetensors").write_bytes(b"fake test checkpoint")
        (self.source / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
        (self.source / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
        (self.source / "unsafe.py").write_text('raise RuntimeError("must never copy")', encoding="utf-8")
        self.destination = self.root / "staged"

    def test_requires_pinned_weight_hash_before_creating_any_file(self):
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            stage_snapshot(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_stages_only_explicit_data_allowlist_and_rejects_overwrite(self):
        with patch("scripts.stage_smol_checkpoint.WEIGHTS_SHA256", sha256(self.source / "model.safetensors")):
            result = stage_snapshot(self.source, self.destination)
        self.assertEqual(set(result), {"model.safetensors", "config.json", "tokenizer.json"})
        self.assertEqual(set(p.name for p in self.destination.iterdir()), set(result))
        self.assertFalse(any(p.is_symlink() for p in self.destination.iterdir()))
        with self.assertRaises(FileExistsError):
            stage_snapshot(self.source, self.destination)

    def test_refuses_missing_tokenizer_and_existing_symlink_destination(self):
        (self.source / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "tokenizer.json"):
            stage_snapshot(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.destination.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            stage_snapshot(self.source, self.destination)


class RealPretrainedVerdictTests(unittest.TestCase):
    def setUp(self):
        self.train = {
            "initial_lora_parameter_sha256": "a" * 64,
            "sham_lora_parameter_sha256": "a" * 64,
            "candidate_lora_parameter_sha256": "b" * 64,
            "candidate_lora_tensor_value_sha256": "b" * 64,
            "control_data": {"sham_is_compute_matched": True,
                             "sham_is_token_exposure_matched": True},
            "training": {"sham": {"steps": 32, "input_tokens_seen": 420},
                         "candidate": {"steps": 32, "input_tokens_seen": 420}},
            "evaluations": {
                name: {"correct": count, "total": 8}
                for name, count in {
                    "baseline": 0, "sham": 0, "corrupted": 1, "candidate": 2
                }.items()
            },
        }
        self.reload = {
            "process_id_differs": True,
            "matches_original_candidate_trials": True,
            "reloaded_lora_tensor_value_sha256": "b" * 64,
        }
        self.audit = {"status": "PASS"}

    def test_even_improvement_never_graduates_from_public_pilot(self):
        result = verdict(self.train, self.reload, self.audit)
        self.assertTrue(result["demonstrated_semantic_advantage_over_all_controls"])
        self.assertEqual(result["candidate_minimum_count_margin"], 1)
        self.assertEqual(result["status"], "EVIDENCE_ONLY_DO_NOT_PROMOTE")
        self.assertFalse(result["graduation_eligible"])

    def test_format_only_or_zero_gain_is_not_reasoning_improvement(self):
        self.train["evaluations"]["candidate"]["correct"] = 0
        result = verdict(self.train, self.reload, self.audit)
        self.assertFalse(result["demonstrated_semantic_advantage_over_all_controls"])
        self.assertFalse(result["graduation_eligible"])

    def test_refuses_unverified_or_unchanged_adapter(self):
        self.audit["status"] = "INCOMPLETE"
        with self.assertRaisesRegex(ValueError, "not complete"):
            verdict(self.train, self.reload, self.audit)
        self.audit["status"] = "PASS"
        self.reload["matches_original_candidate_trials"] = False
        with self.assertRaisesRegex(ValueError, "reload"):
            verdict(self.train, self.reload, self.audit)
        self.reload["matches_original_candidate_trials"] = True
        self.train["candidate_lora_parameter_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "did not change"):
            verdict(self.train, self.reload, self.audit)

    def test_refuses_forged_compute_matched_sham(self):
        for tamper in ("digest", "steps", "tokens", "claim"):
            with self.subTest(tamper=tamper):
                original = self.train
                self.train = {**original,
                              "control_data": dict(original["control_data"]),
                              "training": {arm: dict(record)
                                           for arm, record in original["training"].items()}}
                if tamper == "digest":
                    self.train["sham_lora_parameter_sha256"] = "c" * 64
                elif tamper == "steps":
                    self.train["training"]["sham"]["steps"] = 31
                elif tamper == "tokens":
                    self.train["training"]["sham"]["input_tokens_seen"] = 419
                else:
                    self.train["control_data"]["sham_is_compute_matched"] = False
                with self.assertRaisesRegex(ValueError, "sham"):
                    verdict(self.train, self.reload, self.audit)
                self.train = original


if __name__ == "__main__":
    unittest.main()
