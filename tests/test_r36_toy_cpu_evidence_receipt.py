"""Source-only TOY CPU report preservation and hostile receipt tests."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import qualify_toy_cpu_evidence_r36 as evidence
from scripts.qualify_v2_checkpoint_cpu_r22 import STATUS as CHECKPOINT_STATUS
from scripts.qualify_v2_weight_updates_cpu_r25 import STATUS as WEIGHT_STATUS


SOURCE_HEAD = "a" * 40
SOURCE_TREE = "b" * 40
RUN_ID = "r36.test-run"


def reports():
    weights = {
        "status": WEIGHT_STATUS,
        "physical_cpu_toy_executed": True,
        "actual_trainable_tensor_updates_both_cycles": True,
        "frozen_base_weights_unchanged": True,
        "adamw_moments_advanced_both_cycles": True,
        "frozen_tensor_sha256": ["1" * 64] * 3,
        "trainable_tensor_sha256": ["2" * 64, "3" * 64, "4" * 64],
        "optimizer_moments_sha256": ["5" * 64, "6" * 64],
        "optimizer_steps": [4, 8],
        "pretrained_model_loaded": False,
        "host_custody_verified": False,
        "neural_generalization_verified": False,
        "graduation_authorized": False,
        "persistent_host_artifacts": False,
    }
    checkpoint = {
        "status": CHECKPOINT_STATUS,
        "physical_cpu_toy_executed": True,
        "pretrained_model_loaded": False,
        "maze_training_prompts_executed": True,
        "optimizer_fullstate_continuity_matches_uninterrupted_toy": True,
        "final_fresh_fullstate_restore_verified": True,
        "checkpoint_cycle_receipt_sha256": ["7" * 64, "8" * 64],
        "toy_final_state_sha256": "9" * 64,
        "synthetic_identity_pins": True,
        "independent_host_custody_verified": False,
        "independent_antirollback_verified": False,
        "neural_generalization_verified": False,
        "graduation_authorized": False,
        "persistent_host_artifacts": False,
    }
    return weights, checkpoint


class ToyEvidenceReceiptTests(unittest.TestCase):
    def make(self):
        return evidence.create_receipt(*reports(), source_head=SOURCE_HEAD,
                                       source_tree=SOURCE_TREE, run_id=RUN_ID)

    def inspect(self, raw, *, digest=None):
        return evidence.inspect_receipt(
            raw, expected_receipt_sha256=digest or sha256(raw).hexdigest(),
            expected_source_head=SOURCE_HEAD,
            expected_source_tree=SOURCE_TREE, expected_run_id=RUN_ID,
        )

    def test_valid_receipt_preserves_both_raw_probe_reports_without_authority(self):
        raw = self.make()
        self.assertEqual(json.loads(raw)["weight_report"], reports()[0])
        self.assertEqual(json.loads(raw)["checkpoint_report"], reports()[1])
        result = self.inspect(raw)
        self.assertEqual(result["status"], "SOURCE_ONLY_NONAUTHORIZING")
        self.assertIs(result["physical_toy_evidence_consistent"], True)
        for key in ("graduation_authorized", "deployment_authorized",
                    "identity_consent_transfer_authorized", "host_custody_verified",
                    "pretrained_model_learning_verified"):
            self.assertIs(result[key], False)

    def test_resealed_tampering_is_rejected_by_independently_pinned_digest(self):
        raw = self.make()
        original_digest = sha256(raw).hexdigest()
        forged = json.loads(raw)
        forged["weight_report"]["trainable_tensor_sha256"][1] = "f" * 64
        forged_raw = evidence.canonical_bytes(forged)
        with self.assertRaises(evidence.EvidenceError):
            self.inspect(forged_raw, digest=original_digest)

    def test_source_and_run_pins_are_independent_of_receipt(self):
        raw = self.make()
        for field, bad in (("expected_source_head", "c" * 40),
                           ("expected_source_tree", "d" * 40),
                           ("expected_run_id", "other-run")):
            kwargs = dict(expected_receipt_sha256=sha256(raw).hexdigest(),
                          expected_source_head=SOURCE_HEAD,
                          expected_source_tree=SOURCE_TREE,
                          expected_run_id=RUN_ID)
            kwargs[field] = bad
            with self.subTest(field=field), self.assertRaises(evidence.EvidenceError):
                evidence.inspect_receipt(raw, **kwargs)

    def test_authority_field_upgrade_and_false_as_integer_are_rejected(self):
        for mutate in (
            lambda d: d["weight_report"].__setitem__("graduation_authorized", True),
            lambda d: d["checkpoint_report"].__setitem__("independent_antirollback_verified", 0),
            lambda d: d.__setitem__("deployment_authorized", True),
        ):
            forged = json.loads(self.make())
            mutate(forged)
            with self.assertRaises(evidence.EvidenceError):
                self.inspect(evidence.canonical_bytes(forged))

    def test_malformed_unknown_duplicate_and_oversize_json_fail_closed(self):
        raw = self.make()
        bad = (
            raw[:-1] + b"garbage}",
            raw.replace(b'"run_id":', b'"run_id":"evil","run_id":', 1),
            raw.replace(b'"schema":', b'"unknown_effect":true,"schema":', 1),
            b"{" + b" " * 32769 + b"}",
        )
        for index, forged in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(evidence.EvidenceError):
                self.inspect(forged)

    def test_noop_or_duplicate_optimizer_state_cannot_be_packaged(self):
        for corrupt in (
            lambda w, c: w["trainable_tensor_sha256"].__setitem__(2, w["trainable_tensor_sha256"][1]),
            lambda w, c: w["optimizer_moments_sha256"].__setitem__(1, w["optimizer_moments_sha256"][0]),
            lambda w, c: w.__setitem__("optimizer_steps", [4, True]),
            lambda w, c: c["checkpoint_cycle_receipt_sha256"].__setitem__(1, c["checkpoint_cycle_receipt_sha256"][0]),
        ):
            weight, checkpoint = reports()
            corrupt(weight, checkpoint)
            with self.assertRaises(evidence.EvidenceError):
                evidence.create_receipt(weight, checkpoint, source_head=SOURCE_HEAD,
                                        source_tree=SOURCE_TREE, run_id=RUN_ID)

    def test_preflight_does_not_create_output_and_execute_publishes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "toy-evidence.json"
            kwargs = ["--output", str(target), "--source-head", SOURCE_HEAD,
                      "--source-tree", SOURCE_TREE, "--run-id", RUN_ID]
            self.assertEqual(evidence.main(kwargs), 0)
            self.assertFalse(target.exists())
            weight, checkpoint = reports()
            with patch.object(evidence, "_run_probe", side_effect=(weight, checkpoint)) as runner:
                self.assertEqual(evidence.main(["--execute", *kwargs]), 0)
                self.assertEqual(runner.call_count, 2)
                self.assertEqual(self.inspect(target.read_bytes())["status"],
                                 "SOURCE_ONLY_NONAUTHORIZING")
                with self.assertRaises(evidence.EvidenceError):
                    evidence.main(["--execute", *kwargs])
            self.assertEqual(self.inspect(target.read_bytes())["status"],
                             "SOURCE_ONLY_NONAUTHORIZING")

    def test_invalid_output_is_rejected_before_starting_physical_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            invalid = Path(tmp) / "missing-parent" / "evidence.json"
            with patch.object(evidence, "_run_probe") as runner:
                with self.assertRaises(evidence.EvidenceError):
                    evidence.main(["--execute", "--output", str(invalid),
                                   "--source-head", SOURCE_HEAD,
                                   "--source-tree", SOURCE_TREE,
                                   "--run-id", RUN_ID])
                runner.assert_not_called()

    def test_second_probe_failure_never_publishes_partial_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "evidence.json"
            with patch.object(evidence, "_run_probe",
                              side_effect=(reports()[0], RuntimeError("no checkpoint"))):
                with self.assertRaises(RuntimeError):
                    evidence.main(["--execute", "--output", str(target),
                                   "--source-head", SOURCE_HEAD,
                                   "--source-tree", SOURCE_TREE,
                                   "--run-id", RUN_ID])
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
