"""R12 witness-gated public development/recovery source evidence (no host attestation)."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.neural_resume_checkpoint import (
    FullCheckpointContext, FullCheckpointReceipt, _context_json,
    _json_bytes, _manifest_digest,
)
from labyrinth.qualification_evidence_r12 import (
    WitnessRecord, inspect_witnessed_public_recovery,
)
import test_preregistered_minilm_campaign_r11 as original_tests


def digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def snapshot(directory: Path, context: FullCheckpointContext, weight: bytes) -> FullCheckpointReceipt:
    """Create actual source-inspectable manifest/tensor bytes, not a Torch save."""
    directory.mkdir()
    header = {"tensor_000000": {"dtype": "F32", "shape": [1],
                                "data_offsets": [0, 4]}}
    packed = json.dumps(header, separators=(",", ":")).encode()
    tensors = len(packed).to_bytes(8, "little") + packed + weight
    (directory / "state.safetensors").write_bytes(tensors)
    manifest = {
        "schema": "labyrinth.full-neural-checkpoint.v1",
        "context": _context_json(context),
        "optimizer": "AdamW",
        "optimizer_groups": [["adapter.weight"]],
        "trainable_names": ["adapter.weight"],
        "model_tree": {"type": "dict", "items": [
            [{"type": "scalar", "value": "adapter.weight"},
             {"type": "tensor", "key": "tensor_000000"}]
        ]},
        "optimizer_tree": {"type": "dict", "items": []},
        "rng_tree": {"type": "dict", "items": []},
        "numpy_rng": None,
        "tensor_sha256": digest(tensors),
    }
    manifest_bytes = _json_bytes(manifest)
    (directory / "manifest.json").write_bytes(manifest_bytes)
    return FullCheckpointReceipt(_manifest_digest(manifest_bytes), context,
                                 ("adapter.weight",))


class QualificationEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = original_tests.PreregisteredMiniLMCampaignTests(
            "test_all_preregistered_seeds_and_budgets_score_without_graduation")
        source.setUp()
        self.addCleanup(source.doCleanups)
        self.campaign = source.campaign
        self.reports = source.receipts()
        exposures = (("occ-1", digest(b"train-1")),
                     ("occ-2", digest(b"train-2")))
        self.origin_context = FullCheckpointContext(
            "run-1", "arithmetic", digest(b"model"), digest(b"prior"),
            0, 0, exposures,
        )
        self.before_dir = self.root / "before"
        self.after_dir = self.root / "after"
        self.before = snapshot(self.before_dir, self.origin_context, b"\x00\x00\x80?")
        self.after_context = replace(
            self.origin_context, prior_checkpoint_sha256=self.before.sha256,
            cursor=1,
        )
        self.after = snapshot(self.after_dir, self.after_context, b"\x00\x00\x00@")
        self.witness = WitnessRecord(
            witness_id="receipt-A", authority_id="NAS-TRUST-01",
            campaign_sha256=self.campaign.sha256,
            origin_checkpoint_sha256=self.before.sha256,
            successor_checkpoint_sha256=self.after.sha256,
            precommit_sequence=10, first_outcome_sequence=11,
            recovery_sequence=12,
        )
        self.used: set[str] = set()
        self.head = 12

    def evaluate(self, *, witness=None, campaign=None, before=None, after=None,
                 reports=None, before_dir=None, after_dir=None,
                 verify=True, reserve=True):
        def reserve_once(witness_id, campaign_digest, sequence):
            if not reserve or witness_id in self.used:
                return False
            self.used.add(witness_id)
            return (campaign_digest, sequence) == (self.campaign.sha256, 12)

        return inspect_witnessed_public_recovery(
            campaign=campaign or self.campaign,
            reports=self.reports if reports is None else reports,
            origin_receipt=before or self.before,
            origin_dir=before_dir or self.before_dir,
            successor_receipt=after or self.after,
            successor_dir=after_dir or self.after_dir,
            witness=witness or self.witness,
            expected_authority_id="NAS-TRUST-01",
            authenticate_witness=lambda proof: verify and proof == (witness or self.witness),
            read_authoritative_sequence=lambda authority: self.head,
            reserve_witness_once=reserve_once,
        )

    def test_valid_grid_bytes_and_witness_remain_provisional(self):
        result = self.evaluate()
        self.assertEqual(result.status, "WITNESSED_PUBLIC_DEVELOPMENT_ONLY")
        self.assertEqual(len(result.public_score.run_results), 4)
        self.assertTrue(result.snapshot_bytes_checked)
        self.assertTrue(result.trainable_bytes_changed)
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.physical_model_verified)
        self.assertFalse(result.examiner_custody_verified)

    def test_replay_and_false_authentication_rejected(self):
        self.evaluate()
        with self.assertRaisesRegex(ValueError, "replay|reservation"):
            self.evaluate()
        with self.assertRaisesRegex(ValueError, "authenticat"):
            self.evaluate(witness=replace(self.witness, witness_id="receipt-B"),
                          verify=False)

    def test_posthoc_selection_and_incomplete_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "pre-outcome|sequence"):
            self.evaluate(witness=replace(self.witness, precommit_sequence=11))
        with self.assertRaisesRegex(ValueError, "preregistered runs"):
            self.evaluate(reports=self.reports[:-1])

    def test_drift_source_model_tokenizer_seed_budget_rejected(self):
        for key, value in (("source_tree_sha256", digest(b"different source")),
                           ("pretrained_model_sha256", digest(b"different model")),
                           ("tokenizer_sha256", digest(b"different tokenizer")),
                           ("runs", self.campaign.runs[:-1])):
            changed = replace(self.campaign, **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "campaign|manifest"):
                self.evaluate(campaign=changed)

    def test_stale_rollback_and_wrong_authority_rejected(self):
        self.head = 11
        with self.assertRaisesRegex(ValueError, "stale|rollback|head"):
            self.evaluate()
        self.head = 13
        with self.assertRaisesRegex(ValueError, "stale|rollback|head"):
            self.evaluate()
        with self.assertRaisesRegex(ValueError, "authority"):
            self.evaluate(witness=replace(self.witness, authority_id="other-host"))

    def test_altered_checkpoint_bytes_and_same_weights_rejected(self):
        tensor_path = self.after_dir / "state.safetensors"
        original = tensor_path.read_bytes()
        tensor_path.write_bytes(original[:-4] + b"FAIL")
        with self.assertRaisesRegex(ValueError, "tensor|digest|byte"):
            self.evaluate()
        tensor_path.write_bytes(original)
        duplicate_dir = self.root / "same"
        duplicate_context = self.after_context
        duplicate_receipt = snapshot(duplicate_dir, duplicate_context, b"\x00\x00\x80?")
        with self.assertRaisesRegex(ValueError, "trainable.*unchanged"):
            self.evaluate(after=duplicate_receipt, after_dir=duplicate_dir,
                          witness=replace(self.witness,
                            successor_checkpoint_sha256=duplicate_receipt.sha256))

    def test_cannot_turn_self_reported_receipt_into_graduation(self):
        changed = replace(self.after, graduation_authorized=True)
        with self.assertRaisesRegex(ValueError, "graduation|provisional"):
            self.evaluate(after=changed)

    def test_five_arm_pilot_is_not_misrepresented_as_four_arm_campaign(self):
        first = self.reports[0]
        report = json.loads(Path(first.report_path).read_text(encoding="utf-8"))
        arms = report["arms"]
        # PR #62 opt-in pilot: 3-arm correct/zero/inverted + zero_lr/randomized.
        arms["inverted"] = deepcopy(arms.pop("corrupted_labels"))
        arms.pop("baseline")
        arms["zero_lr"] = deepcopy(arms["zero_update"])
        arms["randomized_labels"] = deepcopy(arms["inverted"])
        self.assertEqual(set(arms), {"correct_labels", "zero_update", "inverted",
                                      "zero_lr", "randomized_labels"})
        changed_bytes = original_tests.canonical(report)
        Path(first.report_path).write_bytes(changed_bytes)
        changed = (replace(first, report_sha256=digest(changed_bytes)),) + self.reports[1:]
        with self.assertRaisesRegex(ValueError, "four arms required"):
            self.evaluate(reports=changed)

    def test_witness_checkpoint_digest_change_and_invalid_tensor_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "witness.*digest"):
            self.evaluate(witness=replace(self.witness,
                         successor_checkpoint_sha256=digest(b"different")))
        tensor_path = self.after_dir / "state.safetensors"
        manifest_path = self.after_dir / "manifest.json"
        raw = tensor_path.read_bytes()
        old_header_len = int.from_bytes(raw[:8], "little")
        old_header = json.loads(raw[8:8 + old_header_len])
        old_header["tensor_000000"]["shape"] = [2]  # 8 bytes needed, only 4 exist
        new_header = json.dumps(old_header, separators=(",", ":")).encode()
        changed_tensor = len(new_header).to_bytes(8, "little") + new_header + raw[8 + old_header_len:]
        tensor_path.write_bytes(changed_tensor)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["tensor_sha256"] = digest(changed_tensor)
        changed_manifest = _json_bytes(manifest)
        manifest_path.write_bytes(changed_manifest)
        changed_receipt = replace(self.after, sha256=_manifest_digest(changed_manifest))
        with self.assertRaisesRegex(ValueError, "dtype/shape byte length"):
            self.evaluate(after=changed_receipt,
                          witness=replace(self.witness,
                            successor_checkpoint_sha256=changed_receipt.sha256))

    def test_null_or_noncallable_host_callbacks_rejected(self):
        with self.assertRaisesRegex(ValueError, "reservation"):
            self.evaluate(reserve=False)


if __name__ == "__main__":
    unittest.main()
