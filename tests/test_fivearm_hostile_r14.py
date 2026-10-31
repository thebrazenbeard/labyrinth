"""Hostile offline five-arm checks; files are synthetic, never neural evidence."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from labyrinth.fivearm_campaign_admission_r13 import FiveArmCampaignError
from scripts import qualify_maze_multiseed_protocol_cpu as protocol
from test_fivearm_campaign_admission_r13 import FiveArmCampaignAdmissionTests


class FiveArmCampaignHostileR14(TestCase):
    def setUp(self) -> None:
        # Reuse the real audit fixture. Its checkpoint bytes are explicit fakes.
        self.fixture = FiveArmCampaignAdmissionTests(
            "test_complete_five_arm_grid_reloads_every_child_and_stays_provisional")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def _job_root(self, index: int) -> Path:
        name = self.fixture.plan["manifest"]["jobs"][index]["run_name"]
        return self.fixture.root / "runs" / name

    def _rewrite_report(self, index: int, data: dict) -> dict[str, str]:
        run = self._job_root(index)
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        (run / "report.json").write_bytes(encoded)
        hashes = dict(self.fixture.report_hashes)
        hashes[run.name] = sha256(encoded).hexdigest()
        return hashes

    def test_audit_rechecks_prior_checkpoint_if_modified_during_later_child(self) -> None:
        """An audit result must not certify bytes replaced mid-campaign."""
        first = self._job_root(0) / "correct" / "cycle-0" / "head.safetensors"
        original = protocol.audit_child
        second = self.fixture.plan["manifest"]["jobs"][1]["run_name"]
        changes = []

        def audited_then_mutate(job, plan, root):
            row = original(job, plan, root)
            if job["run_name"] == second:
                first.write_bytes(first.read_bytes() + b"AFTER-PRIOR-READ")
                changes.append(True)
            return row

        with patch.object(protocol, "audit_child", side_effect=audited_then_mutate):
            with self.assertRaises(FiveArmCampaignError):
                self.fixture.audit()
        self.assertEqual(changes, [True])

    def test_extreme_nesting_refuses_with_public_error_not_recursion_crash(self) -> None:
        """Bounded JSON bytes alone does not bound parser recursion depth."""
        run = self._job_root(0)
        hostile = b"[" * 1400 + b"0" + b"]" * 1400
        (run / "report.json").write_bytes(hostile)
        hashes = dict(self.fixture.report_hashes)
        hashes[run.name] = sha256(hostile).hexdigest()
        with self.assertRaises(FiveArmCampaignError):
            self.fixture.audit(expected_report_sha256_by_run=hashes)

    def test_report_larger_than_bounded_limit_refused_despite_matching_pin(self) -> None:
        run = self._job_root(0)
        hostile = (run / "report.json").read_bytes() + b" " * (8 * 1024 * 1024)
        (run / "report.json").write_bytes(hostile)
        hashes = dict(self.fixture.report_hashes)
        hashes[run.name] = sha256(hostile).hexdigest()
        with self.assertRaisesRegex(FiveArmCampaignError, "bounded size"):
            self.fixture.audit(expected_report_sha256_by_run=hashes)

    def test_nested_symlink_in_ancestor_is_refused(self) -> None:
        """A linked intermediate directory cannot bypass artifact provenance."""
        run = self._job_root(0)
        adapter = run / "correct" / "cycle-0" / "adapter"
        renamed = adapter.with_name("real-adapter")
        adapter.rename(renamed)
        try:
            adapter.symlink_to(renamed, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("local symlinks unsupported")
        with self.assertRaises(FiveArmCampaignError):
            self.fixture.audit()

    def test_forged_witness_does_not_change_source_only_evidence_class(self) -> None:
        """Even an in-process true callback cannot attest independent custody."""
        result = self.fixture.audit(
            preoutcome_witness=b"arbitrary-pretend-witness",
            verify_external_preoutcome_witness=lambda *_: True,
        )
        self.assertFalse(result.physical_model_verified)
        self.assertFalse(result.evaluator_custody_verified)
        self.assertFalse(result.graduation_authorized)
        self.assertIn("SOURCE_AUDIT_ONLY", result.status)

    def test_swapped_runner_or_frozen_model_pin_never_passes(self) -> None:
        """Five arms with self-reported base identity cannot override pins."""
        for name in ("expected_runner_sha256", "expected_protocol_sha256",
                     "expected_pretrained_base_sha256"):
            with self.subTest(name=name), self.assertRaises(FiveArmCampaignError):
                self.fixture.audit(**{name: "0" * 64})

    def test_duplicate_json_at_nested_control_entry_refused(self) -> None:
        run = self._job_root(0)
        report = run / "report.json"
        original = report.read_bytes()
        self.assertIn(b'"initial_lora_sha256":', original)
        hostile = original.replace(
            b'"initial_lora_sha256":',
            b'"initial_lora_sha256":"0", "initial_lora_sha256":', 1)
        report.write_bytes(hostile)
        hashes = dict(self.fixture.report_hashes)
        hashes[run.name] = sha256(hostile).hexdigest()
        with self.assertRaises(FiveArmCampaignError):
            self.fixture.audit(expected_report_sha256_by_run=hashes)

    def test_reordered_aggregate_cells_cannot_be_postselected(self) -> None:
        path = self.fixture.root / "result.json"
        result = json.loads(path.read_text())
        result["results"].reverse()
        path.write_bytes(protocol._canonical(result))
        with self.assertRaisesRegex(FiveArmCampaignError, "result"):
            self.fixture.audit()

    def test_control_arm_schema_reordering_cannot_remap_the_arms(self) -> None:
        run = self._job_root(0)
        report = json.loads((run / "report.json").read_text())
        report["control_arms"] = list(reversed(report["control_arms"]))
        pinned = self._rewrite_report(0, report)
        with self.assertRaises(FiveArmCampaignError):
            self.fixture.audit(expected_report_sha256_by_run=pinned)

    def test_self_reported_head_or_adapter_sha_does_not_override_disk(self) -> None:
        run = self._job_root(0)
        path = run / "report.json"
        original = json.loads(path.read_text())
        for field in ("head_file_sha256", "adapter_file_sha256"):
            with self.subTest(field=field):
                report = json.loads(json.dumps(original))
                report["arms"]["correct"]["cycles"][0][field] = "e" * 64
                pinned = self._rewrite_report(0, report)
                with self.assertRaises(FiveArmCampaignError):
                    self.fixture.audit(expected_report_sha256_by_run=pinned)

    def test_cross_seed_same_frozen_base_hash_is_not_custody_attestation(self) -> None:
        # Cross-seed consistency of an asserted base hash is necessary but not
        # an external proof that pretrained model bytes were physically read.
        for job in self.fixture.plan["manifest"]["jobs"]:
            report = json.loads((self.fixture.root / "runs" / job["run_name"]
                                 / "report.json").read_text())
            self.assertEqual(report["pretrained_base_sha256"], "f" * 64)
        evidence = self.fixture.audit()
        self.assertFalse(evidence.physical_model_verified)
        self.assertFalse(evidence.evaluator_custody_verified)

    def test_replacing_report_after_child_audit_is_detected(self) -> None:
        original = protocol.audit_child
        first = self._job_root(0)
        changed = []

        def audited_then_rewrite(job, plan, root):
            result = original(job, plan, root)
            if root == first:
                report_path = root / "report.json"
                report_path.write_bytes(report_path.read_bytes() + b" ")
                changed.append(True)
            return result

        with patch.object(protocol, "audit_child", side_effect=audited_then_rewrite):
            with self.assertRaisesRegex(FiveArmCampaignError, "changed"):
                self.fixture.audit()
        self.assertEqual(changed, [True])

    def test_removed_saved_cell_cannot_be_postselected(self) -> None:
        from shutil import move

        run = self._job_root(0)
        # A recoverable rename outside the corpus root models a missing cell.
        displaced = self.fixture.root.parent / "displaced-cell"
        move(str(run), str(displaced))
        with self.assertRaises(FiveArmCampaignError):
            self.fixture.audit()


if __name__ == "__main__":
    import unittest

    unittest.main()
