"""Synthetic artifact tests: NOT neural execution or external witness custody."""

from __future__ import annotations

from hashlib import sha256
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import qualify_maze_multiseed_protocol_cpu as protocol
from labyrinth.fivearm_campaign_admission_r13 import (
    FiveArmCampaignEvidence,
    FiveArmCampaignError,
    audit_fivearm_campaign_from_disk,
)


class FiveArmCampaignAdmissionTests(unittest.TestCase):
    """Existing synthetic child fixture exercises the REAL offline child auditor."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "fivearm"
        self.root.mkdir()
        (self.root / "runs").mkdir()
        self.spec = protocol.ProtocolSpec(
            preregistered_controls=True, seeds=(1101, 2202, 3303),
            steps=(8, 16),
        )
        self.plan = protocol.build_plan(self.spec)
        (self.root / "plan.json").write_bytes(protocol._canonical(self.plan))
        fixture = importlib.import_module(
            "test_qualify_maze_multiseed_protocol_cpu").MultiSeedProtocolTests()
        self.rows = []
        self.report_hashes = {}
        for job in self.plan["manifest"]["jobs"]:
            run = self.root / "runs" / job["run_name"]
            fixture._fake_child(run, job, self.plan)
            self.rows.append(protocol.audit_child(job, self.plan, run))
            self.report_hashes[job["run_name"]] = sha256(
                (run / "report.json").read_bytes()).hexdigest()
        result = {
            "schema": protocol.SCHEMA,
            "plan_sha256": protocol._digest(self.plan),
            "qualification": "PUBLIC_MULTISEED_DEVELOPMENT_ONLY",
            "results": self.rows,
            "complete_jobs": len(self.rows),
            "public_aggregate": protocol.summarize_public_controls(self.rows, self.spec),
            "private_final_exam": "NOT_RUN",
            "prior_skill_retention": "NOT_RUN",
            "independent_model_and_evaluator_custody": "NOT_ESTABLISHED",
            "crash_resume": "NOT_SUPPORTED",
            "graduation_authorized": False,
            "preregistered_controls": "FIVE_ARM_PUBLIC_UNSEALED_NOT_INDEPENDENT",
        }
        (self.root / "result.json").write_bytes(protocol._canonical(result))
        self.expected_plan_sha256 = sha256(protocol._canonical(self.plan)).hexdigest()
        self.witness = b"synthetic-test-preoutcome-witness-does-not-establish-time"

    def audit(self, **overrides):
        arguments = {
            "spec": self.spec,
            "campaign_root": self.root,
            "expected_plan_sha256": self.expected_plan_sha256,
            "expected_runner_sha256": self.plan["manifest"]["runner_sha256"],
            "expected_protocol_sha256": self.plan["manifest"]["protocol_sha256"],
            "expected_pretrained_base_sha256": "f" * 64,
            "expected_report_sha256_by_run": self.report_hashes,
            "preoutcome_witness": self.witness,
            "verify_external_preoutcome_witness": lambda witness, plan_sha: (
                witness == self.witness and plan_sha == self.expected_plan_sha256
            ),
        }
        arguments.update(overrides)
        return audit_fivearm_campaign_from_disk(**arguments)

    def test_complete_five_arm_grid_reloads_every_child_and_stays_provisional(self):
        evidence = self.audit()
        self.assertIsInstance(evidence, FiveArmCampaignEvidence)
        self.assertEqual(evidence.run_count, 6)
        self.assertEqual(evidence.control_arms, (
            "correct", "zero_update", "inverted", "zero_lr", "randomized_labels"))
        self.assertEqual(evidence.checkpoint_files_checked, 120)
        self.assertTrue(evidence.all_cells_correct_exceed_every_control)
        self.assertEqual(sha256(evidence.public_aggregate_json).hexdigest(),
                         evidence.public_aggregate_sha256)
        self.assertEqual(evidence.status, "FIVE_ARM_PUBLIC_SOURCE_AUDIT_ONLY")
        self.assertFalse(evidence.physical_model_verified)
        self.assertFalse(evidence.evaluator_custody_verified)
        self.assertFalse(evidence.graduation_authorized)

    def test_missing_external_preoutcome_witness_refuses_admission(self):
        for change in ({"preoutcome_witness": b""},
                       {"verify_external_preoutcome_witness": None},
                       {"verify_external_preoutcome_witness": lambda *_: False}):
            with self.subTest(change=str(change)), self.assertRaises(FiveArmCampaignError):
                self.audit(**change)

    def test_postselected_or_missing_cell_is_rejected(self):
        incomplete = dict(self.report_hashes)
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaisesRegex(FiveArmCampaignError, "complete|grid"):
            self.audit(expected_report_sha256_by_run=incomplete)
        with self.assertRaisesRegex(FiveArmCampaignError, "plan"):
            self.audit(expected_plan_sha256="a" * 64)

    def test_source_and_model_pin_mismatch_rejected(self):
        for parameter in ("expected_runner_sha256", "expected_protocol_sha256",
                          "expected_pretrained_base_sha256"):
            with self.subTest(parameter=parameter), self.assertRaises(FiveArmCampaignError):
                self.audit(**{parameter: "0" * 64})

    def test_swapped_report_or_checkpoint_bytes_fail_reload(self):
        first, second = self.plan["manifest"]["jobs"][:2]
        first_path = self.root / "runs" / first["run_name"] / "report.json"
        second_path = self.root / "runs" / second["run_name"] / "report.json"
        first_bytes, second_bytes = first_path.read_bytes(), second_path.read_bytes()
        first_path.write_bytes(second_bytes)
        with self.assertRaises(FiveArmCampaignError):
            self.audit()
        first_path.write_bytes(first_bytes)
        checkpoint = (self.root / "runs" / first["run_name"] / "zero_lr"
                      / "cycle-1" / "adapter" / "adapter_model.safetensors")
        checkpoint.write_bytes(checkpoint.read_bytes() + b"TAMPER")
        with self.assertRaises(FiveArmCampaignError):
            self.audit()

    def test_four_arm_path_or_five_arm_deletion_is_not_silently_translated(self):
        with self.assertRaisesRegex(FiveArmCampaignError, "five-arm"):
            self.audit(spec=protocol.ProtocolSpec(preregistered_controls=False))
        first = self.plan["manifest"]["jobs"][0]
        report_path = self.root / "runs" / first["run_name"] / "report.json"
        data = json.loads(report_path.read_text())
        del data["arms"]["zero_lr"]
        report_path.write_text(json.dumps(data), encoding="utf-8")
        modified = dict(self.report_hashes)
        modified[first["run_name"]] = sha256(report_path.read_bytes()).hexdigest()
        with self.assertRaises(FiveArmCampaignError):
            self.audit(expected_report_sha256_by_run=modified)

    def test_plan_and_result_replay_mutation_are_rejected(self):
        result_path = self.root / "result.json"
        result = json.loads(result_path.read_text())
        result["complete_jobs"] -= 1
        result_path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(FiveArmCampaignError, "result"):
            self.audit()
        result_path.write_bytes(b"{}")
        plan_path = self.root / "plan.json"
        plan_path.write_bytes(plan_path.read_bytes() + b" ")
        with self.assertRaisesRegex(FiveArmCampaignError, "plan"):
            self.audit()

    def test_duplicate_json_fields_cannot_hide_a_second_control_schema(self):
        first = self.plan["manifest"]["jobs"][0]
        report_path = self.root / "runs" / first["run_name"] / "report.json"
        original = report_path.read_bytes()
        self.assertIn(b'"schema":', original)
        changed = original.replace(b'"schema":', b'"schema":"forged", "schema":', 1)
        report_path.write_bytes(changed)
        modified = dict(self.report_hashes)
        modified[first["run_name"]] = sha256(changed).hexdigest()
        with self.assertRaisesRegex(FiveArmCampaignError, "duplicate|audit"):
            self.audit(expected_report_sha256_by_run=modified)

    def test_symlinked_checkpoint_artifact_is_rejected_even_if_bytes_match(self):
        first = self.plan["manifest"]["jobs"][0]
        child = self.root / "runs" / first["run_name"]
        adapter = child / "correct" / "cycle-0" / "adapter" / "adapter_model.safetensors"
        sibling = adapter.with_name("replayed.safetensors")
        adapter.rename(sibling)
        try:
            adapter.symlink_to(sibling)
        except (OSError, NotImplementedError):
            self.skipTest("sandbox filesystem does not support a local symlink")
        with self.assertRaisesRegex(FiveArmCampaignError, "symlink"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
