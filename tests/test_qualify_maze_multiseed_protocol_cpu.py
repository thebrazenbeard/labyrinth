"""Source-only tests; synthetic files are NOT pretrained neural executions."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.training_scope import TrainingScopeSpec
from scripts import qualify_maze_multiseed_protocol_cpu as target
from scripts.qualify_maze_conditioned_minilm_cpu import (
    build_training_cycles, select_training_batch,
)


class MultiSeedProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = target.ProtocolSpec()

    def _fake_child(self, run_root: Path, job: dict, plan: dict) -> dict:
        """Fake reports test the AUDITOR; they are never physical evidence."""
        run_root.mkdir(parents=True)
        model_sha = "f" * 64
        data = {
            "schema": target.CHILD_SCHEMA,
            "harness_sha256": plan["manifest"]["runner_sha256"],
            "model_id": target.MODEL_ID,
            "scope": job["scope"],
            "scope_fingerprint": job["scope_fingerprint"],
            "steps_per_cycle": job["steps_per_cycle"],
            "weakness_feedback_provenance": "OPERATOR_DECLARED_NOT_MEASURED",
            "private_final_exam": "NOT_RUN",
            "graduation_authorized": False,
            "crash_recovery": "NOT_TESTED",
            "pretrained_base_sha256": model_sha,
            "arms": {},
        }
        scope = TrainingScopeSpec.from_dict(job["scope"])
        cycles = build_training_cycles(
            scope, weak_cycle_one=(plan["manifest"]["weak_skill"],))
        for arm in target.SUPPORTED_ARMS:
            initial = "a" * 64
            after0 = ("a" if arm == "zero_update" else
                      "b" if arm == "correct" else "c") * 64
            after1 = ("a" if arm == "zero_update" else
                      "d" if arm == "correct" else "e") * 64
            initial_lora = "1" * 64
            after0_lora = ("1" if arm == "zero_update" else
                           "2" if arm == "correct" else "3") * 64
            after1_lora = ("1" if arm == "zero_update" else
                           "4" if arm == "correct" else "5") * 64
            rows = []
            for index, cycle in enumerate(cycles):
                checkpoint = run_root / arm / f"cycle-{index}"
                (checkpoint / "adapter").mkdir(parents=True)
                (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(
                    f"FAKE-ADAPTER:{arm}:{index}".encode())
                (checkpoint / "head.safetensors").write_bytes(
                    f"FAKE-HEAD:{arm}:{index}".encode())
                sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append({
                    "cycle": index,
                    "maze_room_ids": job["cycle_room_ids"][index],
                    "maze_exposure_ids": job["cycle_exposure_ids"][index],
                    "training_visit_exposure_ids": [
                        row.exposure_id
                        for step in range(job["steps_per_cycle"])
                        for row in select_training_batch(cycle, step)
                    ],
                    "training_visits": job["steps_per_cycle"] * 4,
                    "optimizer_updates": (
                        0 if arm == "zero_update" else job["steps_per_cycle"]),
                    "pretrained_base_sha256": model_sha,
                    "frozen_peft_sha256_before": model_sha,
                    "frozen_peft_sha256_after": model_sha,
                    "trainable_sha256_before": initial if index == 0 else after0,
                    "trainable_sha256_after": after0 if index == 0 else after1,
                    "lora_sha256_before": initial_lora if index == 0 else after0_lora,
                    "lora_sha256_after": after0_lora if index == 0 else after1_lora,
                    "public_development_accuracy": (
                        (0.4 if index == 0 else 0.8) if arm == "correct" else
                        (0.5 if arm == "zero_update" else 0.2)),
                    "public_transfer_accuracy": (
                        0.7 if arm == "correct" else
                        0.5 if arm == "zero_update" else 0.2),
                    "adapter_file_sha256": sha(
                        checkpoint / "adapter" / "adapter_model.safetensors"),
                    "head_file_sha256": sha(checkpoint / "head.safetensors"),
                })
            data["arms"][arm] = {
                "initial_trainable_sha256": initial,
                "initial_lora_sha256": initial_lora,
                "final_checkpoint_readback": {
                    "status": "SELF_REPORTED_FRESH_INSTANCE_PASS"},
                "cycles": rows,
            }
        (run_root / "report.json").write_text(json.dumps(data), encoding="utf-8")
        return data

    def test_read_only_plan_has_distinct_seeds_and_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(target.main([
                    "--output-parent", parent, "--run-name", "dryrun"]), 0)
            report = json.loads(out.getvalue())
            self.assertFalse(report["execute"])
            self.assertFalse((Path(parent) / "dryrun").exists())
            self.assertEqual(len(report["manifest"]["jobs"]), 6)
            self.assertEqual(len(set(job["run_name"]
                                     for job in report["manifest"]["jobs"])), 6)
            self.assertEqual(report["manifest"]["private_final_exam"], "NOT_RUN")
            self.assertFalse(report["manifest"]["graduation_authorized"])
            self.assertEqual(report["manifest"]["prior_skill_retention"],
                             "NOT_TESTED_SAME_SKILL_PUBLIC_RECHECK_ONLY")

    def test_unsupported_zero_lr_and_randomized_fail_closed(self) -> None:
        for control in ("zero_lr", "randomized"):
            with self.subTest(control=control):
                with self.assertRaisesRegex(ValueError, "backend lacks requested"):
                    target.ProtocolSpec(required_arms=(*target.SUPPORTED_ARMS, control))
        with self.assertRaisesRegex(ValueError, "independently held prior-skill"):
            target.ProtocolSpec(require_protected_retention=True)

    def test_seed_and_compute_limits_fail_closed(self) -> None:
        invalid = (
            {"seeds": (1, 1, 2)},
            {"seeds": (1, 2)},
            {"seeds": (1, 2, 3, 4, 5, 6)},
            {"seeds": (1, 2, 2**31)},
            {"steps": (8, 8)},
            {"steps": (0, 16)},
            {"steps": (8, 16, 4)},
            {"weak_skill": "private_final"},
            {"learning_rate": 0.0},
            {"required_arms": ("correct", "inverted", "zero_update")},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                target.ProtocolSpec(**values)

    def test_child_command_binds_seed_and_no_shell(self) -> None:
        job = target.build_plan(self.spec)["manifest"]["jobs"][0]
        command = target.child_command(job, Path("/tmp/safe"), self.spec)
        self.assertIsInstance(command, list)
        self.assertEqual(command[0], target.sys.executable)
        self.assertIn("--scope-file", command)
        self.assertIn("--execute", command)
        self.assertIn("seed-0000001101-steps-08", command)
        with self.assertRaises(ValueError):
            target.child_command({**job, "steps_per_cycle": 900}, Path("/tmp"), self.spec)

    def test_synthetic_audit_validates_control_continuity_not_neural_execution(self) -> None:
        plan = target.build_plan(self.spec)
        job = plan["manifest"]["jobs"][0]
        with tempfile.TemporaryDirectory() as parent:
            run = Path(parent) / job["run_name"]
            self._fake_child(run, job, plan)
            summary = target.audit_child(job, plan, run)
            self.assertEqual(summary["checkpoint_files_checked"], 12)
            self.assertTrue(summary["public_correct_advantage_both"]["public_development"])
            self.assertEqual(summary["physical_execution"],
                             "CHILD_SELF_REPORTED_UNATTESTED")
            self.assertEqual(summary["independent_retention"], "NOT_RUN")
            self.assertEqual(summary["pretrained_base_sha256"], "f" * 64)

    def test_tampered_checkpoint_and_visit_fail(self) -> None:
        plan = target.build_plan(self.spec)
        job = plan["manifest"]["jobs"][0]
        with tempfile.TemporaryDirectory() as parent:
            run = Path(parent) / job["run_name"]
            data = self._fake_child(run, job, plan)
            file = run / "correct" / "cycle-0" / "head.safetensors"
            file.write_bytes(b"CORRUPTED")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                target.audit_child(job, plan, run)
            data["arms"]["correct"]["cycles"][0]["training_visit_exposure_ids"] = []
            (run / "report.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Maze exposures"):
                target.audit_child(job, plan, run)

    def test_initial_trainable_commitment_must_bind_cycle_zero(self) -> None:
        plan = target.build_plan(self.spec)
        job = plan["manifest"]["jobs"][0]
        with tempfile.TemporaryDirectory() as parent:
            run = Path(parent) / job["run_name"]
            data = self._fake_child(run, job, plan)
            for arm in target.SUPPORTED_ARMS:
                data["arms"][arm]["initial_trainable_sha256"] = "e" * 64
            (run / "report.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initial.*cycle"):
                target.audit_child(job, plan, run)

    def test_malformed_trainable_hash_rejected(self) -> None:
        plan = target.build_plan(self.spec)
        job = plan["manifest"]["jobs"][0]
        with tempfile.TemporaryDirectory() as parent:
            run = Path(parent) / job["run_name"]
            data = self._fake_child(run, job, plan)
            data["arms"]["correct"]["cycles"][1]["trainable_sha256_after"] = "garbage"
            (run / "report.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tensor commitment"):
                target.audit_child(job, plan, run)

    def test_lora_specific_commitment_and_regeneration_are_required(self) -> None:
        plan = target.build_plan(self.spec)
        job = plan["manifest"]["jobs"][0]
        with tempfile.TemporaryDirectory() as parent:
            run = Path(parent) / job["run_name"]
            data = self._fake_child(run, job, plan)
            data["arms"]["correct"]["initial_lora_sha256"] = "f" * 64
            (run / "report.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initial.*LoRA"):
                target.audit_child(job, plan, run)

            run = Path(parent) / "second-lora-fixture"
            data = self._fake_child(run, job, plan)
            data["arms"]["correct"]["cycles"][1]["lora_sha256_before"] = "f" * 64
            (run / "report.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "regeneration.*LoRA"):
                target.audit_child(job, plan, run)

            run = Path(parent) / "third-lora-fixture"
            data = self._fake_child(run, job, plan)
            data["arms"]["inverted"]["cycles"][1]["lora_sha256_after"] = (
                data["arms"]["inverted"]["cycles"][1]["lora_sha256_before"])
            (run / "report.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "updated.*LoRA"):
                target.audit_child(job, plan, run)

    def test_graduation_or_private_exam_claim_rejected(self) -> None:
        plan = target.build_plan(self.spec)
        job = plan["manifest"]["jobs"][0]
        with tempfile.TemporaryDirectory() as parent:
            run = Path(parent) / job["run_name"]
            data = self._fake_child(run, job, plan)
            data["graduation_authorized"] = True
            (run / "report.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatches"):
                target.audit_child(job, plan, run)

    def test_failing_child_preserves_isolated_failure_record(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            with patch.object(target.subprocess, "run", return_value=subprocess.CompletedProcess(
                    args=["fake"], returncode=7, stdout="", stderr="test-only")):
                with self.assertRaisesRegex(RuntimeError, "neural child failed"):
                    target.execute_plan(self.spec, parent, "failure01", timeout_seconds_per_run=60)
            root = Path(parent) / "failure01"
            self.assertTrue((root / "plan.json").is_file())
            self.assertTrue((root / "failure.json").is_file())
            reason = json.loads((root / "failure.json").read_text())
            self.assertEqual(reason["reason"], "CHILD_FAILED")
            self.assertFalse(reason["graduation_authorized"])
            self.assertFalse((root / "result.json").exists())

    def test_success_exit_malformed_report_preserves_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            with patch.object(target.subprocess, "run", return_value=subprocess.CompletedProcess(
                    args=["fake"], returncode=0, stdout="child exited zero", stderr="")):
                with self.assertRaisesRegex(RuntimeError, "audit failed"):
                    target.execute_plan(self.spec, parent, "malformed01", timeout_seconds_per_run=60)
            root = Path(parent) / "malformed01"
            self.assertTrue((root / "plan.json").is_file())
            self.assertEqual(json.loads((root / "failure.json").read_text())["reason"],
                             "CHILD_AUDIT_FAILED")
            self.assertFalse((root / "result.json").exists())

    def test_mocked_successful_orchestration_audits_all_six_children(self) -> None:
        # The children are synthetic fixture writers; this is source integration
        # verification, never a CPU neural-training result.
        with tempfile.TemporaryDirectory() as parent:
            plan = target.build_plan(self.spec)
            jobs = {job["run_name"]: job for job in plan["manifest"]["jobs"]}
            calls = []

            def fake_child(command, **kwargs):
                self.assertIsInstance(command, list)
                self.assertNotIn("shell", kwargs)
                run_name = command[command.index("--run-name") + 1]
                calls.append(run_name)
                self._fake_child(Path(parent) / "synthetic" / "runs" / run_name,
                                 jobs[run_name], plan)
                return subprocess.CompletedProcess(
                    args=command, returncode=0, stdout="test only", stderr="")

            with patch.object(target.subprocess, "run", side_effect=fake_child):
                result = target.execute_plan(
                    self.spec, parent, "synthetic", timeout_seconds_per_run=60)
            self.assertEqual(len(calls), 6)
            self.assertEqual(len(set(calls)), 6)
            self.assertEqual(result["complete_jobs"], 6)
            self.assertEqual(result["qualification"], "PUBLIC_MULTISEED_DEVELOPMENT_ONLY")
            self.assertEqual(result["public_aggregate"]["fixed_public_splits"][
                "16"]["public_development"]["positive_seed_count"], 3)
            self.assertTrue((Path(parent) / "synthetic" / "result.json").is_file())
            self.assertEqual(result["private_final_exam"], "NOT_RUN")
            self.assertFalse(result["graduation_authorized"])

    def test_existing_directory_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            (Path(parent) / "occupied").mkdir()
            with self.assertRaises(FileExistsError):
                target.execute_plan(self.spec, parent, "occupied")

    def test_multiseed_aggregation_rejects_missing_or_swapped_base(self) -> None:
        rows = []
        for seed in self.spec.seeds:
            for step in self.spec.steps:
                rows.append({
                    "seed": seed, "steps_per_cycle": step,
                    "pretrained_base_sha256": "b" * 64,
                    "scores": {
                        "correct": {"public_development": .9, "public_transfer": .8},
                        "zero_update": {"public_development": .5, "public_transfer": .6},
                        "inverted": {"public_development": .1, "public_transfer": .3},
                    },
                })
        result = target.summarize_public_controls(rows, self.spec)
        self.assertEqual(result["fixed_public_splits"]["16"]["public_development"][
            "positive_seed_count"], 3)
        self.assertAlmostEqual(result["fixed_public_splits"]["8"][
            "public_transfer"]["mean_correct_minus_zero"], .2)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            target.summarize_public_controls(rows[:-1], self.spec)
        rows[1]["pretrained_base_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "base changed"):
            target.summarize_public_controls(rows, self.spec)


if __name__ == "__main__":
    unittest.main()
