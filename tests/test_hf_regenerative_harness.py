"""Offline contract checks for the host-only two-cycle HF qualification harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from labyrinth.training_scope import TrainingScopeSpec
from scripts.qualify_hf_regenerative import _command, _run_bounded_child, preflight


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_hf_regenerative.py"


class HFRegenerativeHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "local-model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"fixture-only, not actual weights")
        (self.model / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.scope = self.root / "scope.json"
        self.scope.write_text(json.dumps(TrainingScopeSpec(
            objective_id="arithmetic-harness", domain="numerical",
            skills=("arithmetic",), specificity=2, scale=2,
            max_cycles=2, seed=42,
        ).to_dict()), encoding="utf-8")

    def args(self):
        return [
            sys.executable, str(SCRIPT),
            "--model-path", str(self.model),
            "--scope-json", str(self.scope),
            "--output-parent", str(self.outputs),
            "--run-name", "bounded-run",
            "--model-origin-claim", "operator-local-pretrained",
        ]

    def plan(self):
        return preflight(argparse.Namespace(
            model_path=str(self.model), scope_json=str(self.scope),
            output_parent=str(self.outputs), run_name="bounded-run",
            model_origin_claim="operator-local-pretrained", execute=False,
            authority_ref=None, seed=42, train_count=8, eval_count=8,
            steps=4, max_tokens=256, max_new_tokens=16,
            learning_rate=0.0002, lora_rank=4, lora_alpha=8,
            device="cpu", answer_only=True,
        ))

    def test_dry_run_is_nonmutating_and_labels_unverified_evidence(self):
        before = tuple(self.outputs.iterdir())
        result = subprocess.run(self.args(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["schema"], "labyrinth.hf-two-cycle-plan.v1")
        self.assertEqual(plan["cycle_count"], 2)
        self.assertEqual(plan["qualification"], "NOT_RUN")
        self.assertEqual(plan["base_model_origin_evidence"], "OPERATOR_CLAIM_ONLY")
        self.assertEqual(plan["device_resource_qualification"], "NOT_RUN")
        self.assertFalse(plan["execute"])
        self.assertEqual(tuple(self.outputs.iterdir()), before)
        self.assertEqual(len(plan["planned_cycle_dirs"]), 2)

    def test_preflight_rejects_existing_output_and_missing_tokenizer(self):
        (self.outputs / "bounded-run").mkdir()
        result = subprocess.run(self.args(), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)

        (self.outputs / "bounded-run").rmdir()
        (self.model / "tokenizer.json").unlink()
        result = subprocess.run(self.args(), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local tokenizer", result.stderr)

    def test_scope_must_cover_two_supported_cycles(self):
        altered = TrainingScopeSpec(
            objective_id="unsupported", domain="numerical",
            skills=("program_reasoning",), specificity=2, scale=2,
            max_cycles=2, seed=42,
        )
        self.scope.write_text(json.dumps(altered.to_dict()), encoding="utf-8")
        result = subprocess.run(self.args(), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("arithmetic", result.stderr)

    def test_harness_rejects_unbounded_optimization_budget(self):
        result = subprocess.run(self.args() + ["--steps", "17"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("budget exceeds", result.stderr)
        self.assertEqual(tuple(self.outputs.iterdir()), ())

    def test_run_requires_explicit_execution_authority_label(self):
        result = subprocess.run(self.args() + ["--execute"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("authority-ref", result.stderr)
        self.assertEqual(tuple(self.outputs.iterdir()), ())

    def test_cycle_one_pins_prior_report_and_lora_value_digest(self):
        plan = self.plan()
        report_path = self.root / "prior-report.json"
        report_path.write_text('{"dummy":"report"}', encoding="utf-8")
        first = _command(plan, 0, predecessor=None,
                         predecessor_report_path=None, authority_ref="operator-123")
        self.assertIn("--scope-json", first)
        self.assertNotIn("--predecessor-lora-sha256", first)

        digest = "1" * 64
        successor = _command(
            plan, 1, predecessor={"candidate_lora_tensor_value_sha256": digest},
            predecessor_report_path=report_path, authority_ref="operator-123",
        )
        self.assertIn("--predecessor-report-sha256", successor)
        self.assertEqual(successor[successor.index("--predecessor-lora-sha256") + 1], digest)
        self.assertEqual(successor[successor.index("--resume-authority-ref") + 1], "operator-123")
        self.assertNotIn("--weak-skills", successor)
        with self.assertRaisesRegex(ValueError, "LoRA digest"):
            _command(plan, 1, predecessor={"candidate_lora_tensor_value_sha256": "bad"},
                     predecessor_report_path=report_path, authority_ref="operator-123")

    def test_hanging_child_is_bounded_and_owned_receipts_remain(self):
        run_root = self.outputs / "fake-run"
        run_root.mkdir()
        began = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            _run_bounded_child(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                run_root=run_root, phase="cycle-0-train", timeout_seconds=0.2,
            )
        self.assertLess(time.monotonic() - began, 6.0)
        receipt = json.loads((run_root / "child-cycle-0-train-outcome.json").read_text())
        self.assertEqual(receipt["status"], "TIMEOUT")
        self.assertEqual(receipt["phase"], "cycle-0-train")
        self.assertTrue((run_root / "child-cycle-0-train-output.log").is_file())
        self.assertTrue((run_root / "child-cycle-0-train-start.json").is_file())
        if os.name == "posix":
            with self.assertRaises(ProcessLookupError):
                os.kill(receipt["pid"], 0)

    def test_child_failure_is_reported_without_overwrite(self):
        run_root = self.outputs / "fake-run"
        run_root.mkdir()
        with self.assertRaises(subprocess.CalledProcessError):
            _run_bounded_child(
                [sys.executable, "-c", "raise SystemExit(7)"],
                run_root=run_root, phase="cycle-0-train", timeout_seconds=3,
            )
        receipt = json.loads((run_root / "child-cycle-0-train-outcome.json").read_text())
        self.assertEqual(receipt["status"], "FAILED")
        self.assertEqual(receipt["exit_code"], 7)
        with self.assertRaises(FileExistsError):
            _run_bounded_child(
                [sys.executable, "-c", "raise SystemExit(0)"],
                run_root=run_root, phase="cycle-0-train", timeout_seconds=3,
            )

    def test_explicit_child_timeout_budget_is_bounded_before_writes(self):
        result = subprocess.run(self.args() + ["--child-timeout-seconds", "14401"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("child timeout", result.stderr)
        self.assertEqual(tuple(self.outputs.iterdir()), ())


if __name__ == "__main__":
    unittest.main()
