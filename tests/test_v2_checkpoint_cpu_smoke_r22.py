"""Opt-in R22 CPU toy qualification: source preflight and conditional physical test.

The physical test, when actually executed, uses Torch/AdamW, generated Maze
TRAIN examples and v2 full-state save/load. It is not pretrained MiniLM,
independent causal generalization, host custody, or anti-rollback proof.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from unittest import TestCase, mock, skipUnless

from scripts import qualify_v2_checkpoint_cpu_r22 as smoke


class ToyV2CpuSourceTests(TestCase):
    def test_readonly_cli_preflight_never_creates_training_sandbox(self):
        output = io.StringIO()
        with mock.patch.object(smoke.tempfile, "TemporaryDirectory",
                               side_effect=AssertionError("unexpected write")):
            with redirect_stdout(output):
                code = smoke.main([])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "PREFLIGHT_ONLY_NO_EFFECT")
        self.assertFalse(result["execute"])
        self.assertFalse(result["physical_cpu_toy_executed"])
        self.assertFalse(result["pretrained_model_loaded"])
        self.assertFalse(result["graduation_authorized"])

    def test_explicit_execute_without_dependencies_returns_skip_not_pass(self):
        output = io.StringIO()
        with mock.patch.object(smoke, "missing_dependencies",
                               return_value=("torch", "safetensors")):
            with mock.patch.object(smoke, "execute_toy_cpu_smoke",
                                   side_effect=AssertionError("must not train")):
                with redirect_stdout(output):
                    code = smoke.main(["--execute"])
        self.assertEqual(code, 3)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "SKIPPED_NO_PHYSICAL_EXECUTION")
        self.assertFalse(result["physical_cpu_toy_executed"])
        self.assertEqual(result["missing_dependencies"], ["torch", "safetensors"])
        self.assertFalse(result["graduation_authorized"])

    def test_generated_maze_plan_not_a_generic_checkpoint_fixture(self):
        plan = smoke._plan()
        self.assertEqual(plan.total_steps, 8)
        self.assertEqual([cycle.plan.cycle_index for cycle in plan.cycles], [0, 1])
        self.assertEqual(len(plan.exposures()), 32)
        binding = smoke._binding(plan)
        binding.validate()
        self.assertEqual(binding.source_sha256, plan.plan_sha256)
        self.assertEqual(binding.cycle_count, 2)

    def test_direct_execute_fails_explicitly_if_physical_deps_missing(self):
        with mock.patch.object(smoke, "missing_dependencies", return_value=("torch",)):
            with self.assertRaisesRegex(RuntimeError, "dependencies unavailable"):
                smoke.execute_toy_cpu_smoke()


@skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors")
    and os.name != "nt",
    "optional Torch/safetensors CPU path or POSIX journal fsync unavailable",
)
class ToyV2CpuPhysicalTests(TestCase):
    def test_real_toy_torch_adamw_two_cycle_resume_and_fresh_reload(self):
        result = smoke.execute_toy_cpu_smoke()
        self.assertEqual(result["status"], smoke.STATUS)
        self.assertTrue(result["physical_cpu_toy_executed"])
        self.assertTrue(result["maze_training_prompts_executed"])
        self.assertTrue(result["optimizer_fullstate_continuity_matches_uninterrupted_toy"])
        self.assertTrue(result["final_fresh_fullstate_restore_verified"])
        self.assertEqual(len(result["checkpoint_cycle_receipt_sha256"]), 2)
        self.assertNotEqual(*result["checkpoint_cycle_receipt_sha256"])
        for digest in result["checkpoint_cycle_receipt_sha256"]:
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertFalse(result["pretrained_model_loaded"])
        self.assertFalse(result["independent_host_custody_verified"])
        self.assertFalse(result["independent_antirollback_verified"])
        self.assertFalse(result["neural_generalization_verified"])
        self.assertFalse(result["graduation_authorized"])
        self.assertFalse(result["persistent_host_artifacts"])
