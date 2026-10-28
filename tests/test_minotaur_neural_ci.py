"""Contract for the separately installed Minotaur checkpoint CI runtime."""
from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "minotaur_scratch_v1"
WORKFLOW = ROOT / ".github" / "workflows" / "minotaur-neural-qualification.yml"
CHECKPOINT_SHA256 = "ce2d1a8a7e91f32f58ad01de2304ca6241e2f3f4c2a0227447764503d7fd77ec"


class NeuralQualificationWorkflowContractTests(unittest.TestCase):
    def test_optional_job_is_pinned_and_separate_from_core_dependencies(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('python-version: "3.12"', workflow)
        self.assertIn("runs-on: ubuntu-22.04", workflow)
        self.assertIn("timeout-minutes: 5", workflow)
        self.assertIn("torch==2.5.1+cpu", workflow)
        self.assertIn("safetensors==0.4.5", workflow)
        self.assertIn("numpy==1.26.4", workflow)
        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cpu", workflow)
        self.assertNotIn("--no-deps", workflow)
        self.assertIn("labyrinth/minotaur_*.py", workflow)
        self.assertIn("tests/test_minotaur_*.py", workflow)
        self.assertIn("tests.test_minotaur_v3_ablation", workflow)
        self.assertIn("tests.test_minotaur_scratch_v2", workflow)
        self.assertIn("tests.test_minotaur_combinatorial_curriculum", workflow)
        self.assertIn("tests.test_minotaur_combinatorial_training", workflow)
        self.assertIn("models/minotaur_transition_v4_development/**", workflow)
        self.assertIn("models/minotaur_algebraic_v5_development/**", workflow)
        self.assertIn("tests.test_minotaur_algebraic_checkpoint_v5", workflow)
        self.assertIn("tests.test_minotaur_algebraic_v5", workflow)
        self.assertIn("tests.test_minotaur_v6_spectral", workflow)
        self.assertIn("tests.test_minotaur_v6_latent", workflow)
        self.assertIn("tests.test_minotaur_v6_holdouts", workflow)
        self.assertIn("tests.test_minotaur_v6_mul_eval", workflow)
        self.assertIn("tests.test_minotaur_v6_multiseed", workflow)
        self.assertIn("tests.test_minotaur_v6_group_audit", workflow)
        self.assertIn("tests.test_minotaur_v6_group", workflow)
        self.assertIn("tests.test_minotaur_v7_numeric", workflow)
        self.assertIn("tests.test_minotaur_v7_benchmark", workflow)
        self.assertIn("tests.test_minotaur_v8_structure", workflow)
        self.assertIn("tests.test_minotaur_v8_examiner", workflow)
        self.assertIn("tests.test_minotaur_v8_qualification", workflow)
        self.assertIn("tests.test_minotaur_v8_transfer", workflow)

        self.assertIn("tests.test_minotaur_transition_cli", workflow)
        self.assertIn("tests.test_minotaur_transition_v4", workflow)
        self.assertIn("tests.test_minotaur_atomic_training", workflow)

        self.assertIn("timeout 90s", workflow)
        self.assertIn("actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683", workflow)
        self.assertIn("actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065", workflow)
        self.assertIn("dependencies = []", project)

    def test_actual_committed_checkpoint_reloads_and_one_guarded_trace_runs(self):
        try:
            import safetensors.torch  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("only the dedicated optional-dependency CI job installs torch+safetensors")

        from labyrinth.minotaur_guarded import predict_guarded_trace
        from labyrinth.minotaur_scratch import reload_scratch

        restored = reload_scratch(CHECKPOINT)
        report = json.loads((CHECKPOINT / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(restored["trained_weight_sha256"], report["trained_weight_sha256"])
        self.assertEqual(restored["eval_prediction_sha256"], report["eval_prediction_sha256"])

        # One fixed, malformed-at-step-three public trace bounds neural inference work.
        question = (
            "Execute this register program in order. ADD, SUB, and MUL update the named "
            "target register using its current value and the operand. An operand is either "
            "a literal integer or the current value of the named register. After each "
            "instruction, replace the updated value with its nonnegative remainder modulo "
            "97. Other registers do not change.\nInitial registers: A=1, B=93\n"
            "1. ADD A 3\n2. MUL B 9\n3. ADD A B\nWhat is the final value of A? "
            "Respond with exactly one line: ANSWER: <integer>."
        )
        verdict = predict_guarded_trace(
            CHECKPOINT, question,
            ("Step 1: A=4, B=93", "Step 2: A=4, B=61", "Step 3: A=66, B=61"),
        )
        self.assertEqual(verdict["verified_checkpoint_sha256"], CHECKPOINT_SHA256)
        self.assertEqual(verdict["neural_first_error_step"], 0)
        self.assertEqual(verdict["disposition"], "ABSTAIN")
        self.assertEqual(verdict["graduation_authority"], "none")
        self.assertEqual(verdict["qualification"], "NOT_ESTABLISHED")
