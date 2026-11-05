"""Read-only preflight and control-oracle checks for MiniLM LoRA ablation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_minilm_ablation_cpu.py"
SPEC = importlib.util.spec_from_file_location("qualify_minilm_ablation_cpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MiniLMLoRAAblationContractTests(unittest.TestCase):
    def test_standalone_cli_loads_sibling_pilot_without_installed_scripts_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--output-parent", tmp,
                "--run-name", "standalone",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["execute"])
            self.assertFalse((Path(tmp) / "standalone").exists())

    def test_arm_design_freezes_separate_parameter_families(self):
        self.assertEqual(MODULE.ARMS,
                         ("full", "head_only", "lora_only", "zero_update"))
        self.assertEqual(MODULE.pilot.validate_fixtures(),
                         {"cycle0": 12, "cycle1": 12,
                          "heldout": 10, "transfer": 10})

    def test_observed_differences_are_bounded_not_qualifications(self):
        scores = {"full": .9, "head_only": .7, "lora_only": .6,
                  "zero_update": .5}
        observed = MODULE.compare_arms(scores)
        self.assertAlmostEqual(observed["full_minus_head_only"], .2)
        self.assertAlmostEqual(observed["lora_only_minus_zero"], .1)
        self.assertAlmostEqual(observed["head_only_minus_zero"], .2)
        self.assertTrue(observed["full_outperforms_head_only"])
        self.assertNotIn("graduation_authorized", observed)
        self.assertNotIn("causal_proven", observed)

    def test_incomplete_or_nonfinite_comparisons_rejected(self):
        good = {"full": .9, "head_only": .7, "lora_only": .6,
                "zero_update": .5}
        for corrupt in (
            {"full": .9, "head_only": .7},
            {**good, "head_only": float("nan")},
            {**good, "lora_only": float("inf")},
            {**good, "zero_update": -0.1},
            {**good, "full": True},
        ):
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                MODULE.compare_arms(corrupt)

    def test_default_plan_never_creates_run_or_loads_ml(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("builtins.print") as printed:
                self.assertEqual(MODULE.main([
                    "--output-parent", tmp, "--run-name", "ablation1",
                ]), 0)
            payload = json.loads(printed.call_args.args[0])
            self.assertFalse(payload["execute"])
            self.assertFalse(payload["graduation_authorized"])
            self.assertEqual(payload["arms"], list(MODULE.ARMS))
            self.assertFalse((Path(tmp) / "ablation1").exists())

    def test_existing_run_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            (parent / "used").mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.pilot.plan_output(parent, "used")
            for name in ("", ".hidden", "../outside", "bad/subdir"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    MODULE.pilot.plan_output(parent, name)


if __name__ == "__main__":
    unittest.main()
