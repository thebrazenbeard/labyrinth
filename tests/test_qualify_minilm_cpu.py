"""Offline MiniLM pilot preflight and reporting invariants (no ML dependency)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_minilm_cpu.py"
SPEC = importlib.util.spec_from_file_location("qualify_minilm_cpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MiniLMCPUProbeTests(unittest.TestCase):
    def test_frozen_public_fixture_has_nonoverlapping_balanced_splits(self):
        sizes = MODULE.validate_fixtures()
        self.assertEqual(sizes, {"cycle0": 12, "cycle1": 12,
                                 "heldout": 10, "transfer": 10})
        self.assertEqual(set(MODULE.ARMS), {"correct", "zero_update", "inverted"})

    def test_output_run_preflight_never_overwrites_existing_run(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            candidate = MODULE.plan_output(parent, "minilm-cycle-01")
            self.assertEqual(candidate.parent, parent.resolve())
            self.assertFalse(candidate.exists())
            candidate.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.plan_output(parent, "minilm-cycle-01")
            for name in ("", "../escape", "bad/path", ".hidden", "a" * 70):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    MODULE.plan_output(parent, name)

    def test_control_comparison_only_reports_observed_bounded_values(self):
        result = MODULE.comparison({"correct": .9, "zero_update": .5,
                                    "inverted": .1})
        self.assertTrue(result["correct_outperforms_both"])
        self.assertAlmostEqual(result["correct_minus_zero"], .4)
        self.assertAlmostEqual(result["correct_minus_inverted"], .8)
        for bad in ({"correct": 1},
                    {"correct": 1, "zero_update": 0, "inverted": -1},
                    {"correct": 1, "zero_update": 0, "inverted": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MODULE.comparison(bad)

    def test_read_only_plan_does_not_load_or_initialize_ml(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("builtins.print") as printed:
                self.assertEqual(MODULE.main(["--output-parent", temp,
                                              "--run-name", "probe"]), 0)
            payload = json.loads(printed.call_args.args[0])
            self.assertEqual(payload["schema"], MODULE.SCHEMA)
            self.assertFalse(payload["execute"])
            self.assertFalse(payload["graduation_authorized"])
            self.assertFalse((Path(temp) / "probe").exists())

    def test_zero_update_sham_runs_optimizer_at_zero_learning_rate(self):
        self.assertEqual(MODULE.optimizer_learning_rate("zero_update", .002), 0.0)
        self.assertEqual(MODULE.optimizer_learning_rate("correct", .002), .002)
        self.assertEqual(MODULE.optimizer_learning_rate("inverted", .002), .002)
        with self.assertRaises(ValueError):
            MODULE.optimizer_learning_rate("unknown", .002)

    def test_reload_checks_full_heldout_logits_not_just_parameter_hashes(self):
        original = [[.2, .8], [.9, .1]]
        self.assertEqual(MODULE.reload_logit_delta(original, original), 0.0)
        self.assertAlmostEqual(
            MODULE.reload_logit_delta(original, [[.200001, .799999], [.9, .1]]),
            1e-6,
        )
        with self.assertRaises(RuntimeError):
            MODULE.reload_logit_delta(original, [[.2, .81], [.9, .1]])
        with self.assertRaises(RuntimeError):
            MODULE.reload_logit_delta(original, [[.2, .8]])
        with self.assertRaises(RuntimeError):
            MODULE.reload_logit_delta(original, [[float("nan"), .8], [.9, .1]])


if __name__ == "__main__":
    unittest.main()
