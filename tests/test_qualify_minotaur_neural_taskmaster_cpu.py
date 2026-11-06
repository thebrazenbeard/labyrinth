"""Bounded source controls for the opt-in pretrained-feature neural Minotaur."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from scripts import qualify_minotaur_neural_taskmaster_cpu as probe


try:
    import torch
except ImportError:
    torch = None


class NeuralTaskmasterSourceTests(unittest.TestCase):
    def test_exposed_public_fixtures_are_balanced_and_nonoverlapping(self):
        self.assertEqual(probe.validate_fixtures(), {
            "cycle0": 16, "cycle1": 16, "heldout": 16, "transfer": 16,
        })
        seen = set()
        for split, rows in probe.fixtures().items():
            self.assertEqual({label for _, label in rows}, {0, 1, 2, 3})
            self.assertEqual([sum(label == i for _, label in rows) for i in range(4)],
                             [4, 4, 4, 4])
            for prompt, label in rows:
                self.assertIn(label, range(4))
                self.assertNotIn("tactic=", prompt.casefold())
                self.assertNotIn(prompt, seen, split)
                seen.add(prompt)

    def test_training_visit_schedule_is_bounded_and_deterministic(self):
        visits = probe.training_indices(8)
        self.assertEqual(visits, probe.training_indices(8))
        self.assertEqual(len(visits), 8)
        self.assertTrue(all(len(batch) == 8 for batch in visits))
        self.assertTrue(all(0 <= i < 16 for batch in visits for i in batch))
        self.assertGreater(len({i for batch in visits for i in batch}), 8)
        for invalid in (0, -1, 129, 1.1, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                probe.training_indices(invalid)

    def test_comparison_requires_all_finite_control_arms(self):
        result = probe.compare_arms({
            "correct": 0.875, "zero_update": 0.25, "corrupted": 0.125,
        })
        self.assertTrue(result["correct_outperforms_both"])
        self.assertAlmostEqual(result["correct_minus_zero"], 0.625)
        self.assertAlmostEqual(result["correct_minus_corrupted"], 0.75)
        for bad in (
            {"correct": 1.0, "zero_update": 0.0},
            {"correct": 0.8, "zero_update": 0.3, "corrupted": float("nan")},
            {"correct": 0.8, "zero_update": -0.1, "corrupted": 0.2},
            {"correct": True, "zero_update": 0.2, "corrupted": 0.2},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.compare_arms(bad)

    def test_preflight_requires_new_bounded_run_directory(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(probe.plan_output(root, "test-neural-taskmaster"),
                             root / "test-neural-taskmaster")
            for bad in ("../escape", "../", "", "a b", "a\\b", "~", "."):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    probe.plan_output(root, bad)
            (root / "used").mkdir()
            with self.assertRaises(FileExistsError):
                probe.plan_output(root, "used")
            with self.assertRaises(ValueError):
                probe.plan_output(root / "not-here", "new")

    def test_cli_default_is_read_only_and_never_graduates(self):
        with TemporaryDirectory() as temp:
            completed = subprocess.run([
                sys.executable, "-m", "scripts.qualify_minotaur_neural_taskmaster_cpu",
                "--output-parent", temp, "--run-name", "planned",
            ], text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(completed.stdout)
            self.assertEqual(data["schema"], probe.SCHEMA)
            self.assertFalse(data["execute"])
            self.assertFalse(data["graduation_authorized"])
            self.assertFalse((Path(temp) / "planned").exists())

    def test_invalid_cli_budget_rejects_before_execution(self):
        with TemporaryDirectory() as temp:
            for extra in (("--steps-per-cycle", "0"),
                          ("--steps-per-cycle", "129"),
                          ("--learning-rate", "0"),
                          ("--seed", "-1")):
                with self.subTest(extra=extra):
                    r = subprocess.run([
                        sys.executable, "-m", "scripts.qualify_minotaur_neural_taskmaster_cpu",
                        "--output-parent", temp, "--run-name", "invalid",
                        *extra,
                    ], text=True, capture_output=True, check=False)
                    self.assertNotEqual(r.returncode, 0)
                    self.assertFalse((Path(temp) / "invalid").exists())


@unittest.skipIf(torch is None, "optional CPU Torch unavailable")
class NeuralTaskmasterTorchTests(unittest.TestCase):
    def test_actual_head_update_and_control_exposure_integrity(self):
        torch.manual_seed(123)
        encoded = torch.randn(16, 4)
        labels = torch.tensor([i % 4 for i in range(16)], dtype=torch.int64)
        outcomes = {}
        for arm in probe.ARMS:
            torch.manual_seed(456)
            head = torch.nn.Linear(4, 4)
            outcomes[arm] = probe.train_head_cycle(
                head, encoded, labels, arm=arm, steps=8, learning_rate=0.01)
        self.assertEqual(len({outcomes[a]["before_sha256"] for a in probe.ARMS}), 1)
        self.assertEqual(len({tuple(outcomes[a]["visits"]) for a in probe.ARMS}), 1)
        self.assertEqual(outcomes["zero_update"]["optimizer_updates"], 0)
        self.assertEqual(outcomes["correct"]["optimizer_updates"], 8)
        self.assertEqual(outcomes["corrupted"]["optimizer_updates"], 8)
        self.assertEqual(outcomes["zero_update"]["before_sha256"],
                         outcomes["zero_update"]["after_sha256"])
        self.assertNotEqual(outcomes["correct"]["before_sha256"],
                            outcomes["correct"]["after_sha256"])
        self.assertNotEqual(outcomes["corrupted"]["before_sha256"],
                            outcomes["corrupted"]["after_sha256"])
        self.assertNotEqual(outcomes["correct"]["after_sha256"],
                            outcomes["corrupted"]["after_sha256"])


if __name__ == "__main__":
    unittest.main()
