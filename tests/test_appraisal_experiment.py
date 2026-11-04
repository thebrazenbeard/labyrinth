"""No-dependency synthetic causal appraisal and fresh-process persistence tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from labyrinth.appraisal_experiment import (
    AppraisalHead, SEEDS, _evaluation_cases, _train_probability,
    evaluate, run, shuffled_feedback, train,
)


class SyntheticAppraisalTests(unittest.TestCase):
    def test_preregistered_three_seed_matched_controls(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                report, head, trace = run(seed)
                self.assertEqual(report["propensities_recorded"], 3200)
                self.assertEqual(len(trace), 3200)
                arms = report["arms"]
                candidate = arms["candidate"]["expected_payoff"]
                self.assertGreater(candidate - arms["frozen"]["expected_payoff"], 0.08)
                self.assertGreater(candidate - arms["disabled_head"]["expected_payoff"], 0.08)
                self.assertGreater(candidate - arms["shuffled_feedback"]["expected_payoff"], 0.06)
                self.assertEqual(arms["candidate"], evaluate(head, seed))
                self.assertEqual(arms["frozen"], arms["disabled_head"])
                uncertainty = report["posthoc_uncertainty"]
                self.assertFalse(uncertainty["preregistered"])
                self.assertLess(uncertainty["radius_per_delta"], 0.081)
                for control, difference in uncertainty[
                    "candidate_minus_controls_expected_payoff"
                ].items():
                    self.assertAlmostEqual(
                        difference, candidate - arms[control]["expected_payoff"])
                    self.assertGreater(difference - uncertainty["radius_per_delta"], 0)

    def test_each_logged_behavior_propensity_matches_before_reward_update(self):
        head, trace = train(11, steps=256)
        reconstructed = AppraisalHead.zero()
        self.assertEqual(len(trace), 256)
        for index, row in enumerate(trace):
            self.assertEqual(row["step"], index)
            self.assertIn(row["action"], (0, 1))
            self.assertIn(row["reward"], (0, 1))
            self.assertGreaterEqual(row["context"], -1)
            self.assertLessEqual(row["context"], 1)
            p0 = _train_probability(reconstructed, row["context"])
            expected_p = p0 if row["action"] == 0 else 1 - p0
            self.assertAlmostEqual(row["propensity"], expected_p, places=14)
            self.assertGreaterEqual(row["propensity"], 0.1 - 1e-14)
            self.assertLessEqual(row["propensity"], 0.9 + 1e-14)
            reconstructed.learn(row["context"], row["action"], row["reward"])
        self.assertEqual(reconstructed.weights, head.weights)

    def test_shuffled_reward_multiset_not_label_alignment(self):
        head, trace = train(29, steps=2048)
        candidate_before = [row[:] for row in head.weights]
        trace_before = [dict(row) for row in trace]
        corrupted = shuffled_feedback(29, trace)
        self.assertIsNot(corrupted, head)
        self.assertEqual(head.weights, candidate_before)
        self.assertEqual(trace, trace_before)
        self.assertNotEqual(head.weights, corrupted.weights)
        self.assertNotEqual(evaluate(head, 29), evaluate(corrupted, 29))

    def test_heldout_cases_not_train_context_replay(self):
        _, trace = train(71, steps=64)
        eval_contexts = [x for x, _, _ in _evaluation_cases(71, 100)]
        # Clipped boundary cells can legitimately repeat; the evaluation
        # *sampling process* is held out, not a globally unique-case filter.
        self.assertGreater(len(set(eval_contexts)), 90)
        self.assertFalse(set(eval_contexts).intersection(
            row["context"] for row in trace))

    def test_fresh_python_interpreter_reloads_without_training_trace(self):
        seed = 11
        report, head, trace = run(seed)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "appraisal.v1.json"
            original_sha = head.save_new(checkpoint)
            self.assertEqual(len(original_sha), 64)
            self.assertEqual(AppraisalHead.load(checkpoint).weights, head.weights)
            output = subprocess.check_output([
                sys.executable, "-m", "labyrinth.appraisal_experiment",
                "evaluate", "--checkpoint", str(checkpoint),
                "--seed", str(seed), "--eval-cases", "1200",
            ], text=True)
            self.assertEqual(json.loads(output), report["arms"]["candidate"])
            self.assertEqual(len(trace), report["propensities_recorded"])

    def test_checkpoint_is_new_only_and_integrity_bound(self):
        head, _ = train(11, steps=24)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "head.json"
            head.save_new(checkpoint)
            with self.assertRaises(FileExistsError):
                head.save_new(checkpoint)
            loaded = AppraisalHead.load(checkpoint)
            self.assertEqual(loaded.weights, head.weights)
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            record["weights"][0][0] += 0.01
            checkpoint.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                AppraisalHead.load(checkpoint)

    def test_rejects_invalid_checkpoint_shapes_nan_and_bounds(self):
        for weights in ([[]], [[0, 0], [0, float("nan")]],
                        [[0, 0], [False, 0]], [[0, 0], [101, 0]]):
            with self.subTest(weights=weights):
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "bad.json"
                    with self.assertRaises(ValueError):
                        AppraisalHead(weights).save_new(checkpoint)
                    self.assertFalse(checkpoint.exists())

    def test_rejects_bad_budgets_without_generating_data(self):
        for bad in (0, -1, True, 2**31):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    train(bad, steps=10)
        with self.assertRaises(ValueError):
            train(11, steps=100001)
        with self.assertRaises(ValueError):
            evaluate(AppraisalHead.zero(), 11, cases=0)


if __name__ == "__main__":
    unittest.main()
