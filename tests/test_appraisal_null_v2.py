"""Independent shuffled-online null falsifiers for the synthetic appraisal toy."""

from __future__ import annotations

import unittest

from labyrinth.appraisal_experiment import AppraisalHead, _train_probability, run, train
from labyrinth.appraisal_null_v2 import evaluate_triangular, run_v2, train_online_null


class AppraisalOnlineNullV2Tests(unittest.TestCase):
    def test_online_null_has_own_pre_feedback_action_probabilities(self):
        _, candidate = train(11, steps=256)
        baseline = [row["action"] for row in candidate]
        head, trace = train_online_null(11, [row["reward"] for row in candidate],
                                      replicate=0, include_trace=True)
        self.assertEqual(len(trace), len(candidate))
        self.assertCountEqual([r["reward"] for r in trace],
                              [r["reward"] for r in candidate])
        self.assertGreater(sum(x["action"] != y for x, y in zip(trace, baseline)), 20)
        reconstruction = AppraisalHead.zero()
        for i, row in enumerate(trace):
            self.assertEqual(row["step"], i)
            p0 = _train_probability(reconstruction, row["context"])
            self.assertAlmostEqual(row["propensity"],
                                   p0 if row["action"] == 0 else 1 - p0, places=14)
            reconstruction.learn(row["context"], row["action"], row["reward"])
        self.assertEqual(reconstruction.weights, head.weights)

    def test_fixed_null_replicate_and_distinct_replicates(self):
        _, candidate = train(29, steps=96)
        rewards = [row["reward"] for row in candidate]
        h0, r0 = train_online_null(29, rewards, replicate=0, include_trace=True)
        h0b, r0b = train_online_null(29, rewards, replicate=0, include_trace=True)
        h1, r1 = train_online_null(29, rewards, replicate=1, include_trace=True)
        self.assertEqual(r0, r0b)
        self.assertEqual(h0.weights, h0b.weights)
        self.assertNotEqual(r0, r1)
        self.assertNotEqual(h0.weights, h1.weights)

    def test_frozen_triangular_cases_and_existing_grid_are_distinct(self):
        head = AppraisalHead.zero()
        a = run(11, train_steps=128, eval_cases=128)[0]["arms"]["frozen"]
        b = evaluate_triangular(head, 11, cases=128)
        self.assertNotEqual(a["expected_payoff"], b["expected_payoff"])
        self.assertEqual(b["cases"], 128)

    def test_protocol_defined_100_online_nulls_each_preselected_seed(self):
        for seed in (11, 29, 71):
            with self.subTest(seed=seed):
                r = run_v2(seed, train_steps=3200, eval_cases=1200,
                           null_repeats=100)
                self.assertEqual(r["seed"], seed)
                self.assertEqual(r["null_repeats"], 100)
                self.assertEqual(r["null_logged_propensities"], 320000)
                self.assertIn(r["strong_causal_criterion"], ("PASS", "FAIL"))
                self.assertEqual(set(r["evaluation"]), {"A_grid", "B_triangular"})
                for sampler in r["evaluation"].values():
                    nulls = sampler["nulls"]
                    self.assertEqual(nulls["count"], 100)
                    self.assertLessEqual(nulls["p05"], nulls["median"])
                    self.assertLessEqual(nulls["median"], nulls["p95"])
                    self.assertLessEqual(nulls["p95"], nulls["max"])
                    self.assertLessEqual(nulls["within_0p06_or_above_fraction"], 1)
                    self.assertGreaterEqual(nulls["within_0p06_or_above_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
