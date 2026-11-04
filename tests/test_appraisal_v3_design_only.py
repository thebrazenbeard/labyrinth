"""DESIGN-ONLY V3 source checks; MUST NOT sample outcomes or train/evaluate."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from labyrinth import appraisal_v3 as v3


class AppraisalV3DesignOnlyTests(unittest.TestCase):
    def test_exact_locked_constants_and_cpu_upper_bounds(self):
        manifest = v3.protocol_manifest()
        self.assertEqual(tuple(manifest["seeds"]), (101, 307, 911, 2027, 4049))
        self.assertEqual(manifest["null_repeats"], 250)
        self.assertEqual(manifest["train_steps"], 3200)
        self.assertEqual(manifest["eval_cases_per_family"], 1500)
        self.assertEqual(manifest["null_p95_index_zero_based"], 237)
        self.assertEqual(manifest["max_training_actions"], 5 * 3200 * 251)
        self.assertEqual(manifest["max_evaluation_context_policy_pairs"],
                         5 * 1500 * 2 * 252)
        self.assertEqual(manifest["max_wall_seconds"], 900)
        self.assertEqual(manifest["eval_reward_draws"], 5 * 1500 * 2 * 252)
        self.assertEqual(manifest["null_trace_sha256_count"], 1250)
        self.assertEqual(manifest["realized_outcomes_role"],
                         "secondary_descriptive_not_for_selection")
        self.assertEqual(manifest["status"], "DESIGN_ONLY_NOT_EXECUTED")

    def test_rng_seed_tuple_packing_is_injective_over_declared_domains(self):
        all_ids = []
        for domain in v3.DOMAINS:
            for seed in v3.SEEDS:
                for replicate in (0, 1, 249, 250, 1023):
                    all_ids.append(v3.rng_seed(domain, seed, replicate))
        self.assertEqual(len(all_ids), len(set(all_ids)))
        for bad in (-1, True, 1024):
            with self.subTest(replicate=bad):
                with self.assertRaises(ValueError):
                    v3.rng_seed("null_actions", 101, bad)

    def test_p95_uses_frozen_nearest_rank_index_237(self):
        scores = [i / 1000 for i in range(250)]
        result = v3._summary(scores, candidate=0.2)
        self.assertEqual(result["p95"], 0.237)
        self.assertEqual(result["p05"], 0.012)
        self.assertEqual(result["median"], (0.124 + 0.125) / 2)

    def test_checkpoint_probe_grid_and_exact_equality_contract(self):
        manifest = v3.protocol_manifest()
        self.assertEqual(len(manifest["h2_probe_context_hex"]), 15)
        self.assertEqual(manifest["h2_probe_context_hex"][0], (-1.0).hex())
        self.assertEqual(manifest["h2_probe_context_hex"][-1], (1.0).hex())
        self.assertEqual(manifest["checkpoint_equality"],
                         "exact_float64_values_and_expected_payoff")

    def test_context_digest_is_order_sensitive_on_artificial_nonrandom_values(self):
        # These are hand-constructed constants, NEVER V3 context-generator draws.
        artificial = [0.0] * 1500
        digest = v3.context_digest("C", 101, artificial)
        self.assertEqual(len(digest), 64)
        artificial[5] = 0.5
        self.assertNotEqual(v3.context_digest("C", 101, artificial), digest)
        with self.assertRaises(ValueError):
            v3.context_digest("C", 101, artificial[:-1])

    def test_protocol_command_never_samples_outcome_generators(self):
        # No V3 training, null updates, C/D context draws or evaluation here.
        with patch.object(v3, "_contexts", side_effect=AssertionError("must not sample")), \
             patch.object(v3, "_actual_run", side_effect=AssertionError("must not run")):
            self.assertEqual(v3.protocol_manifest()["protocol"], v3.PROTOCOL)
            self.assertEqual(v3.main(["protocol"]), 0)

    def test_bad_source_head_rejected_before_experiment(self):
        with patch.object(v3, "_actual_run", side_effect=AssertionError("must not run")):
            with self.assertRaises(ValueError):
                v3._qualified_source_head("not-an-exact-head")
            with self.assertRaises(RuntimeError):
                v3._qualified_source_head("0" * 40)


if __name__ == "__main__":
    unittest.main()
