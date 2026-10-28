"""Ten-seed V6 spectral qualification controls; no torch requirement for analysis."""
import unittest
from labyrinth.minotaur_v6_multiseed import SeedSweepConfig, summarize_runs

def make(seed,score,constant=0.02,frozen=0.01):
    return {
        "config":{"seed":seed},
        "model":{"parameter_count":37537},
        "split":{"train_count":7524,"holdout_count":1885,"reversals_grouped":True},
        "candidate":{"train":{"accuracy":1.0,"count":7524},
                     "holdout":{"accuracy":score,"count":1885}},
        "controls":{"constant":{"train":{"accuracy":constant},
                                "holdout":{"accuracy":constant}},
                    "frozen_random":{"train":{"accuracy":frozen},
                                     "holdout":{"accuracy":frozen}}},
        "training":{"optimizer_steps":1280,"initial_weight_sha256":"a"+str(seed),
                    "trained_weight_sha256":"b"+str(seed)},
        "pair_group_audit":{"canonical_groups":947,"directed_rows":1885,
                            "correct_complete_groups":round(score*947),
                            "pair_complete_accuracy":score,
                            "directed_row_accuracy":score,
                            "reversal_consistency_rate":1.0,
                            "prediction_sha256":"f"*64,
                            "independent_label_replay":"minotaur_v6_mul_eval.recompute_mul_label"}
    }

class TenSeedSpectralTests(unittest.TestCase):
    def test_config_validates_ten_distinct_seeds(self):
        SeedSweepConfig().validate()
        for bad in (SeedSweepConfig(seeds=(1,1)), SeedSweepConfig(seeds=(True,)),
                    SeedSweepConfig(seeds=tuple(range(11))),
                    SeedSweepConfig(epochs=0),SeedSweepConfig(rank=0)):
            with self.assertRaises(ValueError): bad.validate()

    def test_strict_gate_uses_all_runs_and_worst_seed(self):
        seeds=list(range(10))
        passing=[make(s,.97) for s in seeds]
        result=summarize_runs(passing)
        self.assertEqual(result["passing_seeds"],10)
        self.assertAlmostEqual(result["mean_group_accuracy"],.97)
        self.assertEqual(result["runs"][0]["canonical_groups"],947)
        self.assertTrue(result["criterion_95pct_on_9_of_10_seeds"])
        failing=passing[:-1]+[make(9,.716)]
        result=summarize_runs(failing)
        self.assertFalse(result["criterion_95pct_on_9_of_10_seeds"])
        self.assertEqual(result["worst_seed_accuracy"],.716)
        self.assertEqual(result["passing_seeds"],9)

    def test_rejects_missing_bad_or_duplicate_runs(self):
        runs=[make(i,.95) for i in range(10)]
        for bad in ([*runs[:-1],runs[0]], [*runs[:-1],{}],
                    [*runs[:-1],make(9,1.1)],
                    [*runs[:-1],make(9,.95,constant=.98)]):
            with self.assertRaises(ValueError): summarize_runs(bad)

    def test_no_single_seed_or_below_baseline_is_promotion(self):
        scores=[make(i,.02) for i in range(10)]
        result=summarize_runs(scores)
        self.assertFalse(result["criterion_95pct_on_9_of_10_seeds"])
        self.assertEqual(result["promotion_recommendation"],"DO_NOT_PROMOTE")
