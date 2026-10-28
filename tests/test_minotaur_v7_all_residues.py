"""Every-output-residue stress protocol for numeric V7 neural multiplier."""
import unittest
from labyrinth.minotaur_v7_all_residues import StressConfig, summarize_stress, run_stress

class AllResidueStressTests(unittest.TestCase):
    def test_configuration_is_predeclared_and_bounded(self):
        StressConfig().validate()
        for bad in (StressConfig(seeds=(4,4)), StressConfig(epochs=0),
                    StressConfig(batch_size=0), StressConfig(seed= True)):
            with self.assertRaises(ValueError): bad.validate()

    def test_incomplete_diagnostic_never_qualifies(self):
        results = [{"seed":17,"label":3,"heldout_correct":0,"heldout_count":96,
                    "heldout_accuracy":0.0,"frozen_accuracy":0.02,"constant_accuracy":0.0,
                    "train_excludes_label":True,"optimizer_updates":1,
                    "weights_changed":True},
                   {"seed":19,"label":3,"heldout_correct":1,"heldout_count":96,
                    "heldout_accuracy":1/96,"frozen_accuracy":0.03,"constant_accuracy":0.0,
                    "train_excludes_label":True,"optimizer_updates":1,
                    "weights_changed":True}]
        report=summarize_stress(results,seeds=(17,19),labels=(3,))
        self.assertEqual(report["tested_label_count"],1)
        self.assertFalse(report["all_97_outputs_evaluated"])
        self.assertEqual(report["promotion_recommendation"],"DO_NOT_PROMOTE")

    def test_source_frozen_real_neural_stress_has_actual_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional neural runtime")
        config=StressConfig(seed=17,seeds=(17,19),epochs=1,batch_size=2048,
                            max_optimizer_seconds=10.0)
        report=run_stress(config,labels=(3,53),require_clean_source=False)
        self.assertEqual(report["aggregate"]["tested_label_count"],2)
        self.assertFalse(report["aggregate"]["all_97_outputs_evaluated"])
        self.assertEqual(len(report["runs"]),4)
        for row in report["runs"]:
            self.assertTrue(row["train_excludes_label"])
            self.assertGreater(row["optimizer_updates"],0)
            self.assertTrue(row["weights_changed"])
            self.assertGreater(row["heldout_count"],0)
            self.assertEqual(row["qualification"],"NOT_ESTABLISHED")
