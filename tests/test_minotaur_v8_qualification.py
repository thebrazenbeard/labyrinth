"""Independent-prediction, model-only V8 public qualification contracts."""
import unittest
from labyrinth.minotaur_v8_qualification import QualificationConfig, run_public_exam

class V8PublicExamTests(unittest.TestCase):
    def test_config_bounds(self):
        QualificationConfig().validate()
        for invalid in (QualificationConfig(seeds=(1,1)),
                        QualificationConfig(epochs=0),
                        QualificationConfig(seed=True),
                        QualificationConfig(max_optimizer_seconds=0.)):
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_model_submits_only_predictions_and_excludes_missing_label(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch neural CI")
        report=run_public_exam(QualificationConfig(
            seed=119, seeds=(120,121), epochs=1,
            hidden_size=20, batch_size=2048,max_optimizer_seconds=8.))
        self.assertEqual(report["qualification"],"NOT_ESTABLISHED")
        self.assertEqual(report["graduation_authority"],"none")
        self.assertEqual(len(report["runs"]),2)
        for run in report["runs"]:
            self.assertEqual(run["training"]["target_53_train_rows"],0)
            self.assertEqual(run["training"]["input_53_train_rows"],0)
            self.assertEqual(run["exam"]["verdict"],"SCORED")
            self.assertEqual(set(run["exam"]["groups"]),
                {"independent_holdout","output_53","input_53","modulus_101"})
            self.assertEqual(len(run["exam"]["prediction_sha256"]),64)
            self.assertGreater(run["training"]["optimizer_steps"],0)

