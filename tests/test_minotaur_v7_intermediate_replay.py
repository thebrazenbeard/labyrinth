"""Independent V7 multi-head prediction replay integration contracts."""
import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None, "Optional PyTorch belongs to neural qualification CI")
class IntermediateAuditIntegrationTests(unittest.TestCase):
    def test_frozen_predictions_replayed_by_independent_evaluator(self):
        from labyrinth.minotaur_v7_intermediate import IntermediateConfig, run_intermediate_study
        from labyrinth.minotaur_v7_blindish_eval import build_protocol, audit_predictions
        config=IntermediateConfig(seeds=(131,132),seed=20261028,epochs=1,
                                  max_optimizer_seconds=8.0,batch_size=1024)
        captured={}
        def sink(seed,challenge,predictions):
            self.assertNotIn(challenge,captured.setdefault(seed,{}))
            captured[seed][challenge]=predictions
        study=run_intermediate_study(config,prediction_sink=sink)
        protocol=build_protocol(seed=config.seed,holdout_fraction=config.holdout_fraction)
        for run in study["runs"]:
            seed=run["seed"]
            self.assertEqual(set(captured[seed]),set(protocol.challenges))
            audited=audit_predictions(protocol,captured[seed])
            self.assertTrue(audited["integrity"]["ok"])
            for model_name,external_name in (
                ("independent_holdout","unseen_group_pairs"),
                ("absent_output_53","missing_output_53"),
                ("absent_operand_53","missing_input_53"),
                ("modulus_101","mod101")):
                self.assertEqual(
                    audited["cases"][external_name]["correct"],
                    run["evaluation"][model_name]["correct"])
                self.assertEqual(
                    audited["cases"][external_name]["count"],
                    run["evaluation"][model_name]["count"])
            self.assertEqual(audited["qualification"],"NOT_ESTABLISHED")

