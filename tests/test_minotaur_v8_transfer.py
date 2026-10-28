"""Contracts for the preregistered cross-modulus V8 transfer study."""

import inspect
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch belongs to neural CI")
class MinotaurV8TransferTests(unittest.TestCase):
    def _module(self):
        import labyrinth.minotaur_v8_transfer as transfer
        return transfer

    def test_preregistered_training_moduli_and_strict_97_residue_53_holdout(self):
        transfer = self._module()
        split = transfer.build_transfer_split(seed=17, dev_fraction=0.05)
        self.assertEqual(split.train_moduli, (83, 89, 97))
        self.assertEqual(split.final_modulus, 101)
        rows97 = split.train[97]
        self.assertEqual(int(((rows97.x == 53) | (rows97.y == 53)).sum()), 0)
        self.assertEqual(int((rows97.target == 53).sum()), 0)
        self.assertGreater(len(split.dev[97]), 0)
        self.assertTrue(torch.all((split.strict_input_53.x == 53) |
                                  (split.strict_input_53.y == 53)))
        self.assertTrue(torch.all(split.strict_output_53.target == 53))
        self.assertFalse(torch.any((split.strict_output_53.x == 53) |
                                   (split.strict_output_53.y == 53)))
        train_pairs = set(zip(rows97.x.tolist(), rows97.y.tolist()))
        self.assertFalse(train_pairs & set(zip(split.dev[97].x.tolist(), split.dev[97].y.tolist())))

    def test_forward_is_learned_and_has_no_product_target_or_lookup_table(self):
        transfer = self._module()
        model = transfer.build_transfer_model(transfer.TransferConfig(seed=4, width=20, depth=2))
        source = inspect.getsource(model.forward)
        self.assertNotIn("multiplication_targets", source)
        self.assertNotIn("target", inspect.signature(model.forward).parameters)
        self.assertLess(sum(p.numel() for p in model.parameters()), 25_000)
        x = torch.tensor([0, 1, 53, 96], dtype=torch.long)
        y = torch.tensor([96, 33, 1, 4], dtype=torch.long)
        p = torch.full_like(x, 97)
        with patch.object(transfer, "multiplication_targets", side_effect=RuntimeError):
            output = model(x, y, p)
        self.assertEqual(output.shape, (4,))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output, model(y, x, p)))

    def test_two_seed_report_keeps_dev_and_final_101_separate(self):
        transfer = self._module()
        report = transfer.run_study(transfer.TransferConfig(
            seeds=(5, 6), width=16, depth=1, epochs=1, batch_size=2048,
            max_optimizer_seconds=8.0, dev_fraction=0.03,
        ))
        self.assertEqual(report["schema"], "labyrinth.minotaur.v8-transfer.v1")
        self.assertEqual(report["protocol"]["train_moduli"], [83, 89, 97])
        self.assertEqual(report["protocol"]["final_evaluation_modulus"], 101)
        self.assertEqual(len(report["runs"]), 2)
        for run in report["runs"]:
            self.assertEqual(set(run["controls"]), {"constant", "frozen_random"})
            self.assertEqual(run["split"]["mod97_input_53_train_rows"], 0)
            self.assertEqual(run["split"]["mod97_output_53_train_rows"], 0)
            self.assertIn("development", run)
            self.assertIn("final_modulus_101", run["evaluation"])
            self.assertNotIn("final_modulus_101", run["development"])
            self.assertGreater(run["training"]["optimizer_steps"], 0)
            self.assertLessEqual(run["training"]["optimizer_seconds"], 8.0)
        self.assertEqual(report["promotion_recommendation"], "DO_NOT_PROMOTE")


if __name__ == "__main__":
    unittest.main()
