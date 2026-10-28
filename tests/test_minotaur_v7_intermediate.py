"""TDD contracts for the V7 intermediate multi-head experiment."""
import inspect
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch belongs to neural CI")
class MinotaurV7IntermediateTests(unittest.TestCase):
    def _module(self):
        import labyrinth.minotaur_v7_intermediate as intermediate
        return intermediate

    def test_group_split_has_no_53_training_labels_or_operands(self):
        m = self._module()
        split = m.build_intermediate_split(seed=17, holdout_fraction=.10)
        train = set(zip(split.train.x.tolist(), split.train.y.tolist()))
        self.assertTrue(all(53 not in (x, y, z) for x, y, z in
                            zip(split.train.x.tolist(), split.train.y.tolist(), split.train.remainder.tolist())))
        self.assertTrue(all((y, x) in train for x, y in train))
        self.assertTrue(all(z == 53 for z in split.absent_output.remainder.tolist()))
        self.assertTrue(all(53 in (x, y) for x, y in
                            zip(split.absent_operand.x.tolist(), split.absent_operand.y.tolist())))
        self.assertTrue(m.audit_group_pairs(split)["all_disjoint"])

    def test_forward_is_symmetric_multihead_and_never_calls_offline_teacher(self):
        m = self._module()
        model = m.build_intermediate_model(m.IntermediateConfig(seed=4, width=24, depth=2))
        x = torch.tensor([0, 1, 53, 96], dtype=torch.long)
        y = torch.tensor([96, 33, 1, 4], dtype=torch.long)
        modulus = torch.full_like(x, 97)
        with patch.object(m, "offline_targets", side_effect=RuntimeError):
            out = model(x, y, modulus)
        self.assertEqual(set(out), {"remainder", "quotient", "product_scale"})
        self.assertTrue(all(value.shape == (4,) and torch.isfinite(value).all() for value in out.values()))
        reverse = model(y, x, modulus)
        self.assertTrue(all(torch.equal(out[key], reverse[key]) for key in out))
        self.assertLess(sum(p.numel() for p in model.parameters()), 30_000)
        forward = inspect.getsource(m.IntermediateNetwork.forward)
        self.assertNotIn("x*y", forward.replace(" ", ""))
        self.assertNotIn("torch.remainder", forward)

    def test_two_seed_report_includes_auxiliary_losses_group_audit_and_controls(self):
        m = self._module()
        report = m.run_intermediate_study(m.IntermediateConfig(
            seeds=(5, 6), width=20, depth=2, epochs=1, batch_size=1024,
            max_optimizer_seconds=8.0, holdout_fraction=.05,
        ))
        self.assertEqual(report["schema"], "labyrinth.minotaur.v7-intermediate.v1")
        self.assertEqual(len(report["runs"]), 2)
        self.assertEqual(report["group_pair_audit"]["all_disjoint"], True)
        self.assertIn("quotient", report["auxiliary_supervision"])
        for run in report["runs"]:
            self.assertEqual(set(run["controls"]), {"constant", "frozen_random"})
            self.assertEqual(run["split"]["target_53_train_rows"], 0)
            self.assertEqual(run["split"]["operand_53_train_rows"], 0)
            self.assertIn("absent_output_53", run["evaluation"])
            self.assertIn("absent_operand_53", run["evaluation"])
            self.assertIn("modulus_101", run["evaluation"])
            self.assertGreater(run["training"]["optimizer_steps"], 0)
            self.assertLessEqual(run["training"]["optimizer_seconds"], 8.0)


if __name__ == "__main__":
    unittest.main()
