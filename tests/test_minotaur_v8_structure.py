"""TDD contracts for the V8 learned digit-state multiplication experiment."""

import inspect
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch belongs to neural CI")
class MinotaurV8StructureTests(unittest.TestCase):
    def _module(self):
        import labyrinth.minotaur_v8_structure as structure
        return structure

    def test_partition_has_separate_train_dev_and_reserved_challenges(self):
        v8 = self._module()
        split = v8.build_structure_split(seed=17, dev_fraction=.10)
        train = set(zip(split.train.x.tolist(), split.train.y.tolist()))
        dev = set(zip(split.dev.x.tolist(), split.dev.y.tolist()))
        self.assertTrue(train.isdisjoint(dev))
        self.assertTrue(all(53 not in (x, y, target)
                            for x, y, target in zip(split.train.x.tolist(), split.train.y.tolist(), split.train.target.tolist())))
        self.assertTrue(all(target == 53 for target in split.missing_output.target.tolist()))
        self.assertEqual(len(split.missing_output), 94)
        self.assertTrue(all(x == 53 or y == 53
                            for x, y in zip(split.unseen_operand.x.tolist(), split.unseen_operand.y.tolist())))
        self.assertEqual(len(split.mod101_test), 101 ** 2)

    def test_digit_state_model_has_no_label_table_or_teacher_in_forward(self):
        v8 = self._module()
        model = v8.build_structure_model(v8.StructureConfig(seed=4, hidden_size=40, layers=1))
        self.assertLess(sum(p.numel() for p in model.parameters()), 150_000)
        self.assertFalse(any(isinstance(m, torch.nn.Embedding) for m in model.modules()))
        source = inspect.getsource(model.forward)
        self.assertNotIn("multiplication_targets", source)
        self.assertNotIn("remainder", source)
        x = torch.tensor([0, 1, 53, 96], dtype=torch.long)
        y = torch.tensor([96, 33, 1, 4], dtype=torch.long)
        modulus = torch.full_like(x, 97)
        with patch.object(v8, "multiplication_targets", side_effect=RuntimeError):
            answer = model(x, y, modulus)
        self.assertEqual(answer.shape, (4,))
        self.assertTrue(torch.isfinite(answer).all())

    def test_two_predeclared_seed_study_has_negative_controls_and_split_metrics(self):
        v8 = self._module()
        report = v8.run_structure_study(v8.StructureConfig(
            seeds=(71, 72), hidden_size=32, layers=1, epochs=1,
            batch_size=1024, max_optimizer_seconds=8.0, dev_fraction=.05,
        ))
        self.assertEqual(report["schema"], "labyrinth.minotaur.v8-structure.v1")
        self.assertEqual(len(report["runs"]), 2)
        for run in report["runs"]:
            self.assertEqual(set(run["controls"]), {"constant", "frozen_random", "shuffled_labels"})
            self.assertEqual(set(run["evaluation"]), {"dev", "missing_output_53", "unseen_operand_53", "modulus_101"})
            self.assertEqual(run["split"]["missing_output_train_rows"], 0)
            self.assertEqual(run["split"]["operand_53_train_rows"], 0)
            self.assertGreater(run["training"]["optimizer_steps"], 0)
            self.assertLessEqual(run["training"]["optimizer_seconds"], 8.0)


if __name__ == "__main__":
    unittest.main()
