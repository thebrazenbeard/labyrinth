"""Contracts for the split-respecting V5 learned-operator trainer."""
import unittest

from labyrinth.minotaur_algebraic_training_v5 import train_atomic_operator


class _TinyNeuralOperator:
    """A learned MLP-like operator, deliberately not a lookup-table model."""
    def __new__(cls, torch):
        class Operator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.op = torch.nn.Embedding(2, 4)
                self.value = torch.nn.Embedding(8, 4)
                self.head = torch.nn.Sequential(torch.nn.Linear(12, 12), torch.nn.Tanh(),
                                                torch.nn.Linear(12, 3))
                self.forward_argument_counts = []

            def forward(self, op, x, y):
                # Recording the interface catches accidental label-as-feature calls.
                self.forward_argument_counts.append(3)
                return self.head(torch.cat((self.op(op), self.value(x), self.value(y)), dim=1))
        return Operator()


class AlgebraicTrainingV5Tests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        self.torch = torch
        torch.manual_seed(41)
        torch.use_deterministic_algorithms(True)
        self.model = _TinyNeuralOperator(torch)
        # Dev contains both a pair repeated from training and genuinely unseen pairs.
        self.train = [(op, x, y, (op + x + 2 * y) % 3)
                      for op in range(2) for x in range(3) for y in range(3)]
        self.dev = [(0, 0, 0, 0), (1, 2, 2, 1), (0, 5, 6, 2), (1, 6, 5, 2)]

    def test_updates_weights_and_reports_seen_and_unseen_dev_pairs(self):
        report = train_atomic_operator(self.model, self.train, self.dev, steps=30,
                                       batch_size=6, lr=0.03, time_budget=10.0,
                                       torch=self.torch)
        self.assertEqual(report["optimizer_updates"], 30)
        self.assertTrue(report["weight_digest"]["changed"])
        self.assertEqual(report["frozen_baseline"]["per_op"]["0"]["seen_pairs"]["total"], 1)
        self.assertEqual(report["final"]["per_op"]["1"]["truly_unseen_pairs"]["total"], 1)
        self.assertEqual(report["final"]["macro"]["all"]["operations"], 2)
        self.assertIsNotNone(report["final"]["macro"]["all"]["mean_confidence"])
        self.assertTrue(all(count == 3 for count in self.model.forward_argument_counts))

    def test_requires_one_fixed_training_length_and_never_accepts_test_rows(self):
        with self.assertRaises(ValueError):
            train_atomic_operator(self.model, self.train, self.dev, batch_size=3, torch=self.torch)
        with self.assertRaises(ValueError):
            train_atomic_operator(self.model, self.train, self.dev, steps=1, epochs=1,
                                  batch_size=3, torch=self.torch)


if __name__ == "__main__":
    unittest.main()
