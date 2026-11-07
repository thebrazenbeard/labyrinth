"""Skeptic hostile controls for source-only real-model evaluation boundaries.

Tests are standalone and touch no model, host service, or repository network.
Historically RED tests deliberately target duplicated trust assumptions.
"""

from dataclasses import replace
from contextlib import nullcontext
import unittest
from unittest.mock import patch

from labyrinth import switchback_circuit as world, switchback_oracle as oracle
from labyrinth.minotaur_algebraic_eval_v5 import AtomicEvaluationCase
from labyrinth.minotaur_public_evidence import evaluate_compact_public


class CanonicalModelTrialBoundary(unittest.TestCase):
    def test_script_only_can_score_perfect_public_diagnostic_without_training(self):
        class FakeTensor:
            ndim = 0

            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeTorch:
            long = "not_torch"

            @staticmethod
            def tensor(value, dtype):
                return value

            @staticmethod
            def no_grad():
                return nullcontext()

        class NeverTrainedScript:
            training = False
            calls = 0

            def eval(self):
                self.training = False

            def train(self, mode):
                self.training = mode

            def predict(self, op, x, y):
                self.calls += 1
                return FakeTensor((x + y if op == 0 else
                                   x - y if op == 1 else x * y) % 97)

        cases = [AtomicEvaluationCase("a", "ADD", 11, 17, 28),
                 AtomicEvaluationCase("b", "SUB", 8, 13, 92),
                 AtomicEvaluationCase("c", "MUL", 13, 17, 27)]
        scripted = NeverTrainedScript()
        report = evaluate_compact_public(
            scripted, cases, [], [(0, 1, 2, 3)], FakeTorch)
        self.assertEqual(report["disposition"], "PUBLIC_DIAGNOSTIC")
        self.assertEqual(report["atomic"]["metrics"]["exact_accuracy"], 1.0)
        self.assertEqual(scripted.calls, 3)
        self.assertEqual(report["qualification"], "NOT_ESTABLISHED")
        self.assertEqual(report["graduation_authority"], "none")

    def test_overridden_case_equality_cannot_admit_noncanonical_rules(self):
        canonical = world.make_case(0)
        forged = replace(canonical, target=(canonical.target + 1) % 8)
        self.assertFalse(oracle._valid_case(forged))
        with patch.object(world.Case, "__eq__", return_value=True):
            self.assertFalse(
                oracle._valid_case(forged),
                "class-wide Python equality spoof fooled canonical Case check",
            )

    def test_overridden_case_equality_cannot_open_forged_probe_session(self):
        canonical = world.make_case(0)
        forged = replace(canonical, mechanic_signature="0" * 64)
        with patch.object(world.Case, "__eq__", return_value=True):
            with self.assertRaises(ValueError):
                world.ProbeSession(forged)


if __name__ == "__main__":
    unittest.main()
