"""Adversarial attempt-budget witness tests, separate from implementation tests."""

import unittest

from labyrinth.epistemic_shift_exam import MAX_ATTEMPTS, make_case
from labyrinth.epistemic_shift_verifier import adjudicate_claim, independent_replay


class AttemptBudgetTrust(unittest.TestCase):
    def test_subclass_length_lies_cannot_exceed_examination_budget(self):
        class LyingList(list):
            def __len__(self):
                return 4

        case = make_case(0, split="test")
        attempts = LyingList([(0, 0)] * (MAX_ATTEMPTS + 12))
        self.assertEqual(len(attempts), 4)
        self.assertEqual(list.__len__(attempts), 22)
        self.assertIsNone(independent_replay(case, attempts))
        result = adjudicate_claim(case,
                                  {"kind": "HYPOTHESIS", "kernel": "edge"},
                                  attempts, [0] * 22, [])
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(result["verified_success"])

    def test_tuple_subclass_iteration_lies_cannot_alter_replay(self):
        class LyingTuple(tuple):
            def __len__(self):
                return 1

            def __iter__(self):
                return iter(((0, 0),) * 30)

        case = make_case(1, split="test")
        attempts = LyingTuple(((0, 0),))
        self.assertEqual(len(attempts), 1)
        self.assertIsNone(independent_replay(case, attempts))

    def test_exact_builtins_preserve_ten_attempt_boundary(self):
        case = make_case(0, split="test")
        for constructor in (list, tuple):
            for amount in (0, 4, MAX_ATTEMPTS):
                replay = independent_replay(case, constructor([(0, 0)] * amount))
                self.assertIsNotNone(replay)
                self.assertEqual(replay[0].interventions, amount)
            self.assertIsNone(independent_replay(
                case, constructor([(0, 0)] * (MAX_ATTEMPTS + 1))))


if __name__ == "__main__":
    unittest.main()
