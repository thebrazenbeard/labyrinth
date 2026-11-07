"""Independent second-pass adversarial inputs to the Oracle Chamber verifier."""

import unittest

from labyrinth.oracle_chamber_exam import make_case
from labyrinth.oracle_chamber_verifier import replay_answers, verify_unsat_bound


class OracleChamberInputBoundaryTests(unittest.TestCase):
    def test_overbudget_question_sequence_cannot_spoof_list_length(self):
        class LyingList(list):
            def __len__(self):
                return 1

        case = make_case(8)
        self.assertIsNone(replay_answers(case, LyingList(["alpha"] * 8)))

    def test_certificate_declared_rules_cannot_spoof_integer_equality(self):
        class EqualsEverything:
            def __eq__(self, _other):
                return True

        case = make_case(8, sealed=True)
        result = verify_unsat_bound(case, {"declared_rules": {
            "secrets": EqualsEverything(), "binary_answers": 2,
            "max_queries": 5, "lie_budget": 2,
        }})
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["certificate_verified"])


if __name__ == "__main__":
    unittest.main()
