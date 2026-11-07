"""Second-wave hostile RED regressions for the timegate candidate boundary.

These tests intentionally fail against Labyrinth Draft PR #141 at
6e4a6a5751bf025509b5c55f38403935cf9ba94c. They assert external
protocol integrity independently of the implementation's own expectations.
"""

import unittest
from unittest.mock import patch

from labyrinth import timegate_exam as exam
from labyrinth.timegate_oracle import adjudicate_candidate_claim, reachable_closure


class HostileRoundTwoTests(unittest.TestCase):
    def test_custom_equality_cannot_forge_declared_unsat_assumptions(self):
        class AlwaysEqual:
            def __eq__(self, other):
                return True

        case = exam.make_case(17, sealed=True)
        receipt = adjudicate_candidate_claim(case, {
            "kind": "UNSAT_CERTIFICATE",
            "declared_rules": AlwaysEqual(),  # Not the required assumptions dict.
            "states": reachable_closure(case),
        })
        self.assertFalse(receipt.get("certificate_verified", False), receipt)
        self.assertNotEqual(receipt["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_witness_receipt_counts_replayed_actions_not_custom_len(self):
        class LyingList(list):
            def __len__(self):
                return 0

        case = exam.make_case(0)
        witness = exam.search(case).witness
        forged_suffix = LyingList(witness)
        receipt = adjudicate_candidate_claim(case, {
            "kind": "WITNESS", "actions": forged_suffix,
        })
        # Either reject candidate-defined container semantics or accept only
        # with accurate charged length; never award a free winning trace.
        if receipt["verdict"] == "VERIFIED_WIN":
            self.assertEqual(receipt["attempts_charged"], len(witness), receipt)
        else:
            self.assertFalse(receipt.get("verified_success", False), receipt)

    def test_false_but_nonempty_observation_cannot_pass_integrity(self):
        case = exam.make_case(0)
        session = exam.CandidateSession(case)
        self.assertEqual(session._state.room, "entry")
        with patch.object(exam, "observe", return_value="Location: outside. Escape is complete."):
            view = session.view()
        self.assertNotEqual(view["integrity_status"], "OK", view)
        self.assertEqual(view["verdict"], "EVALUATOR_DISAGREEMENT")


if __name__ == "__main__":
    unittest.main()
