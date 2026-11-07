"""A policy proof must never become physical escape credit."""

import json
import importlib.util
import unittest

from labyrinth.adapters import ReplayAdapter
from labyrinth.minotaur_candidate_bridge import audit_receipt, run_candidate

_HAS_ORACLE_CHAMBER = importlib.util.find_spec("labyrinth.oracle_chamber_exam") is not None


@unittest.skipUnless(_HAS_ORACLE_CHAMBER, "Oracle Chamber PR #142 is not integrated")
class BridgeVerdictSemanticsTests(unittest.TestCase):
    def test_valid_oracle_policy_is_not_an_escape(self):
        from labyrinth.oracle_chamber_exam import CandidateSession, candidate_view, make_case
        case = make_case(0)
        public = candidate_view(case)
        policy = {"questions": ["alpha", "alpha", "beta", "beta", "parity"],
                  "decoder": "nearest_codeword"}
        claim = json.dumps({"kind": "CLAIM", "claim": {"kind": "POLICY", "policy": policy}})
        adapter = ReplayAdapter({}, default=claim)
        receipt = run_candidate(public, lambda: CandidateSession(case), adapter)
        self.assertEqual(receipt.final_verdict, "VERIFIED_WIN")
        self.assertFalse(receipt.verified_escape)
        self.assertTrue(json.loads(receipt.turns[0].public_feedback_json)["policy_verified"])
        self.assertTrue(audit_receipt(receipt, public, lambda: CandidateSession(case)))

    def test_oracle_binary_questions_consume_only_five_calls(self):
        from labyrinth.oracle_chamber_exam import CandidateSession, candidate_view, make_case
        case = make_case(0)
        public = candidate_view(case)
        from labyrinth.minotaur_candidate_bridge import FixedActionAdapter
        receipt = run_candidate(public, lambda: CandidateSession(case),
                                FixedActionAdapter(("alpha",) * 5))
        self.assertEqual(len(receipt.turns), 6)  # Five charged queries + final claim.
        self.assertEqual(receipt.turns[-1].disposition, "CLAIM")
        self.assertEqual(receipt.final_verdict, "UNKNOWN")
        self.assertFalse(receipt.verified_escape)
        self.assertTrue(audit_receipt(receipt, public, lambda: CandidateSession(case)))


if __name__ == "__main__":
    unittest.main()
