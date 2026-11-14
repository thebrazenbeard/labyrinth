"""Independent RED: replay must not trust the same patched core adjudicator.

Exact original affected source: 5aeb3f106da90d401e47f735a0f85588c04646b9.
Mutation is test-local and restores source; no case/weights are modified.
"""

from hashlib import sha256
import unittest
from unittest.mock import patch

from labyrinth.switchback_circuit import ProbeSession, make_case
from labyrinth import switchback_oracle as oracle


class IndependentSessionOracle(unittest.TestCase):
    def test_false_verified_win_from_patched_core_adjudicator_is_not_self_certifying(self):
        case = make_case(14)  # independent DUT exhaustive test proves UNSAT
        claim = {"kind": "CLAIM_UNSAT"}
        genuine = oracle.adjudicate(case, claim)
        self.assertEqual(genuine["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        forged_core = dict(genuine)
        forged_core["verdict"] = "VERIFIED_WIN"
        forged_core["verified_success"] = True
        forged_core.pop("sha256")
        forged_core["sha256"] = sha256(oracle._canonical(forged_core).encode()).hexdigest()
        with patch.object(oracle, "adjudicate", return_value=forged_core):
            receipt = ProbeSession(case).submit(claim)
            self.assertFalse(receipt["verified_success"]
                             and oracle.verify_receipt_replay(receipt),
                             "core scorer and replay share mutated assumption, forgery certified")


if __name__ == "__main__":
    unittest.main()
