"""Independent RED: probe-bearing exam receipts must bind actual observations.

Original failing source: 0021d05516b90f216bdf36c9a10a6a1ac1e3d87d.
"""

import unittest

from labyrinth.switchback_circuit import ProbeSession, make_case
from labyrinth.switchback_controls import MEMORIZED_SEED0_PATH
from labyrinth.switchback_oracle import verify_receipt_replay


class StatefulReceiptEvidence(unittest.TestCase):
    def test_zero_probe_and_two_probe_sessions_cannot_share_identical_receipt(self):
        case = make_case(0)
        proposal = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
        cold = ProbeSession(case)
        cold_receipt = cold.submit(proposal)
        observed = ProbeSession(case)
        self.assertEqual(observed.probe("0")["remaining_probes"], 15)
        self.assertEqual(observed.probe("RESET")["remaining_probes"], 14)
        observed_receipt = observed.submit(proposal)
        self.assertEqual(cold_receipt["verdict"], "VERIFIED_WIN")
        self.assertEqual(observed_receipt["verdict"], "VERIFIED_WIN")
        self.assertTrue(verify_receipt_replay(cold_receipt))
        self.assertTrue(verify_receipt_replay(observed_receipt))
        self.assertNotEqual(cold_receipt["sha256"], observed_receipt["sha256"],
                            "actual probe trajectory omitted from ostensibly replayable evidence")

    def test_candidate_retained_proposal_reference_cannot_poison_final_receipt(self):
        session = ProbeSession(make_case(0))
        supplied = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
        first = session.submit(supplied)
        self.assertEqual(first["verdict"], "VERIFIED_WIN")
        self.assertTrue(verify_receipt_replay(first))
        supplied["actions"][0] = "FORGED"
        readback = session.submit(supplied)
        self.assertEqual(readback["verdict"], "VERIFIED_WIN")
        self.assertTrue(verify_receipt_replay(readback),
                        "caller-mutated input changed already accepted private evidence")
        frozen_proposal = readback.get("proposal")
        if frozen_proposal is None:
            frozen_proposal = readback.get("adjudication", {}).get("proposal")
        self.assertEqual(frozen_proposal["actions"][0],
                         MEMORIZED_SEED0_PATH[0])


if __name__ == "__main__":
    unittest.main()
