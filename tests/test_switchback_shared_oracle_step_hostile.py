"""RED: one shared oracle_step fault must not self-certify a false Switchback win.

This source-only fault injection tests the claimed independence of receipt
verification. It is NOT a neural or separately held private oracle result.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from labyrinth import switchback_circuit as dut
from labyrinth import switchback_oracle as oracle


class SharedTransitionOracleFalsifier(unittest.TestCase):
    def test_zero_probe_one_commit_false_win_cannot_replay_as_verified(self):
        case = oracle.oracle_case(0)
        proposal = {"kind": "PLAN", "actions": ["COMMIT"]}
        before = oracle.adjudicate(case, proposal)
        self.assertEqual(before["verdict"], "VERIFIED_FAILURE")
        self.assertFalse(dut.step(case, dut.State(case.initial), "COMMIT").won)

        original_step = oracle.oracle_step

        def poisoned_step(current_case, state, action):
            if action == "COMMIT":
                # Precisely one compromised shared transition oracle, not a
                # forged receipt checksum or a change to source DUT.step.
                return oracle.State(state.register, state.ticks, state.pulses,
                                    state.injected_after, committed=True, won=True)
            return original_step(current_case, state, action)

        with patch.object(oracle, "oracle_step", side_effect=poisoned_step):
            candidate = oracle.adjudicate(case, proposal)
            purportedly_independent_replay = oracle.verify_receipt_replay(candidate)

        # Either a separately implemented trusted transition check catches
        # the bad verdict inside adjudicate, OR replay must reject the result.
        # Today BOTH generator and replay share oracle_step and this is RED.
        self.assertFalse(candidate["verdict"] == "VERIFIED_WIN"
                         and purportedly_independent_replay,
                         "shared oracle_step accepted a false zero-probe win")


if __name__ == "__main__":
    unittest.main()
