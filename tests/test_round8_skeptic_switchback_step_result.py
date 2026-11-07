"""Independent RED: malformed DUT transition returns must be charged and fatal.

Original source 0021d05516b90f216bdf36c9a10a6a1ac1e3d87d; also bypassed
the first attempted exception-containment patch (which caught throws only).
"""

import unittest
from unittest.mock import patch

from labyrinth import switchback_circuit as dut
from labyrinth.switchback_controls import MEMORIZED_SEED0_PATH


class InvalidTransitionReturn(unittest.TestCase):
    def test_none_return_cannot_grant_unrecorded_free_retry(self):
        session = dut.ProbeSession(dut.make_case(0))
        try:
            with patch.object(dut, "step", return_value=None):
                session.probe("0")
        except (AttributeError, TypeError):
            pass  # pre-fix exception must still close/charge the session
        try:
            finished = session.view()["finished"]
        except (AttributeError, TypeError):
            finished = False
        self.assertTrue(finished, "malformed transition corrupted session view")
        final = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertFalse(final["verified_success"],
                         "None transition returned an uncharged retry plus verified win")

    def test_malformed_state_field_cannot_escape_as_exception_and_retry(self):
        session = dut.ProbeSession(dut.make_case(0))
        malformed = dut.State(register=None, ticks=1, pulses=1)
        try:
            with patch.object(dut, "step", return_value=malformed):
                session.probe("0")
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            finished = session.view()["finished"]
        except (AttributeError, TypeError):
            finished = False
        self.assertTrue(finished, "malformed State fields corrupted session view")
        final = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertFalse(final["verified_success"])


if __name__ == "__main__":
    unittest.main()
