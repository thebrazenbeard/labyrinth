"""Independent shared-data-constructor mutation bypass.

Both evaluator engines construct the same State dataclass; patched constructor
may launder an impossible COMMIT as a win even with two transition engines.
"""

import unittest
from unittest.mock import patch

from labyrinth import switchback_oracle as oracle
from labyrinth.switchback_circuit import State, make_case


class SharedStateConstructorFault(unittest.TestCase):
    def test_single_constructor_won_bit_fault_cannot_prove_win(self):
        case = make_case(0)
        plan = ["COMMIT"]  # Always loses: no post-change input ever injected.
        self.assertEqual(oracle.verify_plan(case, plan)["verdict"], "VERIFIED_FAILURE")
        original_init = State.__init__

        def tamper_terminal_won(self, register, ticks=0, pulses=0,
                                injected_after=False, committed=False, won=False):
            return original_init(self, register, ticks, pulses,
                                 injected_after, committed,
                                 True if committed else won)

        with patch.object(State, "__init__", new=tamper_terminal_won):
            receipt = oracle.adjudicate(case, {"kind": "PLAN", "actions": plan})
            replayable = oracle.verify_receipt_replay(receipt)
            # One corrupted constructor must not make two unindependent
            # transition engines certify the same impossible terminal state.
            self.assertFalse(receipt.get("verified_success") is True and replayable,
                             "shared State.__init__ won-bit flip self-certified win")


if __name__ == "__main__":
    unittest.main()
