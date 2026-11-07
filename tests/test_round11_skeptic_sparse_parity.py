"""Independent RED: sampled parity misses an essential reachable clock tick.

Target Experimenter post-6339 WIP parity guard; exact seed0 switch_at=3.
The patched oracle_step fails only at tick1, omitted by sampled guard.
"""

import unittest
from unittest.mock import patch

from labyrinth.switchback_circuit import ProbeSession, make_case
from labyrinth import switchback_oracle as oracle


class IncompleteTransitionParity(unittest.TestCase):
    def test_mutated_reachable_tick1_cannot_pass_independent_parity(self):
        case = make_case(0)
        self.assertEqual(case.switch_at, 3)
        original_step = oracle.oracle_step

        def faulty_step(world, state, action):
            return state if state.ticks == 1 else original_step(world, state, action)

        with patch.object(oracle, "oracle_step", side_effect=faulty_step):
            self.assertFalse(oracle.independent_transition_parity(case),
                             "sampled parity omitted reachable tick1")

    def test_tick1_shared_kernel_fault_cannot_prove_unsat(self):
        case = make_case(0)
        original_step = oracle.oracle_step

        def faulty_step(world, state, action):
            return state if state.ticks == 1 else original_step(world, state, action)

        with patch.object(oracle, "oracle_step", side_effect=faulty_step):
            result = ProbeSession(case).submit({"kind": "CLAIM_UNSAT"})
            self.assertFalse(
                result.get("verdict") == "PROVEN_UNSAT_UNDER_DECLARED_RULES"
                and oracle.verify_receipt_replay(result),
                "missed transition mutation produced replay-certified false UNSAT",
            )


if __name__ == "__main__":
    unittest.main()
