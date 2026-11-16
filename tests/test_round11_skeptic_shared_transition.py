"""Independent RED: two finite solvers sharing one faulty oracle are NOT independent.

Targets Experimenter WIP after source 5aeb3f1. Seed zero has independent
verified winning plans; a broken oracle transition cannot prove it UNSAT.
"""

import unittest
from unittest.mock import patch

from labyrinth.switchback_circuit import ProbeSession, make_case
from labyrinth import switchback_oracle as oracle


class CorrelatedSearchFault(unittest.TestCase):
    def test_bfs_and_dfs_must_not_certify_unsat_from_same_mutated_transition(self):
        case = make_case(0)
        self.assertEqual(oracle.search(case)["verdict"], "VERIFIED_WIN")
        with patch.object(oracle, "oracle_step", side_effect=lambda case, state, action: state):
            try:
                receipt = ProbeSession(case).submit({"kind": "CLAIM_UNSAT"})
            except (TypeError, ValueError):
                return  # explicit fail-closed invalid transition is acceptable
            self.assertFalse(
                receipt.get("verdict") == "PROVEN_UNSAT_UNDER_DECLARED_RULES"
                and oracle.verify_receipt_replay(receipt),
                "mutated shared transition let BFS and DFS self-certify false UNSAT",
            )


if __name__ == "__main__":
    unittest.main()
