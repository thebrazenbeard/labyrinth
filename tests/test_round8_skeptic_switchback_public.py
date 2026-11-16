"""Independent hostile tests: public-state collisions and finite reachability.

These results apply to source-only Switchback, not an independently custodied
candidate experiment or neural model. Initially tested against author WIP.
"""

from itertools import product
import unittest

from labyrinth.switchback_circuit import (
    CONTROLS, MAX_MOVES, State, make_case, public_view, step,
)
from labyrinth.switchback_oracle import search, verify_plan


def dut_state_reachable_win(case):
    """Independent DUT dynamic exploration, not oracle.search/step/verify."""
    active = {State(case.initial)}
    for _ in range(MAX_MOVES):
        advanced = set()
        for state in active:
            if step(case, state, "COMMIT").won:
                return True
            for action in ("0", "1", "WAIT"):
                if action != "WAIT" and state.pulses >= case.max_pulses:
                    continue
                next_state = step(case, state, action)
                if next_state != state and not next_state.committed:
                    advanced.add(next_state)
        active = advanced
    return False


class SwitchbackIndependentHostile(unittest.TestCase):
    def test_identical_public_input_has_opposite_sat_status(self):
        win, unsat = make_case(13), make_case(14)
        self.assertEqual(public_view(win), public_view(unsat))
        self.assertEqual(search(win)["verdict"], "VERIFIED_WIN")
        self.assertEqual(search(unsat)["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertTrue(dut_state_reachable_win(win))
        self.assertFalse(dut_state_reachable_win(unsat))

    def test_100_seed_oracle_reachability_matches_independent_dut(self):
        for seed in range(100):
            case = make_case(seed)
            oracle_wins = search(case)["verdict"] == "VERIFIED_WIN"
            self.assertEqual(dut_state_reachable_win(case), oracle_wins,
                             f"seed {seed} oracle/DUT divergence")

    def test_unsat_14_exhaustively_rejects_every_legal_input_prefix(self):
        case = make_case(14)
        tried = 0
        for length in range(MAX_MOVES):
            for prefix in product(("0", "1", "WAIT"), repeat=length):
                state = State(case.initial)
                legal = True
                for action in prefix:
                    if action != "WAIT" and state.pulses >= case.max_pulses:
                        legal = False
                        break
                    state = step(case, state, action)
                if legal:
                    tried += 1
                    self.assertFalse(step(case, state, "COMMIT").won,
                                     f"counterexample prefix {prefix!r}")
        self.assertGreater(tried, 1000)

    def test_oracle_seed13_witness_works_through_separate_dut(self):
        case = make_case(13)
        witness = search(case)["witness"]
        self.assertEqual(verify_plan(case, witness)["verdict"], "VERIFIED_WIN")
        state = State(case.initial)
        for action in witness:
            state = step(case, state, action)
        self.assertTrue(state.won)


if __name__ == "__main__":
    unittest.main()
