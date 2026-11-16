"""Independent Wave12 exhaustive-parity hostile mutation controls.

Run against the exact frozen Switchback source, not this test-only branch's
older copy of labyrinth. The source checkout must be the process cwd.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from labyrinth import switchback_circuit as dut
from labyrinth import switchback_oracle as oracle
from labyrinth.switchback_controls import MEMORIZED_SEED0_PATH


class SwitchbackWave12Destroyer(unittest.TestCase):
    def test_bounded_clock_pulse_flags_all_action_states(self):
        """Compare every legal bounded state tuple, including unreachable states.

        This targets tick=0, tick=MAX_MOVES, rule-change boundary, pulse
        exhaustion, and terminal flags without shared State.__eq__ semantics.
        """
        fields = ("register", "ticks", "pulses", "injected_after", "committed", "won")
        for seed in (0, 1, 9):
            case = dut.make_case(seed)
            self.assertTrue(oracle.independent_transition_parity(case))
            for tick in range(dut.MAX_MOVES + 1):
                for register in range(8):
                    for pulses in range(case.max_pulses + 1):
                        for injected in (False, True):
                            for committed, won in ((False, False), (True, False),
                                                    (True, True)):
                                state = dut.State(register, tick, pulses, injected,
                                                  committed, won)
                                for action in dut.CONTROLS:
                                    with self.subTest(seed=seed, tick=tick,
                                                      register=register, pulses=pulses,
                                                      injected=injected, committed=committed,
                                                      won=won, action=action):
                                        actual = dut.step(case, state, action)
                                        expected = oracle.oracle_step(case, state, action)
                                        self.assertEqual(tuple(getattr(actual, f) for f in fields),
                                                         tuple(getattr(expected, f) for f in fields))

    def test_seed_zero_first_tick_oracle_mutation_is_rejected(self):
        case = dut.make_case(0)
        original = oracle.oracle_step

        def poisoned(c, state, action):
            result = original(c, state, action)
            if c.seed == 0 and state.ticks == 0 and action == "1":
                return dut.State((result.register + 1) % 8, result.ticks,
                                 result.pulses, result.injected_after,
                                 result.committed, result.won)
            return result

        with patch.object(oracle, "oracle_step", side_effect=poisoned):
            self.assertFalse(oracle.independent_transition_parity(case))
            receipt = oracle.adjudicate(case, {"kind": "PLAN", "actions":
                                                    list(MEMORIZED_SEED0_PATH)})
            self.assertEqual(receipt["verdict"], "UNKNOWN")
            self.assertFalse(receipt["verified_success"])

    def test_seed_zero_first_tick_dut_mutation_is_rejected(self):
        case = dut.make_case(0)
        original = dut.step

        def poisoned(c, state, action):
            result = original(c, state, action)
            if c.seed == 0 and state.ticks == 0 and action == "1":
                return dut.State((result.register + 1) % 8, result.ticks,
                                 result.pulses, result.injected_after,
                                 result.committed, result.won)
            return result

        with patch.object(dut, "step", side_effect=poisoned):
            self.assertFalse(oracle.independent_transition_parity(case))
            session = dut.ProbeSession(case)
            view = session.probe("1")
            self.assertTrue(view["finished"])
            self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"})["verdict"],
                             "EVALUATOR_DISAGREEMENT")

    def test_unreachable_kernel_truth_table_mutation_is_rejected(self):
        """A hidden altered truth-table entry is not excused as unreachable."""
        case = dut.make_case(0)
        original = dut.kernel

        def poisoned(register, bit, rule):
            result = original(register, bit, rule)
            if register == 7 and bit == 1 and rule == case.after:
                return (result + 1) % 8
            return result

        with patch.object(dut, "kernel", side_effect=poisoned):
            self.assertFalse(oracle.independent_transition_parity(case))
            result = oracle.adjudicate(case, {"kind": "CLAIM_UNSAT"})
            self.assertEqual(result["verdict"], "UNKNOWN")
            self.assertEqual(result["review"]["reason"],
                             "INDEPENDENT_TRANSITION_DISAGREEMENT")

    def test_single_verify_plan_mutation_cannot_self_certify_false_win(self):
        """A false plan verifier must not also be its own receipt witness."""
        case = dut.make_case(0)
        losing_plan = {"kind": "PLAN", "actions": ["COMMIT"]}
        self.assertEqual(oracle.verify_plan(case, ["COMMIT"])["verdict"],
                         "VERIFIED_FAILURE")
        forged_truth = {"verdict": "VERIFIED_WIN", "reason": "FAULT_INJECTED",
                        "steps_checked": 1, "counterexample": None}
        with patch.object(oracle, "verify_plan", return_value=forged_truth):
            receipt = oracle.adjudicate(case, losing_plan)
            accepted = (receipt["verified_success"], oracle.verify_receipt_replay(receipt))
            self.assertEqual(accepted, (False, False))

    def test_single_probe_trace_verifier_mutation_cannot_self_certify_receipt(self):
        """Forged public trace must not inherit authority from one patched API."""
        case = dut.make_case(0)
        forged_trace = [{"action": "FORGED", "valid": True, "applied": True,
                         "before": "000", "after": "111"}]
        self.assertFalse(oracle.verify_probe_trace(case, forged_trace))
        with patch.object(oracle, "verify_probe_trace", return_value=True):
            try:
                receipt = oracle.adjudicate_session(case, forged_trace,
                           {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
            except ValueError:
                # Best defense: reject the doctored trace before issuing a receipt.
                return
            self.assertFalse(oracle.verify_receipt_replay(receipt))

    def test_single_shared_state_constructor_mutation_cannot_invent_win(self):
        """Both DUT and oracle use one State constructor: a common-mode fault."""
        case = dut.make_case(0)
        losing_plan = ["0", "0", "WAIT", "1", "COMMIT"]
        self.assertEqual(oracle.verify_plan(case, losing_plan)["verdict"],
                         "VERIFIED_FAILURE")
        original = dut.State.__init__

        def shifted(self, register, ticks=0, pulses=0, injected_after=False,
                    committed=False, won=False):
            return original(self, (register + 1) % 8, ticks, pulses,
                            injected_after, committed, won)

        with patch.object(dut.State, "__init__", new=shifted):
            parity = oracle.independent_transition_parity(case)
            receipt = oracle.adjudicate(case, {"kind": "PLAN", "actions": losing_plan})
            replay = oracle.verify_receipt_replay(receipt)
            self.assertEqual((parity, receipt["verified_success"], replay),
                             (False, False, False))

    def test_seed_zero_receipt_mutation_replay_fails_closed(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        session.probe("1")
        receipt = session.submit({"kind": "PLAN", "actions":
                                                   list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertTrue(oracle.verify_receipt_replay(receipt))
        original = oracle.oracle_step

        def poisoned(c, state, action):
            if c.seed == 0 and state.ticks == 0 and action == "1":
                return state
            return original(c, state, action)

        with patch.object(oracle, "oracle_step", side_effect=poisoned):
            self.assertFalse(oracle.verify_receipt_replay(receipt))

    def test_equality_spoof_cannot_certify_divergent_register(self):
        """Do not trust DUT-owned State.__eq__ as independent field parity."""
        case = dut.make_case(0)
        original = oracle.oracle_step

        def poisoned(c, state, action):
            result = original(c, state, action)
            if c.seed == 0 and state.ticks == 0 and action == "1":
                return dut.State((result.register + 1) % 8, result.ticks,
                                 result.pulses, result.injected_after,
                                 result.committed, result.won)
            return result

        def spoof(left, right):
            if type(right) is not dut.State:
                return NotImplemented
            return (left.ticks, left.pulses, left.injected_after,
                    left.committed, left.won) == (right.ticks, right.pulses,
                                                  right.injected_after,
                                                  right.committed, right.won)

        with (patch.object(dut.State, "__eq__", new=spoof),
              patch.object(oracle, "oracle_step", side_effect=poisoned)):
            self.assertFalse(oracle.independent_transition_parity(case))


if __name__ == "__main__":
    unittest.main()
