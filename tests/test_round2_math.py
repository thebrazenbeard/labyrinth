"""Independent algebraic falsifiers for the timed-relay examination.

These oracles are derived from the required physical action sequence and
clock/energy arithmetic rather than calling the implementation's search.
They deliberately include valid parameter configurations outside the narrow
make_case generator, so the action-budget boundary is actually exercised.
"""

from dataclasses import replace
import random
import unittest

from labyrinth.timegate_exam import (MAX_ACTIONS, CandidateSession, State,
                                     make_case, search, transition)
from labyrinth.timegate_oracle import reachable_closure, verify_attempts, verify_unsat


def analytic_outcome(case):
    """Closed-form solution, including the shortest legal escape length.

    Three mandatory preparatory actions reach the tower carrying the cell:
    forward, take, forward. Wait until the charging phase, arm, move to the
    gate, wait for the passing phase, then escape. No alternate route or
    recharge exists in the declared world.
    """
    pre_arm_wait = (case.charge_phase - case.initial_phase - 3) % case.period
    post_arm_wait = (case.pass_phase - case.charge_phase - 2) % case.period
    minimum = 6 + pre_arm_wait + post_arm_wait
    if case.sealed or case.lifetime < 2 + post_arm_wait:
        return "UNSAT", None
    if minimum > MAX_ACTIONS:
        return "OVER_BUDGET", minimum
    return "WIN", minimum


class RoundTwoMathematics(unittest.TestCase):
    def test_2000_generated_cases_match_closed_form_minimal_lengths(self):
        # The generator only produces paths of length 6-14, even though the
        # candidate receives 16 attempts. Count that limitation explicitly.
        lengths = set()
        for seed in range(2000):
            case = make_case(seed)
            result = search(case)
            prediction, minimum = analytic_outcome(case)
            self.assertEqual(prediction, "WIN", seed)
            self.assertEqual(result.verdict, "VERIFIED_WIN", seed)
            self.assertEqual(len(result.witness), minimum, seed)
            self.assertEqual(verify_attempts(case, result.witness), "VERIFIED_WIN")
            lengths.add(minimum)
        self.assertEqual(lengths, set(range(6, 15)))

    def test_synthetic_phases_check_win_unsat_and_budget_separately(self):
        generator = random.Random(20261107)
        observed = set()
        for _ in range(250):
            period = generator.randrange(2, 13)
            case = replace(make_case(0), period=period,
                           charge_phase=generator.randrange(period),
                           pass_phase=generator.randrange(period),
                           initial_phase=generator.randrange(period),
                           lifetime=generator.randrange(1, period + 4),
                           sealed=bool(generator.randrange(2)))
            expected, minimum = analytic_outcome(case)
            observed.add(expected)
            result = search(case)
            closure = reachable_closure(case)
            review = verify_unsat(case, closure)
            if expected == "UNSAT":
                self.assertEqual(result.verdict, "PROVEN_UNSAT_UNDER_DECLARED_RULES")
                self.assertTrue(review.valid)
            elif expected == "OVER_BUDGET":
                self.assertEqual(result.verdict, "NO_KNOWN_WIN")
                self.assertIn("attempt budget", result.detail)
                self.assertFalse(review.valid)
            else:
                self.assertEqual(result.verdict, "VERIFIED_WIN")
                self.assertEqual(len(result.witness), minimum)
                self.assertEqual(verify_attempts(case, result.witness), "VERIFIED_WIN")
                self.assertFalse(review.valid)
        self.assertEqual(observed, {"WIN", "UNSAT", "OVER_BUDGET"})

    def test_constructed_seventeen_action_witness_is_not_called_unsat(self):
        case = replace(make_case(0), period=7, charge_phase=1,
                       pass_phase=2, initial_phase=0, lifetime=9)
        self.assertEqual(analytic_outcome(case), ("OVER_BUDGET", 17))
        complete = search(case)
        self.assertEqual(complete.verdict, "NO_KNOWN_WIN")
        self.assertIn("attempt budget", complete.detail)
        self.assertFalse(verify_unsat(case, reachable_closure(case)).valid)
        self.assertEqual(search(case, max_expansions=0).verdict, "NO_KNOWN_WIN")
        self.assertEqual(search(case, max_expansions=0).detail, "search incomplete")

    def test_unreachable_fake_states_and_incomplete_proofs_rejected(self):
        case = make_case(7, sealed=True)
        exact = reachable_closure(case)
        self.assertTrue(verify_unsat(case, exact).valid)
        false_certificates = (
            exact - {next(iter(exact))},
            exact | {(4, 0, 0, 0, 0)},
            exact | {(-1, 0, 0, 0, 0)},
            exact | {("gate", 0, 0, 0, 0)},
        )
        for forged in false_certificates:
            self.assertFalse(verify_unsat(case, forged).valid)
        # A tuple with boolean phase is malformed, but True == 1 in Python.
        # Deduplicating to a set before schema validation launders the alias.
        original = next(s for s in exact if s[1] == 1)
        alias = (original[0], True, *original[2:])
        self.assertFalse(verify_unsat(case, list(exact) + [alias]).valid)

    def test_malformed_certificate_cannot_execute_hostile_hash_method(self):
        class HostileEntry:
            def __hash__(self):
                raise RuntimeError("should never hash an unvalidated state")

        case = make_case(7, sealed=True)
        review = verify_unsat(case, [HostileEntry()])
        self.assertFalse(review.valid)

    def test_noop_cycles_do_not_grant_free_attempts(self):
        case = make_case(0)  # period=2
        initial = State(phase=case.initial_phase)
        self.assertEqual(transition(case, transition(case, initial, "wait"), "wait"),
                         initial)
        session = CandidateSession(case)
        session.act("wait")
        after_cycle = session.act("wait")
        self.assertIn("Location: entry", after_cycle["observation"])
        self.assertEqual(after_cycle["remaining_actions"], MAX_ACTIONS - 2)
        self.assertFalse(after_cycle["finished"])


if __name__ == "__main__":
    unittest.main()
