"""Second-wave red-team regressions for independent timegate adjudication."""

import unittest
from unittest import mock

from labyrinth import timegate_exam as exam
from labyrinth.timegate_oracle import (adjudicate_candidate_claim, reachable_closure,
                                       replay_snapshot, verify_attempts, verify_unsat)


class RoundTwoAdversarialTests(unittest.TestCase):
    def test_certificate_must_not_accept_deceptive_equality(self):
        class ForgedRules:
            def __eq__(self, other):
                return True

        case = exam.make_case(2, sealed=True)
        response = adjudicate_candidate_claim(case, {
            "kind": "UNSAT_CERTIFICATE", "declared_rules": ForgedRules(),
            "states": reachable_closure(case),
        })
        self.assertEqual(response["verdict"], "UNKNOWN")
        self.assertFalse(response.get("certificate_verified", False))

    def test_witness_cannot_spoof_attempt_charge_via_len(self):
        class LyingList(list):
            def __len__(self):
                return 0

        case = exam.make_case(0)
        witness = exam.search(case).witness
        response = adjudicate_candidate_claim(case, {
            "kind": "WITNESS", "actions": LyingList(witness),
        })
        self.assertFalse(response.get("verified_success", False))
        self.assertNotEqual(response["verdict"], "VERIFIED_WIN")

    def test_overbudget_replay_cannot_be_hidden_by_deceptive_length(self):
        class LyingList(list):
            def __len__(self):
                return 1

        case = exam.make_case(0)
        too_long = ["wait"] * 10 + list(exam.search(case).witness)
        self.assertGreater(len(too_long), exam.MAX_ACTIONS)
        self.assertEqual(verify_attempts(case, LyingList(too_long)), "UNKNOWN")
        self.assertIsNone(replay_snapshot(case, LyingList(too_long)))

    def test_invalid_certificate_state_cannot_disappear_during_deduplication(self):
        case = exam.make_case(7, sealed=True)
        closure = reachable_closure(case)
        alias_target = next(state for state in closure if state[1] == 1)
        forged_bool_phase = (alias_target[0], True, *alias_target[2:])
        self.assertNotEqual(type(forged_bool_phase[1]), int)
        result = verify_unsat(case, list(closure) + [forged_bool_phase])
        self.assertFalse(result.valid)

    def test_malformed_certificate_cannot_execute_candidate_hash(self):
        class HostileEntry:
            def __hash__(self):
                raise RuntimeError("candidate-controlled hash invoked")

        case = exam.make_case(7, sealed=True)
        result = verify_unsat(case, [HostileEntry()])
        self.assertFalse(result.valid)
        receipt = adjudicate_candidate_claim(case, {
            "kind": "UNSAT_CERTIFICATE",
            "states": [HostileEntry()],
            "declared_rules": {"period": case.period,
                               "charge_phase": case.charge_phase,
                               "pass_phase": case.pass_phase,
                               "initial_phase": case.initial_phase,
                               "lifetime": case.lifetime, "sealed": case.sealed},
        })
        self.assertEqual(receipt["verdict"], "UNKNOWN")

    def test_local_solver_cannot_certify_false_impossibility(self):
        case = exam.make_case(1)
        self.assertEqual(exam.search(case).verdict, "VERIFIED_WIN")

        def suppress_all_actions(_case, state, _action):
            return state

        with mock.patch.object(exam, "transition", side_effect=suppress_all_actions):
            result = exam.search(case)
        self.assertEqual(result.verdict, "EVALUATOR_DISAGREEMENT")

    def test_truthy_but_false_observation_is_terminal_disagreement(self):
        case = exam.make_case(0)
        session = exam.CandidateSession(case)
        with mock.patch.object(exam, "observe", return_value="Location: outside. Escape complete."):
            result = session.view()
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])


if __name__ == "__main__":
    unittest.main()
