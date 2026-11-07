"""Regression probes for second-wave verifier fault containment."""

import unittest
from unittest import mock

from labyrinth.diverse_exam import CandidateSession, make_exam, search
from labyrinth.diverse_exam_verifier import (verify, verify_attempts,
                                            verify_unsat_certificate)


class DiverseVerifierHardeningTests(unittest.TestCase):
    def test_independent_verifier_crash_is_sticky_terminal(self):
        session = CandidateSession(make_exam(0))
        with mock.patch("labyrinth.diverse_exam_verifier._advance",
                        side_effect=RuntimeError("injected verifier crash")):
            with self.assertRaisesRegex(RuntimeError, "independent verifier failed"):
                session.act("north")
        state = session.view()
        self.assertTrue(state["finished"])
        self.assertEqual(state["verdict"], "EVALUATOR_ERROR")
        self.assertEqual(state["action_status"], "EVALUATOR_ERROR")
        self.assertEqual(session.attempts(), ("north",))
        with self.assertRaisesRegex(ValueError, "finished"):
            session.act("north")

    def test_verifier_crash_only_on_victory_fails_closed(self):
        case = make_exam(0)
        session = CandidateSession(case)
        path = search(case).witness
        for move in path[:-1]:
            session.act(move)
        with mock.patch("labyrinth.diverse_exam_verifier.verify",
                        side_effect=RuntimeError("verification unavailable")):
            result = session.act(path[-1])
        self.assertTrue(result["finished"])
        self.assertEqual(result["verdict"], "EVALUATOR_ERROR")
        with self.assertRaisesRegex(ValueError, "finished"):
            session.act("north")

    def test_adversarial_list_length_cannot_erase_attempts(self):
        class LyingList(list):
            def __len__(self):
                return 1

        case = make_exam(0)
        actual = ["take"] * 6 + list(search(case).witness)
        self.assertGreater(len(actual), 12)
        options = {"family": case.family, "rotation": case.rotation,
                   "sealed": case.sealed}
        self.assertEqual(verify(**options, actions=actual).outcome, "INVALID_TRACE")
        self.assertEqual(verify(**options, actions=LyingList(actual)).outcome, "INVALID_TRACE")
        self.assertEqual(verify_attempts(**options, attempts=LyingList(actual)).outcome,
                         "INVALID_TRACE")

    def test_certificate_bool_alias_is_rejected_before_hash_dedup(self):
        case = make_exam(0, sealed=True)
        states = search(case).reachable
        certificate = [(s.room, s.bits, s.enabled) for s in states]
        aliased = next(row for row in certificate if row[1] == 1)
        malformed = (aliased[0], True, aliased[2])
        result = verify_unsat_certificate(family=case.family,
                                          rotation=case.rotation,
                                          sealed=case.sealed,
                                          claimed_states=certificate + [malformed])
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
