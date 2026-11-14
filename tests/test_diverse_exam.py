"""Property and metamorphic controls for public multi-family toy dilemmas."""

import json
import unittest
from unittest.mock import patch

import labyrinth.diverse_exam as exam_module
from labyrinth.diverse_exam import (ACTIONS, CandidateSession, FAMILIES, MAX_ACTIONS, State,
                          family_fingerprint, make_exam, public_view, search,
                          step, world_fingerprint)
from labyrinth.diverse_exam_verifier import verify, verify_attempts, verify_unsat_certificate


class DiverseExamTests(unittest.TestCase):
    def test_genuinely_different_semantic_families(self):
        cases = [make_exam(seed) for seed in range(3)]
        self.assertEqual({case.family for case in cases}, set(FAMILIES))
        self.assertEqual(len({family_fingerprint(case) for case in cases}), 3)
        self.assertEqual(len({search(case).witness for case in cases}), 3)
        hard_coded = search(cases[0]).witness
        self.assertEqual(sum(verify(family=c.family, rotation=c.rotation,
                                    sealed=False, actions=hard_coded).outcome == "VERIFIED_WIN"
                             for c in cases), 1)

    def test_seed_rotation_changes_labelled_traces(self):
        for offset in range(3):
            cases = [make_exam(offset + 3 * turn) for turn in range(4)]
            self.assertEqual(len({world_fingerprint(case) for case in cases}), 4)
            self.assertEqual(len({search(case).witness for case in cases}), 4)
            self.assertEqual(len({family_fingerprint(case) for case in cases}), 1)

    def test_near_twins_have_real_witness_or_closed_unsat(self):
        for seed in range(48):
            open_case, closed_case = make_exam(seed), make_exam(seed, sealed=True)
            found = search(open_case)
            self.assertEqual(found.status, "VERIFIED_WIN", (seed, found))
            self.assertLessEqual(len(found.witness), MAX_ACTIONS)
            self.assertEqual(verify(family=open_case.family, rotation=open_case.rotation,
                                    sealed=False, actions=found.witness).outcome, "VERIFIED_WIN")
            closed = search(closed_case)
            self.assertEqual(closed.status, "UNSAT_DECLARED_FINITE_WORLD", (seed, closed))
            self.assertNotIn("freedom", {state.room for state in closed.reachable})
            self.assertTrue(all(step(closed_case, state, action) in closed.reachable
                                for state in closed.reachable for action in ACTIONS))
            self.assertNotEqual(verify(family=closed_case.family, rotation=closed_case.rotation,
                                       sealed=True, actions=found.witness).outcome, "VERIFIED_WIN")
            self.assertNotEqual(world_fingerprint(open_case), world_fingerprint(closed_case))

    def test_bounded_search_abstains_not_false_impossibility(self):
        for seed in range(15):
            for sealed in (False, True):
                self.assertEqual(search(make_exam(seed, sealed=sealed), max_expansions=0).status,
                                 "UNSOLVED")

    def test_paraphrase_unchanged_world_and_no_evaluator_metadata(self):
        for seed in range(12):
            case = make_exam(seed, split="test", variant=3, sealed=bool(seed % 2))
            body = public_view(case)
            self.assertEqual(set(body), {"prompt", "legal_actions", "remaining_actions", "observation"})
            for banned in ("case_id", "sealed", "family", "seed", "gold", "witness", "fingerprint"):
                self.assertNotIn(banned, json.dumps(body).lower())
            self.assertEqual(world_fingerprint(case), world_fingerprint(make_exam(seed, variant=99,
                                                                                   sealed=case.sealed)))
            surface_forms = {public_view(make_exam(seed, variant=i,
                                                    sealed=case.sealed))["prompt"]
                             for i in range(32)}
            self.assertGreater(len(surface_forms), 1)

    def test_candidate_session_observations_and_action_budget(self):
        for seed in range(12):
            case = make_exam(seed)
            session = CandidateSession(case)
            self.assertEqual(session.view()["remaining_actions"], MAX_ACTIONS)
            for action in search(case).witness:
                view = session.act(action)
                self.assertEqual(set(view), {"observation", "remaining_actions", "finished",
                                             "action_status", "verdict"})
                self.assertFalse(any(k in view for k in ("case_id", "sealed", "family", "seed")))
            self.assertTrue(view["finished"])
            self.assertEqual(view["verdict"], "VERIFIED_WIN")
            self.assertEqual(verify(family=case.family, rotation=case.rotation,
                                    sealed=case.sealed, actions=session.actions()).outcome,
                             "VERIFIED_WIN")
            with self.assertRaises(ValueError):
                session.act("north")
            probe = CandidateSession(case)
            self.assertEqual(probe.act("teleport")["action_status"], "INVALID_ACTION")
            self.assertEqual(probe.view()["remaining_actions"], MAX_ACTIONS - 1)

    def test_invalid_action_spam_exhausts_budget_and_cannot_be_winning_receipt(self):
        case = make_exam(1)
        session = CandidateSession(case)
        for attempt in range(MAX_ACTIONS):
            result = session.act({"malformed": attempt} if attempt % 2 else "teleport")
            self.assertEqual(result["action_status"], "INVALID_ACTION")
            self.assertEqual(result["remaining_actions"], MAX_ACTIONS - attempt - 1)
        self.assertEqual(result["verdict"], "EXHAUSTED")
        self.assertTrue(result["finished"])
        with self.assertRaises(ValueError):
            session.act("north")
        self.assertEqual(len(session.attempts()), MAX_ACTIONS)
        self.assertEqual(session.actions(), ())
        self.assertEqual(verify_attempts(family=case.family, rotation=case.rotation,
                                         sealed=False, attempts=session.attempts()).outcome,
                         "NOT_WON")

        # Invalid probes are costed no-op attempts, not permanent failure:
        # a later legal escape can win if it still fits the attempt budget.
        case = make_exam(0)
        session = CandidateSession(case)
        session.act("teleport")
        for move in search(case).witness:
            result = session.act(move)
        self.assertEqual(result["verdict"], "VERIFIED_WIN")
        self.assertEqual(len(session.attempts()), len(session.actions()) + 1)
        self.assertEqual(verify(family=case.family, rotation=case.rotation,
                                sealed=False, actions=session.actions()).outcome, "VERIFIED_WIN")
        self.assertEqual(verify_attempts(family=case.family, rotation=case.rotation,
                                         sealed=False, attempts=session.attempts()).outcome,
                         "VERIFIED_WIN")

    def test_independent_unsat_certificate_closed_and_near_twin_falsifies(self):
        for seed in range(48):
            sealed = make_exam(seed, sealed=True)
            certified = {(state.room, state.bits, state.enabled)
                         for state in search(sealed).reachable}
            good = verify_unsat_certificate(family=sealed.family, rotation=sealed.rotation,
                                            sealed=True, claimed_states=certified)
            self.assertTrue(good.valid, (seed, good))
            self.assertGreater(good.reachable_count, 0)
            open_case = make_exam(seed)
            wrong = verify_unsat_certificate(family=open_case.family, rotation=open_case.rotation,
                                             sealed=False, claimed_states=certified)
            self.assertFalse(wrong.valid, (seed, wrong))
            self.assertIn("goal reachable", wrong.reason)

            missing_start = certified - {("foyer", 0, False)}
            self.assertFalse(verify_unsat_certificate(family=sealed.family,
                             rotation=sealed.rotation, sealed=True,
                             claimed_states=missing_start).valid)
            self.assertFalse(verify_unsat_certificate(family=sealed.family,
                             rotation=sealed.rotation, sealed=True,
                             claimed_states=certified | {("freedom", 0, False)}).valid)

    def test_replay_matches_transition_on_all_reachable_shortest_prefixes(self):
        for seed in range(12):
            for sealed in (False, True):
                case = make_exam(seed, sealed=sealed)
                seen = {State()}
                frontier = [(State(), ())]
                while frontier:
                    state, prefix = frontier.pop(0)
                    if state.room == "freedom":
                        continue
                    for action in ACTIONS:
                        updated = step(case, state, action)
                        trace = prefix + (action,)
                        if len(trace) > MAX_ACTIONS:
                            continue  # Replay contract is a bounded attempt, not arbitrary reachability.
                        replay = verify(family=case.family, rotation=case.rotation,
                                        sealed=sealed, actions=trace)
                        self.assertEqual(replay.final_room, updated.room, (seed, sealed, trace))
                        self.assertEqual(replay.outcome == "VERIFIED_WIN", updated.room == "freedom")
                        if updated not in seen:
                            seen.add(updated)
                            frontier.append((updated, trace))

    def test_malformed_or_forged_claim_is_not_witness(self):
        for broken in ("WIN", ["teleport"], ["north"] * 13,
                       ["leave"], ["north", "leave"], None):
            outcome = verify(family="relic", rotation=0, sealed=False, actions=broken).outcome
            self.assertNotEqual(outcome, "VERIFIED_WIN")
        for bad in ("0", -1, True, 2**31):
            with self.assertRaises(ValueError):
                make_exam(bad)

    def test_candidate_session_rejects_suppressed_valid_win(self):
        """An incorrect engine must not silently erase a verifier-proven escape."""
        case = make_exam(0)
        witness = search(case).witness
        actual_step = step

        def suppress_winning_leave(exam, state, action):
            if action == "leave" and state.room == "gate":
                return state
            return actual_step(exam, state, action)

        session = CandidateSession(case)
        with patch.object(exam_module, "step", side_effect=suppress_winning_leave):
            for action in witness[:-1]:
                session.act(action)
            with self.assertRaisesRegex(RuntimeError, "verifier disagree"):
                session.act(witness[-1])
        self.assertTrue(session.view()["finished"])
        self.assertEqual(session.view()["verdict"], "REPLAY_DIVERGENCE")
        with self.assertRaises(ValueError):
            session.act("leave")

    def test_candidate_session_rejects_early_transition_divergence(self):
        """Mismatch must abort at the first affected action, even before a win."""
        case = make_exam(1)
        first_action = search(case).witness[0]
        session = CandidateSession(case)
        with patch.object(exam_module, "step", return_value=State()):
            with self.assertRaisesRegex(RuntimeError, "verifier disagree"):
                session.act(first_action)
        self.assertTrue(session.view()["finished"])
        self.assertEqual(session.view()["verdict"], "REPLAY_DIVERGENCE")

    def test_candidate_session_rejects_hidden_inventory_divergence(self):
        """An unchanged room does not excuse a broken inventory transition."""
        case = make_exam(0)  # relic; taking the item changes bits, not room
        witness = search(case).witness
        actual_step = step

        def suppress_relic_pickup(exam, state, action):
            if action == "take" and state.room == "archive":
                return state
            return actual_step(exam, state, action)

        session = CandidateSession(case)
        with patch.object(exam_module, "step", side_effect=suppress_relic_pickup):
            for action in witness:
                if action == "take":
                    with self.assertRaisesRegex(RuntimeError, "verifier disagree"):
                        session.act(action)
                    break
                session.act(action)
            else:
                self.fail("relic witness did not attempt a pickup")
        self.assertEqual(session.view()["verdict"], "REPLAY_DIVERGENCE")

    def test_candidate_session_seals_engine_exception_and_preserves_receipt(self):
        """An exception during a winning command cannot leave the session live."""
        case = make_exam(0)
        witness = search(case).witness
        session = CandidateSession(case)
        for action in witness[:-1]:
            session.act(action)

        with patch.object(exam_module, "step", side_effect=RuntimeError("injected step fault")):
            with self.assertRaisesRegex(RuntimeError, "injected step fault") as caught:
                session.act(witness[-1])

        self.assertEqual(session.attempts(), witness)
        self.assertTrue(session.view()["finished"])
        self.assertEqual(session.view()["verdict"], "EVALUATOR_ERROR")
        self.assertEqual(session.view()["action_status"], "EVALUATOR_ERROR")
        self.assertIs(session._transition_error, caught.exception)
        self.assertEqual(verify_attempts(family=case.family, rotation=case.rotation,
                                         sealed=case.sealed, attempts=session.attempts()).outcome,
                         "VERIFIED_WIN")
        with self.assertRaises(ValueError):
            session.act("leave")

    def test_candidate_session_seals_malformed_transition_results(self):
        """A bad return must never replace the session's last valid state."""
        case = make_exam(0)
        for invalid_result in (None, State("unknown_room"),
                               State("foyer", True, False),
                               State("foyer", 0, "yes")):
            with self.subTest(return_value=invalid_result):
                session = CandidateSession(case)
                previous_observation = session.view()["observation"]
                with patch.object(exam_module, "step", return_value=invalid_result):
                    with self.assertRaisesRegex(TypeError, "invalid State") as caught:
                        session.act("north")
                self.assertEqual(session.attempts(), ("north",))
                self.assertEqual(session._state, State())
                self.assertEqual(previous_observation, public_view(case)["observation"])
                self.assertIsNone(session.view()["observation"])
                self.assertTrue(session.view()["finished"])
                self.assertEqual(session.view()["verdict"], "EVALUATOR_ERROR")
                self.assertIs(session._transition_error, caught.exception)
                with self.assertRaises(ValueError):
                    session.act("east")

    def test_candidate_session_seals_observation_exception(self):
        """A successful transition without deliverable feedback must halt play."""
        case = make_exam(0)
        session = CandidateSession(case)
        session.act("north")
        with patch.object(exam_module, "observe", side_effect=RuntimeError("injected observation fault")):
            with self.assertRaisesRegex(RuntimeError, "injected observation fault") as caught:
                session.act("east")
        self.assertEqual(session.attempts(), ("north", "east"))
        self.assertEqual(session._state.room, "archive")
        self.assertTrue(session.view()["finished"])
        self.assertEqual(session.view()["verdict"], "EVALUATOR_ERROR")
        self.assertEqual(session.view()["action_status"], "EVALUATOR_ERROR")
        self.assertIsNone(session.view()["observation"])
        self.assertIs(session._observation_error, caught.exception)
        with self.assertRaises(ValueError):
            session.act("west")

    def test_candidate_session_rejects_malformed_observations(self):
        """A missing or non-text observation must not allow the attempt to continue."""
        case = make_exam(0)
        for invalid in (None, "", "   ", True):
            with self.subTest(observation=invalid):
                session = CandidateSession(case)
                with patch.object(exam_module, "observe", return_value=invalid):
                    with self.assertRaisesRegex(TypeError, "invalid observation"):
                        session.act("north")
                self.assertEqual(session.attempts(), ("north",))
                self.assertTrue(session.view()["finished"])
                self.assertEqual(session.view()["verdict"], "EVALUATOR_ERROR")
                self.assertIsNone(session.view()["observation"])
                with self.assertRaises(ValueError):
                    session.act("east")


if __name__ == "__main__":
    unittest.main()
