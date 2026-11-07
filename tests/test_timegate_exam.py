"""Adversarial controls for the public timed-relay Minotaur examination."""

import unittest
from unittest import mock

from labyrinth import timegate_exam as dut
from labyrinth.timegate_baseline import run_public_rule_parser
from labyrinth.timegate_oracle import (adjudicate_candidate_claim,
                                       adjudicate_external_solution, reachable_closure,
                                       test_hypothetical_manual_release, verify_attempts,
                                       verify_unsat)


def _project(state):
    return (dut.ROOMS.index(state.room), state.phase, int(state.carrying),
            int(state.available), state.energy)


def _engine_closure(case):
    pending = [dut.start(case)]
    reached = {pending[0]}
    while pending:
        state = pending.pop()
        if state.room == "outside":
            continue
        for action in dut.ACTIONS:
            result = dut.transition(case, state, action)
            if result not in reached:
                reached.add(result)
                pending.append(result)
    return frozenset(_project(s) for s in reached)


class TimegateHostileTests(unittest.TestCase):
    def test_all_open_120_have_independent_replayable_witness(self):
        for seed in range(120):
            with self.subTest(seed=seed):
                case = dut.make_case(seed)
                solution = dut.search(case)
                self.assertEqual(solution.verdict, "VERIFIED_WIN")
                self.assertLessEqual(len(solution.witness), dut.MAX_ACTIONS)
                self.assertEqual(verify_attempts(case, solution.witness), "VERIFIED_WIN")
                session = dut.CandidateSession(case)
                for action in solution.witness:
                    last = session.act(action)
                self.assertEqual(last["verdict"], "VERIFIED_WIN")

    def test_open_and_unsat_twins_differ_by_real_rules(self):
        for seed in range(120):
            for flags in ({"sealed": True}, {"weak_cell": True}):
                with self.subTest(seed=seed, flags=flags):
                    blocked = dut.make_case(seed, **flags)
                    verdict = dut.search(blocked)
                    self.assertEqual(verdict.verdict, "PROVEN_UNSAT_UNDER_DECLARED_RULES")
                    proof = reachable_closure(blocked)
                    self.assertTrue(verify_unsat(blocked, proof).valid)
                    self.assertNotEqual(verify_attempts(blocked, dut.search(dut.make_case(seed)).witness),
                                        "VERIFIED_WIN")
                    # Independent invariant: a one-turn cell expires during
                    # the required tower-to-gate move, before 'escape'.
                    if flags.get("weak_cell"):
                        self.assertEqual(blocked.lifetime, 1)

    def test_forged_or_incomplete_unsat_certificate_rejected(self):
        case = dut.make_case(37, sealed=True)
        all_states = reachable_closure(case)
        self.assertTrue(verify_unsat(case, all_states).valid)
        self.assertFalse(verify_unsat(case, all_states - {next(iter(all_states))}).valid)
        self.assertFalse(verify_unsat(case, all_states | {(4, 0, 0, 0, 0)}).valid)
        self.assertFalse(verify_unsat(dut.make_case(37), all_states).valid)

    def test_engine_and_independent_oracle_complete_reachability_match(self):
        for seed in range(32):
            for flags in ({}, {"sealed": True}, {"weak_cell": True}):
                case = dut.make_case(seed, **flags)
                self.assertEqual(_engine_closure(case), reachable_closure(case),
                                 (seed, flags))

    def test_public_paraphrase_does_not_change_mechanical_problem(self):
        for seed in range(20):
            first = dut.make_case(seed)
            descriptions = {dut.candidate_view(dut.make_case(seed, variant=i))["prompt"]
                            for i in range(60)}
            self.assertGreater(len(descriptions), 3)
            self.assertEqual(dut.fingerprint(first),
                             dut.fingerprint(dut.make_case(seed, variant=50)))
            for leaked in ("case_id", "seed", "sealed", "gold_trace", "fingerprint"):
                self.assertNotIn(leaked, str(dut.candidate_view(first)).lower())

    def test_invalid_attempt_consumes_budget_and_advances_clock(self):
        case = dut.make_case(18)
        session = dut.CandidateSession(case)
        initial = dut.start(case)
        first = session.act({"unknown": "teleport"})
        self.assertEqual(first["remaining_actions"], dut.MAX_ACTIONS - 1)
        self.assertIn(f"Clock: {(initial.phase + 1) % case.period}/", first["observation"])
        for _ in range(dut.MAX_ACTIONS - 1):
            final = session.act("teleport")
        self.assertEqual(final["verdict"], "VERIFIED_FAILURE")
        self.assertTrue(final["finished"])
        with self.assertRaises(ValueError):
            session.act("forward")
        self.assertEqual(verify_attempts(case, session.attempts()), "VERIFIED_FAILURE")

    def test_no_known_win_is_not_unsat_proof(self):
        for case in (dut.make_case(7), dut.make_case(7, sealed=True)):
            self.assertEqual(dut.search(case, max_expansions=0).verdict, "NO_KNOWN_WIN")

    def test_unexpected_rule_proposal_preserved_not_auto_rejected(self):
        sealed = dut.make_case(1, sealed=True)
        proposed = "Could the cell be relayed mechanically without energizing the sealed panel?"
        challenge = adjudicate_external_solution(sealed, actions=["throw_cell", "escape"],
                                                  proposed_rule_change=proposed)
        self.assertEqual(challenge["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(challenge["proposal"], proposed)
        self.assertEqual(adjudicate_external_solution(sealed, actions=["throw_cell"])["verdict"],
                         "VERIFIED_FAILURE")

    def test_explicit_abstention_is_not_fake_graduation(self):
        sealed = dut.make_case(17, sealed=True)
        session = dut.CandidateSession(sealed)
        self.assertIn("ABSTAIN", dut.candidate_view(sealed)["submission_kinds"])
        review = session.submit({"kind": "ABSTAIN", "assessment": "NO_KNOWN_WIN"})
        self.assertEqual(review["verdict"], "NO_KNOWN_WIN")
        self.assertFalse(review["verified_success"])
        self.assertTrue(session.view()["finished"])
        with self.assertRaises(ValueError):
            session.submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})

    def test_candidate_can_prove_declared_unsat_without_global_claim(self):
        sealed = dut.make_case(17, sealed=True)
        declared = {"period": sealed.period, "charge_phase": sealed.charge_phase,
                    "pass_phase": sealed.pass_phase, "initial_phase": sealed.initial_phase,
                    "lifetime": sealed.lifetime, "sealed": sealed.sealed}
        receipt = dut.CandidateSession(sealed).submit({
            "kind": "UNSAT_CERTIFICATE", "states": reachable_closure(sealed),
            "declared_rules": declared})
        self.assertEqual(receipt["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(receipt["verified_success"])
        self.assertFalse(receipt["escape_verified"])
        self.assertTrue(receipt["certificate_verified"])
        session = dut.CandidateSession(sealed)
        session.submit({"kind": "UNSAT_CERTIFICATE", "states": reachable_closure(sealed),
                        "declared_rules": declared})
        self.assertEqual(session.view()["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        reopened = dut.make_case(17)
        forged = adjudicate_candidate_claim(reopened, {
            "kind": "UNSAT_CERTIFICATE", "states": reachable_closure(sealed),
            "declared_rules": declared})
        self.assertEqual(forged["verdict"], "UNKNOWN")
        self.assertFalse(forged["verified_success"])
        missing_assumptions = adjudicate_candidate_claim(sealed, {
            "kind": "UNSAT_CERTIFICATE", "states": reachable_closure(sealed)})
        self.assertEqual(missing_assumptions["verdict"], "UNKNOWN")

    def test_submitted_witness_cannot_erase_prior_charged_attempts(self):
        case = dut.make_case(0)
        shortest = dut.search(case).witness
        session = dut.CandidateSession(case)
        for _ in range(dut.MAX_ACTIONS - len(shortest) + 1):
            session.act("teleport")
        receipt = session.submit({"kind": "WITNESS", "actions": shortest})
        self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
        self.assertFalse(receipt["verified_success"])
        self.assertGreater(receipt["attempts_charged"], dut.MAX_ACTIONS)
        clean = dut.CandidateSession(case).submit({"kind": "WITNESS", "actions": shortest})
        self.assertEqual(clean["verdict"], "VERIFIED_WIN")
        self.assertTrue(clean["verified_success"])
        self.assertTrue(clean["escape_verified"])
        self.assertFalse(clean["certificate_verified"])

    def test_terminal_failure_cannot_be_laundered_into_unsat_certificate(self):
        sealed = dut.make_case(17, sealed=True)
        session = dut.CandidateSession(sealed)
        for _ in range(dut.MAX_ACTIONS):
            result = session.act("teleport")
        self.assertEqual(result["verdict"], "VERIFIED_FAILURE")
        declared = {"period": sealed.period, "charge_phase": sealed.charge_phase,
                    "pass_phase": sealed.pass_phase, "initial_phase": sealed.initial_phase,
                    "lifetime": sealed.lifetime, "sealed": sealed.sealed}
        with self.assertRaisesRegex(ValueError, "terminal session"):
            session.submit({"kind": "UNSAT_CERTIFICATE", "declared_rules": declared,
                            "states": reachable_closure(sealed)})
        self.assertEqual(session.view()["verdict"], "VERIFIED_FAILURE")

    def test_terminal_victory_cannot_be_overwritten_by_abstention(self):
        case = dut.make_case(0)
        session = dut.CandidateSession(case)
        for move in dut.search(case).witness:
            result = session.act(move)
        self.assertEqual(result["verdict"], "VERIFIED_WIN")
        with self.assertRaisesRegex(ValueError, "terminal session"):
            session.submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
        self.assertEqual(session.view()["verdict"], "VERIFIED_WIN")

    def test_mutated_environment_suppressing_escape_is_fatal_disagreement(self):
        case = dut.make_case(0)
        witness = dut.search(case).witness
        original = dut.transition

        def suppress_escape(config, state, action):
            if action == "escape" and state.room == "gate":
                return state
            return original(config, state, action)

        session = dut.CandidateSession(case)
        with mock.patch.object(dut, "transition", side_effect=suppress_escape):
            for move in witness:
                result = session.act(move)
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(result["integrity_status"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])
        self.assertEqual(session.attempts(), witness)
        self.assertEqual(verify_attempts(case, session.attempts()), "VERIFIED_WIN")
        with self.assertRaisesRegex(ValueError, "finished"):
            session.act("wait")
        with self.assertRaisesRegex(ValueError, "terminal session"):
            session.submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})

    def test_transition_exception_is_charged_and_terminal_not_silent(self):
        case = dut.make_case(1)
        session = dut.CandidateSession(case)
        with mock.patch.object(dut, "transition", side_effect=RuntimeError("broken")):
            result = session.act("forward")
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(result["remaining_actions"], dut.MAX_ACTIONS - 1)
        self.assertTrue(result["finished"])
        self.assertEqual(session.attempts(), ("forward",))

    def test_observation_exception_is_terminal_evaluator_error(self):
        case = dut.make_case(1)
        session = dut.CandidateSession(case)
        with mock.patch.object(dut, "observe", side_effect=RuntimeError("broken")):
            result = session.view()
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(result["integrity_status"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])
        self.assertEqual(result["remaining_actions"], dut.MAX_ACTIONS)
        with self.assertRaisesRegex(ValueError, "finished"):
            session.act("forward")

    def test_malformed_observation_is_not_treated_as_valid_feedback(self):
        case = dut.make_case(1)
        for malformed in (None, "", "   ", True):
            with self.subTest(payload=malformed):
                session = dut.CandidateSession(case)
                with mock.patch.object(dut, "observe", return_value=malformed):
                    result = session.view()
                self.assertTrue(result["finished"])
                self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
                self.assertEqual(result["integrity_status"], "EVALUATOR_DISAGREEMENT")
                self.assertEqual(session.attempts(), ())
                with self.assertRaisesRegex(ValueError, "finished"):
                    session.act("forward")

    def test_independent_oracle_exception_is_terminal_evaluator_error(self):
        case = dut.make_case(1)
        session = dut.CandidateSession(case)
        with mock.patch("labyrinth.timegate_oracle.replay_snapshot",
                        side_effect=RuntimeError("oracle broken")):
            result = session.view()
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["finished"])
        self.assertEqual(session.attempts(), ())

    def test_novel_rule_claim_preserved_without_automatic_validation(self):
        case = dut.make_case(2, sealed=True)
        proposed = "A relay may transmit mechanical force without the sealed electrical panel."
        receipt = dut.CandidateSession(case).submit({"kind": "NOVEL_RULE",
                                                     "rule_change": proposed,
                                                     "actions": ["mechanical_pulse"]})
        self.assertEqual(receipt["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertFalse(receipt["verified_success"])
        self.assertEqual(receipt["proposal"], proposed)

    def test_independent_counterfactual_replay_can_overturn_old_unsat_conditionally(self):
        case = dut.make_case(2, sealed=True)
        self.assertEqual(dut.search(case).verdict, "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        surprising = ["forward", "take", "forward", "forward", "manual_release"]
        tested = test_hypothetical_manual_release(case, surprising)
        self.assertEqual(tested["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertTrue(tested["counterfactual_verified"])
        self.assertEqual(tested["original_rules_verdict"], "VERIFIED_FAILURE")
        self.assertFalse(test_hypothetical_manual_release(case,
                        ["forward", "manual_release"])["counterfactual_verified"])

    def test_counterfactual_replay_rejects_non_string_equality_spoof(self):
        class FakeAction:
            def __eq__(self, other):
                return other == "manual_release"

        case = dut.make_case(0, sealed=True)
        forged = ["forward", "take", "forward", "forward", FakeAction()]
        self.assertFalse(test_hypothetical_manual_release(case, forged)[
            "counterfactual_verified"])

    def test_claim_kind_rejects_non_string_equality_spoof(self):
        class FakeLabel:
            def __eq__(self, other):
                return other in ("WITNESS", "UNKNOWN")

        case = dut.make_case(0)
        forged = adjudicate_candidate_claim(case, {
            "kind": FakeLabel(), "actions": dut.search(case).witness})
        self.assertEqual(forged["verdict"], "UNKNOWN")
        self.assertFalse(forged["verified_success"])
        bogus_abstain = adjudicate_candidate_claim(case, {
            "kind": "ABSTAIN", "assessment": FakeLabel()})
        self.assertEqual(bogus_abstain["verdict"], "UNKNOWN")
        self.assertEqual(bogus_abstain["reason"], "invalid abstention label")

    def test_zero_update_fixed_script_fails_structural_transfer(self):
        script = dut.search(dut.make_case(0)).witness
        actual = sum(verify_attempts(dut.make_case(i), script) == "VERIFIED_WIN"
                     for i in range(120))
        self.assertEqual(actual, 16)  # Reproducible, not an assumed 0% baseline.
        self.assertEqual(len({dut.fingerprint(dut.make_case(i)) for i in range(120)}), 86)

    def test_zero_update_public_rule_parser_solves_every_open_world(self):
        outcomes = {"open": [], "sealed": [], "weak_cell": []}
        for seed in range(120):
            for kind, flags in (("open", {}), ("sealed", {"sealed": True}),
                                ("weak_cell", {"weak_cell": True})):
                case = dut.make_case(seed, **flags)
                session = dut.CandidateSession(case)
                result = run_public_rule_parser(dut.candidate_view(case),
                                                session.act, session.submit)
                outcomes[kind].append(result.verdict)
        self.assertEqual(outcomes["open"].count("VERIFIED_WIN"), 120)
        self.assertEqual(outcomes["sealed"].count("UNKNOWN"), 120)
        self.assertEqual(outcomes["weak_cell"].count("UNKNOWN"), 120)

    def test_test_the_tester_mutant_is_rejected_by_independent_oracle(self):
        sealed = dut.make_case(0, sealed=True)
        original = dut.transition

        def broken_seal(case, state, action):
            result = original(case, state, action)
            if action == "arm" and state.room == "tower" and state.carrying:
                return dut.State(result.room, result.phase, False, result.available,
                                 case.lifetime)
            return result

        with mock.patch.object(dut, "transition", side_effect=broken_seal):
            claimed = dut.search(sealed)
        self.assertEqual(claimed.verdict, "VERIFIED_WIN")
        self.assertEqual(verify_attempts(sealed, claimed.witness), "VERIFIED_FAILURE")


if __name__ == "__main__":
    unittest.main()
