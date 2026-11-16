"""Adversarial source-only tests for generated nonlinear rule-change exams."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from itertools import product
import io
import json
import unittest
from unittest import mock

from labyrinth import switchback_circuit as dut
from labyrinth.switchback_cli import main
from labyrinth.switchback_controls import (MEMORIZED_SEED0_PATH, memorized_control,
                                          privileged_symbolic_control, published_lookup_control)
from labyrinth.switchback_oracle import (_canonical, _rule_output, adjudicate,
                                         oracle_step, search, verify_plan, verify_probe_trace,
                                         verify_receipt_replay)


def _invariant(case):
    """Fixed points/image size are invariant under arbitrary state relabeling."""
    return tuple(tuple(sorted((len({_rule_output(r, b, rule) for r in range(8)}),
                               sum(_rule_output(r, b, rule) == r for r in range(8)))
                              for b in (0, 1))) for rule in (case.before, case.after))


def _resign(receipt):
    modified = deepcopy(receipt)
    modified.pop("sha256", None)
    modified["sha256"] = sha256(_canonical(modified).encode()).hexdigest()
    return modified


class SwitchbackHostileTests(unittest.TestCase):
    def test_strict_factory_seed_bounds(self):
        for seed in (None, True, -1, 2**31, 1.0, "1"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                dut.make_case(seed)

    def test_seeded_cases_deterministic_and_distinct(self):
        for seed in (0, 1, 2, 7, 100, 2**31 - 1):
            self.assertEqual(dut.make_case(seed), dut.make_case(seed))
        self.assertNotEqual(dut.make_case(1).case_id, dut.make_case(2).case_id)

    def test_pre_and_post_rule_truth_tables_actually_differ_all_seed_cases(self):
        for seed in range(100):
            case = dut.make_case(seed)
            before = tuple(_rule_output(r, b, case.before) for r in range(8) for b in (0, 1))
            after = tuple(_rule_output(r, b, case.after) for r in range(8) for b in (0, 1))
            self.assertNotEqual(before, after)

    def test_100_distinct_mechanical_signatures_not_merely_seed_labels(self):
        cases = [dut.make_case(seed) for seed in range(100)]
        self.assertEqual(len({case.mechanic_signature for case in cases}), 100)
        # Even if a hostile evaluator permits state relabelings AND input
        # token swaps, these exact conjugacy invariants differ in 84 groups.
        self.assertEqual(len({_invariant(case) for case in cases}), 84)

    def test_heldout_100_to_199_contains_genuinely_new_mechanics(self):
        train = [dut.make_case(seed) for seed in range(100)]
        heldout = [dut.make_case(seed) for seed in range(100, 200)]
        train_inv = {_invariant(c) for c in train}
        heldout_inv = {_invariant(c) for c in heldout}
        self.assertEqual(len(heldout_inv), 88)
        self.assertEqual(len(heldout_inv - train_inv), 71)
        self.assertEqual(sum(_invariant(c) not in train_inv for c in heldout), 78)

    def test_five_generators_are_represented_in_sample(self):
        cases = [dut.make_case(i) for i in range(100)]
        self.assertEqual({r.family for c in cases for r in (c.before, c.after)},
                         set(range(5)))

    def test_candidate_public_view_does_not_return_latent_rule_or_seed(self):
        for seed in range(10):
            public = dut.public_view(dut.make_case(seed))
            self.assertEqual(set(public), {"prompt", "initial_register", "target_register",
                                           "plan_actions", "probe_actions", "maximum_moves", "switch_at",
                                           "maximum_pulses"})
            self.assertNotIn("seed", str(public))
            self.assertNotIn("case_id", str(public))
            self.assertNotIn("mechanic_signature", str(public))
            self.assertEqual(public["probe_actions"], ["0", "1", "WAIT", "RESET"])
            self.assertEqual(public["plan_actions"], ["0", "1", "WAIT", "COMMIT"])

    def test_public_probe_actions_replay_without_private_rule_leak(self):
        for seed in (0, 1, 2, 13, 14):
            session = dut.ProbeSession(dut.make_case(seed))
            for action in ("0", "WAIT", "1", "RESET", "1"):
                view = session.probe(action)
                self.assertNotIn("seed", str(view))
                self.assertNotIn("mechanic_signature", str(view))
            self.assertTrue(verify_probe_trace(dut.make_case(seed), view["observations"]))

    def test_identical_zero_probe_public_cases_can_split_after_one_intervention(self):
        left, right = dut.make_case(13), dut.make_case(14)
        self.assertEqual(dut.public_view(left), dut.public_view(right))
        self.assertNotEqual(search(left)["verdict"], search(right)["verdict"])
        s, t = dut.ProbeSession(left), dut.ProbeSession(right)
        self.assertNotEqual(s.probe("0")["current_register"],
                            t.probe("0")["current_register"])

    def test_same_public_view_and_family_pair_not_same_mechanics_or_answer(self):
        first, second = dut.make_case(3), dut.make_case(2513)
        self.assertEqual(dut.public_view(first), dut.public_view(second))
        self.assertEqual((first.before.family, first.after.family),
                         (second.before.family, second.after.family))
        self.assertNotEqual(first.mechanic_signature, second.mechanic_signature)
        self.assertNotEqual(search(first)["verdict"], search(second)["verdict"])

    def test_same_mechanics_digest_not_same_scenario_or_answer(self):
        first, second = dut.make_case(44), dut.make_case(3566)
        self.assertEqual(first.mechanic_signature, second.mechanic_signature)
        self.assertNotEqual(first.case_id, second.case_id)
        self.assertNotEqual(search(first)["verdict"], search(second)["verdict"])

    def test_public_view_nested_mutation_cannot_rewrite_private_probes(self):
        session = dut.ProbeSession(dut.make_case(0))
        displayed = session.probe("0")
        displayed["observations"][0]["action"] = "WAIT"
        self.assertEqual(session.view()["observations"][0]["action"], "0")

    def test_probe_engine_mutation_fails_closed_before_valid_submission(self):
        session = dut.ProbeSession(dut.make_case(0))
        with mock.patch.object(dut, "step", side_effect=lambda case, state, action: state):
            session.probe("0")
        receipt = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(receipt["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(receipt["verified_success"])

    def test_crashing_probe_transition_charges_attempt_and_seals_session(self):
        session = dut.ProbeSession(dut.make_case(0))
        with mock.patch.object(dut, "step", side_effect=RuntimeError("mutated engine")):
            result = session.probe("0")
        self.assertTrue(result["finished"])
        self.assertEqual(len(result["observations"]), 1)
        self.assertEqual(result["remaining_probes"], dut.MAX_PROBES - 1)
        self.assertIsNone(result["observations"][0]["after"])
        receipt = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(receipt["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(receipt["reason"], "BLOCKED_PROBE_EXECUTION")
        self.assertFalse(receipt["verified_success"])

    def test_malformed_or_spoofed_probe_return_never_allows_free_retry(self):
        class LyingState(dut.State):
            def __eq__(self, other):
                return True
        bad_states = (None, LyingState(7), dut.State(True),
                      dut.State(8), dut.State(0, ticks=1.5),
                      dut.State(0, pulses=100), dut.State(0, won=1))
        for candidate in bad_states:
            with self.subTest(candidate=repr(candidate)):
                session = dut.ProbeSession(dut.make_case(0))
                with mock.patch.object(dut, "step", return_value=candidate):
                    view = session.probe("0")
                self.assertTrue(view["finished"])
                self.assertEqual(view["remaining_probes"], dut.MAX_PROBES - 1)
                self.assertEqual(view["observations"][0]["after"], None)
                receipt = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
                self.assertEqual(receipt["verdict"], "EVALUATOR_DISAGREEMENT")
                self.assertFalse(receipt["verified_success"])

    def test_public_probe_invalid_actions_charge_and_terminally_block_unserializable(self):
        session = dut.ProbeSession(dut.make_case(0))
        self.assertEqual(session.probe("INVALID")["remaining_probes"], dut.MAX_PROBES - 1)
        view = session.probe(True)
        self.assertTrue(view["finished"])
        self.assertEqual(len(view["observations"]), 2)
        result = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(result["verified_success"])

    def test_recognized_action_can_be_budget_blocked_without_false_application(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        for _ in range(case.max_pulses):
            record = session.probe("0")["observations"][-1]
            self.assertTrue(record["valid"])
            self.assertTrue(record["applied"])
        view = session.probe("0")
        self.assertEqual(view["observations"][-1]["action"], "0")
        self.assertTrue(view["observations"][-1]["valid"])
        self.assertFalse(view["observations"][-1]["applied"])
        self.assertTrue(verify_probe_trace(case, view["observations"]))

    def test_one_shot_public_session_submission_cannot_be_changed_to_win(self):
        session = dut.ProbeSession(dut.make_case(0))
        first = session.submit({"kind": "PLAN", "actions": ["COMMIT"]})
        self.assertEqual(first["verdict"], "VERIFIED_FAILURE")
        second = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(second, first)

    def test_session_receipt_distinguishes_zero_probe_from_two_real_probes(self):
        case = dut.make_case(0)
        proposal = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
        without = dut.ProbeSession(case).submit(proposal)
        with_probes = dut.ProbeSession(case)
        with_probes.probe("0")
        with_probes.probe("RESET")
        observed = with_probes.submit(proposal)
        self.assertEqual(without["verdict"], "VERIFIED_WIN")
        self.assertEqual(observed["verdict"], "VERIFIED_WIN")
        self.assertEqual(without["probe_count"], 0)
        self.assertEqual(observed["probe_count"], 2)
        self.assertNotEqual(without["sha256"], observed["sha256"])
        self.assertTrue(verify_receipt_replay(without))
        self.assertTrue(verify_receipt_replay(observed))

    def test_caller_owned_nested_proposal_cannot_mutate_finalized_evidence(self):
        session = dut.ProbeSession(dut.make_case(0))
        proposal = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
        first = session.submit(proposal)
        self.assertTrue(verify_receipt_replay(first))
        proposal["actions"][0] = "FORGED"
        later = session.submit(proposal)
        self.assertEqual(first, later)
        self.assertTrue(verify_receipt_replay(later))
        self.assertEqual(later["adjudication"]["proposal"]["actions"][0], "0")

    def test_crashing_submission_evaluator_seals_terminal_without_escape(self):
        from labyrinth import switchback_oracle as oracle
        session = dut.ProbeSession(dut.make_case(0))
        with mock.patch.object(oracle, "adjudicate_session", side_effect=RuntimeError("poison")):
            first = session.submit({"kind": "CLAIM_UNSAT"})
        self.assertEqual(first["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(first["reason"], "BLOCKED_SUBMISSION_EVALUATOR")
        self.assertFalse(first["verified_success"])
        self.assertEqual(first, session.submit({"kind": "PLAN", "actions": list(
            MEMORIZED_SEED0_PATH)}))

    def test_crashing_independent_probe_replay_seals_terminal(self):
        from labyrinth import switchback_oracle as oracle
        session = dut.ProbeSession(dut.make_case(0))
        with mock.patch.object(oracle, "verify_probe_trace", side_effect=RuntimeError("poison")):
            result = session.submit({"kind": "CLAIM_UNSAT"})
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(result["reason"], "BLOCKED_SUBMISSION_EVALUATOR")
        self.assertEqual(result, session.submit({"kind": "PLAN", "actions": list(
            MEMORIZED_SEED0_PATH)}))

    def test_forged_or_malformed_success_receipt_cannot_become_session_win(self):
        from labyrinth import switchback_oracle as oracle
        for value in ({"verdict": "VERIFIED_WIN", "verified_success": True}, None):
            with self.subTest(value=value):
                session = dut.ProbeSession(dut.make_case(0))
                with mock.patch.object(oracle, "adjudicate_session", return_value=value):
                    first = session.submit({"kind": "CLAIM_UNSAT"})
                self.assertEqual(first["verdict"], "EVALUATOR_DISAGREEMENT")
                self.assertFalse(first["verified_success"])
                self.assertEqual(first, session.submit({"kind": "PLAN", "actions": list(
                    MEMORIZED_SEED0_PATH)}))

    def test_shared_session_receipt_generator_cannot_self_certify_forged_win(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(2)
        honest = dut.ProbeSession(case).submit({"kind": "CLAIM_UNSAT"})
        self.assertEqual(honest["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        forged = deepcopy(honest)
        forged["adjudication"]["verdict"] = "VERIFIED_WIN"
        forged["adjudication"]["verified_success"] = True
        forged["adjudication"] = _resign(forged["adjudication"])
        forged["verdict"] = "VERIFIED_WIN"
        forged["verified_success"] = True
        forged = _resign(forged)
        self.assertFalse(verify_receipt_replay(forged))
        session = dut.ProbeSession(case)
        with mock.patch.object(oracle, "adjudicate_session", return_value=forged):
            receipt = session.submit({"kind": "CLAIM_UNSAT"})
        self.assertEqual(receipt["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(receipt["verified_success"])
        self.assertEqual(receipt, session.submit({"kind": "PLAN", "actions": list(
            MEMORIZED_SEED0_PATH)}))

    def test_lower_level_adjudicate_cannot_self_certify_forged_win(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(2)
        honest = adjudicate(case, {"kind": "CLAIM_UNSAT"})
        forged = deepcopy(honest)
        forged["verdict"] = "VERIFIED_WIN"
        forged["verified_success"] = True
        forged = _resign(forged)
        self.assertFalse(verify_receipt_replay(forged))
        session = dut.ProbeSession(case)
        with mock.patch.object(oracle, "adjudicate", return_value=forged):
            receipt = session.submit({"kind": "CLAIM_UNSAT"})
        self.assertEqual(receipt["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(receipt["verified_success"])

    def test_mutated_search_complete_unsat_without_closure_is_not_proof(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(0)
        fake = {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "witness": None,
                "complete": True, "nodes_explored": 1}
        with mock.patch.object(oracle, "search", return_value=fake):
            receipt = adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(receipt["review"]["reason"], "COUNTEREXAMPLE_WIN_EXISTS")
        self.assertEqual(verify_plan(case, receipt["review"]["counterexample"])["verdict"],
                         "VERIFIED_WIN")

    def test_shared_oracle_transition_mutation_cannot_prove_false_unsat(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(0)
        self.assertEqual(search(case)["verdict"], "VERIFIED_WIN")
        with mock.patch.object(oracle, "oracle_step", side_effect=lambda c, s, a: s):
            false_proof = adjudicate(case, {"kind": "CLAIM_UNSAT"})
            self.assertNotEqual(false_proof["verdict"],
                                "PROVEN_UNSAT_UNDER_DECLARED_RULES")
            self.assertFalse(false_proof["verified_success"])
            self.assertEqual(false_proof["review"]["reason"],
                             "INDEPENDENT_TRANSITION_DISAGREEMENT")
            session = dut.ProbeSession(case)
            quarantined = session.submit({"kind": "CLAIM_UNSAT"})
            self.assertEqual(quarantined["verdict"], "EVALUATOR_DISAGREEMENT")
            self.assertFalse(quarantined["verified_success"])

    def test_intermediate_tick_shared_oracle_mutation_cannot_hide_in_sampling_gap(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(0)
        original = oracle.oracle_step
        def hidden_tick(c, state, action):
            return state if state.ticks == 1 else original(c, state, action)
        with mock.patch.object(oracle, "oracle_step", side_effect=hidden_tick):
            self.assertFalse(oracle.independent_transition_parity(case))
            review = adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertNotEqual(review["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(review["review"]["reason"], "INDEPENDENT_TRANSITION_DISAGREEMENT")

    def test_forged_early_commit_win_fails_full_reachable_parity(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(0)
        original = oracle.oracle_step
        def forged_commit(c, state, action):
            if action == "COMMIT":
                return dut.State(state.register, state.ticks, state.pulses,
                                 state.injected_after, True, True)
            return original(c, state, action)
        with mock.patch.object(oracle, "oracle_step", side_effect=forged_commit):
            self.assertFalse(oracle.independent_transition_parity(case))
            review = adjudicate(case, {"kind": "PLAN", "actions": ["COMMIT"]})
        self.assertFalse(review["verified_success"])
        self.assertEqual(review["verdict"], "UNKNOWN")

    def test_single_forged_verify_plan_result_cannot_certify_unearned_win(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(0)
        self.assertFalse(oracle.independent_plan_won(case, ["COMMIT"]))
        fake = {"verdict": "VERIFIED_WIN", "reason": "FAKE_SCORER_WIN",
                "counterexample": None, "steps_checked": 1}
        with mock.patch.object(oracle, "verify_plan", return_value=fake):
            receipt = adjudicate(case, {"kind": "PLAN", "actions": ["COMMIT"]})
            self.assertFalse(receipt["verified_success"])
            self.assertNotEqual(receipt["verdict"], "VERIFIED_WIN")
            self.assertFalse(verify_receipt_replay(receipt))
            session = dut.ProbeSession(case)
            quarantined = session.submit({"kind": "PLAN", "actions": ["COMMIT"]})
            self.assertFalse(quarantined["verified_success"])

    def test_single_forged_probe_verifier_cannot_certify_unobserved_trace(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(0)
        fake_trace = [{"action": "FORGED", "valid": True, "applied": True,
                       "before": "000", "after": "111"}]
        honest = dut.ProbeSession(case).submit({"kind": "PLAN", "actions": list(
            MEMORIZED_SEED0_PATH)})
        forged = deepcopy(honest)
        forged["probe_count"] = 1
        forged["probe_trace"] = fake_trace
        forged["probe_sha256"] = sha256(_canonical({"trace": fake_trace}).encode()).hexdigest()
        forged = _resign(forged)
        with mock.patch.object(oracle, "verify_probe_trace", return_value=True):
            with self.assertRaises(ValueError):
                oracle.adjudicate_session(case, fake_trace, {"kind": "PLAN",
                                                             "actions": list(MEMORIZED_SEED0_PATH)})
            self.assertFalse(verify_receipt_replay(forged))

    def test_state_object_equality_spoof_cannot_launder_transition_mismatch(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(0)
        original = oracle.oracle_step
        def hidden_tick(c, state, action):
            return state if state.ticks == 1 else original(c, state, action)
        with (mock.patch.object(dut.State, "__eq__", return_value=True),
              mock.patch.object(oracle, "oracle_step", side_effect=hidden_tick)):
            self.assertFalse(oracle.independent_transition_parity(case))
            review = adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertEqual(review["review"]["reason"],
                         "INDEPENDENT_TRANSITION_DISAGREEMENT")
        self.assertFalse(review["verified_success"])

    def test_shared_state_constructor_mutation_cannot_launder_false_win(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(0)
        original_constructor = dut.State.__init__
        def shifted_constructor(self, register, ticks=0, pulses=0,
                                injected_after=False, committed=False, won=False):
            return original_constructor(self, (register + 1) % 8, ticks,
                                        pulses, injected_after, committed, won)
        with mock.patch.object(dut.State, "__init__", shifted_constructor):
            self.assertFalse(oracle.independent_transition_parity(case))
            review = adjudicate(case, {"kind": "PLAN", "actions": [
                "0", "0", "WAIT", "1", "COMMIT"]})
            self.assertFalse(review["verified_success"])
            self.assertEqual(review["review"]["reason"],
                             "INDEPENDENT_TRANSITION_DISAGREEMENT")

    def test_provenance_verifier_exception_fails_closed_boolean(self):
        from labyrinth import switchback_oracle as oracle
        receipt = adjudicate(dut.make_case(0), {"kind": "PLAN", "actions": list(
            MEMORIZED_SEED0_PATH)})
        with mock.patch.object(oracle, "adjudicate", side_effect=RuntimeError("poison")):
            self.assertFalse(verify_receipt_replay(receipt))

    def test_sixteenth_legal_probe_still_allows_one_final_answer(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        for _ in range(dut.MAX_PROBES):
            last = session.probe("RESET")
        self.assertEqual(last["remaining_probes"], 0)
        self.assertFalse(last["finished"])
        receipt = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertEqual(receipt["probe_count"], dut.MAX_PROBES)
        self.assertTrue(verify_receipt_replay(receipt))

    def test_seventeenth_probe_terminally_forfeits_no_late_win(self):
        session = dut.ProbeSession(dut.make_case(0))
        for _ in range(dut.MAX_PROBES):
            session.probe("RESET")
        last = session.probe("RESET")
        self.assertTrue(last["finished"])
        self.assertEqual(session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})[
            "verdict"], "VERIFIED_FAILURE")

    def test_oracle_outputs_agree_with_distinct_dut_kernel_all_contexts(self):
        for seed in range(75):
            case = dut.make_case(seed)
            for rule in (case.before, case.after):
                for register in range(8):
                    for bit in (0, 1):
                        self.assertEqual(dut.kernel(register, bit, rule),
                                         _rule_output(register, bit, rule))

    def test_oracle_step_differential_state_and_clock_across_rule_boundary(self):
        for seed in range(20):
            case = dut.make_case(seed)
            for tick in range(0, 7):
                for register in range(8):
                    for pulses in range(case.max_pulses + 1):
                        state = dut.State(register, tick, pulses, bool(tick % 2))
                        for action in dut.CONTROLS:
                            self.assertEqual(dut.step(case, state, action),
                                             oracle_step(case, state, action))

    def test_seed_zero_canonical_witness_is_verified(self):
        case = dut.make_case(0)
        witness = search(case)["witness"]
        self.assertEqual(witness, list(MEMORIZED_SEED0_PATH))
        self.assertEqual(verify_plan(case, witness)["verdict"], "VERIFIED_WIN")

    def test_unexpected_valid_path_not_in_cached_answer_is_accepted(self):
        case = dut.make_case(0)
        alternate = ["0", "WAIT", "0", "0", "COMMIT"]
        self.assertNotEqual(alternate, search(case)["witness"])
        report = verify_plan(case, alternate)
        self.assertEqual(report["verdict"], "VERIFIED_WIN")
        receipt = adjudicate(case, {"kind": "PLAN", "actions": alternate})
        self.assertTrue(receipt["verified_success"])
        self.assertFalse(receipt["graduation_eligible"])
        self.assertTrue(verify_receipt_replay(receipt))

    def test_absent_postchange_input_cannot_fake_commit(self):
        case = dut.make_case(0)
        receipt = adjudicate(case, {"kind": "PLAN", "actions": ["WAIT"] * 4 + ["COMMIT"]})
        self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")

    def test_invalid_plan_shapes_are_rejected_without_python_eq_spoof(self):
        class Liar:
            def __eq__(self, other):
                return other == "COMMIT"
        case = dut.make_case(0)
        for actions in ([], ["0"] * 10, ["0", "DOOR", "COMMIT"],
                        ["0", True, "COMMIT"], ["0", Liar(), "COMMIT"],
                        ("0", "COMMIT")):
            self.assertEqual(verify_plan(case, actions)["verdict"], "VERIFIED_FAILURE")

    def test_action_after_commit_cannot_flip_terminal_result(self):
        case = dut.make_case(0)
        result = verify_plan(case, [*MEMORIZED_SEED0_PATH, "WAIT"])
        self.assertEqual(result["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(result["reason"], "ACTION_AFTER_COMMIT")

    def test_maximum_pulse_budget_is_enforced(self):
        case = dut.make_case(0)
        result = verify_plan(case, ["0"] * (case.max_pulses + 1) + ["COMMIT"])
        self.assertEqual(result["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(result["reason"], "PULSE_BUDGET_EXCEEDED")

    def test_exact_search_distinguishes_exhaustive_unsat_from_budgeted_unknown(self):
        case = dut.make_case(2)
        limited = search(case, max_nodes=1)
        self.assertEqual(limited["verdict"], "NO_KNOWN_WIN")
        self.assertFalse(limited["complete"])
        exhaustive = search(case)
        self.assertEqual(exhaustive["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertTrue(exhaustive["complete"])
        self.assertIsNone(exhaustive["witness"])

    def test_true_unsat_claim_is_not_escape_graduation_success(self):
        receipt = adjudicate(dut.make_case(2), {"kind": "CLAIM_UNSAT"})
        self.assertEqual(receipt["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(receipt["verified_success"])
        self.assertTrue(verify_receipt_replay(receipt))

    def test_incomplete_or_inconsistent_search_cannot_issue_unsat_proof(self):
        from labyrinth import switchback_oracle as oracle
        case = dut.make_case(2)
        for incomplete in (
            {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "witness": None,
             "complete": False, "nodes_explored": 1},
            {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "witness": None,
             "complete": None, "nodes_explored": 1},
            {"verdict": "NO_KNOWN_WIN", "witness": None,
             "complete": True, "nodes_explored": 1},
        ):
            with self.subTest(incomplete=incomplete):
                with mock.patch.object(oracle, "search", return_value=incomplete):
                    receipt = adjudicate(case, {"kind": "CLAIM_UNSAT"})
                self.assertNotEqual(receipt["verdict"],
                                    "PROVEN_UNSAT_UNDER_DECLARED_RULES")
                self.assertFalse(receipt["verified_success"])

    def test_false_unsat_claim_preserves_actual_counterexample_witness(self):
        case = dut.make_case(0)
        receipt = adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(receipt["review"]["reason"], "COUNTEREXAMPLE_WIN_EXISTS")
        self.assertEqual(verify_plan(case, receipt["review"]["counterexample"])["verdict"],
                         "VERIFIED_WIN")

    def test_unknown_untyped_novel_claim_is_pending_not_novelty_victory(self):
        receipt = adjudicate(dut.make_case(0),
                             {"kind": "NOVEL_RULE", "description": "Maybe a new law exists"})
        self.assertEqual(receipt["verdict"], "UNKNOWN")
        self.assertEqual(receipt["review"]["review_status"], "PENDING_INDEPENDENT_REVIEW")
        self.assertFalse(receipt["verified_success"])
        self.assertTrue(verify_receipt_replay(receipt))

    def test_malformed_proposal_has_unknown_not_auto_novel_label(self):
        receipt = adjudicate(dut.make_case(0), {"kind": "EXECUTE_PYTHON", "source": "pass"})
        self.assertEqual(receipt["verdict"], "UNKNOWN")

    def test_unserializable_proposal_is_rejected_not_normalized_to_false_receipt(self):
        for payload in ({1: "x"}, {"data": (1, 2)}, {"data": {1, 2}},
                        {"data": "x" * 9000}):
            with self.subTest(payload=repr(payload)[:40]), self.assertRaises(ValueError):
                adjudicate(dut.make_case(0), payload)

    def test_world_id_and_rule_subclass_spoof_fails_case_authority(self):
        class LyingStr(str):
            def __eq__(self, other):
                return True
            __hash__ = str.__hash__
        real = dut.make_case(0)
        fake = replace(real, case_id=LyingStr("forged"))
        with self.assertRaises(ValueError):
            search(fake)
        with self.assertRaises(ValueError):
            adjudicate(fake, {"kind": "CLAIM_UNSAT"})

    def test_mutating_dut_kernel_does_not_change_independent_verifier(self):
        case = dut.make_case(0)
        action = ["0", "WAIT", "0", "0", "COMMIT"]
        baseline = verify_plan(case, action)
        with mock.patch.object(dut, "kernel", side_effect=lambda *_: 0):
            self.assertEqual(verify_plan(case, action), baseline)

    def test_receipt_checksum_tamper_fails_replay(self):
        receipt = adjudicate(dut.make_case(0), {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        modified = deepcopy(receipt)
        modified["verified_success"] = False
        self.assertFalse(verify_receipt_replay(modified))

    def test_recomputed_digest_cannot_forge_win_or_uncertainty_verdict(self):
        receipt = adjudicate(dut.make_case(2), {"kind": "CLAIM_UNSAT"})
        modified = deepcopy(receipt)
        modified["verdict"] = "VERIFIED_WIN"
        modified["verified_success"] = True
        self.assertFalse(verify_receipt_replay(_resign(modified)))

    def test_100_seed_nonneural_controls_and_exhaustive_status_counts(self):
        cases = [dut.make_case(i) for i in range(100)]
        memory = [memorized_control(case)["verdict"] for case in cases]
        symbolic = [privileged_symbolic_control(case)["verdict"] for case in cases]
        self.assertEqual(memory.count("VERIFIED_WIN"), 7)
        self.assertEqual(symbolic.count("VERIFIED_WIN"), 67)
        self.assertEqual(symbolic.count("PROVEN_UNSAT_UNDER_DECLARED_RULES"), 33)

    def test_public_source_lookup_memorizer_breaks_apparent_training_scores(self):
        from collections import Counter
        training = Counter(published_lookup_control(dut.make_case(i))["verdict"]
                           for i in range(100))
        heldout = Counter(published_lookup_control(dut.make_case(i))["verdict"]
                          for i in range(100, 200))
        self.assertEqual(training, {
            "VERIFIED_WIN": 56, "PROVEN_UNSAT_UNDER_DECLARED_RULES": 27, "UNKNOWN": 17})
        self.assertEqual(heldout, {
            "UNKNOWN": 86, "VERIFIED_FAILURE": 12,
            "PROVEN_UNSAT_UNDER_DECLARED_RULES": 1, "VERIFIED_WIN": 1})

    def test_cli_prints_replay_valid_json_for_both_controls(self):
        for args in (("--seed", "0"), ("--seed", "2"),
                     ("--seed", "0", "--control", "memorized")):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(list(args)), 0)
            self.assertTrue(verify_receipt_replay(json.loads(out.getvalue())))


if __name__ == "__main__":
    unittest.main()
