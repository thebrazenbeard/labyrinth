"""Independent hostile metamorphic tests of Switchback formal evidence.

Source-only synthetic receipts and a monkeypatched adversarial oracle; these
tests neither execute a model nor establish external authenticated custody.
"""

from copy import deepcopy
from hashlib import sha256
import json
import random
from unittest import mock
import unittest

from labyrinth import switchback_circuit as dut
from labyrinth import switchback_oracle as checker
from labyrinth.switchback_controls import MEMORIZED_SEED0_PATH


def _independent_sha(data):
    text = json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
    return sha256(text.encode()).hexdigest()


def _rehash_session(receipt):
    trial = deepcopy(receipt)
    trial["probe_sha256"] = _independent_sha({"trace": trial["probe_trace"]})
    trial.pop("sha256")
    trial["sha256"] = _independent_sha(trial)
    return trial


class SwitchbackReceiptMathTests(unittest.TestCase):
    def test_malformed_probe_engine_result_must_be_terminally_quarantined(self):
        session = dut.ProbeSession(dut.make_case(0))
        with mock.patch.object(dut, "step", return_value=None):
            view = session.probe("0")
        self.assertTrue(view["finished"])
        self.assertEqual(len(view["observations"]), 1)
        outcome = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(outcome["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(outcome["verified_success"])

    def test_probe_engine_exception_quarantined_no_free_retry(self):
        session = dut.ProbeSession(dut.make_case(0))
        with mock.patch.object(dut, "step", side_effect=RuntimeError("injected")):
            view = session.probe("0")
        self.assertTrue(view["finished"])
        self.assertEqual(view["remaining_probes"], dut.MAX_PROBES - 1)
        self.assertEqual(session.submit({"kind": "PLAN", "actions":
                          list(MEMORIZED_SEED0_PATH)})["verdict"],
                         "EVALUATOR_DISAGREEMENT")

    def test_submission_engine_exception_quarantined_with_stable_terminal_result(self):
        session = dut.ProbeSession(dut.make_case(0))
        with mock.patch.object(checker, "adjudicate_session",
                               side_effect=RuntimeError("injected review crash")):
            result = session.submit({"kind": "CLAIM_UNSAT"})
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(result["verified_success"])
        self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"}), result)
        self.assertTrue(session.view()["finished"])

    def test_independent_probe_replay_exception_quarantined(self):
        session = dut.ProbeSession(dut.make_case(0))
        with mock.patch.object(checker, "verify_probe_trace",
                               side_effect=RuntimeError("injected replay crash")):
            result = session.submit({"kind": "CLAIM_UNSAT"})
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(result["verified_success"])
        self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"}), result)

    def test_malformed_successful_adjudication_return_must_not_be_accepted(self):
        for forged in (None, {"verdict": "VERIFIED_WIN", "verified_success": True}):
            with self.subTest(forged=forged):
                session = dut.ProbeSession(dut.make_case(0))
                with mock.patch.object(checker, "adjudicate_session", return_value=forged):
                    result = session.submit({"kind": "CLAIM_UNSAT"})
                self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
                self.assertFalse(result["verified_success"])
                self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"}), result)

    def test_shared_session_generator_mutation_cannot_self_certify_forged_win(self):
        """A second verifier must not ask the same corrupted generator to vouch."""
        case = dut.make_case(2)  # Independently proven UNSAT, not a winning plan.
        honest_session = dut.ProbeSession(case)
        honest = honest_session.submit({"kind": "CLAIM_UNSAT"})
        self.assertNotEqual(honest["verdict"], "VERIFIED_WIN")
        forged = deepcopy(honest)
        forged["verdict"] = "VERIFIED_WIN"
        forged["verified_success"] = True
        forged["adjudication"]["verdict"] = "VERIFIED_WIN"
        forged["adjudication"]["verified_success"] = True
        forged["adjudication"].pop("sha256")
        forged["adjudication"]["sha256"] = _independent_sha(forged["adjudication"])
        forged.pop("sha256")
        forged["sha256"] = _independent_sha(forged)
        # The sole compromised function is the session receipt generator.
        # A genuinely separate replay/reviewer should catch the forged inner
        # verdict by evaluating CLAIM_UNSAT independently.
        session = dut.ProbeSession(case)
        with mock.patch.object(checker, "adjudicate_session", return_value=forged):
            result = session.submit({"kind": "CLAIM_UNSAT"})
        self.assertNotEqual(result["verdict"], "VERIFIED_WIN")
        self.assertFalse(result["verified_success"])

    def test_proven_unsat_must_require_completed_search_even_if_return_claims_proof(self):
        fake = {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "witness": None,
                "complete": False, "nodes_explored": 1}
        with mock.patch.object(checker, "search", return_value=fake):
            receipt = checker.adjudicate(dut.make_case(2), {"kind": "CLAIM_UNSAT"})
        self.assertNotEqual(receipt["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(receipt["verified_success"])
        self.assertNotEqual(receipt["review"]["reason"], "EXHAUSTIVE_FINITE_SEARCH")

    def test_self_reported_complete_unsat_cannot_replace_independent_closure(self):
        fake = {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "witness": None,
                "complete": True, "nodes_explored": 1}
        # Seed zero has two independently valid winning plans; this single
        # corrupted search result cannot prove a finite impossibility.
        case = dut.make_case(0)
        self.assertEqual(checker.verify_plan(case, list(MEMORIZED_SEED0_PATH))["verdict"],
                         "VERIFIED_WIN")
        with mock.patch.object(checker, "search", return_value=fake):
            result = checker.adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertNotEqual(result["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(result["verified_success"])

    def test_two_search_algorithms_sharing_faulty_oracle_step_cannot_prove_unsat(self):
        """Different BFS/DFS traversals do not imply independent transitions."""
        case = dut.make_case(0)
        self.assertEqual(checker.verify_plan(case, list(MEMORIZED_SEED0_PATH))["verdict"],
                         "VERIFIED_WIN")
        with mock.patch.object(checker, "oracle_step",
                               side_effect=lambda case, state, action: state):
            verdict = checker.adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertNotEqual(verdict["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(verdict["verified_success"])

    def test_two_search_algorithms_sharing_faulty_kernel_cannot_prove_unsat(self):
        case = dut.make_case(0)
        self.assertEqual(checker.verify_plan(case, list(MEMORIZED_SEED0_PATH))["verdict"],
                         "VERIFIED_WIN")
        with mock.patch.object(checker, "_rule_output", return_value=0):
            try:
                result = checker.adjudicate(case, {"kind": "CLAIM_UNSAT"})
            except ValueError:
                return  # Altered oracle invalidated the canonical Case; safe.
        self.assertNotEqual(result["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_dut_side_transition_fault_cannot_promote_oracle_proof(self):
        case = dut.make_case(2)
        with mock.patch.object(dut, "step", side_effect=lambda case, state, action: state):
            result = checker.adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["review"]["reason"],
                         "INDEPENDENT_TRANSITION_DISAGREEMENT")

    def test_tick_one_only_oracle_fault_cannot_hide_between_sampled_boundaries(self):
        case = dut.make_case(0)
        self.assertEqual(case.switch_at, 3)  # Tick1 is not boundary 0,2,3,9.
        authentic = checker.oracle_step

        def tick_one_corruption(case, state, action):
            evolved = authentic(case, state, action)
            if state.ticks == 1 and action == "WAIT":
                from dataclasses import replace
                return replace(evolved, register=(evolved.register + 1) % 8)
            return evolved

        with mock.patch.object(checker, "oracle_step", side_effect=tick_one_corruption):
            receipt = checker.adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertEqual(receipt["verdict"], "UNKNOWN")
        self.assertEqual(receipt["review"]["reason"],
                         "INDEPENDENT_TRANSITION_DISAGREEMENT")

    def test_skeptic_tick_one_full_noop_mutant_blocks_session_certificate(self):
        case = dut.make_case(0)
        true_oracle = checker.oracle_step

        def tick1_noop(case, state, action):
            return state if state.ticks == 1 else true_oracle(case, state, action)

        with mock.patch.object(checker, "oracle_step", side_effect=tick1_noop):
            self.assertFalse(checker.independent_transition_parity(case))
            output = dut.ProbeSession(case).submit({"kind": "CLAIM_UNSAT"})
        self.assertNotEqual(output["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(output["verified_success"])

    def test_zero_probe_commit_cannot_become_win_under_mutated_oracle(self):
        case = dut.make_case(0)
        self.assertEqual(checker.verify_plan(case, ["COMMIT"])["verdict"],
                         "VERIFIED_FAILURE")
        session = dut.ProbeSession(case)
        self.assertEqual(session.submit({"kind": "PLAN", "actions": ["COMMIT"]})["verdict"],
                         "VERIFIED_FAILURE")
        genuine = checker.oracle_step

        def instant_win_mutation(case, state, action):
            if action == "COMMIT":
                from dataclasses import replace
                return replace(state, committed=True, won=True)
            return genuine(case, state, action)

        with mock.patch.object(checker, "oracle_step", side_effect=instant_win_mutation):
            claim = checker.adjudicate(case, {"kind": "PLAN", "actions": ["COMMIT"]})
        self.assertNotEqual(claim["verdict"], "VERIFIED_WIN")
        self.assertFalse(claim["verified_success"])

    def test_capped_search_is_unknown_not_unsat_proof(self):
        fake = {"verdict": "NO_KNOWN_WIN", "witness": None,
                "complete": False, "nodes_explored": 1}
        with mock.patch.object(checker, "search", return_value=fake):
            receipt = checker.adjudicate(dut.make_case(2), {"kind": "CLAIM_UNSAT"})
        self.assertEqual(receipt["verdict"], "NO_KNOWN_WIN")
        self.assertEqual(receipt["review"]["reason"], "SEARCH_INCOMPLETE")
        self.assertFalse(receipt["verified_success"])

    def test_public_receipt_verifier_quarantines_internal_oracle_exceptions(self):
        case = dut.make_case(0)
        plan = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
        plain = checker.adjudicate(case, plan)
        self.assertTrue(checker.verify_receipt_replay(plain))
        with mock.patch.object(checker, "adjudicate",
                               side_effect=RuntimeError("post oracle crash")):
            self.assertFalse(checker.verify_receipt_replay(plain))
        session = dut.ProbeSession(case)
        trace = session.submit(plan)
        self.assertTrue(checker.verify_receipt_replay(trace))
        # Independent verifier must NOT depend on its emitter; if the emitter
        # is missing or crashes later, already valid receipts still replay.
        with mock.patch.object(checker, "adjudicate_session",
                               side_effect=RuntimeError("post session replay crash")):
            self.assertTrue(checker.verify_receipt_replay(trace))

    def test_session_submit_quarantines_post_oracle_verification_crash(self):
        session = dut.ProbeSession(dut.make_case(0))
        with mock.patch.object(checker, "verify_receipt_replay",
                               side_effect=RuntimeError("receipt verify crash")):
            outcome = session.submit({"kind": "PLAN", "actions":
                                      list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(outcome["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertFalse(outcome["verified_success"])

    def test_sixteenth_probe_permitted_seventeenth_terminally_blocks(self):
        case = dut.make_case(0)
        valid = dut.ProbeSession(case)
        for _ in range(dut.MAX_PROBES):
            view = valid.probe("RESET")
        self.assertEqual(view["remaining_probes"], 0)
        self.assertFalse(view["finished"])
        win = valid.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(win["verdict"], "VERIFIED_WIN")
        self.assertEqual(win["probe_count"], dut.MAX_PROBES)
        self.assertTrue(checker.verify_receipt_replay(win))
        blocked = dut.ProbeSession(case)
        for _ in range(dut.MAX_PROBES):
            blocked.probe("RESET")
        after = blocked.probe("RESET")
        self.assertTrue(after["finished"])
        self.assertEqual(after["remaining_probes"], 0)
        self.assertEqual(blocked.submit({"kind": "PLAN", "actions":
                                        list(MEMORIZED_SEED0_PATH)})["verdict"],
                         "VERIFIED_FAILURE")

    def test_candidate_cannot_rewrite_transcript_even_with_all_checksums_updated(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        for action in ("0", "WAIT", "1", "RESET", "1"):
            session.probe(action)
        receipt = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertTrue(checker.verify_receipt_replay(receipt))
        fake_output = deepcopy(receipt)
        old = fake_output["probe_trace"][0]["after"]
        fake_output["probe_trace"][0]["after"] = "111" if old != "111" else "000"
        self.assertFalse(checker.verify_receipt_replay(_rehash_session(fake_output)))
        fake_flag = deepcopy(receipt)
        fake_flag["probe_trace"][0]["valid"] = False
        self.assertFalse(checker.verify_receipt_replay(_rehash_session(fake_flag)))

    def test_all_sixteen_charged_probe_rows_reject_forged_output_history(self):
        rng = random.Random(20310516)
        checked = 0
        for seed in range(20):
            case = dut.make_case(seed)
            session = dut.ProbeSession(case)
            for _ in range(dut.MAX_PROBES):
                session.probe(rng.choice(("0", "1", "WAIT", "RESET", "INVALID")))
            receipt = session.submit({"kind": "CLAIM_UNSAT"})
            self.assertEqual(receipt["probe_count"], dut.MAX_PROBES)
            self.assertTrue(checker.verify_receipt_replay(receipt))
            for index in range(dut.MAX_PROBES):
                modified = deepcopy(receipt)
                modified["probe_trace"][index]["before"] = "NOT_AN_OBSERVED_REGISTER"
                with self.subTest(seed=seed, charged_index=index):
                    self.assertFalse(checker.verify_receipt_replay(
                        _rehash_session(modified)))
                checked += 1
        self.assertEqual(checked, 320)

    def test_altered_rules_or_untyped_revision_cannot_become_formal_win(self):
        case = dut.make_case(0)
        bad = deepcopy(case)
        from dataclasses import replace
        bad = replace(bad, after=replace(bad.after, bias=(bad.after.bias + 1) % 8))
        with self.assertRaises(ValueError):
            checker.adjudicate(bad, {"kind": "CLAIM_UNSAT"})
        untyped = checker.adjudicate(case, {"kind": "NOVEL_RULE",
                                           "description": "Maybe the law changed again"})
        self.assertEqual(untyped["verdict"], "UNKNOWN")
        self.assertFalse(untyped["verified_success"])

    def test_completed_session_is_one_shot_and_result_immutable_to_caller(self):
        session = dut.ProbeSession(dut.make_case(0))
        receipt = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        baseline = deepcopy(receipt)
        receipt["verdict"] = "VERIFIED_FAILURE"
        self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"}), baseline)
        self.assertTrue(session.probe("RESET")["finished"])


if __name__ == "__main__":
    unittest.main()
