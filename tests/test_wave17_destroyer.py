"""Independent Wave17 hostile novel-router evaluator/custody tests.

Source target: 13a5d122a3c5a577023a38a6e32c891676362471.
Tests are source-level fault injection; no model, host service, or network.
"""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from labyrinth import switchback_novel_transfer as dut


class Wave17Destroyer(unittest.TestCase):
    def test_shared_state_constructor_cannot_certify_false_win(self):
        case = dut.make_case(16)
        losing = ["LOAD", "ROUTE", "TICK", "LOAD", "TICK", "COMMIT"]
        self.assertEqual(dut.verify_plan(case, losing)["verdict"], "VERIFIED_FAILURE")
        original = dut.State.__init__

        def shifted(self, left, right, orientation, pending=0, ticks=0,
                    loads=0, routes=0, post_delivery=False, committed=False, won=False):
            original(self, (left + 1) % 4, right, orientation, pending, ticks,
                     loads, routes, post_delivery, committed, won)

        with patch.object(dut.State, "__init__", new=shifted):
            forged = dut.adjudicate(case, {"kind": "PLAN", "actions": losing})
        self.assertFalse(forged["verified_success"])
        self.assertNotEqual(forged["verdict"], "VERIFIED_WIN")

    def test_transition_exception_charges_attempt_and_seals_session(self):
        session = dut.ProbeSession(dut.make_case(0))
        with patch.object(dut, "step", side_effect=RuntimeError("injected step fault")):
            outcome = session.probe("LOAD")
        self.assertTrue(outcome["finished"])
        self.assertEqual(outcome["remaining_probes"], dut.MAX_PROBES - 1)
        self.assertEqual(outcome["observations"][0]["action"], "LOAD")
        self.assertFalse(session.submit({"kind": "PLAN", "actions": ["COMMIT"]})[
            "verified_success"])

    def test_reset_constructor_fault_charges_attempt_and_seals_session(self):
        session = dut.ProbeSession(dut.make_case(0))
        with patch.object(dut, "initial_state", side_effect=RuntimeError("reset broken")):
            outcome = session.probe("RESET")
        self.assertTrue(outcome["finished"])
        self.assertEqual(outcome["remaining_probes"], dut.MAX_PROBES - 1)
        self.assertEqual(outcome["observations"][0]["action"], "RESET")
        self.assertFalse(session.submit({"kind": "PLAN", "actions": ["COMMIT"]})[
            "verified_success"])

    def test_mutated_adjudicator_cannot_self_sign_false_session_win(self):
        session = dut.ProbeSession(dut.make_case(0))
        proposal = {"kind": "PLAN", "actions": ["COMMIT"]}
        fake = {"verdict": "VERIFIED_WIN", "verified_success": True}
        with patch.object(dut, "adjudicate", return_value=fake):
            receipt = session.submit(proposal)
        self.assertFalse(receipt["verified_success"])
        self.assertNotEqual(receipt["verdict"], "VERIFIED_WIN")

    def test_case_equality_spoof_cannot_admit_forged_identity(self):
        case = dut.make_case(0)
        forged = replace(case, target_left=(case.target_left + 1) % 4)
        with patch.object(dut.Case, "__eq__", return_value=True):
            self.assertFalse(dut._valid_case(forged))
            with self.assertRaises(ValueError):
                dut.adjudicate(forged, {"kind": "CLAIM_UNSAT"})

    def test_rule_equality_spoof_cannot_change_seed_genesis(self):
        expected = [dut.make_case(seed) for seed in range(20)]
        with patch.object(dut.RoutingRule, "__eq__", return_value=True):
            actual = [dut.make_case(seed) for seed in range(20)]
            for left, right in zip(expected, actual):
                self.assertEqual(left.mechanic_signature, right.mechanic_signature)
                self.assertEqual(left.case_id, right.case_id)

    def test_invalid_probe_actions_consume_budget_without_escape(self):
        session = dut.ProbeSession(dut.make_case(0))
        for _ in range(dut.MAX_PROBES):
            view = session.probe("NOT_A_ROUTER_ACTION")
        self.assertFalse(view["finished"])
        self.assertEqual(view["remaining_probes"], 0)
        denied = session.probe("NOT_A_ROUTER_ACTION")
        self.assertTrue(denied["finished"])
        self.assertFalse(session.submit({"kind": "PLAN", "actions": ["COMMIT"]})[
            "verified_success"])

    def test_valid_alternate_win_and_false_unsat_are_distinguished(self):
        case = dut.make_case(0)
        alternate = ["ROUTE", "TICK", "LOAD", "TICK", "COMMIT"]
        self.assertEqual(dut.verify_plan(case, alternate)["verdict"], "VERIFIED_WIN")
        win = dut.adjudicate(case, {"kind": "PLAN", "actions": alternate})
        self.assertTrue(win["verified_success"])
        self.assertEqual(len(win["sha256"]), 64)
        claim = dut.adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertEqual(claim["verdict"], "VERIFIED_FAILURE")
        self.assertFalse(claim["verified_success"])

    def test_single_false_unsat_search_claim_rejected_by_independent_dfs(self):
        case = dut.make_case(0)
        fake = {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "complete": True,
                "witness": None, "nodes_explored": 1}
        with patch.object(dut, "search", return_value=fake):
            receipt = dut.adjudicate(case, {"kind": "CLAIM_UNSAT"})
        self.assertNotEqual(receipt["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(receipt["verified_success"])

    def test_single_false_plan_scorer_claim_rejected_by_independent_replay(self):
        case = dut.make_case(0)
        with patch.object(dut, "verify_plan", return_value={"verdict": "VERIFIED_WIN"}):
            receipt = dut.adjudicate(case, {"kind": "PLAN", "actions": ["COMMIT"]})
        self.assertEqual(receipt["verdict"], "UNKNOWN")
        self.assertFalse(receipt["verified_success"])

    def test_valid_reset_restores_public_initial_state_and_charges_probe(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        session.probe("LOAD")
        after = session.probe("RESET")
        self.assertEqual(after["current_levels"],
                         [case.initial_left, case.initial_right])
        self.assertEqual(after["remaining_probes"], dut.MAX_PROBES - 2)
        self.assertEqual([item["action"] for item in after["observations"]],
                         ["LOAD", "RESET"])

    def test_public_view_and_receipt_are_detached_from_callers(self):
        session = dut.ProbeSession(dut.make_case(0))
        initial = session.view()
        initial["target_levels"][0] = 999
        self.assertNotIn(999, session.view()["target_levels"])
        viewed = session.probe("LOAD")
        viewed["observations"][0]["after"][0] = 999
        self.assertNotIn(999, session.view()["observations"][0]["after"])
        receipt = session.submit({"kind": "PLAN", "actions": ["COMMIT"]})
        receipt["adjudication"]["verdict"] = "VERIFIED_WIN"
        self.assertNotEqual(session.submit({"kind": "CLAIM_UNSAT"})["verdict"],
                            "VERIFIED_WIN")


if __name__ == "__main__":
    unittest.main()
