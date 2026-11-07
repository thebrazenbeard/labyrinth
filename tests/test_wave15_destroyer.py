"""Wave15 independent hostile receipt and finite Switchback admission tests.

Pinned target: 2345c9c69ad0882cb32c8a584a3d0677af986123.
Tests deliberately expose direct case-identity admission weakness, while
preserving negative controls for receipt replay and malformed inputs.
"""

from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import replace
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from labyrinth import switchback_circuit as dut
from labyrinth import switchback_oracle as oracle
from labyrinth.switchback_controls import MEMORIZED_SEED0_PATH


def resign(receipt):
    altered = deepcopy(receipt)
    unsigned = {k: v for k, v in altered.items() if k != "sha256"}
    altered["sha256"] = sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    return altered


def direct_finite_witness(case):
    """Separate bounded DUT graph walk; never call oracle.search or closure."""
    start = dut.State(case.initial)
    frontier = deque([(start, ())])
    seen = {(start, 0)}
    while frontier:
        state, path = frontier.popleft()
        if len(path) >= dut.MAX_MOVES:
            continue
        for action in dut.CONTROLS:
            next_state = dut.step(case, state, action)
            next_path = (*path, action)
            if next_state.committed:
                if next_state.won:
                    return list(next_path)
            elif next_state != state and (next_state, len(next_path)) not in seen:
                seen.add((next_state, len(next_path)))
                frontier.append((next_state, next_path))
    return None


class Wave15Destroyer(unittest.TestCase):
    def test_mutated_case_eq_cannot_admit_forged_winning_rules(self):
        """Exact primitive provenance must not depend on Case.__eq__."""
        canonical = dut.make_case(0)
        losing_plan = ["0", "0", "WAIT", "1", "COMMIT"]
        self.assertEqual(oracle.verify_plan(canonical, losing_plan)["verdict"],
                         "VERIFIED_FAILURE")
        # This plan ends at register 1 whereas the canonical target is 0.
        fabricated = replace(canonical, target=1)
        self.assertNotEqual(fabricated.target, canonical.target)
        with patch.object(dut.Case, "__eq__", return_value=True):
            admission = oracle._valid_case(fabricated)
            parity = oracle.independent_transition_parity(fabricated)
            with self.assertRaisesRegex(ValueError, "forged case"):
                oracle.adjudicate(fabricated, {"kind": "PLAN",
                                                    "actions": losing_plan})
            self.assertEqual((admission, parity), (False, False))

    def test_cross_seed_and_case_identity_receipt_replay_rejected(self):
        canonical = oracle.adjudicate(dut.make_case(0),
                 {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertTrue(oracle.verify_receipt_replay(canonical))
        wrong_seed = deepcopy(canonical)
        wrong_seed["seed"] = 1
        self.assertFalse(oracle.verify_receipt_replay(resign(wrong_seed)))
        wrong_id = deepcopy(canonical)
        wrong_id["case_id"] = dut.make_case(1).case_id
        self.assertFalse(oracle.verify_receipt_replay(resign(wrong_id)))
        wrong_mechanic = deepcopy(canonical)
        wrong_mechanic["mechanic_signature"] = dut.make_case(1).mechanic_signature
        self.assertFalse(oracle.verify_receipt_replay(resign(wrong_mechanic)))

    def test_nested_session_receipt_tampering_rejected_even_if_resigned(self):
        session = dut.ProbeSession(dut.make_case(0))
        session.probe("1")
        receipt = session.submit({"kind": "PLAN", "actions":
                                 list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertTrue(oracle.verify_receipt_replay(receipt))
        for candidate in (
                lambda r: r.update(verdict="PROVEN_UNSAT_UNDER_DECLARED_RULES"),
                lambda r: r["adjudication"]["review"].update(reason="FORGED"),
                lambda r: r["probe_trace"][0].update(action="RESET"),
                lambda r: r["adjudication"].update(source_commit="FABRICATED")):
            forged = deepcopy(receipt)
            candidate(forged)
            self.assertFalse(oracle.verify_receipt_replay(resign(forged)))

    def test_malformed_plans_never_certify_win(self):
        case = dut.make_case(0)
        for candidate in (
                [], ["COMMIT", "WAIT"], ["WAIT"] * dut.MAX_MOVES + ["COMMIT"],
                ["0", True, "COMMIT"], ["0", 1, "COMMIT"],
                ["0", "ＣＯＭＭＩＴ"], ["WAIT", "WAIT"],
                ["0"] * dut.MAX_MOVES, "0", ("0", "COMMIT")):
            with self.subTest(actions=candidate):
                review = oracle.verify_plan(case, candidate)
                self.assertNotEqual(review["verdict"], "VERIFIED_WIN")
                if type(candidate) is tuple:
                    with self.assertRaises(ValueError):
                        oracle.adjudicate(case, {"kind": "PLAN", "actions": candidate})
                else:
                    emitted = oracle.adjudicate(case, {"kind": "PLAN",
                                                      "actions": candidate})
                    self.assertFalse(emitted["verified_success"])

    def test_false_unsat_is_rejected_against_direct_dut_witness(self):
        sat, unsat = 0, 0
        for seed in range(24):
            case = dut.make_case(seed)
            witness = direct_finite_witness(case)
            receipt = oracle.adjudicate(case, {"kind": "CLAIM_UNSAT"})
            if witness is None:
                unsat += 1
                self.assertEqual(receipt["verdict"],
                                 "PROVEN_UNSAT_UNDER_DECLARED_RULES")
            else:
                sat += 1
                self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
                win = oracle.adjudicate(case, {"kind": "PLAN", "actions": witness})
                self.assertEqual(win["verdict"], "VERIFIED_WIN")
                self.assertTrue(oracle.verify_receipt_replay(win))
            self.assertTrue(oracle.verify_receipt_replay(receipt))
        self.assertGreater(sat, 0)
        self.assertGreater(unsat, 0)

    def test_mutable_proposal_and_probe_inputs_do_not_rewrite_receipts(self):
        case = dut.make_case(0)
        proposal = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
        original = deepcopy(proposal)
        core = oracle.adjudicate(case, proposal)
        proposal["actions"][0] = "COMMIT"
        self.assertEqual(core["proposal"], original)
        self.assertTrue(oracle.verify_receipt_replay(core))
        session = dut.ProbeSession(case)
        session.probe("1")
        trace = session.view()["observations"]
        receipt = oracle.adjudicate_session(case, trace, original)
        trace[0]["after"] = "111"
        self.assertTrue(oracle.verify_receipt_replay(receipt))
        self.assertNotEqual(receipt["probe_trace"], trace)


if __name__ == "__main__":
    unittest.main()
