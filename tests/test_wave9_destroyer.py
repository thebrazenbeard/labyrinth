"""Independent Switchback Circuit falsifiers, exact source 0021d055.

RED tests characterize public probe budget/fail-closed violations. Passing
exploit controls expose published-source leakage, not trained reasoning.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from labyrinth import switchback_circuit as dut
from labyrinth.switchback_controls import MEMORIZED_SEED0_PATH
from labyrinth import switchback_oracle as witness
from labyrinth.switchback_oracle import adjudicate, search, verify_probe_trace


class SwitchbackDestroyer(unittest.TestCase):
    def test_forged_complete_unsat_search_cannot_defeat_independent_closure(self):
        case = dut.make_case(0)  # witness already known to exist
        bogus = {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES",
                 "witness": None, "nodes_explored": 1, "complete": True}
        with patch.object(witness, "search", return_value=bogus):
            receipt = witness.adjudicate(case, {"kind": "CLAIM_UNSAT"})
            self.assertEqual(receipt["verdict"], "VERIFIED_FAILURE")
            self.assertEqual(receipt["review"]["reason"], "COUNTEREXAMPLE_WIN_EXISTS")
            self.assertTrue(witness.verify_receipt_replay(receipt))

    def test_reset_constructor_exception_is_charged_and_quarantined(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        with patch.object(dut, "State", side_effect=RuntimeError("reset constructor failed")):
            displayed = session.probe("RESET")
        self.assertTrue(displayed["finished"])
        self.assertEqual(displayed["remaining_probes"], dut.MAX_PROBES - 1)
        self.assertEqual(len(displayed["observations"]), 1)
        self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"})["verdict"],
                         "EVALUATOR_DISAGREEMENT")

    def test_probe_pulse_quota_distinguishes_syntax_from_effect(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        for _ in range(case.max_pulses):
            session.probe("0")
        displayed = session.probe("0")
        denied = displayed["observations"][-1]
        self.assertTrue(denied["valid"])
        self.assertFalse(denied["applied"])
        self.assertEqual(displayed["pulses_used"], case.max_pulses)
        self.assertEqual(displayed["remaining_probes"], dut.MAX_PROBES - case.max_pulses - 1)

    def test_reset_mutation_must_not_escape_continuous_oracle_parity(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        session.probe("0")
        genuine_state_class = dut.State
        with patch.object(dut, "State", side_effect=lambda initial: genuine_state_class(
                (initial + 1) % 8)):
            displayed = session.probe("RESET")
        self.assertFalse(verify_probe_trace(case, displayed["observations"]))
        self.assertTrue(displayed["finished"])
        self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"})["verdict"],
                         "EVALUATOR_DISAGREEMENT")

    def test_independent_adjudicator_exception_must_seal_error_receipt(self):
        session = dut.ProbeSession(dut.make_case(0))
        with patch.object(witness, "adjudicate",
                          side_effect=RuntimeError("independent verifier fault")):
            result = session.submit({"kind": "CLAIM_UNSAT"})
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(session.view()["finished"])
        self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"}), result)

    def test_malformed_transition_output_must_not_poison_evaluator_state(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        with patch.object(dut, "step", return_value="NOT_A_STATE"):
            session.probe("0")
        self.assertIs(type(session._state), dut.State)
        self.assertTrue(session.view()["finished"])
        self.assertEqual(session.view()["remaining_probes"], dut.MAX_PROBES - 1)

    def test_faulty_transition_must_not_expose_unverified_observation(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        with patch.object(dut, "step", side_effect=lambda case, state, action: state):
            displayed = session.probe("0")
        self.assertFalse(verify_probe_trace(case, displayed["observations"]))
        # The host already has an independently reproducible counterexample;
        # continuing to show unfinished task feedback contaminates adaptation.
        self.assertTrue(displayed["finished"])
        result = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertEqual(result["verdict"], "EVALUATOR_DISAGREEMENT")

    def test_engine_exception_must_seal_session_and_charge_valid_probe(self):
        session = dut.ProbeSession(dut.make_case(0))
        with patch.object(dut, "step", side_effect=RuntimeError("injected transition fault")):
            session.probe("0")
        view = session.view()
        self.assertTrue(view["finished"])
        self.assertEqual(view["remaining_probes"], dut.MAX_PROBES - 1)
        self.assertEqual(len(view["observations"]), 1)
        result = session.submit({"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)})
        self.assertIn(result["verdict"], ("EVALUATOR_DISAGREEMENT", "EVALUATOR_ERROR"))

    def test_final_submission_after_exactly_maximum_probes_is_adjudicated(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        view = {}
        for _ in range(dut.MAX_PROBES):
            view = session.probe("RESET")
        self.assertEqual(view["remaining_probes"], 0)
        self.assertTrue(verify_probe_trace(case, view["observations"]))
        proposal = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
        self.assertEqual(adjudicate(case, proposal)["verdict"], "VERIFIED_WIN")
        self.assertEqual(session.submit(proposal)["verdict"], "VERIFIED_WIN")

    def test_public_metadata_enables_84_of_100_known_heldout_seed_recoveries(self):
        mapping = defaultdict(list)
        for seed in range(100, 200):
            case = dut.make_case(seed)
            mapping[json.dumps(dut.public_view(case), sort_keys=True)].append(seed)
        unambiguous = sum(len(seeds) == 1 for seeds in mapping.values())
        self.assertEqual(unambiguous, 84)
        self.assertEqual(len(mapping), 92)

    def test_one_public_probe_identifies_all_100_known_heldout_private_seeds(self):
        def observed_key(seed):
            case = dut.make_case(seed)
            initial = json.dumps(dut.public_view(case), sort_keys=True)
            # One candidate-visible intervention resolves collisions.
            response = dut.ProbeSession(case).probe("1")["current_register"]
            return initial, response

        dictionary = {observed_key(seed): seed for seed in range(100, 200)}
        self.assertEqual(len(dictionary), 100)
        for seed in range(100, 200):
            self.assertEqual(dictionary[observed_key(seed)], seed)

    def test_independent_dut_reachability_matches_oracle_unsat_100_seeds(self):
        def can_win(case):
            frontier = {dut.State(case.initial)}
            for _ in range(dut.MAX_MOVES):
                if any(dut.step(case, state, "COMMIT").won for state in frontier):
                    return True
                next_states = set()
                for state in frontier:
                    for action in ("0", "1", "WAIT"):
                        after = dut.step(case, state, action)
                        if after != state:
                            next_states.add(after)
                frontier = next_states
            return False

        verdicts = []
        for seed in range(100):
            case = dut.make_case(seed)
            witness_exists = can_win(case)
            actual = search(case)["verdict"]
            self.assertEqual(witness_exists, actual == "VERIFIED_WIN", seed)
            verdicts.append(actual)
        self.assertEqual(verdicts.count("PROVEN_UNSAT_UNDER_DECLARED_RULES"), 33)

    def test_candidate_receipts_and_observations_return_detached_copies(self):
        session = dut.ProbeSession(dut.make_case(0))
        first = session.probe("0")
        first["observations"][0]["action"] = "WAIT"
        first["probe_actions"].clear()
        first["plan_actions"].clear()
        self.assertEqual(session.view()["observations"][0]["action"], "0")
        proposal = {"kind": "PLAN", "actions": list(MEMORIZED_SEED0_PATH)}
        receipt = session.submit(proposal)
        original = deepcopy(receipt)
        receipt["verdict"] = "FORGED"
        self.assertEqual(session.submit(proposal), original)


if __name__ == "__main__":
    unittest.main()
