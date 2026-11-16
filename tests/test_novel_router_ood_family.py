"""Separate primitive-state oracle for graph-threshold OOD transfer family."""

from collections import Counter, deque
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import io
import json
import unittest
from unittest.mock import patch

from labyrinth import novel_router_ood_family as dut


# Reference tuple: bits, sealed, ticks, seals_used, post_seal_pulse.
def independent_initial(case):
    return (case.initial_bits, 0, 0, 0, False)


def independent_pulse(case, bits, sealed):
    # Explicitly compute every result bit from the SAME pre-pulse bitmask.
    outputs = []
    for node in range(5):
        if sealed & (1 << node):
            outputs.append((bits >> node) & 1)
        else:
            contributing = sum((bits >> j) & 1 for j in range(5)
                               if (case.parents[node] >> j) & 1)
            outputs.append(int(contributing >= case.thresholds[node]))
    return sum(bit << node for node, bit in enumerate(outputs))


def independent_next(case, state, action):
    bits, sealed, ticks, seals_used, post = state
    if ticks >= dut.MAX_MOVES:
        return None
    if action.startswith("TOGGLE_"):
        index = int(action.split("_")[-1])
        if not 0 <= index < 5 or sealed & (1 << index):
            return None
        return bits ^ (1 << index), sealed, ticks + 1, seals_used, post
    if action.startswith("SEAL_"):
        index = int(action.split("_")[-1])
        if (not 0 <= index < 5 or sealed & (1 << index) or
                seals_used >= case.max_seals):
            return None
        return bits, sealed | (1 << index), ticks + 1, seals_used + 1, post
    if action == "PULSE":
        updated = independent_pulse(case, bits, sealed)
        return (updated, sealed, ticks + 1, seals_used,
                post or seals_used == case.max_seals)
    raise AssertionError("not an OOD graph action")


def independent_goal(case, state):
    return (state[0] == case.target_bits and state[1] == case.target_sealed and
            state[3] == case.max_seals and state[4])


def independent_witness(case):
    @lru_cache(None)
    def solve(state):
        if state[2] >= dut.MAX_MOVES:
            return None
        if independent_goal(case, state):
            return ("COMMIT",)
        if state[2] >= dut.MAX_MOVES - 1:
            return None
        # Different order from source BFS: PULSE first, seals before toggles.
        order = ("PULSE", *reversed(dut.SEAL_ACTIONS),
                 *reversed(dut.TOGGLE_ACTIONS))
        for action in order:
            advanced = independent_next(case, state, action)
            if advanced is not None:
                suffix = solve(advanced)
                if suffix is not None:
                    return (action, *suffix)
        return None

    witness = solve(independent_initial(case))
    return witness, solve.cache_info().currsize


def independent_play(case, actions):
    if (type(actions) not in (tuple, list) or not 1 <= len(actions) <= dut.MAX_MOVES
            or actions[-1] != "COMMIT"):
        return False
    state = independent_initial(case)
    for action in actions[:-1]:
        if action not in dut.PLAN_ACTIONS[:-1]:
            return False
        state = independent_next(case, state, action)
        if state is None:
            return False
    return independent_goal(case, state)


class CausalLatchOODIndependentTests(unittest.TestCase):
    def test_seed_genesis_is_canonical_and_graph_nontrivial(self):
        families = set()
        for seed in range(200):
            case = dut.make_case(seed)
            data = sha256(f"causal-latch-ood.v1:{seed}".encode()).digest()
            expected_parents = tuple(
                ((data[node] & 31) | (1 << ((node + 1) % 5))) & ~(1 << node)
                for node in range(5))
            expected_thresholds = tuple(
                1 + data[5 + node] % expected_parents[node].bit_count()
                for node in range(5))
            self.assertEqual(case.parents, expected_parents)
            self.assertEqual(case.thresholds, expected_thresholds)
            self.assertEqual(case.max_seals, 2)
            self.assertEqual(case.target_sealed.bit_count(), 2)
            self.assertTrue(all(mask & (1 << node) == 0 for node, mask in
                                enumerate(case.parents)))
            families.add((case.parents, case.thresholds))
        self.assertGreater(len(families), 50)
        print("OOD_DISTINCT_GRAPH_RULES_200", len(families))

    def test_public_seed_block_identity_leakage_is_explicitly_measured(self):
        # This is a source-public block, not a secret hosted experiment. A
        # memorizer that *knows* [100,200) can precompute public fingerprints.
        heldout = [dut.make_case(seed) for seed in range(100, 200)]
        keys = [(c.initial_bits, c.target_bits, c.target_sealed, c.max_seals)
                for c in heldout]
        counts = Counter(keys)
        identifiable = sum(counts[key] == 1 for key in keys)
        self.assertEqual(identifiable, 100)
        # Unique in this inspected 100-case catalogue is NOT a claim of
        # identity recovery over the unknown 2**31 seed universe.
        print("OOD_BOUNDED_CATALOGUE_IDENTITY_LEAK", {
            "catalogue": "[100,200)", "cases": 100,
            "identifiable_from_initial_public_fields": identifiable,
        })

    def test_synchronous_counterexample_and_irreversible_seal(self):
        witness = None
        for seed in range(30):
            case = dut.make_case(seed)
            for bits in range(32):
                from_snapshot = independent_pulse(case, bits, 0)
                sequential = bits
                for node in range(5):
                    active = sum((sequential >> i) & 1 for i in range(5)
                                 if (case.parents[node] >> i) & 1)
                    sequential = ((sequential | (1 << node)) if
                                  active >= case.thresholds[node] else
                                  (sequential & ~(1 << node)))
                if from_snapshot != sequential:
                    witness = (seed, bits, from_snapshot, sequential)
                    break
            if witness is not None:
                break
        self.assertIsNotNone(witness)
        case = dut.make_case(2)
        state = dut.step(case, dut.initial_state(case), "SEAL_0")
        with self.assertRaises(ValueError):
            dut.step(case, state, "TOGGLE_0")
        after_pulse = dut.step(case, state, "PULSE")
        self.assertEqual(after_pulse.bits & 1, state.bits & 1)
        print("OOD_SYNCHRONOUS_VS_SEQUENTIAL", witness)

    def test_independent_dp_200_case_outcomes_and_alternate_wins(self):
        counts = Counter()
        max_reference = max_source = 0
        for seed in range(200):
            case = dut.make_case(seed)
            independent, explored = independent_witness(case)
            max_reference = max(max_reference, explored)
            source = dut.search(case)
            max_source = max(max_source, source["nodes_explored"])
            self.assertTrue(source["complete"], seed)
            if independent is None:
                counts["UNSAT"] += 1
                self.assertEqual(source["verdict"],
                                 "PROVEN_UNSAT_UNDER_DECLARED_RULES", seed)
                self.assertIsNone(source["witness"])
            else:
                counts["WIN"] += 1
                self.assertEqual(source["verdict"], "VERIFIED_WIN", seed)
                self.assertTrue(independent_play(case, independent))
                self.assertTrue(independent_play(case, source["witness"]))
                self.assertEqual(dut.verify_plan(case, list(independent))["verdict"],
                                 "VERIFIED_WIN")
        self.assertGreater(counts["WIN"], 0)
        self.assertGreater(counts["UNSAT"], 0)
        self.assertEqual(counts["WIN"] + counts["UNSAT"], 200)
        case = dut.make_case(3)
        plan_a = ["SEAL_0", "SEAL_4", "PULSE", "COMMIT"]
        plan_b = ["SEAL_4", "SEAL_0", "PULSE", "COMMIT"]
        self.assertTrue(independent_play(case, plan_a))
        self.assertTrue(independent_play(case, plan_b))
        self.assertEqual(dut.adjudicate(case, {"kind": "PLAN", "actions": plan_b})
                         ["verdict"], "VERIFIED_WIN")
        print("OOD_DP200", dict(counts), {"max_reference_states": max_reference,
                                         "max_source_nodes": max_source})

    def test_reference_all_reachable_state_transitions_30_seeds(self):
        checks = 0
        for seed in range(30):
            case = dut.make_case(seed)
            initial = independent_initial(case)
            frontier = deque([initial])
            seen = {initial}
            while frontier:
                state = frontier.popleft()
                as_source = dut.State(*state)
                for action in dut.PLAN_ACTIONS[:-1]:
                    expected = independent_next(case, state, action)
                    try:
                        actual = dut.step(case, as_source, action)
                    except ValueError:
                        self.assertIsNone(expected, (seed, state, action))
                    else:
                        self.assertIsNotNone(expected, (seed, state, action))
                        self.assertEqual((actual.bits, actual.sealed, actual.ticks,
                                          actual.seals_used, actual.post_seal_pulse),
                                         expected, (seed, state, action))
                    if expected is not None and expected not in seen:
                        seen.add(expected)
                        frontier.append(expected)
                    checks += 1
        self.assertGreater(checks, 1_000)
        print("OOD_INDEPENDENT_TRANSITIONS", {"seeds": 30, "checks": checks})

    def test_declared_resources_and_incomplete_search_never_forge_unsat(self):
        for seed in range(60):
            capped = dut.search(dut.make_case(seed), max_nodes=1)
            self.assertEqual(capped["verdict"], "NO_KNOWN_WIN")
            self.assertFalse(capped["complete"])
        case = dut.make_case(3)
        self.assertEqual(dut.verify_plan(case, ["SEAL_0", "TOGGLE_0", "COMMIT"])
                         ["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(dut.verify_plan(case, ["COMMIT", "PULSE"])["verdict"],
                         "VERIFIED_FAILURE")
        self.assertFalse(dut.adjudicate(case, {"kind": "NOVEL_RULE",
                    "description": "A hidden edge might behave differently."})["verified_success"])
        forged = replace(case, target_bits=float(case.target_bits))
        self.assertFalse(dut._valid_case(forged))
        with self.assertRaises(ValueError):
            dut.adjudicate(forged, {"kind": "CLAIM_UNSAT"})

    def test_probe_budget_legal_boundary_and_unqualified_receipt(self):
        case = dut.make_case(2)
        probe = dut.ProbeSession(case)
        view = probe.view()
        self.assertTrue({"seed", "parents", "thresholds", "case_id"}.isdisjoint(view))
        for count in range(dut.MAX_PROBES):
            current = probe.probe("PULSE")
            self.assertEqual(len(current["observations"]), count + 1)
        receipt = probe.submit({"kind": "PLAN", "actions": ["COMMIT"]})
        self.assertEqual(receipt["probe_count"], dut.MAX_PROBES)
        self.assertFalse(receipt["private_custody"])
        self.assertFalse(receipt["neural_learning_verified"])
        self.assertFalse(receipt["graduation_eligible"])
        self.assertEqual(receipt["evidence_class"], "SOURCE_PUBLIC_SYNTHETIC")
        self.assertEqual(probe.submit({"kind": "CLAIM_UNSAT"}), receipt)

        over = dut.ProbeSession(case)
        for _ in range(dut.MAX_PROBES):
            over.probe("PULSE")
        over.probe("PULSE")
        self.assertEqual(over.submit({"kind": "PLAN", "actions": ["COMMIT"]})
                         ["reason"], "PROBE_BUDGET_EXCEEDED")

    def test_cli_runs_a_real_public_case_and_validates_witness(self):
        plan = dut.search(dut.make_case(2))["witness"]
        self.assertTrue(plan)
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            self.assertEqual(dut.main(["--seed", "2", "--plan", *plan]), 0)
        receipt = json.loads(buffer.getvalue())
        self.assertEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertFalse(receipt["private_custody"])

    def test_corrupted_shared_scorer_or_state_constructor_cannot_forge_win(self):
        case = dut.make_case(2)
        plan = {"kind": "PLAN", "actions": ["COMMIT"]}
        self.assertEqual(dut.adjudicate(case, plan)["verdict"], "VERIFIED_FAILURE")
        with patch.object(dut, "verify_plan", return_value={
                "verdict": "VERIFIED_WIN", "reason": "INJECTED"}):
            result = dut.adjudicate(case, plan)
            self.assertEqual(result["verdict"], "UNKNOWN")
            self.assertFalse(result["verified_success"])

        original = dut.State.__init__

        def forged_genesis(self, bits, sealed=0, ticks=0, seals_used=0,
                           post_seal_pulse=False, committed=False, won=False):
            return original(self, case.target_bits, case.target_sealed,
                            ticks, case.max_seals, True, committed, won)

        with patch.object(dut.State, "__init__", forged_genesis):
            result = dut.adjudicate(case, plan)
            self.assertNotEqual(result["verdict"], "VERIFIED_WIN")
            self.assertFalse(result["verified_success"])
            self.assertEqual(dut.search(case)["verdict"], "UNKNOWN")
            with self.assertRaises(ValueError):
                dut.ProbeSession(case)

    def test_corrupted_step_cannot_prove_unsat_for_known_winnable_case(self):
        case = dut.make_case(2)
        self.assertIsNotNone(independent_witness(case)[0])
        with patch.object(dut, "step", side_effect=lambda c, state, action: state):
            claimed = dut.search(case)
            self.assertEqual(claimed["verdict"], "UNKNOWN")
            self.assertFalse(claimed["complete"])
            review = dut.adjudicate(case, {"kind": "CLAIM_UNSAT"})
            self.assertEqual(review["verdict"], "VERIFIED_FAILURE")
            self.assertTrue(independent_play(case, review["counterexample"]))
            session = dut.ProbeSession(case)
            session.probe("TOGGLE_0")
            self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"})["reason"],
                             "INDEPENDENT_TRANSITION_DISAGREEMENT")

    def test_submission_cannot_promote_source_scorer_fake_success(self):
        case = dut.make_case(2)
        proposal = {"kind": "PLAN", "actions": ["COMMIT"]}
        self.assertEqual(dut.adjudicate(case, proposal)["verdict"],
                         "VERIFIED_FAILURE")
        forged = dict(dut.adjudicate(case, proposal))
        forged["verdict"] = "VERIFIED_WIN"
        forged["verified_success"] = True
        forged["reason"] = "INJECTED_SHARED_SCORER_WIN"
        unsigned = {key: value for key, value in forged.items() if key != "sha256"}
        forged["sha256"] = sha256(dut._canonical(unsigned).encode()).hexdigest()
        session = dut.ProbeSession(case)
        with patch.object(dut, "adjudicate", return_value=forged):
            receipt = session.submit(proposal)
        self.assertFalse(receipt["verified_success"])
        self.assertNotEqual(receipt["verdict"], "VERIFIED_WIN")
        self.assertTrue(session.view()["finished"])

    def test_reset_cannot_silently_import_corrupt_constructor_state(self):
        case = dut.make_case(2)
        session = dut.ProbeSession(case)
        malformed = dut.State(case.initial_bits ^ 1)
        with patch.object(dut, "initial_state", return_value=malformed):
            view = session.probe("RESET")
        self.assertTrue(view["finished"])
        self.assertEqual(view["current_bits"], f"{case.initial_bits:05b}")
        result = session.submit({"kind": "PLAN", "actions": ["COMMIT"]})
        self.assertFalse(result["verified_success"])
        self.assertIn("INDEPENDENT", result["reason"])

    def test_probe_and_submit_faults_remain_charged_terminal_events(self):
        case = dut.make_case(2)
        for fault in (RuntimeError("source probe crashed"), object()):
            session = dut.ProbeSession(case)
            with patch.object(dut, "step", side_effect=fault if isinstance(
                    fault, Exception) else None, return_value=fault):
                view = session.probe("TOGGLE_0")
            self.assertTrue(view["finished"])
            self.assertEqual(len(view["observations"]), 1)
            self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"})["reason"],
                             "INDEPENDENT_TRANSITION_DISAGREEMENT")

        submission = dut.ProbeSession(case)
        with patch.object(dut, "adjudicate", side_effect=RuntimeError("scorer crash")):
            failed = submission.submit({"kind": "CLAIM_UNSAT"})
        self.assertTrue(submission.view()["finished"])
        self.assertFalse(failed["verified_success"])
        self.assertEqual(failed["reason"], "BLOCKED_SUBMISSION_EVALUATOR")
        self.assertEqual(submission.submit({"kind": "PLAN", "actions": ["COMMIT"]}),
                         failed)


if __name__ == "__main__":
    unittest.main()
