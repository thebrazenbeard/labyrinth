"""Independent hostile oracle for the source-public delayed-packet challenge.

No production transition, search or rule helpers are called by the reference
solver. A separate tuple-state model and backward finite recurrence provide
the differential witness for all generated cases under test.
"""

from collections import Counter, defaultdict, deque
from dataclasses import replace
from functools import lru_cache
import unittest
from unittest.mock import patch

from labyrinth import switchback_novel_transfer as dut

# The novel module must be runnable on canonical main even if the earlier
# Switchback examination is still an unmerged draft. Preserve its published
# action contract as an explicit reference; verify live source if installed.
EXPECTED_OLD_CONTROLS = ("0", "1", "WAIT", "COMMIT")
try:
    from labyrinth.switchback_circuit import CONTROLS as OLD_CONTROLS
except ModuleNotFoundError as error:
    if error.name != "labyrinth.switchback_circuit":
        raise
    OLD_CONTROLS = EXPECTED_OLD_CONTROLS


# Tuple: left, right, orientation, pending, ticks, loads, routes, post_delivery.
def reference_initial(case):
    return (case.initial_left, case.initial_right, case.initial_orientation,
            0, 0, 0, 0, False)


def reference_transition(case, state, action):
    a, b, direction, pending, tick, used_loads, used_routes, post = state
    if tick >= dut.MAX_MOVES:
        return None
    rule = case.after if tick >= case.switch_at else case.before
    if action == "LOAD":
        if pending != 0 or used_loads >= case.max_loads:
            return None
        cargo = ((a + (b << 1) + direction + rule.payload_shift) % 3) + 1
        return a, b, direction, cargo, tick + 1, used_loads + 1, used_routes, post
    if action == "ROUTE":
        if used_routes >= case.max_routes:
            return None
        return a, b, direction ^ 1, pending, tick + 1, used_loads, used_routes + 1, post
    if action == "PURGE":
        if not pending:
            return None
        return a, b, direction, 0, tick + 1, used_loads, used_routes, post
    if action == "TICK":
        if not pending:
            return a, b, direction, 0, tick + 1, used_loads, used_routes, post
        output_port = (direction ^ rule.lane_bias ^ ((a ^ b) & 1))
        tanks = [a, b]
        tanks[output_port] = (tanks[output_port] + pending + rule.coupling) % 4
        tanks[1 - output_port] = (
            tanks[1 - output_port] + (1 if pending % 2 and rule.coupling % 2 else 0)) % 4
        return (tanks[0], tanks[1], direction, 0, tick + 1, used_loads,
                used_routes, post or tick >= case.switch_at)
    raise AssertionError("nonexistent reference action")


def reference_goal(case, state):
    a, b, _, pending, tick, _, _, post = state
    return (not pending and post and tick >= case.switch_at and
            a == case.target_left and b == case.target_right)


def reference_plan_wins(case, plan):
    if (type(plan) not in (tuple, list) or not 1 <= len(plan) <= dut.MAX_MOVES
            or plan[-1] != "COMMIT"):
        return False
    state = reference_initial(case)
    for action in plan[:-1]:
        if action not in dut.PLAN_ACTIONS[:-1]:
            return False
        state = reference_transition(case, state, action)
        if state is None:
            return False
    return reference_goal(case, state)


def reference_backward_witness(case):
    @lru_cache(None)
    def recurse(state):
        if state[4] >= dut.MAX_MOVES:
            return None
        if reference_goal(case, state):
            return ("COMMIT",)
        if state[4] >= dut.MAX_MOVES - 1:
            return None
        # Deliberately different action ordering from source forward BFS.
        for action in ("TICK", "PURGE", "ROUTE", "LOAD"):
            evolved = reference_transition(case, state, action)
            if evolved is not None:
                suffix = recurse(evolved)
                if suffix is not None:
                    return (action,) + suffix
        return None

    solution = recurse(reference_initial(case))
    return solution, recurse.cache_info().currsize


def independent_forward_reachable(case):
    initial = reference_initial(case)
    frontier = deque([initial])
    visited = {initial}
    while frontier:
        state = frontier.popleft()
        yield state
        for action in dut.PLAN_ACTIONS[:-1]:
            advanced = reference_transition(case, state, action)
            if advanced is not None and advanced not in visited:
                visited.add(advanced)
                frontier.append(advanced)


def as_dut_state(state):
    return dut.State(*state)


def matching_state(state):
    return (state.left, state.right, state.orientation, state.pending,
            state.ticks, state.loads, state.routes, state.post_delivery)


class NovelRouterIndependentMathTests(unittest.TestCase):
    def test_mechanically_new_public_actions_and_delayed_state(self):
        self.assertEqual(tuple(OLD_CONTROLS), EXPECTED_OLD_CONTROLS)
        self.assertNotEqual(set(dut.PLAN_ACTIONS), set(OLD_CONTROLS))
        self.assertTrue({"LOAD", "ROUTE", "TICK", "PURGE"}.isdisjoint(OLD_CONTROLS))
        self.assertGreater(4 * 4, 8)  # two visible four-level reservoirs
        distinguishable = None
        for seed in range(200):
            case = dut.make_case(seed)
            initial = reference_initial(case)
            loaded = reference_transition(case, initial, "LOAD")
            if loaded is None:
                continue
            self.assertEqual(loaded[:2], initial[:2])  # load is delayed
            self.assertNotEqual(loaded[3], 0)         # pending packet survives
            without_load = reference_transition(case, initial, "TICK")
            with_load = reference_transition(case, loaded, "TICK")
            if with_load[:2] != without_load[:2]:
                distinguishable = (seed, with_load[:2], without_load[:2])
                break
        self.assertIsNotNone(distinguishable)
        most_visible = max((len({state[:2] for state in
                                 independent_forward_reachable(dut.make_case(seed))}), seed)
                           for seed in range(100))
        # A *single* episode can reach every one of the 16 visible reservoir
        # pairs: more than the eight states of the old three-bit register.
        self.assertEqual(most_visible[0], 16)
        print("NOVEL_ROUTER_DELAYED_STATE", distinguishable)
        print("NOVEL_ROUTER_REACHABLE_PUBLIC_LEVELS", most_visible)

    def test_hidden_rules_are_distinguishable_through_authorized_probes(self):
        grouped = defaultdict(list)
        for seed in range(1000):
            case = dut.make_case(seed)
            metadata = (case.initial_left, case.initial_right,
                        case.initial_orientation, case.target_left,
                        case.target_right, case.switch_at,
                        case.max_loads, case.max_routes)
            grouped[metadata].append(case)
        discriminator = None
        for group in grouped.values():
            for index, first in enumerate(group):
                for second in group[index + 1:]:
                    if first.before == second.before and first.after == second.after:
                        continue
                    first_loaded = reference_transition(first, reference_initial(first),
                                                        "LOAD")
                    second_loaded = reference_transition(second, reference_initial(second),
                                                         "LOAD")
                    if first_loaded != second_loaded:
                        discriminator = (first.seed, second.seed, ("LOAD",))
                        break
                    first_delivered = reference_transition(first, first_loaded, "TICK")
                    second_delivered = reference_transition(second, second_loaded, "TICK")
                    if first_delivered != second_delivered:
                        discriminator = (first.seed, second.seed, ("LOAD", "TICK"))
                        break
                if discriminator is not None:
                    break
            if discriminator is not None:
                break
        self.assertIsNotNone(discriminator)
        print("NOVEL_ROUTER_PUBLIC_RULE_DISCRIMINATOR", discriminator)

    def test_independent_backward_oracle_matches_source_closure_across_200(self):
        counts = Counter()
        maximum_reference_states = 0
        maximum_author_nodes = 0
        for seed in range(200):
            case = dut.make_case(seed)
            independent, explored = reference_backward_witness(case)
            maximum_reference_states = max(maximum_reference_states, explored)
            author = dut.search(case)
            maximum_author_nodes = max(maximum_author_nodes, author["nodes_explored"])
            self.assertTrue(author["complete"], seed)
            if independent is None:
                counts["UNSAT"] += 1
                self.assertEqual(author["verdict"],
                                 "PROVEN_UNSAT_UNDER_DECLARED_RULES", seed)
                self.assertIsNone(author["witness"])
                self.assertEqual(dut.adjudicate(case, {"kind": "CLAIM_UNSAT"})["verdict"],
                                 "PROVEN_UNSAT_UNDER_DECLARED_RULES")
            else:
                counts["WIN"] += 1
                self.assertEqual(author["verdict"], "VERIFIED_WIN", seed)
                self.assertTrue(reference_plan_wins(case, independent))
                self.assertTrue(reference_plan_wins(case, author["witness"]))
                self.assertEqual(dut.verify_plan(case, list(independent))["verdict"],
                                 "VERIFIED_WIN")
                rejected = dut.adjudicate(case, {"kind": "CLAIM_UNSAT"})
                self.assertEqual(rejected["verdict"], "VERIFIED_FAILURE")
                self.assertTrue(reference_plan_wins(case, rejected["counterexample"]))
        self.assertGreater(counts["WIN"], 0)
        self.assertGreater(counts["UNSAT"], 0)
        print("NOVEL_ROUTER_DP200", dict(counts), {
            "max_reference_states": maximum_reference_states,
            "max_source_nodes": maximum_author_nodes,
        })

    def test_independent_all_reachable_transitions_40_cases(self):
        compared = 0
        for seed in range(40):
            case = dut.make_case(seed)
            for state in independent_forward_reachable(case):
                for action in dut.PLAN_ACTIONS[:-1]:
                    expected = reference_transition(case, state, action)
                    try:
                        actual = dut.step(case, as_dut_state(state), action)
                    except ValueError:
                        self.assertIsNone(expected, (seed, state, action))
                    else:
                        self.assertIsNotNone(expected, (seed, state, action))
                        self.assertEqual(matching_state(actual), expected,
                                         (seed, state, action))
                    compared += 1
        self.assertGreater(compared, 1_000)
        print("NOVEL_ROUTER_TRANSITION_PARITY", {"seeds": 40,
                                                 "state_action_checks": compared})

    def test_alternate_winning_plans_are_not_rejected_by_cached_solution(self):
        chosen = None
        for seed in range(30):
            case = dut.make_case(seed)
            primary = dut.search(case)["witness"]
            alternative, _ = reference_backward_witness(case)
            if (primary is not None and alternative is not None and
                    tuple(primary) != alternative):
                chosen = (seed, primary, alternative)
                self.assertTrue(reference_plan_wins(case, primary))
                self.assertTrue(reference_plan_wins(case, alternative))
                self.assertEqual(dut.adjudicate(case, {"kind": "PLAN",
                    "actions": list(alternative)})["verdict"], "VERIFIED_WIN")
                break
        self.assertIsNotNone(chosen)
        print("NOVEL_ROUTER_ALTERNATE_WIN", chosen)

    def test_bounded_cap_never_proves_unsat_and_resource_violations_fail(self):
        for seed in range(50):
            bounded = dut.search(dut.make_case(seed), max_nodes=1)
            self.assertEqual(bounded["verdict"], "NO_KNOWN_WIN")
            self.assertFalse(bounded["complete"])
        case = dut.make_case(0)
        self.assertEqual(dut.verify_plan(case, ["PURGE", "COMMIT"])["verdict"],
                         "VERIFIED_FAILURE")
        self.assertEqual(dut.verify_plan(case, ["COMMIT", "TICK"])["verdict"],
                         "VERIFIED_FAILURE")
        self.assertEqual(dut.verify_plan(case, ["INVALID"])["verdict"],
                         "VERIFIED_FAILURE")
        with self.assertRaises(ValueError):
            dut.search(case, max_nodes=0)

    def test_probe_budget_and_explicit_source_public_custody(self):
        case = dut.make_case(0)
        session = dut.ProbeSession(case)
        public = session.view()
        self.assertTrue({"before", "after", "seed", "case_id",
                         "mechanic_signature"}.isdisjoint(public))
        self.assertEqual(public["maximum_probes"], dut.MAX_PROBES)
        for n in range(dut.MAX_PROBES):
            view = session.probe("TICK")
            self.assertEqual(len(view["observations"]), n + 1)
        receipt = session.submit({"kind": "PLAN", "actions": ["COMMIT"]})
        self.assertEqual(receipt["probe_count"], dut.MAX_PROBES)
        self.assertFalse(receipt["private_custody"])
        self.assertFalse(receipt["graduation_eligible"])
        self.assertFalse(receipt["neural_learning_verified"])
        self.assertEqual(receipt["evidence_class"], "SOURCE_PUBLIC_SYNTHETIC")
        self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"}), receipt)

        excess = dut.ProbeSession(case)
        for _ in range(dut.MAX_PROBES):
            excess.probe("TICK")
        excess.probe("TICK")
        self.assertTrue(excess.view()["finished"])
        self.assertEqual(excess.submit({"kind": "PLAN", "actions": ["COMMIT"]})
                         ["reason"], "PROBE_BUDGET_EXCEEDED")
        novel = dut.adjudicate(case, {"kind": "NOVEL_RULE",
                                    "description": "An alternative law could reroute pending packets."})
        self.assertEqual(novel["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertFalse(novel["verified_success"])

    def test_canonical_case_identity_rejects_numeric_type_laundering(self):
        valid = dut.make_case(0)
        self.assertTrue(dut._valid_case(valid))
        forged = replace(valid, target_left=float(valid.target_left))
        self.assertFalse(dut._valid_case(forged))
        with self.assertRaises(ValueError):
            dut.adjudicate(forged, {"kind": "CLAIM_UNSAT"})
        with self.assertRaises(ValueError):
            dut.ProbeSession(forged)

    def test_forged_case_remains_invalid_under_monkeypatched_object_equality(self):
        valid = dut.make_case(0)
        forged = replace(valid, target_right=(valid.target_right + 1) % 4)
        self.assertFalse(dut._valid_case(forged))
        with patch.object(dut.Case, "__eq__", return_value=True):
            self.assertFalse(dut._valid_case(forged))
            with self.assertRaises(ValueError):
                dut.adjudicate(forged, {"kind": "CLAIM_UNSAT"})
            with self.assertRaises(ValueError):
                dut.ProbeSession(forged)

    def test_seeded_genesis_does_not_depend_on_routing_rule_eq(self):
        canonical = dut.make_case(0)
        with patch.object(dut.RoutingRule, "__eq__", return_value=True):
            mutated = dut.make_case(0)
            self.assertEqual(mutated.mechanic_signature, canonical.mechanic_signature)
            self.assertEqual(mutated.after.lane_bias, canonical.after.lane_bias)
            self.assertEqual(mutated.after.coupling, canonical.after.coupling)
            self.assertEqual(mutated.after.payload_shift,
                             canonical.after.payload_shift)
            self.assertTrue(dut._valid_case(canonical))

    def test_proof_closure_cannot_be_spoofed_by_state_equality_or_hash(self):
        case = dut.make_case(0)
        self.assertIsNotNone(reference_backward_witness(case)[0])
        with patch.object(dut.State, "__eq__", return_value=True), patch.object(
                dut.State, "__hash__", return_value=0):
            outcome = dut.search(case)
            self.assertEqual(outcome["verdict"], "VERIFIED_WIN")
            self.assertTrue(reference_plan_wins(case, outcome["witness"]))

    def test_shared_plan_scorer_cannot_self_certify_fabricated_win(self):
        case = dut.make_case(0)
        wrong_plan = {"kind": "PLAN", "actions": ["COMMIT"]}
        self.assertEqual(dut.adjudicate(case, wrong_plan)["verdict"],
                         "VERIFIED_FAILURE")
        with patch.object(dut, "verify_plan", return_value={
                "verdict": "VERIFIED_WIN", "reason": "INJECTED"}):
            result = dut.adjudicate(case, wrong_plan)
            self.assertEqual(result["verdict"], "UNKNOWN")
            self.assertFalse(result["verified_success"])
            self.assertEqual(result["reason"], "INDEPENDENT_SCORER_DISAGREEMENT")
            # Even a corrupt source search must not award a false win.
            self.assertEqual(dut.search(case)["verdict"], "UNKNOWN")

    def test_corrupted_source_step_cannot_turn_known_win_into_false_unsat(self):
        case = dut.make_case(0)
        self.assertIsNotNone(reference_backward_witness(case)[0])
        with patch.object(dut, "step", side_effect=lambda c, s, action: s):
            claimed = dut.search(case)
            self.assertEqual(claimed["verdict"], "UNKNOWN")
            self.assertFalse(claimed["complete"])
            review = dut.adjudicate(case, {"kind": "CLAIM_UNSAT"})
            self.assertEqual(review["verdict"], "VERIFIED_FAILURE")
            self.assertTrue(reference_plan_wins(case, review["counterexample"]))

    def test_source_probe_mutation_cannot_emit_unverified_observation(self):
        session = dut.ProbeSession(dut.make_case(0))
        with patch.object(dut, "step", side_effect=lambda c, s, action: s):
            view = session.probe("LOAD")
            self.assertTrue(view["finished"])
            self.assertEqual(session.submit({"kind": "CLAIM_UNSAT"})["reason"],
                             "INDEPENDENT_TRANSITION_DISAGREEMENT")


if __name__ == "__main__":
    unittest.main()
