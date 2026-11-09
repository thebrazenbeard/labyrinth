"""Independent backward-DP certificate for Switchback Circuit finite claims.

This solver does not invoke the author's BFS, oracle_step, or kernel. It
constructs a complete acyclic decision recurrence over the finite time,
register, pulse-budget, and post-change-injection state. Tests remain synthetic
source evidence, not private model evaluation or neural-learning proof.
"""

from functools import lru_cache
from unittest import mock
import random
import unittest

from labyrinth.switchback_circuit import (
    CONTROLS, MAX_MOVES, MAX_PROBES, ProbeSession, State, make_case, step,
)
from labyrinth import switchback_oracle as author
from labyrinth.switchback_controls import MEMORIZED_SEED0_PATH


def independent_kernel(register, bit, rule):
    """Bit-slice equations written separately from both production paths."""
    occupied = tuple(index for index in range(3) if (register >> index) & 1)
    selected = sum(bool((rule.mask >> index) & 1) for index in occupied)
    if rule.family == 0:
        return ((2 * register + bit) ^ (4 * (selected % 2)) ^ rule.bias) % 8
    if rule.family == 1:
        rotated = (register // 4) | ((register % 4) * 2)
        return (rotated + bit + rule.bias) % 8
    if rule.family == 2:
        return ((2 * register + bit) ^
                (rule.mask if len(occupied) >= rule.threshold else rule.bias)) % 8
    if rule.family == 3:
        return (register + bit + selected + rule.bias) % 8
    if rule.family == 4:
        toggle = rule.mask if selected >= rule.threshold else 0
        return (register ^ toggle ^ bit ^ rule.bias) % 8
    raise AssertionError("unknown family")


def independent_witness(case):
    """Return one winning path iff it exists, proving all branches by recursion.

    Every nonterminal action advances tick exactly once; COMMIT consumes one
    action and terminally tests the target. There are no cycles in this DAG.
    """
    @lru_cache(None)
    def reachable(reg, tick, pulses, post_input):
        if tick >= MAX_MOVES:
            return None
        if tick >= case.switch_at and post_input and reg == case.target:
            return ("COMMIT",)
        if tick >= MAX_MOVES - 1:
            return None  # cannot make another input AND then COMMIT
        # Different order from author's forward BFS; WAIT first is intentional.
        after_wait = reachable(reg, tick + 1, pulses, post_input)
        if after_wait is not None:
            return ("WAIT", *after_wait)
        if pulses == case.max_pulses:
            return None
        changed = tick >= case.switch_at
        rule = case.after if changed else case.before
        for bit in (1, 0):
            new_reg = independent_kernel(reg, bit, rule)
            remaining = reachable(new_reg, tick + 1, pulses + 1,
                                  post_input or changed)
            if remaining is not None:
                return (str(bit), *remaining)
        return None

    path = reachable(case.initial, 0, 0, False)
    return path, reachable.cache_info().currsize


def independent_play(case, plan):
    register, tick, pulses, post_input = case.initial, 0, 0, False
    if (type(plan) not in (tuple, list) or not 1 <= len(plan) <= MAX_MOVES
            or any(type(a) is not str or a not in CONTROLS for a in plan)):
        return False
    for index, action in enumerate(plan):
        if tick >= MAX_MOVES:
            return False
        if action == "COMMIT":
            return (index == len(plan) - 1 and tick >= case.switch_at
                    and post_input and register == case.target)
        if action == "WAIT":
            tick += 1
            continue
        if pulses >= case.max_pulses:
            return False
        revised = tick >= case.switch_at
        register = independent_kernel(register, int(action),
                                      case.after if revised else case.before)
        pulses += 1
        tick += 1
        post_input |= revised
    return False


class SwitchbackIndependentMathTests(unittest.TestCase):
    def test_independent_backward_recurrence_decides_400_generated_cases(self):
        cases = list(range(400)) + [2**31 - 1, 10_000_003, 314_159_265]
        successes = failures = 0
        max_states = 0
        max_bfs_nodes = 0
        for seed in cases:
            case = make_case(seed)
            expected, visited = independent_witness(case)
            max_states = max(max_states, visited)
            claimed = author.search(case)
            max_bfs_nodes = max(max_bfs_nodes, claimed["nodes_explored"])
            with self.subTest(seed=seed):
                self.assertTrue(claimed["complete"])
                self.assertLessEqual(claimed["nodes_explored"], 100_000)
                # All legal nonterminal transitions advance tick. Therefore
                # every searched state is among at most 8 registers, two
                # post-input flags, (pulse_cap+1) pulse counters, and nine
                # tick values; source BFS cannot exhaust its 100k node cap.
                self.assertLessEqual(
                    claimed["nodes_explored"],
                    MAX_MOVES * 8 * (case.max_pulses + 1) * 2,
                )
                if expected is None:
                    failures += 1
                    self.assertEqual(claimed["verdict"],
                                     "PROVEN_UNSAT_UNDER_DECLARED_RULES")
                    self.assertIsNone(claimed["witness"])
                else:
                    successes += 1
                    self.assertEqual(claimed["verdict"], "VERIFIED_WIN")
                    self.assertTrue(independent_play(case, expected))
                    self.assertTrue(independent_play(case, claimed["witness"]))
                    self.assertEqual(author.verify_plan(case, list(expected))["verdict"],
                                     "VERIFIED_WIN")
        self.assertGreater(successes, 0)
        self.assertGreater(failures, 0)
        print("SWITCHBACK_INDEPENDENT_DP", {"cases": len(cases),
              "SAT": successes, "UNSAT": failures,
              "max_cached_states_per_case": max_states,
              "max_author_bfs_nodes_per_case": max_bfs_nodes})

    def test_two_distinct_seed_zero_winning_plans_survive_independent_verifier(self):
        case = make_case(0)
        alternatives = (MEMORIZED_SEED0_PATH,
                        ("0", "WAIT", "0", "0", "COMMIT"))
        self.assertNotEqual(*alternatives)
        for actions in alternatives:
            self.assertTrue(independent_play(case, actions))
            self.assertEqual(author.verify_plan(case, list(actions))["verdict"],
                             "VERIFIED_WIN")
        self.assertIsNotNone(independent_witness(case)[0])

    def test_rule_change_boundary_and_pulse_cap_for_100_seeds(self):
        for seed in range(100):
            case = make_case(seed)
            for tick in (case.switch_at - 1, case.switch_at, case.switch_at + 1):
                for pulses in (0, case.max_pulses):
                    for register in range(8):
                        for bit in (0, 1):
                            before = State(register, tick, pulses, False)
                            expected = (before if pulses == case.max_pulses else
                                State(independent_kernel(register, bit,
                                        case.after if tick >= case.switch_at else case.before),
                                      tick + 1, pulses + 1, tick >= case.switch_at))
                            with self.subTest(seed=seed, tick=tick, pulses=pulses,
                                              register=register, bit=bit):
                                self.assertEqual(step(case, before, str(bit)), expected)
                                self.assertEqual(author.oracle_step(case, before,
                                                                   str(bit)), expected)

    def test_resource_cap_does_not_forge_unsat_certificate(self):
        for seed in range(70):
            case = make_case(seed)
            bounded = author.search(case, max_nodes=1)
            self.assertEqual(bounded["verdict"], "NO_KNOWN_WIN")
            self.assertFalse(bounded["complete"])
            self.assertIsNone(bounded["witness"])

    def test_mutated_author_bfs_is_detected_by_independent_solver(self):
        case = make_case(0)
        self.assertIsNotNone(independent_witness(case)[0])
        # An external corruptor that freezes every oracle transition will
        # falsely label this winnable case UNSAT. The second solver disagrees.
        with mock.patch.object(author, "oracle_step", side_effect=lambda c, s, a: s):
            mutant = author.search(case)
        self.assertEqual(mutant["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertTrue(independent_play(case, MEMORIZED_SEED0_PATH))

    def test_public_probe_reset_and_switch_observations_match_independent_model(self):
        rng = random.Random(20261107)
        for seed in range(50):
            case = make_case(seed)
            probe = ProbeSession(case)
            reg, tick, pulse, injected = case.initial, 0, 0, False
            for _ in range(MAX_PROBES - 1):
                action = rng.choice(("0", "1", "WAIT", "RESET", "INVALID"))
                if action == "RESET":
                    reg, tick, pulse, injected = case.initial, 0, 0, False
                elif action == "WAIT":
                    if tick < MAX_MOVES:
                        tick += 1
                elif action in ("0", "1") and tick < MAX_MOVES and pulse < case.max_pulses:
                    revised = tick >= case.switch_at
                    reg = independent_kernel(reg, int(action),
                                             case.after if revised else case.before)
                    tick += 1
                    pulse += 1
                    injected |= revised
                view = probe.probe(action)
                with self.subTest(seed=seed, action=action,
                                  count=len(view["observations"])):
                    self.assertEqual(view["current_register"], format(reg, "03b"))
                    self.assertEqual(view["ticks"], tick)
                    self.assertEqual(view["pulses_used"], pulse)
                    self.assertTrue(author.verify_probe_trace(
                        case, view["observations"]))
            self.assertEqual(view["remaining_probes"], 1)


if __name__ == "__main__":
    unittest.main()
