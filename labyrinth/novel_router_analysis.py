"""Independent mathematics for source-public delayed-router identifiability.

The reference state is a primitive tuple; no production step, search,
verify_plan, or probe engine participates in its calculations. All results
are bounded synthetic source diagnostics, never model or private-trial proof.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import replace
from functools import lru_cache
from itertools import product

from .switchback_novel_transfer import Case, MAX_MOVES, MAX_PROBES, RoutingRule


# (left, right, orientation, pending, ticks, loads, routes, post_delivery)
PROBES = ("LOAD", "TICK", "ROUTE", "PURGE", "RESET")
NONTERMINAL = ("TICK", "PURGE", "ROUTE", "LOAD")


def initial(case: Case) -> tuple[int, int, int, int, int, int, int, bool]:
    return (case.initial_left, case.initial_right, case.initial_orientation,
            0, 0, 0, 0, False)


def public_initial_signature(case: Case) -> tuple[int, ...]:
    """All stable initial case fields in the candidate's actual view."""
    return (case.initial_left, case.initial_right, case.initial_orientation,
            case.target_left, case.target_right, case.switch_at,
            case.max_loads, case.max_routes)


def reference_advance(case: Case, state: tuple, action: str) -> tuple | None:
    """Independent raw-tuple law including resource validity and RESET."""
    left, right, facing, cargo, time, loaded, rotated, delivered_after = state
    if action == "RESET":
        return initial(case)
    if time >= MAX_MOVES:
        return None
    law = case.after if time >= case.switch_at else case.before
    if action == "LOAD":
        if cargo or loaded == case.max_loads:
            return None
        # Payload depends on the law active when loading; delivery may use
        # the other rule across a switch boundary.
        payload = (left + (right + right) + facing + law.payload_shift) % 3 + 1
        return (left, right, facing, payload, time + 1,
                loaded + 1, rotated, delivered_after)
    if action == "ROUTE":
        if rotated == case.max_routes:
            return None
        return (left, right, 1 - facing, cargo, time + 1,
                loaded, rotated + 1, delivered_after)
    if action == "PURGE":
        if cargo == 0:
            return None
        return (left, right, facing, 0, time + 1,
                loaded, rotated, delivered_after)
    if action == "TICK":
        if cargo:
            output = (facing + law.lane_bias + ((left ^ right) % 2)) % 2
            lanes = [left, right]
            lanes[output] = (lanes[output] + cargo + law.coupling) % 4
            cross = int(bool(cargo % 2) and bool(law.coupling % 2))
            lanes[1 - output] = (lanes[1 - output] + cross) % 4
            left, right = lanes
        return (left, right, facing, 0, time + 1, loaded, rotated,
                delivered_after or (cargo != 0 and time >= case.switch_at))
    raise ValueError("unknown raw reference action")


def reference_goal(case: Case, state: tuple) -> bool:
    left, right, _, cargo, time, _, _, delivered_after = state
    return (cargo == 0 and delivered_after and time >= case.switch_at and
            left == case.target_left and right == case.target_right)


def reference_witness(case: Case) -> tuple[tuple[str, ...] | None, int]:
    """Complete acyclic backward DP, no calls to production search/step."""
    @lru_cache(maxsize=None)
    def solve(state: tuple) -> tuple[str, ...] | None:
        if state[4] >= MAX_MOVES:
            return None
        if reference_goal(case, state):
            return ("COMMIT",)
        if state[4] >= MAX_MOVES - 1:
            return None
        for action in NONTERMINAL:
            evolved = reference_advance(case, state, action)
            if evolved is None:
                continue
            continuation = solve(evolved)
            if continuation is not None:
                return (action, *continuation)
        return None

    result = solve(initial(case))
    return result, solve.cache_info().currsize


def reference_play(case: Case, actions: object) -> bool:
    if (type(actions) not in (list, tuple) or not 1 <= len(actions) <= MAX_MOVES
            or actions[-1] != "COMMIT"):
        return False
    state = initial(case)
    for action in actions[:-1]:
        if action not in NONTERMINAL:
            return False
        state = reference_advance(case, state, action)
        if state is None:
            return False
    return reference_goal(case, state)


def full_law_signature(rule: RoutingRule) -> tuple[int, ...]:
    """Complete LOAD and TICK tables, not source parameter hashes.

    Includes every pair of visible reservoir levels, both orientations, and
    every pending payload 1..3 at delivery. This is all phase-local physics
    rather than a sample restricted to generated initial states.
    """
    table = []
    for left, right, facing in product(range(4), range(4), range(2)):
        table.append((left + 2 * right + facing + rule.payload_shift) % 3 + 1)
        for cargo in (1, 2, 3):
            tanks = [left, right]
            outlet = (facing ^ rule.lane_bias ^ ((left ^ right) & 1))
            tanks[outlet] = (tanks[outlet] + cargo + rule.coupling) % 4
            if cargo & 1 and rule.coupling & 1:
                tanks[1 - outlet] = (tanks[1 - outlet] + 1) % 4
            table.extend(tanks)
    return tuple(table)


def all_law_equivalence() -> dict[tuple[int, ...], tuple[RoutingRule, ...]]:
    groups: dict[tuple[int, ...], list[RoutingRule]] = defaultdict(list)
    for bias, coupling, shift in product(range(2), range(4), range(3)):
        rule = RoutingRule(bias, coupling, shift)
        groups[full_law_signature(rule)].append(rule)
    return {key: tuple(values) for key, values in groups.items()}


def counterfactual_pair_observability(base: Case) -> dict[str, object]:
    """Exact finite behavior classes for all 24×23 changed-law pairs.

    Fix one public initial context and substitute every admissible distinct
    pair of phase rules. These are counterfactual synthetic rule variants,
    NOT canonical seeded cases or secret evaluator receipts. A hash-free
    structural-intern table assigns equal IDs iff the complete observable
    finite transition trees (without resets) are identical. RESET returns to
    the same initial state, hence cannot distinguish equal initial trees.
    """
    laws = [RoutingRule(a, b, c) for a, b, c in
            product(range(2), range(4), range(3))]
    ids: dict[tuple, int] = {}

    def intern(key: tuple) -> int:
        if key not in ids:
            ids[key] = len(ids) + 1
        return ids[key]

    groups: dict[int, list[tuple[RoutingRule, RoutingRule]]] = defaultdict(list)
    for first, second in product(laws, repeat=2):
        if first == second:
            continue
        world = replace(base, before=first, after=second)

        @lru_cache(maxsize=None)
        def characterize(state: tuple) -> int:
            successors = []
            for action in NONTERMINAL:
                child = reference_advance(world, state, action)
                successors.append(0 if child is None else characterize(child))
            return intern((state, tuple(successors)))

        semantic_id = characterize(initial(world))
        groups[semantic_id].append((first, second))

    larger_classes = [v for v in groups.values() if len(v) > 1]
    first_equivalent = (larger_classes[0][:2] if larger_classes else None)
    return {
        "counterfactual_worlds": 24 * 23,
        "distinct_bounded_observation_classes": len(groups),
        "equivalent_world_count": sum(len(v) for v in larger_classes),
        "class_size_histogram": dict(sorted(Counter(map(len, groups.values())).items())),
        "largest_class": max(map(len, groups.values())),
        "example_equivalent_pair": first_equivalent,
        "public_context": public_initial_signature(base),
    }


def shortest_public_discriminator(first: Case, second: Case) -> tuple[str, ...] | None:
    """Complete pair-product BFS within twenty charged legal probes.

    A public state includes both lane levels, pending cargo, orientation,
    tick/load/route counters and post-delivery flag. A failed operation is
    itself visible as `applied:false`. RESET is included and charged.
    """
    if public_initial_signature(first) != public_initial_signature(second):
        raise ValueError("public initial views differ already")
    start = (initial(first), initial(second))
    frontier = deque([(start, ())])
    seen = {start}
    while frontier:
        (state_a, state_b), prefix = frontier.popleft()
        if len(prefix) >= MAX_PROBES:
            continue
        for action in PROBES:
            evolved_a = reference_advance(first, state_a, action)
            evolved_b = reference_advance(second, state_b, action)
            if (evolved_a is None) != (evolved_b is None):
                return prefix + (action,)
            if evolved_a is None:
                continue
            if evolved_a != evolved_b:
                return prefix + (action,)
            pair = (evolved_a, evolved_b)
            if pair not in seen:
                seen.add(pair)
                frontier.append((pair, prefix + (action,)))
    return None


def probe_catalogue_signature(case: Case, actions: tuple[str, ...]) -> tuple:
    """Reconstruct the public trace for a frozen source-known seed catalogue."""
    state = initial(case)
    observations = []
    for action in actions:
        evolved = reference_advance(case, state, action)
        applied = evolved is not None
        if applied:
            state = evolved
        observations.append((action, applied, state))
    return public_initial_signature(case), tuple(observations)


def retrospective_unique_count(cases: list[Case], actions: tuple[str, ...]) -> int:
    signatures = Counter(probe_catalogue_signature(case, actions) for case in cases)
    return sum(signatures[probe_catalogue_signature(case, actions)] == 1
               for case in cases)


def analyze_public_collision_classes(cases: list[Case]) -> dict[str, object]:
    by_view: dict[tuple[int, ...], list[Case]] = defaultdict(list)
    for case in cases:
        by_view[public_initial_signature(case)].append(case)
    equivalent_pairs: list[tuple[int, int]] = []
    different_pairs: list[tuple[int, int, tuple[str, ...]]] = []
    max_shortest = 0
    for group in by_view.values():
        for index, first in enumerate(group):
            for second in group[index + 1:]:
                witness = shortest_public_discriminator(first, second)
                if witness is None:
                    equivalent_pairs.append((first.seed, second.seed))
                else:
                    different_pairs.append((first.seed, second.seed, witness))
                    max_shortest = max(max_shortest, len(witness))
    return {
        "cases": len(cases),
        "initial_public_buckets": len(by_view),
        "ambiguous_public_buckets": sum(len(group) > 1 for group in by_view.values()),
        "same_initial_public_pairs": len(equivalent_pairs) + len(different_pairs),
        "bounded_probe_equivalent_pairs": tuple(equivalent_pairs),
        "bounded_probe_distinguishable_pairs": len(different_pairs),
        "max_shortest_discriminator": max_shortest,
        "first_discriminator": different_pairs[0] if different_pairs else None,
    }
