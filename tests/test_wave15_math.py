"""Independent finite-state and observational custody of Switchback heldout claims.

No author search, oracle_step, kernel, or candidate scoring logic is used to
construct this reference. Author search is called only *after* independent
classification for differential comparison. This is public source evidence.
"""

from collections import Counter, defaultdict, deque
from functools import lru_cache
from itertools import product
import unittest

from labyrinth.switchback_circuit import MAX_MOVES, Rule, make_case
from labyrinth.switchback_oracle import search


def reference_law(register, bit, law):
    """Independent three-bit equations, unpacked from truth-table specification."""
    occupied = {position for position in range(3) if register & (1 << position)}
    overlap = len(occupied & {position for position in range(3)
                              if law.mask & (1 << position)})
    family = law.family
    if family == 0:
        return ((2 * register + bit) ^ (4 if overlap % 2 else 0) ^ law.bias) % 8
    if family == 1:
        return (2 * (register % 4) + register // 4 + bit + law.bias) % 8
    if family == 2:
        return ((2 * register + bit) ^
                (law.mask if len(occupied) >= law.threshold else law.bias)) % 8
    if family == 3:
        return (register + bit + overlap + law.bias) % 8
    if family == 4:
        return (register ^ (law.mask if overlap >= law.threshold else 0)
                ^ bit ^ law.bias) % 8
    raise AssertionError("outside declared five-family law")


def reference_witness(case):
    """Finite backward recurrence, independent of forward production search.

    The four-state tuple is (register, elapsed ticks, consumed pulses,
    whether a post-switch input was applied). All nonterminal steps consume
    one tick; COMMIT is terminal and requires a remaining move slot.
    """
    @lru_cache(None)
    def solve(register, ticks, pulses, injected):
        if ticks >= MAX_MOVES:
            return None
        if ticks >= case.switch_at and injected and register == case.target:
            return ("COMMIT",)
        if ticks >= MAX_MOVES - 1:
            return None
        waited = solve(register, ticks + 1, pulses, injected)
        if waited is not None:
            return ("WAIT",) + waited
        if pulses >= case.max_pulses:
            return None
        after_switch = ticks >= case.switch_at
        active_law = case.after if after_switch else case.before
        for bit in (1, 0):
            nxt = reference_law(register, bit, active_law)
            found = solve(nxt, ticks + 1, pulses + 1, injected or after_switch)
            if found is not None:
                return (str(bit),) + found
        return None

    result = solve(case.initial, 0, 0, False)
    return result, solve.cache_info().currsize


def public_initial_key(case):
    return case.initial, case.target, case.switch_at, case.max_pulses


def reference_probe(case, state, action):
    register, ticks, pulses = state
    if action == "RESET":
        return case.initial, 0, 0
    if ticks >= MAX_MOVES:
        return state
    if action == "WAIT":
        return register, ticks + 1, pulses
    if pulses >= case.max_pulses:
        return state
    active_law = case.after if ticks >= case.switch_at else case.before
    return reference_law(register, int(action), active_law), ticks + 1, pulses + 1


def shortest_public_discriminator(first, second):
    """Pair-product BFS on public probing states, never hidden trial IDs.

    Both cases must expose the same initial public metadata. RESET has no
    extra existential distinguishing power because it returns both worlds
    to their shared initial state; every distinguishable reset suffix can
    instead start at time zero. Invalid commands likewise reveal no new law.
    """
    if public_initial_key(first) != public_initial_key(second):
        raise ValueError("not publicly ambiguous at time zero")
    start = (first.initial, second.initial, 0, 0)
    visited = {start}
    frontier = deque([(start, ())])
    while frontier:
        (left, right, ticks, pulses), actions = frontier.popleft()
        if ticks >= MAX_MOVES:
            continue
        for action in ("0", "1", "WAIT"):
            next_left = reference_probe(first, (left, ticks, pulses), action)
            next_right = reference_probe(second, (right, ticks, pulses), action)
            if next_left[0] != next_right[0]:
                return actions + (action,)
            state = (next_left[0], next_right[0], next_left[1], next_left[2])
            if state not in visited:
                visited.add(state)
                frontier.append((state, actions + (action,)))
    return None


def truth_table(law):
    return tuple(reference_law(register, bit, law)
                 for register in range(8) for bit in (0, 1))


class SwitchbackWave15IndependentMath(unittest.TestCase):
    def test_declared_840_rule_parameters_are_only_378_distinct_laws(self):
        groups = defaultdict(list)
        for family, mask, bias, threshold in product(
                range(5), range(1, 8), range(8), range(1, 4)):
            law = Rule(family, mask, bias, threshold)
            groups[truth_table(law)].append(law)
        self.assertEqual(sum(map(len, groups.values())), 840)
        self.assertEqual(len(groups), 378)
        self.assertTrue(any(len(group) > 1 for group in groups.values()))
        self.assertTrue(all(len(table) == 16 for table in groups))
        print("W15_RULE_EQUIVALENCE", {
            "parameterizations": 840,
            "distinct_transition_functions": len(groups),
            "largest_equivalence_class": max(map(len, groups.values())),
            "multiplicity_histogram": dict(sorted(Counter(map(len, groups.values())).items())),
        })

    def test_independent_dp_disproves_no_mislabeled_case_in_1000_seed_sample(self):
        block_counts = []
        max_states = 0
        for low, high in ((0, 100), (100, 200), (200, 1000)):
            result = Counter()
            for seed in range(low, high):
                case = make_case(seed)
                witness, visited = reference_witness(case)
                max_states = max(max_states, visited)
                expected = ("VERIFIED_WIN" if witness is not None else
                            "PROVEN_UNSAT_UNDER_DECLARED_RULES")
                claimed = search(case)
                self.assertTrue(claimed["complete"], seed)
                self.assertEqual(claimed["verdict"], expected, seed)
                if witness is not None:
                    self.assertLessEqual(len(witness), MAX_MOVES)
                    self.assertEqual(witness[-1], "COMMIT")
                    self.assertIsNotNone(claimed["witness"])
                else:
                    self.assertIsNone(claimed["witness"])
                result[expected] += 1
            block_counts.append(result)
        train, heldout, further = block_counts
        self.assertEqual(train["VERIFIED_WIN"], 67)
        self.assertEqual(heldout["VERIFIED_WIN"], 72)
        all_counts = train + heldout + further
        self.assertEqual(all_counts["VERIFIED_WIN"], 717)
        self.assertEqual(all_counts["PROVEN_UNSAT_UNDER_DECLARED_RULES"], 283)
        print("W15_INDEPENDENT_FINITE_LABELS", {
            "cases": 1000, "train": dict(train), "heldout": dict(heldout),
            "total": dict(all_counts), "max_cached_dp_states": max_states,
        })

    def test_public_collision_and_opposite_label_cases_are_probe_separable(self):
        by_key = defaultdict(list)
        labels = {}
        for seed in range(1000):
            case = make_case(seed)
            by_key[public_initial_key(case)].append(case)
            labels[seed] = reference_witness(case)[0] is not None
        self.assertEqual(len(by_key), 569)
        mixed = [cases for cases in by_key.values()
                 if len({labels[case.seed] for case in cases}) > 1]
        self.assertEqual(len(mixed), 80)
        # Optimistic empirical zero-probe classifier sees all labels in the
        # sample and chooses the majority label of every public-key bucket.
        # It is an *in-sample* ceiling under uniform sample-seed weighting,
        # never a forward prediction guarantee.
        retrospective_majority = sum(max(Counter(labels[case.seed] for case in cases)
                                         .values()) for cases in by_key.values())
        self.assertEqual(retrospective_majority, 916)
        opposite_pairs = 0
        max_probe_length = 0
        all_collision_pairs = 0
        indistinguishable_pairs = 0
        indistinguishable_examples = []
        for cases in by_key.values():
            for left_index in range(len(cases)):
                for right in cases[left_index + 1:]:
                    left = cases[left_index]
                    all_collision_pairs += 1
                    distinguisher = shortest_public_discriminator(left, right)
                    if distinguisher is None:
                        indistinguishable_pairs += 1
                        indistinguishable_examples.append({
                            "seeds": (left.seed, right.seed),
                            "same_mechanic_signature": (
                                left.mechanic_signature == right.mechanic_signature),
                            "same_full_truth_tables": (
                                truth_table(left.before) == truth_table(right.before) and
                                truth_table(left.after) == truth_table(right.after)),
                            "same_label": labels[left.seed] == labels[right.seed],
                        })
                    else:
                        max_probe_length = max(max_probe_length, len(distinguisher))
                        self.assertLessEqual(len(distinguisher), 16)
                        before_a = (left.initial, 0, 0)
                        before_b = (right.initial, 0, 0)
                        for action in distinguisher:
                            before_a = reference_probe(left, before_a, action)
                            before_b = reference_probe(right, before_b, action)
                        self.assertNotEqual(before_a[0], before_b[0])
                    if labels[left.seed] != labels[right.seed]:
                        opposite_pairs += 1
                        # Complete finite public experiments have to be able
                        # to separate opposite win/no-win claims under the
                        # same public resource/goal metadata.
                        self.assertIsNotNone(distinguisher, (left.seed, right.seed))
        self.assertEqual(indistinguishable_pairs, 1)
        self.assertEqual(indistinguishable_examples[0], {
            "seeds": (118, 555), "same_mechanic_signature": False,
            "same_full_truth_tables": False, "same_label": True,
        })
        print("W15_PUBLIC_OBSERVATIONAL_COLLISIONS", {
            "cases": 1000, "public_metadata_buckets": len(by_key),
            "mixed_label_buckets": len(mixed),
            "in_sample_best_zero_probe_correct": retrospective_majority,
            "same_metadata_pairs": all_collision_pairs,
            "opposite_label_pairs": opposite_pairs,
            "probe_indistinguishable_pairs": indistinguishable_pairs,
            "indistinguishable_examples": indistinguishable_examples,
            "max_shortest_discriminator": max_probe_length,
        })

    def test_public_seed_catalogue_and_family_leakage(self):
        blocks = ([make_case(seed) for seed in range(100)],
                  [make_case(seed) for seed in range(100, 200)])
        observed_families = []
        observed_pairs = []
        observed_signatures = []
        for cases in blocks:
            observed_families.append({rule.family for case in cases
                                      for rule in (case.before, case.after)})
            observed_pairs.append({(case.before.family, case.after.family)
                                   for case in cases})
            observed_signatures.append({case.mechanic_signature for case in cases})
        self.assertEqual(observed_families, [set(range(5)), set(range(5))])
        self.assertEqual(len(observed_signatures[0]), 100)
        self.assertEqual(len(observed_signatures[1]), 100)
        self.assertGreater(len(observed_pairs[0] & observed_pairs[1]), 0)
        pair_freq_train = Counter((c.before.family, c.after.family) for c in blocks[0])
        pair_freq_heldout = Counter((c.before.family, c.after.family) for c in blocks[1])
        # Simpson's effective number of observed *pair categories* is a
        # descriptive concentration measure, not a statistically proven n_eff
        # of independent exam tasks.
        concentration_train = 10000 / sum(count * count for count in
                                            pair_freq_train.values())
        concentration_heldout = 10000 / sum(count * count for count in
                                              pair_freq_heldout.values())
        self.assertLessEqual(concentration_train, len(pair_freq_train))
        self.assertLessEqual(concentration_heldout, len(pair_freq_heldout))

        catalogue = blocks[1]
        metadata = Counter(public_initial_key(case) for case in catalogue)
        enriched = Counter((public_initial_key(case), reference_probe(
            case, (case.initial, 0, 0), "1")[0]) for case in catalogue)
        unique_without_probe = sum(metadata[public_initial_key(case)] == 1
                                   for case in catalogue)
        unique_with_one_probe = sum(enriched[(public_initial_key(case),
                reference_probe(case, (case.initial, 0, 0), "1")[0])] == 1
                                    for case in catalogue)
        self.assertEqual(unique_without_probe, 84)
        self.assertEqual(unique_with_one_probe, 100)
        print("W15_HELDOUT_LEAKAGE", {
            "train_family_pairs": len(observed_pairs[0]),
            "heldout_family_pairs": len(observed_pairs[1]),
            "overlapping_family_pairs": len(observed_pairs[0] & observed_pairs[1]),
            "descriptive_pair_diversity_train": round(concentration_train, 3),
            "descriptive_pair_diversity_heldout": round(concentration_heldout, 3),
            "both_blocks_family_types": sorted(observed_families[0]),
            "heldout_exact_catalogue_identifiable_without_probe": unique_without_probe,
            "heldout_exact_catalogue_identifiable_after_one_probe": unique_with_one_probe,
        })


if __name__ == "__main__":
    unittest.main()
