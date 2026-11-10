"""Public-probe, source-aware zero-update control for Switchback Circuit.

This is a TEST-ONLY finite hypothesis planner, not a neural run. It reads
only session public view/probes, never target seed, private Case, or oracle
search. A finite source-known rule class makes legitimate transfer weaker
than a truly novel mechanism challenge. No graduation evidence is emitted.
"""

from collections import Counter
from functools import lru_cache
from itertools import product
import unittest
from unittest import mock

from labyrinth.switchback_circuit import (
    MAX_MOVES, ProbeSession, Rule, kernel, make_case,
)
from labyrinth.switchback_oracle import search, verify_receipt_replay


_RULES = tuple(Rule(family, mask, bias, threshold)
               for family, mask, bias, threshold in product(
                   range(5), range(1, 8), range(8), range(1, 4)))
_TRANSITIONS = tuple(tuple(tuple(kernel(state, bit, rule)
                                 for bit in (0, 1))
                           for state in range(8))
                     for rule in _RULES)
_ALL_RULE_IDS = tuple(range(len(_RULES)))
_UNIQUE_TRANSITIONS = tuple(dict.fromkeys(_TRANSITIONS))


class _PublicSurface:
    """Expose only candidate-authorized callable methods in the live scope."""

    __slots__ = ("view", "probe", "submit")

    def __init__(self, session):
        self.view = session.view
        self.probe = session.probe
        self.submit = session.submit


def _adaptive_source_prior_policy(surface, *, transitions=None):
    """Finite Bayesian-style rule elimination, with zero parameter updates.

    Selection maximizes number of source-rule hypotheses yielding a hit by
    a future bounded intervention; after each actual public observation the
    rule set is filtered. Ties are deterministic. It never guesses UNSAT.
    """
    initial = surface.view()
    if transitions is None:
        transitions = _TRANSITIONS
    target = int(initial["target_register"], 2)
    switch_at = initial["switch_at"]
    pulse_limit = min(initial["maximum_pulses"],
                      initial["maximum_moves"] - switch_at - 1)
    history = []
    for _ in range(switch_at):
        surface.probe("WAIT")
        history.append("WAIT")

    @lru_cache(maxsize=50000)
    def maximum_hits(belief, register, remaining):
        if remaining <= 0 or not belief:
            return 0
        scores = []
        for bit in (0, 1):
            outcomes = {}
            for rule_id in belief:
                result = transitions[rule_id][register][bit]
                outcomes.setdefault(result, []).append(rule_id)
            count = sum(len(ids) if state == target else
                        maximum_hits(tuple(ids), state, remaining - 1)
                        for state, ids in outcomes.items())
            scores.append(count)
        return max(scores)

    possible = tuple(range(len(transitions)))
    for iteration in range(pulse_limit):
        current = int(surface.view()["current_register"], 2)
        remaining = pulse_limit - iteration
        best = max((0, 1), key=lambda bit: (
            sum(len(ids) if state == target else
                maximum_hits(tuple(ids), state, remaining - 1)
                for state, ids in _partitions(possible, current, bit,
                                              transitions).items()),
            -bit))
        output = surface.probe(str(best))
        history.append(str(best))
        observed = int(output["current_register"], 2)
        possible = tuple(rule_id for rule_id in possible
                         if transitions[rule_id][current][best] == observed)
        if not possible:
            return surface.submit({"kind": "ABSTAIN"}), {
                "reason": "OUTSIDE_PUBLIC_SOURCE_RULE_CLASS",
                "posterior": 0, "probes": len(history), "optimizer_updates": 0,
            }
        if observed == target:
            receipt = surface.submit({"kind": "PLAN", "actions": history + ["COMMIT"]})
            return receipt, {"reason": "WITNESSED_PATH", "posterior": len(possible),
                             "probes": len(history), "optimizer_updates": 0}
    return surface.submit({"kind": "ABSTAIN"}), {
        "reason": "BOUNDED_UNCERTAINTY", "posterior": len(possible),
        "probes": len(history), "optimizer_updates": 0,
    }


def _partitions(rule_ids, state, bit, transitions):
    groups = {}
    for rule_id in rule_ids:
        result = transitions[rule_id][state][bit]
        groups.setdefault(result, []).append(rule_id)
    return groups


class SwitchbackWave12AdaptivePublicControlTests(unittest.TestCase):
    def test_declared_source_prior_has_840_not_100_seed_entries(self):
        self.assertEqual(len(_RULES), 5 * 7 * 8 * 3)
        self.assertEqual(len(_ALL_RULE_IDS), 840)
        self.assertEqual(len(_UNIQUE_TRANSITIONS), 378)
        # Uniform weight over 840 Rule parameterizations overweights some
        # of the 378 actual transition functions. We test both priors below.
        # Rule hypotheses are parameterized finite source laws, no seed or
        # case identifier is looked up during candidate execution.

    def test_public_observations_only_and_no_privileged_search_or_training(self):
        for seed in (0, 2, 13, 14, 100, 101, 109, 193):
            session = ProbeSession(make_case(seed))
            facade = _PublicSurface(session)
            with mock.patch("labyrinth.switchback_oracle.search",
                            side_effect=AssertionError("privileged search forbidden")):
                receipt, meta = _adaptive_source_prior_policy(facade)
            self.assertTrue(verify_receipt_replay(receipt))
            self.assertIn(receipt["verdict"], ("VERIFIED_WIN", "UNKNOWN"))
            self.assertEqual(meta["optimizer_updates"], 0)
            self.assertFalse(receipt["neural_learning_verified"])
            self.assertFalse(receipt["graduation_eligible"])
            self.assertLessEqual(meta["probes"], 8)
            if receipt["verdict"] == "VERIFIED_WIN":
                self.assertEqual(meta["reason"], "WITNESSED_PATH")

    def test_measured_train_and_withheld_seed_controls_no_pseudo_independence(self):
        evidence = {}
        for label, seeds in (("TRAIN_CATALOG_SEEDS", range(100)),
                             ("WITHHELD_SEEDS", range(100, 200))):
            outcomes = Counter()
            posterior_sizes = []
            for seed in seeds:
                case = make_case(seed)
                session = ProbeSession(case)
                receipt, meta = _adaptive_source_prior_policy(_PublicSurface(session))
                self.assertTrue(verify_receipt_replay(receipt))
                self.assertFalse(receipt["neural_learning_verified"])
                outcomes[receipt["verdict"]] += 1
                posterior_sizes.append(meta["posterior"])
            evidence[label] = (outcomes, posterior_sizes)
        train_counts, train_posterior = evidence["TRAIN_CATALOG_SEEDS"]
        heldout_counts, heldout_posterior = evidence["WITHHELD_SEEDS"]
        self.assertEqual(sum(train_counts.values()), 100)
        self.assertEqual(sum(heldout_counts.values()), 100)
        self.assertTrue(all(size >= 0 for size in train_posterior + heldout_posterior))
        self.assertTrue(all(label in ("VERIFIED_WIN", "UNKNOWN")
                            for label in train_counts | heldout_counts))
        self.assertEqual(train_counts,
                         {"VERIFIED_WIN": 44, "UNKNOWN": 56})
        self.assertEqual(heldout_counts,
                         {"VERIFIED_WIN": 39, "UNKNOWN": 61})
        # Only the independent evaluator—not the candidate policy—has direct
        # hidden Case access to compute these *denominators*.
        self.assertEqual(sum(search(make_case(seed))["verdict"] == "VERIFIED_WIN"
                             for seed in range(100)), 67)
        self.assertEqual(sum(search(make_case(seed))["verdict"] == "VERIFIED_WIN"
                             for seed in range(100, 200)), 72)

    def test_deduplicating_equal_truth_tables_changes_greedy_policy_results(self):
        successes = []
        for low in (0, 100):
            counts = Counter()
            for seed in range(low, low + 100):
                receipt, meta = _adaptive_source_prior_policy(
                    _PublicSurface(ProbeSession(make_case(seed))),
                    transitions=_UNIQUE_TRANSITIONS)
                counts[receipt["verdict"]] += 1
                self.assertFalse(receipt["neural_learning_verified"])
                self.assertEqual(meta["optimizer_updates"], 0)
            successes.append(counts["VERIFIED_WIN"])
        self.assertEqual(successes, [47, 40])
        # 840 uniformly weighted parameter objects != 378 equally weighted
        # transition functions; neither prior is an independently justified
        # distribution over external problem mechanics.

    def test_seed_set_is_mechanically_varied_but_independence_unqualified(self):
        for start in (0, 100):
            cases = [make_case(seed) for seed in range(start, start + 100)]
            distinct_cases = len({case.case_id for case in cases})
            distinct_mechanisms = len({case.mechanic_signature for case in cases})
            family_pairs = {(case.before.family, case.after.family) for case in cases}
            raw_families = {r.family for case in cases
                            for r in (case.before, case.after)}
            self.assertEqual(distinct_cases, 100)
            self.assertEqual(distinct_mechanisms, 100)
            self.assertEqual(len(raw_families), 5)
            self.assertLessEqual(len(family_pairs), 25)
            # Distinct case hashes/tables are meaningful parameter variation,
            # NOT 100 separately developed generator mechanisms or n_eff100.


if __name__ == "__main__":
    unittest.main()
