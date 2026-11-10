"""Hostile independent tests for finite exact-cover rescue-allocation tribunal."""

from dataclasses import replace
from itertools import combinations
import random
import unittest

from labyrinth.exact_cover_tribunal import (
    Scenario, adjudicate_unexpected, catalogue_control, check_scenario,
    families, fingerprint, fixed_script_control, independent_subset_oracle,
    novel_rule_challenge, public_view, search_allocation, verify_allocation,
)


def primitive_subset_oracle(s):
    """An independent test-only tuple multiset count, no search internals."""
    target = {site: 1 for site in s.sites}
    for count in range(1, min(len(s.crews), s.budget) + 1):
        for indices in combinations(range(len(s.crews)), count):
            occurrences = {site: 0 for site in s.sites}
            for index in indices:
                for site in s.crews[index]:
                    occurrences[site] += 1
            if occurrences == target:
                return True
    return False


class ExactCoverTribunalTests(unittest.TestCase):
    def test_cyclic_deficiency_proven_only_under_declared_rules(self):
        s = families()["cyclic_deficiency"]
        self.assertFalse(primitive_subset_oracle(s))
        result = search_allocation(s)
        self.assertEqual(result["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertTrue(result["exhaustive"])
        self.assertEqual(result["certificate_scope"],
                         "exact_cover_of_declared_crews_with_declared_budget")

    def test_one_new_declared_crew_changes_mechanics_from_unsat_to_win(self):
        bad = families()["cyclic_deficiency"]
        fixed = families()["repairable_by_declared_team"]
        self.assertEqual(bad.sites, fixed.sites)
        self.assertEqual(bad.budget, fixed.budget)
        self.assertEqual(fixed.crews[:-1], bad.crews)
        self.assertFalse(primitive_subset_oracle(bad))
        self.assertTrue(primitive_subset_oracle(fixed))
        solved = search_allocation(fixed)
        self.assertEqual(solved["verdict"], "VERIFIED_WIN")
        self.assertEqual(verify_allocation(fixed, solved["allocation"])["verdict"],
                         "VERIFIED_WIN")

    def test_another_intelligence_can_overturn_wrong_answer_with_valid_cover(self):
        s = families()["alternative_matching"]
        solver = search_allocation(s)
        self.assertEqual(solver["verdict"], "VERIFIED_WIN")
        other = (5, 6)
        self.assertNotEqual(set(other), set(solver["allocation"]))
        result = adjudicate_unexpected(s, other, "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(result["verdict"], "VERIFIED_WIN")
        self.assertTrue(result["overturned_prior"])
        self.assertTrue(result["independently_replayed"])

    def test_deceptive_repeated_site_false_win_rejected_by_independent_replay(self):
        s = families()["cyclic_deficiency"]
        wrong = (0, 1)  # Two crews have three sites each, but overlap a.
        self.assertEqual(len(wrong) * 3, len(s.sites))
        self.assertEqual(verify_allocation(s, wrong)["verdict"], "VERIFIED_FAILURE")
        # An implementation that counts assigned slots rather than exact sites
        # will falsely agree with itself; this direct-set verifier kills it.
        buggy_solver_thinks_win = len(wrong) * 3 == len(s.sites)
        buggy_replayer_thinks_win = len(wrong) == s.budget
        self.assertTrue(buggy_solver_thinks_win and buggy_replayer_thinks_win)

    def test_invalid_or_over_budget_actions_never_win(self):
        s = families()["repairable_by_declared_team"]
        for value in ((0, 0), (0, 1, 2), (0, True), (0, "4"), [0, 4], (99,), ()):
            with self.subTest(value=value):
                self.assertEqual(verify_allocation(s, value)["verdict"],
                                 "VERIFIED_FAILURE")

    def test_partial_search_never_proves_unsat(self):
        for s in families().values():
            r = search_allocation(s, node_limit=0)
            self.assertEqual(r["verdict"], "NO_KNOWN_WIN")
            self.assertFalse(r["exhaustive"])
        self.assertEqual(search_allocation(s, node_limit=True)["verdict"], "UNKNOWN")

    def test_novel_affordance_is_reviewed_not_summarily_impossible_or_win(self):
        s = families()["cyclic_deficiency"]
        r = novel_rule_challenge(s, {"new_crew": ("d", "e", "f"),
                                      "reason": "Call a second mutually compatible crew"})
        self.assertEqual(r["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(r["independent_verification"], "NOT_PERFORMED")
        self.assertEqual(search_allocation(s)["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(novel_rule_challenge(s, {"new_crew": ("d", "e", "f"),
                           "reason": "R" * 100_000})["verdict"], "UNKNOWN")

    def test_malicious_python_mapping_key_fails_closed(self):
        class LyingKey(str):
            def __eq__(self, other):
                raise RuntimeError("deceptive equality callback")
            __hash__ = str.__hash__

        s = families()["cyclic_deficiency"]
        result = novel_rule_challenge(s, {LyingKey("new_crew"): ("d", "e", "f"),
                                         "reason": "real unused team"})
        self.assertEqual(result["verdict"], "UNKNOWN")

    def test_source_public_memorization_is_not_learning(self):
        s = Scenario(tuple("abcdef"), (("a", "b", "d"), ("c", "e", "f")), 2)
        self.assertNotIn(fingerprint(s), {fingerprint(v) for v in families().values()})
        output = catalogue_control(s)
        self.assertEqual(output["public_fixtures"], 4)
        self.assertEqual(output["exact_matches"], 4)
        self.assertEqual(output["unseen_verdict"], "UNKNOWN")
        self.assertEqual(output["optimizer_updates"], 0)

        fixed = fixed_script_control()
        self.assertEqual(fixed["public_cases"], 4)
        self.assertEqual(fixed["fixed_script_wins"], 0)
        self.assertEqual(fixed["symbolic_wins"], 2)
        self.assertEqual(fixed["optimizer_updates"], 0)
        with self.assertRaises(ValueError):
            fixed_script_control((True, 1))

    def test_perspective_paraphrases_do_not_become_new_structural_examples(self):
        s = families()["cyclic_deficiency"]
        views = [public_view(s, variant=i) for i in range(8)]
        self.assertEqual(len({v["prompt"] for v in views}), 8)
        self.assertEqual({fingerprint(s)}, {fingerprint(s) for _ in views})
        self.assertTrue(all(v["sites"] == list(s.sites) for v in views))

    def test_input_contract_rejects_structural_ambiguity(self):
        good = families()["cyclic_deficiency"]
        cases = (replace(good, sites=good.sites + ("z",)),
                 replace(good, sites=("a",) * 6),
                 replace(good, crews=good.crews + (good.crews[0],)),
                 replace(good, crews=good.crews + (("a", "b", "a"),)),
                 replace(good, budget=True),
                 replace(good, crews=good.crews + (("a", "b", "zzz"),)))
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                check_scenario(case)

    def test_randomized_independent_subset_oracle_differential(self):
        rng = random.Random(20261110)
        checked = 0
        site_names = tuple("abcdefghi")
        for site_count in (3, 6, 9):
            sites = site_names[:site_count]
            possible = list(combinations(sites, 3))
            for _ in range(180):
                rng.shuffle(possible)
                total = rng.randint(1, min(12, len(possible)))
                s = Scenario(sites, tuple(possible[:total]), rng.randint(1, 3))
                expected = primitive_subset_oracle(s)
                self.assertEqual(independent_subset_oracle(s), expected)
                actual = search_allocation(s)
                self.assertEqual(actual["verdict"] == "VERIFIED_WIN", expected, s)
                self.assertIn(actual["verdict"],
                              ("VERIFIED_WIN", "PROVEN_UNSAT_UNDER_DECLARED_RULES"))
                if expected:
                    self.assertEqual(verify_allocation(s, actual["allocation"])["verdict"],
                                     "VERIFIED_WIN")
                checked += 1
        self.assertEqual(checked, 540)


if __name__ == "__main__":
    unittest.main()
