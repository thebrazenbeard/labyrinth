"""Independent finite-model tests beyond the four source-public fixtures."""

from itertools import combinations
import random
import unittest

from labyrinth.exact_cover_tribunal import (
    Scenario, independent_subset_oracle, search_allocation, verify_allocation,
)
from labyrinth.minotaur_public_exam import _independent_cover_exists_dp


def uncovered_set_oracle(scenario: Scenario) -> bool:
    """Set-partition recurrence, independent of the bitmask/source DFS code."""
    target = frozenset(scenario.sites)
    options = tuple(frozenset(crew) for crew in scenario.crews)
    cache = {}

    def possible(uncovered: frozenset[str], remaining: int) -> bool:
        if not uncovered:
            return True
        if remaining <= 0 or len(uncovered) != 3 * remaining:
            return False
        key = (uncovered, remaining)
        if key not in cache:
            # A complete partition must contain a crew that covers the first
            # uncovered site. This works with sets, not the production masks.
            pivot = sorted(uncovered)[0]
            cache[key] = any(pivot in crew and crew <= uncovered
                             and possible(uncovered - crew, remaining - 1)
                             for crew in options)
        return cache[key]

    return any(possible(target, count) for count in range(1, scenario.budget + 1))


class FiniteExamMetamorphicTests(unittest.TestCase):
    def test_exhaustive_six_site_rosters_with_up_to_four_crews(self):
        sites = tuple("abcdef")
        possible_crews = tuple(combinations(sites, 3))
        tested = 0
        for crew_count in range(1, 5):
            for roster in combinations(possible_crews, crew_count):
                for budget in (1, 2):
                    scenario = Scenario(sites, roster, budget)
                    expected = uncovered_set_oracle(scenario)
                    solved = search_allocation(scenario)
                    self.assertEqual(solved["verdict"] == "VERIFIED_WIN", expected,
                                     (roster, budget))
                    self.assertEqual(independent_subset_oracle(scenario), expected)
                    self.assertIs(_independent_cover_exists_dp(scenario), expected)
                    tested += 1
        # 20+190+1140+4845 crew rosters, each at two budget levels.
        self.assertEqual(tested, 12_390)

    def test_heldout_eighteen_site_random_and_planted_covers(self):
        rng = random.Random(20261111)
        seen_win = seen_unsat = 0
        for site_count in (3, 6, 9, 12, 15, 18):
            sites = tuple(f"s{i:02d}" for i in range(site_count))
            all_crews = list(combinations(sites, 3))
            for case_no in range(48):
                rng.shuffle(all_crews)
                budget = rng.randint(1, 6)
                amount = rng.randint(1, 18)
                selected = list(all_crews[:amount])
                if case_no % 2 == 0:
                    # Plant a genuine, disjoint cover (when admitted) mixed
                    # with random decoys; the source search must find it.
                    planted = [tuple(sites[i:i + 3])
                               for i in range(0, site_count, 3)]
                    selected = list(dict.fromkeys(planted + selected))[:18]
                    budget = max(budget, site_count // 3)
                scenario = Scenario(sites, tuple(selected), budget)
                expected = uncovered_set_oracle(scenario)
                source = search_allocation(scenario, node_limit=100_000)
                self.assertIn(source["verdict"],
                              ("VERIFIED_WIN", "PROVEN_UNSAT_UNDER_DECLARED_RULES"))
                self.assertEqual(source["verdict"] == "VERIFIED_WIN", expected,
                                 (site_count, case_no))
                self.assertEqual(independent_subset_oracle(scenario), expected)
                self.assertIs(_independent_cover_exists_dp(scenario), expected)
                if expected:
                    self.assertEqual(
                        verify_allocation(scenario, source["allocation"])["verdict"],
                        "VERIFIED_WIN")
                    seen_win += 1
                else:
                    seen_unsat += 1
        self.assertGreater(seen_win, 100)
        self.assertGreater(seen_unsat, 50)

    def test_site_renaming_and_crew_permutation_preserve_verdict(self):
        rng = random.Random(20261112)
        sites = tuple(f"s{i}" for i in range(12))
        options = list(combinations(sites, 3))
        for _ in range(75):
            rng.shuffle(options)
            scenario = Scenario(sites, tuple(options[:18]), 4)
            base = search_allocation(scenario)["verdict"]
            shuffled_sites = list(sites)
            rng.shuffle(shuffled_sites)
            renaming = dict(zip(sites, shuffled_sites))
            permuted = [tuple(renaming[site] for site in crew)
                        for crew in scenario.crews]
            rng.shuffle(permuted)
            transformed = Scenario(tuple(reversed(sites)), tuple(permuted), 4)
            self.assertEqual(search_allocation(transformed)["verdict"], base)


if __name__ == "__main__":
    unittest.main()
