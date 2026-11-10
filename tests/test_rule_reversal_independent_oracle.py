"""Adversarial full-contract checks for the independent Tribunal oracle."""

import random
from itertools import product
import unittest
from unittest.mock import patch

import labyrinth.rule_reversal_tribunal as tribunal
from labyrinth.rule_reversal_independent_oracle import can_guarantee


def _masks(s):
    return tuple((w.pre_a, w.pre_b, w.post_a, w.post_b) for w in s.worlds)


class RuleReversalIndependentOracleTests(unittest.TestCase):
    def test_full_admitted_six_world_eight_action_contract(self):
        worlds = tuple(tribunal.World(1, 2, 1, 2) for _ in range(6))
        s = tribunal.Scenario(worlds, 8, "full")
        self.assertTrue(can_guarantee(_masks(s), s.budget, s.sensor))
        self.assertEqual(tribunal.search_policy(s)["verdict"], "VERIFIED_WIN")

    def test_oracle_is_physically_separate_from_shared_transition(self):
        s = tribunal.families()["revision_under_feedback"]
        with patch.object(tribunal, "_advance", side_effect=RuntimeError("shared engine broken")):
            self.assertTrue(can_guarantee(_masks(s), s.budget, s.sensor))

    def test_shared_false_unsat_is_vetoed_not_certified(self):
        s = tribunal.families()["revision_under_feedback"]

        def broken_action(world, state, action, sensor):
            return state, "BROKEN", False, False

        with patch.object(tribunal, "_advance", side_effect=broken_action):
            result = tribunal.search_policy(s)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["reason"], "independent_oracle_disagreement")

    def test_incomplete_search_stays_nonproven(self):
        s = tribunal.families()["revision_under_feedback"]
        result = tribunal.search_policy(s, node_limit=0)
        self.assertEqual(result["verdict"], "NO_KNOWN_WIN")
        self.assertFalse(result["exhaustive"])

    def test_seeded_fuzz_uses_full_six_world_eight_action_space(self):
        rng = random.Random(20261110)
        checked = wins = unsat = 0
        for _ in range(2200):
            worlds = tuple(tribunal.World(*(rng.randrange(4) for _ in range(4)))
                           for _ in range(rng.randint(1, 6)))
            s = tribunal.Scenario(worlds, rng.randint(1, 8),
                                  rng.choice(("full", "parity", "sealed")))
            expected = can_guarantee(_masks(s), s.budget, s.sensor)
            solved = tribunal.search_policy(s, node_limit=500000)
            self.assertIn(solved["verdict"],
                          ("VERIFIED_WIN", "PROVEN_UNSAT_UNDER_DECLARED_RULES"), s)
            self.assertEqual(solved["verdict"] == "VERIFIED_WIN", expected, s)
            if expected:
                self.assertEqual(tribunal.replay_policy(s, solved["policy"])["verdict"],
                                 "VERIFIED_WIN")
                wins += 1
            else:
                unsat += 1
            checked += 1
        self.assertEqual(checked, 2200)
        self.assertGreater(wins, 100)
        self.assertGreater(unsat, 100)

    def test_exhaustive_small_post_shift_laws_against_third_oracle(self):
        checked = 0
        for masks in product(range(4), repeat=4):
            first = tribunal.World(1, 2, masks[0], masks[1])
            second = tribunal.World(1, 2, masks[2], masks[3])
            for sensor in ("full", "sealed"):
                for budget in (4, 5):
                    scenario = tribunal.Scenario((first, second), budget, sensor)
                    self.assertEqual(
                        can_guarantee(_masks(scenario), budget, sensor),
                        tribunal.brute_force_oracle(scenario),
                        (masks, sensor, budget),
                    )
                    checked += 1
        self.assertEqual(checked, 1024)


if __name__ == "__main__":
    unittest.main()
