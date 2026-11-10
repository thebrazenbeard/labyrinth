"""Adversarial full-contract checks for the independent Tribunal oracle."""

import random
from itertools import product
import unittest
from unittest.mock import patch

import labyrinth.rule_reversal_tribunal as tribunal
from labyrinth.rule_reversal_independent_oracle import can_guarantee, verifies_witness


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
        witness = tribunal.search_policy(s)["policy"]
        with patch.object(tribunal, "_advance", side_effect=RuntimeError("shared engine broken")):
            self.assertTrue(can_guarantee(_masks(s), s.budget, s.sensor))
            self.assertTrue(verifies_witness(_masks(s), s.budget, s.sensor, witness))

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

    def test_replay_callback_cannot_mutate_emitted_winning_witness(self):
        s = tribunal.families()["revision_under_feedback"]
        original_replay = tribunal.replay_policy

        def dishonest_callback(scenario, candidate):
            verified = original_replay(scenario, candidate)
            self.assertEqual(verified["verdict"], "VERIFIED_WIN")
            candidate.clear()
            candidate["action"] = "COMMIT"
            return verified

        with patch.object(tribunal, "replay_policy", side_effect=dishonest_callback):
            claimed = tribunal.search_policy(s)
        self.assertEqual(claimed["verdict"], "VERIFIED_WIN")
        self.assertEqual(original_replay(s, claimed["policy"])["verdict"],
                         "VERIFIED_WIN", "emitted witness must be the one verified")

    def test_independent_witness_vetoes_shared_fabricated_replay_win(self):
        s = tribunal.families()["revision_under_feedback"]
        original_advance = tribunal._advance

        def forged_commit(world, state, action, sensor):
            if action == "COMMIT":
                return state, "COMMITTED", True, True
            return original_advance(world, state, action, sensor)

        with (patch.object(tribunal, "_advance", side_effect=forged_commit),
              patch.object(tribunal, "replay_policy",
                           return_value={"verdict": "VERIFIED_WIN"})):
            claimed = tribunal.search_policy(s)
        self.assertEqual(claimed["verdict"], "UNKNOWN")
        self.assertEqual(claimed["reason"], "independent_witness_disagreement")

    def test_rich_comparison_subclass_cannot_forge_replay_agreement(self):
        s = tribunal.families()["revision_under_feedback"]

        class Liar(str):
            def __eq__(self, other):
                return other == "VERIFIED_WIN"

            def __ne__(self, other):
                return other != "VERIFIED_WIN"

        with patch.object(tribunal, "replay_policy",
                          return_value={"verdict": Liar("VERIFIED_FAILURE")}):
            claimed = tribunal.search_policy(s)
        self.assertEqual(claimed["verdict"], "UNKNOWN")
        self.assertEqual(claimed["reason"], "solver_replay_disagreement")

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
                self.assertTrue(verifies_witness(_masks(s), s.budget,
                                                s.sensor, solved["policy"]))
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
