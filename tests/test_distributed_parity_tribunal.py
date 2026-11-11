"""Hostile, independent tests of simultaneous private-information parity."""

from dataclasses import replace
import random
import unittest
from unittest.mock import patch

from labyrinth.distributed_parity_tribunal import (
    World, Scenario, adjudicate_unexpected, catalogue_control, check_scenario,
    families, fingerprint, fixed_script_control, independent_dfs_exists,
    independent_gf2_exists, independent_witness_check, novel_rule_challenge,
    public_view, search_policy, verify_policy,
)


def test_owned_oracle(s):
    """A third exhaustive existence oracle, entirely in the test module."""
    n = s.left_observations + s.right_observations
    for mask in range(1 << n):
        bits = [(mask >> i) & 1 for i in range(n)]
        if sum(bits) > s.total_on_budget:
            continue
        if all((bits[w.left_obs] ^ bits[s.left_observations + w.right_obs])
               == w.required_parity for w in s.worlds):
            return True
    return False


class DistributedParityTribunalTests(unittest.TestCase):
    def test_mechanical_fixtures_have_independent_consensus(self):
        expected = {
            "coordinated_parity": "VERIFIED_WIN",
            "missing_corner": "VERIFIED_WIN",
            "and_paradox": "PROVEN_UNSAT_UNDER_DECLARED_RULES",
            "inconsistent_reports": "PROVEN_UNSAT_UNDER_DECLARED_RULES",
        }
        for name, scenario in families().items():
            with self.subTest(name=name):
                verdict = search_policy(scenario)["verdict"]
                self.assertEqual(verdict, expected[name])
                exists = verdict == "VERIFIED_WIN"
                self.assertEqual(independent_dfs_exists(scenario), exists)
                self.assertEqual(independent_gf2_exists(scenario), exists)
                self.assertEqual(test_owned_oracle(scenario), exists)

    def test_minimized_over_budget_independent_dfs_regression(self):
        # An XOR target of 1 cannot be met with zero asserted-one entries.
        # The first DFS version incorrectly accepted a terminal over-budget
        # assignment; the third test-owned oracle catches that false positive.
        s = Scenario((World(0, 0, 1),), 2, 2, 0)
        self.assertFalse(test_owned_oracle(s))
        self.assertFalse(independent_dfs_exists(s))
        self.assertFalse(independent_gf2_exists(s))
        result = search_policy(s)
        self.assertEqual(result["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(result["certificate"]["enumerated"], 16)
        self.assertEqual(result["certificate"]["finite_space_size"], 16)
        self.assertEqual(result["certificate"]["valid_policies_found"], 0)

    def test_exact_witness_and_private_independent_replay(self):
        s = families()["coordinated_parity"]
        policy = ((0, 1), (0, 1))
        self.assertEqual(verify_policy(s, policy)["verdict"], "VERIFIED_WIN")
        self.assertTrue(independent_witness_check(s, policy))
        self.assertEqual(search_policy(s)["verdict"], "VERIFIED_WIN")
        self.assertFalse(independent_witness_check(s, ((0, 1), (0, 0))))
        self.assertEqual(verify_policy(s, ((0, 1), (0, 0)))["verdict"],
                         "VERIFIED_FAILURE")
        for malformed in ([], ((0, True), (0, 1)), ((0, 1), (0, 2)),
                          ((0,), (0, 1)), ((0, 1), (0, 1), ())):
            self.assertFalse(independent_witness_check(s, malformed))
            self.assertEqual(verify_policy(s, malformed)["verdict"],
                             "VERIFIED_FAILURE")

    def test_caps_are_not_impossibility_proofs(self):
        for s in families().values():
            self.assertEqual(search_policy(s, node_limit=0)["verdict"],
                             "NO_KNOWN_WIN")
            self.assertEqual(independent_dfs_exists(s, state_cap=0), None)
            self.assertEqual(independent_gf2_exists(s, state_cap=0), None)
            self.assertEqual(search_policy(s, node_limit=True)["verdict"],
                             "UNKNOWN")
        s = families()["and_paradox"]
        self.assertEqual(search_policy(s, independent_cap=0)["verdict"],
                         "UNKNOWN")

    def test_surprise_can_overturn_wrong_unsat_but_not_corrupt_verifier(self):
        s = families()["coordinated_parity"]
        alternate = ((1, 0), (1, 0))
        answer = adjudicate_unexpected(s, alternate,
                                       "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(answer["verdict"], "VERIFIED_WIN")
        self.assertTrue(answer["overturned_prior"])
        with patch("labyrinth.distributed_parity_tribunal.verify_policy",
                   return_value={"verdict": "VERIFIED_WIN",
                                 "rules_fingerprint": fingerprint(s)}):
            bad = adjudicate_unexpected(s, ((0, 0), (0, 0)),
                                        "VERIFIED_FAILURE")
        self.assertEqual(bad["verdict"], "UNKNOWN")
        with patch("labyrinth.distributed_parity_tribunal.verify_policy",
                   return_value={"verdict": "VERIFIED_FAILURE",
                                 "rules_fingerprint": fingerprint(s)}):
            corrupted = adjudicate_unexpected(s, alternate,
                                               "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(corrupted["verdict"], "UNKNOWN")

    def test_lying_unsat_oracle_is_not_a_certificate(self):
        s = families()["and_paradox"]
        with patch("labyrinth.distributed_parity_tribunal.independent_dfs_exists",
                   return_value=True):
            self.assertEqual(search_policy(s)["verdict"], "UNKNOWN")
        with patch("labyrinth.distributed_parity_tribunal.independent_gf2_exists",
                   return_value=None):
            self.assertEqual(search_policy(s)["verdict"], "UNKNOWN")

    def test_new_communication_rules_are_not_existing_wins(self):
        s = families()["and_paradox"]
        r = novel_rule_challenge(s, {"new_channel": "1-bit herald",
                                      "reason": "send right observation to left"})
        self.assertEqual(r["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(r["independent_verification"], "NOT_PERFORMED")
        for known in ("none", "private observation", "simultaneous output"):
            self.assertEqual(novel_rule_challenge(s, {"new_channel": known,
                                                     "reason": "existing"})["verdict"],
                             "UNKNOWN")
        for disguise in ("\u200b", "\u200d", "\u2060", "\ufeff",
                         "private observation!", "simultaneous output.",
                         " \ufeff ", "...", "None!!!"):
            with self.subTest(disguise=disguise):
                self.assertEqual(novel_rule_challenge(s,
                                 {"new_channel": disguise,
                                  "reason": "not a material new capability"})["verdict"],
                                 "UNKNOWN")

    def test_public_catalogue_script_and_paraphrases_do_not_learn(self):
        public = families()
        result = fixed_script_control()
        self.assertEqual(result["fixed_script_wins"], 0)
        self.assertEqual(result["symbolic_wins"], 2)
        self.assertEqual(result["optimizer_updates"], 0)
        unseen = Scenario((World(0, 0, 0), World(0, 1, 1)), 2, 2, 2)
        table = catalogue_control(unseen)
        self.assertEqual(table["unseen_verdict"], "UNKNOWN")
        self.assertEqual(table["exact_matches"], 4)
        self.assertEqual(table["optimizer_updates"], 0)
        s = public["and_paradox"]
        views = [public_view(s, variant=i) for i in range(8)]
        self.assertEqual(len({v["prompt"] for v in views}), 8)
        self.assertEqual({v["mechanics_fingerprint"] for v in views},
                         {fingerprint(s)})

    def test_exact_schema_rejects_python_bool_and_oversized_world(self):
        s = families()["coordinated_parity"]
        for case in (replace(s, left_observations=True),
                     replace(s, worlds=(World(True, 0, 0),)),
                     replace(s, worlds=(World(0, 0, 2),)),
                     replace(s, left_observations=5),
                     replace(s, total_on_budget=-1),
                     replace(s, worlds=())):
            with self.subTest(case=case), self.assertRaises(ValueError):
                check_scenario(case)

    def test_randomized_differential_and_observation_permutation(self):
        rng = random.Random(0xD157B1)
        for seed in range(512):
            left, right = rng.randint(2, 4), rng.randint(2, 4)
            n_worlds = rng.randint(2, 11)
            worlds = tuple(World(rng.randrange(left), rng.randrange(right),
                                 rng.randrange(2)) for _ in range(n_worlds))
            budget = rng.randint(0, left + right)
            s = Scenario(worlds, left, right, budget)
            expected = test_owned_oracle(s)
            with self.subTest(seed=seed):
                self.assertEqual(independent_gf2_exists(s), expected)
                self.assertEqual(independent_dfs_exists(s), expected)
                self.assertEqual(search_policy(s)["verdict"] == "VERIFIED_WIN",
                                 expected)
                lperm, rperm = list(range(left)), list(range(right))
                rng.shuffle(lperm)
                rng.shuffle(rperm)
                relabel = Scenario(tuple(World(lperm[w.left_obs], rperm[w.right_obs],
                                               w.required_parity) for w in worlds),
                                   left, right, budget)
                self.assertEqual(independent_gf2_exists(relabel), expected)
                self.assertEqual(search_policy(relabel)["verdict"] == "VERIFIED_WIN",
                                 expected)


if __name__ == "__main__":
    unittest.main()
