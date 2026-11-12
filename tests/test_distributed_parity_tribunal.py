"""Hostile, independent tests of simultaneous private-information parity."""

from dataclasses import replace
from itertools import product
import random
import unittest
from unittest.mock import patch

from labyrinth.distributed_parity_tribunal import (
    World, Scenario, adjudicate_unexpected, catalogue_control, check_scenario,
    families, fingerprint, mechanics_fingerprint, fixed_script_control, independent_dfs_exists,
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

    def test_two_correlated_lies_cannot_certify_specific_wrong_policy(self):
        s = families()["coordinated_parity"]
        # Existential solvers may correctly say SOME solution exists; that
        # does not verify THIS proposal, if both direct replay paths lie.
        with (patch("labyrinth.distributed_parity_tribunal.verify_policy",
                    return_value={"verdict": "VERIFIED_WIN",
                                  "rules_fingerprint": fingerprint(s)}),
              patch("labyrinth.distributed_parity_tribunal.independent_witness_check",
                    return_value=True)):
            self.assertEqual(search_policy(s)["verdict"], "UNKNOWN")
            self.assertEqual(adjudicate_unexpected(s, ((0, 0), (0, 0)),
                                                   "VERIFIED_FAILURE")["verdict"],
                             "UNKNOWN")

    def test_new_communication_rules_are_not_existing_wins(self):
        s = families()["and_paradox"]
        r = novel_rule_challenge(s, {"new_channel": "1-bit herald",
                                      "reason": "send right observation to left"})
        self.assertEqual(r["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(r["independent_verification"], "NOT_PERFORMED")
        for genuinely_new in (
                "no wires, but a 1-bit optical beacon",
                "without radio, add a shared semaphore",
                "unchanged isolation except a new 1-bit herald"):
            with self.subTest(genuinely_new=genuinely_new):
                changed = novel_rule_challenge(
                    s, {"new_channel": genuinely_new,
                        "reason": "Separate physical signal added to isolated actors"})
                self.assertEqual(changed["verdict"], "NOVEL_RULESET_CHALLENGE")
        for known in ("none", "private observation", "simultaneous output"):
            self.assertEqual(novel_rule_challenge(s, {"new_channel": known,
                                                     "reason": "existing"})["verdict"],
                             "UNKNOWN")
        for disguise in ("\u200b", "\u200d", "\u2060", "\ufeff",
                         "private observation!", "simultaneous output.",
                         " \ufeff ", "...", "None!!!", "no channel",
                         "no new channel", "simultaneous outputs",
                         "private observations"):
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
                         {mechanics_fingerprint(s)})

    def test_world_order_is_provenance_not_distinct_mechanics(self):
        original = families()["coordinated_parity"]
        reversed_worlds = replace(original,
                                  worlds=tuple(reversed(original.worlds)))
        repeated = replace(original, worlds=original.worlds + original.worlds[:1])
        changed = replace(original, worlds=original.worlds[:-1])
        self.assertNotEqual(fingerprint(original), fingerprint(reversed_worlds))
        self.assertEqual(mechanics_fingerprint(original),
                         mechanics_fingerprint(reversed_worlds))
        self.assertEqual(mechanics_fingerprint(original),
                         mechanics_fingerprint(repeated))
        self.assertNotEqual(mechanics_fingerprint(original),
                            mechanics_fingerprint(changed))
        self.assertEqual(public_view(original)["mechanics_fingerprint"],
                         public_view(reversed_worlds)["mechanics_fingerprint"])
        self.assertNotEqual(public_view(original)["case_fingerprint"],
                            public_view(reversed_worlds)["case_fingerprint"])

    def test_observation_relabeling_is_same_mechanics_not_new_family(self):
        # A consistent renaming of private observation indices cannot become
        # a new independent mechanical transfer family.
        source = Scenario((World(0, 0, 0), World(0, 1, 1),
                           World(1, 2, 1), World(2, 0, 0)), 3, 3, 3)
        lperm, rperm = (2, 0, 1), (1, 2, 0)
        relabeled = Scenario(
            tuple(World(lperm[w.left_obs], rperm[w.right_obs],
                        w.required_parity) for w in source.worlds),
            3, 3, 3)
        self.assertNotEqual(fingerprint(source), fingerprint(relabeled))
        self.assertEqual(mechanics_fingerprint(source),
                         mechanics_fingerprint(relabeled))

    def test_unused_observation_slots_do_not_manufacture_new_mechanics(self):
        # An observation never appearing in a possible world is unreachable.
        # It can always be assigned zero, leaving the win/UNSAT rules intact.
        source = Scenario((World(0, 0, 1), World(0, 1, 0)), 2, 2, 1)
        padded = Scenario((World(2, 1, 1), World(2, 3, 0)), 4, 4, 1)
        self.assertNotEqual(fingerprint(source), fingerprint(padded))
        self.assertEqual(test_owned_oracle(source), test_owned_oracle(padded))
        self.assertEqual(mechanics_fingerprint(source), mechanics_fingerprint(padded))

    def test_swapping_symmetric_actors_keeps_mechanical_identity(self):
        source = Scenario((World(0, 0, 0), World(1, 0, 1),
                           World(2, 1, 0)), 3, 2, 2)
        swapped = Scenario(
            tuple(World(w.right_obs, w.left_obs, w.required_parity)
                  for w in source.worlds), 2, 3, 2)
        self.assertNotEqual(fingerprint(source), fingerprint(swapped))
        self.assertEqual(test_owned_oracle(source), test_owned_oracle(swapped))
        self.assertEqual(mechanics_fingerprint(source), mechanics_fingerprint(swapped))

    def test_unconstrained_output_polarity_is_not_new_transfer_mechanics(self):
        # With the asserted-one budget nonbinding, flipping local output-bit
        # interpretations gives a bijection on feasible complete policies.
        source = Scenario((World(0, 0, 0), World(0, 1, 0),
                           World(1, 0, 0)), 2, 2, 4)
        changed = Scenario(tuple(
            World(w.left_obs, w.right_obs,
                  w.required_parity ^ int(w.left_obs == 0))
            for w in source.worlds), 2, 2, 4)
        self.assertNotEqual(fingerprint(source), fingerprint(changed))
        self.assertEqual(test_owned_oracle(source), test_owned_oracle(changed))
        self.assertEqual(mechanics_fingerprint(source),
                         mechanics_fingerprint(changed))

    def test_surplus_budget_on_unreachable_slots_is_not_new_mechanics(self):
        worlds = (World(0, 0, 0), World(0, 1, 1),
                  World(1, 0, 0))
        source = Scenario(worlds, 4, 4, 4)
        surplus = Scenario(worlds, 4, 4, 8)
        self.assertNotEqual(fingerprint(source), fingerprint(surplus))
        self.assertEqual(test_owned_oracle(source), test_owned_oracle(surplus))
        self.assertEqual(mechanics_fingerprint(source),
                         mechanics_fingerprint(surplus))

    def test_bit_polarity_is_not_a_symmetry_when_budget_binds(self):
        # A zero-ones budget permits all-zero parity, not a single odd edge.
        first = Scenario((World(0, 0, 0),), 2, 2, 0)
        odd = Scenario((World(0, 0, 1),), 2, 2, 0)
        self.assertTrue(test_owned_oracle(first))
        self.assertFalse(test_owned_oracle(odd))
        self.assertNotEqual(mechanics_fingerprint(first),
                            mechanics_fingerprint(odd))

    def test_random_full_budget_gauge_relabeling_preserves_identity(self):
        rng = random.Random(0xD1572027)
        for seed in range(48):
            left, right = rng.randint(2, 4), rng.randint(2, 4)
            pairs = [(a, b) for a in range(left) for b in range(right)]
            chosen = rng.sample(pairs, rng.randint(1, min(len(pairs), 12)))
            worlds = tuple(World(a, b, rng.randrange(2)) for a, b in chosen)
            source = Scenario(worlds, left, right, left + right)
            left_labels = rng.sample(range(left), left)
            right_labels = rng.sample(range(right), right)
            left_flip = tuple(rng.randrange(2) for _ in range(left))
            right_flip = tuple(rng.randrange(2) for _ in range(right))
            changed = Scenario(
                tuple(World(left_labels[w.left_obs], right_labels[w.right_obs],
                            w.required_parity ^ left_flip[w.left_obs]
                            ^ right_flip[w.right_obs]) for w in reversed(worlds)),
                left, right, left + right)
            with self.subTest(seed=seed):
                self.assertEqual(test_owned_oracle(source),
                                 test_owned_oracle(changed))
                self.assertEqual(mechanics_fingerprint(source),
                                 mechanics_fingerprint(changed))
                # Check the actual policy bijection as well as existence.
                for _ in range(3):
                    lbits = tuple(rng.randrange(2) for _ in range(left))
                    rbits = tuple(rng.randrange(2) for _ in range(right))
                    remap_left = [0] * left
                    remap_right = [0] * right
                    for index in range(left):
                        remap_left[left_labels[index]] = lbits[index] ^ left_flip[index]
                    for index in range(right):
                        remap_right[right_labels[index]] = rbits[index] ^ right_flip[index]
                    self.assertEqual(
                        verify_policy(source, (lbits, rbits))["verdict"],
                        verify_policy(changed,
                                      (tuple(remap_left), tuple(remap_right)))["verdict"])

    def test_small_exhaustive_mechanics_groups_never_merge_win_and_unsat(self):
        # Adversarial check against *over*-canonicalization: every 2x2
        # finite single-parity-per-pair scenario, every permitted budget.
        # Different scenarios may share an ID only if independent feasibility
        # agrees; this does not certify all possible semantic equivalences.
        seen = {}
        pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
        for values in product((-1, 0, 1), repeat=4):
            worlds = tuple(World(left, right, parity)
                           for (left, right), parity in zip(pairs, values)
                           if parity >= 0)
            if not worlds:
                continue
            for budget in range(5):
                scenario = Scenario(worlds, 2, 2, budget)
                key = mechanics_fingerprint(scenario)
                exists = test_owned_oracle(scenario)
                with self.subTest(values=values, budget=budget):
                    if key in seen:
                        self.assertEqual(seen[key], exists)
                    else:
                        seen[key] = exists
        conflict = Scenario((World(0, 0, 0), World(0, 0, 1)), 2, 2, 4)
        consistent = Scenario((World(0, 0, 0),), 2, 2, 4)
        self.assertFalse(test_owned_oracle(conflict))
        self.assertTrue(test_owned_oracle(consistent))
        self.assertNotEqual(mechanics_fingerprint(conflict),
                            mechanics_fingerprint(consistent))

    def test_random_unreachable_padding_and_actor_swap(self):
        rng = random.Random(0xD1572026)
        for seed in range(40):
            worlds = tuple(World(rng.randrange(2), rng.randrange(2),
                                 rng.randrange(2))
                           for _ in range(rng.randint(1, 10)))
            budget = rng.randint(0, 4)
            source = Scenario(worlds, 2, 2, budget)
            left_map, right_map = rng.sample(range(4), 2), rng.sample(range(4), 2)
            padded_swapped = Scenario(
                tuple(World(right_map[w.right_obs], left_map[w.left_obs],
                            w.required_parity) for w in reversed(worlds)),
                4, 4, budget)
            with self.subTest(seed=seed):
                self.assertEqual(test_owned_oracle(source),
                                 test_owned_oracle(padded_swapped))
                self.assertEqual(mechanics_fingerprint(source),
                                 mechanics_fingerprint(padded_swapped))
                self.assertNotEqual(fingerprint(source),
                                    fingerprint(padded_swapped))

    def test_random_local_label_permutations_preserve_mechanics(self):
        rng = random.Random(0xC0FFEE)
        for seed in range(40):
            left, right = rng.randint(2, 4), rng.randint(2, 4)
            worlds = tuple(
                World(rng.randrange(left), rng.randrange(right),
                      rng.randrange(2))
                for _ in range(rng.randint(1, 12))
            )
            budget = rng.randint(0, left + right)
            source = Scenario(worlds, left, right, budget)
            left_labels, right_labels = list(range(left)), list(range(right))
            rng.shuffle(left_labels)
            rng.shuffle(right_labels)
            transformed = Scenario(
                tuple(World(left_labels[w.left_obs], right_labels[w.right_obs],
                            w.required_parity) for w in reversed(worlds)),
                left, right, budget)
            with self.subTest(seed=seed):
                self.assertEqual(mechanics_fingerprint(source),
                                 mechanics_fingerprint(transformed))

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
                candidate_rng = random.Random(seed ^ 0xFACE81)
                candidate_bits = tuple(candidate_rng.randrange(2)
                                       for _ in range(left + right))
                candidate = (candidate_bits[:left], candidate_bits[left:])
                candidate_expected = (
                    sum(candidate_bits) <= budget and
                    all((candidate_bits[w.left_obs] ^
                         candidate_bits[left + w.right_obs]) == w.required_parity
                        for w in worlds))
                self.assertEqual(independent_gf2_exists(s, fixed_policy=candidate),
                                 candidate_expected)
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
