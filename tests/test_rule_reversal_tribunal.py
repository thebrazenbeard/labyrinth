"""Independent causal-intervention and adversarial tests for Tribunal v1."""

from itertools import product
import random
import unittest

from labyrinth.rule_reversal_tribunal import (
    Scenario, World, brute_force_oracle, check_scenario, control_trial,
    families, fingerprint, novel_rule_challenge, public_view,
    replay_policy, search_policy,
)


class DeceptiveStr(str):
    __hash__ = str.__hash__

    def __eq__(self, other):
        return True


class RuleReversalTribunalTests(unittest.TestCase):
    def test_rule_reversal_requires_causal_revision_not_lucky_script(self):
        s = families()["revision_under_feedback"]
        result = search_policy(s)
        self.assertEqual(result["verdict"], "VERIFIED_WIN")
        verified = replay_policy(s, result["policy"])
        self.assertEqual(verified["verdict"], "VERIFIED_WIN")
        self.assertEqual(len(verified["world_traces"]), 2)
        self.assertEqual([r["won"] for r in verified["world_traces"]], [True, True])

        # A separate primitive fixed-sequence oracle enumerates all <=5 scripts.
        def fixed_wins(world, script):
            bits, shifted, pre, post = 0, False, 0, 0
            for index, action in enumerate(script):
                if action == "COMMIT":
                    return (index == len(script) - 1 and shifted and pre > 0
                            and post > 0 and bits == 3)
                if action == "SHIFT":
                    if shifted or pre == 0:
                        return False
                    shifted = True
                elif action in ("A", "B"):
                    if shifted:
                        post += 1
                        bits ^= (world.post_a if action == "A" else world.post_b)
                    else:
                        pre += 1
                        bits ^= (world.pre_a if action == "A" else world.pre_b)
                else:
                    return False
            return False

        scripts = [seq for length in range(1, s.budget + 1)
                   for seq in product(("A", "B", "SHIFT", "COMMIT"), repeat=length)
                   if seq[-1] == "COMMIT"]
        self.assertFalse(any(all(fixed_wins(world, script) for world in s.worlds)
                             for script in scripts))
        self.assertGreater(len(scripts), 100)

    def test_rule_shift_changes_same_action_causality(self):
        s = families()["revision_under_feedback"]
        self.assertEqual({w.pre_a for w in s.worlds}, {1})
        self.assertEqual({w.post_a for w in s.worlds}, {2, 3})
        self.assertNotEqual(s.worlds[0].pre_a, s.worlds[0].post_a)

    def test_action_budget_can_make_all_world_policy_unsatisfiable(self):
        s = families()["too_few_interventions"]
        self.assertEqual(search_policy(s)["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(brute_force_oracle(s))

    def test_causal_reachability_not_wording_only(self):
        s = families()["no_y_causality"]
        self.assertEqual(search_policy(s)["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(brute_force_oracle(s))
        changed = families()["common_post_law"]
        self.assertEqual(search_policy(changed)["verdict"], "VERIFIED_WIN")
        self.assertTrue(brute_force_oracle(changed))

    def test_loss_of_observation_can_remove_guaranteed_policy(self):
        s = families()["revision_without_feedback"]
        self.assertEqual(search_policy(s)["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertFalse(brute_force_oracle(s))

    def test_bounded_search_does_not_forge_proof(self):
        s = families()["revision_under_feedback"]
        limited = search_policy(s, node_limit=0)
        self.assertEqual(limited["verdict"], "NO_KNOWN_WIN")
        self.assertFalse(limited["exhaustive"])

    def test_public_view_preserves_private_world_custody_contract(self):
        s = families()["revision_under_feedback"]
        a, b = public_view(s, variant=0), public_view(s, variant=5)
        self.assertNotEqual(a["prompt"], b["prompt"])
        self.assertEqual(fingerprint(s), fingerprint(s))
        self.assertEqual(set(a), {"prompt", "actions", "budget", "sensor", "initial_observation"})
        public = str(a).lower()
        for secret in ("post_a", "pre_b", "worlds", "rules_fingerprint", "target_bits"):
            self.assertNotIn(secret, public)

    def test_invalid_actions_are_charged_and_cannot_win(self):
        s = families()["revision_under_feedback"]
        result = replay_policy(s, {"action": "UNDECLARED"})
        self.assertEqual(result["verdict"], "VERIFIED_FAILURE")
        self.assertTrue(all(t["attempts"] == 1 for t in result["world_traces"]))

    def test_policy_input_is_frozen_before_any_world(self):
        s = families()["revision_under_feedback"]
        fake = {"action": DeceptiveStr("A"), "branches": {}}
        self.assertEqual(replay_policy(s, fake)["verdict"], "VERIFIED_FAILURE")
        self.assertIn("input_error", replay_policy(s, fake))
        cyclic = {"action": "A", "branches": {}}
        cyclic["branches"]["PRE:01"] = cyclic
        self.assertEqual(replay_policy(s, cyclic)["verdict"], "VERIFIED_FAILURE")

    def test_input_types_reject_boolean_masks_and_fake_worlds(self):
        with self.assertRaises(ValueError):
            check_scenario(Scenario((World(True, 2, 1, 1),), 5, "full"))
        with self.assertRaises(ValueError):
            check_scenario(Scenario((World(1, 2, 1, 1),), True, "full"))

    def test_novel_ruleset_is_challenged_not_awarded_a_win(self):
        s = families()["no_y_causality"]
        r = novel_rule_challenge(s, {"action": "request-third-actuator",
                                      "claim": "A distinct causal intervention may reach y"})
        self.assertEqual(r["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(r["independent_verification"], "NOT_PERFORMED")
        self.assertEqual(search_policy(s)["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(novel_rule_challenge(s, {"action": "A", "claim": "trust me"})["verdict"],
                         "UNKNOWN")
        self.assertEqual(novel_rule_challenge(
            s, {DeceptiveStr("action"): "third actuator", "claim": "secret exit"})["verdict"],
            "UNKNOWN")

    def test_zero_update_control_does_not_demonstrate_learning(self):
        s = families()["revision_under_feedback"]
        c = control_trial(s)
        self.assertEqual(c["optimizer_updates"], 0)
        self.assertEqual(c["fixed_policy_verdict"], "VERIFIED_FAILURE")
        self.assertEqual(c["fixed_world_wins"], 1)
        self.assertEqual(c["adaptive_symbolic_verdict"], "VERIFIED_WIN")

    def test_exhaustive_post_shift_mechanics_differential(self):
        # Exhaust the 16×16 pairs of post-shift actuator maps, at both the
        # marginal and sufficient budgets, with and without useful feedback.
        checked = 0
        for a, b, c, d in product(range(4), repeat=4):
            worlds = (World(1, 2, a, b), World(1, 2, c, d))
            for sensor in ("full", "sealed"):
                for budget in (4, 5):
                    s = Scenario(worlds, budget, sensor)
                    expected = brute_force_oracle(s)
                    actual = search_policy(s)
                    self.assertEqual(actual["verdict"] == "VERIFIED_WIN", expected, s)
                    self.assertNotEqual(actual["verdict"], "UNKNOWN")
                    checked += 1
        self.assertEqual(checked, 1024)

    def test_randomized_differential_oracle(self):
        rng = random.Random(20261109)
        checked = 0
        for _ in range(300):
            worlds = tuple(World(*(rng.randrange(4) for _ in range(4)))
                           for _ in range(rng.randint(1, 3)))
            s = Scenario(worlds, rng.randint(1, 5), rng.choice(("full", "sealed", "parity")))
            expected = brute_force_oracle(s)
            actual = search_policy(s)
            self.assertIn(actual["verdict"],
                          ("VERIFIED_WIN", "PROVEN_UNSAT_UNDER_DECLARED_RULES"))
            self.assertEqual(actual["verdict"] == "VERIFIED_WIN", expected, s)
            if expected:
                self.assertEqual(replay_policy(s, actual["policy"])["verdict"], "VERIFIED_WIN")
            checked += 1
        self.assertEqual(checked, 300)


if __name__ == "__main__":
    unittest.main()
