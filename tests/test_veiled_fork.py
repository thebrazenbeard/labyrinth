"""Independent public-seam adversarial tests for Veiled Fork v1."""

import json
from itertools import product
import random
import unittest

from labyrinth.veiled_fork import (
    Probe, Scenario, brute_force_oracle, challenge_novel_rule,
    check_scenario, control_trial, families, find_guaranteed_policy,
    replay_policy,
)


class MaliciousList(list):
    def __len__(self):
        return 0


class MaliciousString(str):
    def __eq__(self, other):
        return True


class StateDependentBranchKey(str):
    """Forge different decisions from the *same* observed string."""

    def __new__(cls, value, first_match_only):
        self = super().__new__(cls, value)
        self.first_match_only = first_match_only
        self.lookups = 0
        return self

    def __hash__(self):
        return hash("same")

    def __eq__(self, other):
        if isinstance(other, StateDependentBranchKey):
            return self is other
        if other == "same":
            self.lookups += 1
            return not self.first_match_only or self.lookups == 1
        return False


class EpistemicExamTests(unittest.TestCase):
    def test_each_hidden_world_has_a_clairvoyant_winning_commit(self):
        scenario = families()["unobservable_fork"]
        for world, exit_ in enumerate(scenario.exits):
            self.assertEqual(exit_, world)
        self.assertEqual(find_guaranteed_policy(scenario)["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_simultaneous_world_policy_is_not_a_pick_the_lucky_case(self):
        s = families()["unobservable_fork"]
        result = replay_policy(s, {"kind": "commit", "exit": 0})
        self.assertEqual([t["won"] for t in result["world_traces"]], [True, False])
        self.assertEqual(result["verdict"], "VERIFIED_FAILURE")

    def test_observation_conditioning_can_guarantee_escape(self):
        s = families()["identifiable_fork"]
        result = find_guaranteed_policy(s)
        self.assertEqual(result["verdict"], "VERIFIED_WIN")
        self.assertEqual(replay_policy(s, result["policy"])["verdict"], "VERIFIED_WIN")

    def test_too_late_to_probe_proves_unsat_not_universe_impossibility(self):
        s = families()["too_late_to_know"]
        result = find_guaranteed_policy(s)
        self.assertEqual(result["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertIn("public observation policy", result["quantifier"])

    def test_two_signals_require_distinct_conditional_actions(self):
        s = families()["two_signal_mosaic"]
        self.assertEqual(find_guaranteed_policy(s)["verdict"], "VERIFIED_WIN")
        self.assertFalse(brute_force_oracle(Scenario(s.exits, s.probes, 2)))
        self.assertTrue(brute_force_oracle(s))

    def test_common_safe_exit_reverses_impossible_conclusion(self):
        # Change the actual mechanics, not merely the word labels.
        s = families()["common_safe_exit"]
        result = find_guaranteed_policy(s)
        self.assertEqual(result["verdict"], "VERIFIED_WIN")
        self.assertEqual(result["policy"], {"kind": "commit", "exit": 1})

    def test_adaptive_probe_policy_beats_every_fixed_two_probe_script(self):
        # For each nonadaptive pair, at least two opposite-exit worlds look
        # identical; the candidate must choose its second probe conditionally.
        s = families()["adaptive_branching_advantage"]
        probes = list(s.probes)
        evaluated = 0
        for first, second in product(probes, repeat=2):
            evaluated += 1
            ambiguous = [(a, b) for a in range(len(s.exits))
                         for b in range(a + 1, len(s.exits))
                         if s.exits[a] != s.exits[b]
                         and (first.answers[a], second.answers[a]) ==
                         (first.answers[b], second.answers[b])]
            self.assertTrue(ambiguous, f"fixed probe pair {first.name}/{second.name} resolved all worlds")
        self.assertEqual(evaluated, 9)
        solved = find_guaranteed_policy(s)
        self.assertEqual(solved["verdict"], "VERIFIED_WIN")
        self.assertEqual(replay_policy(s, solved["policy"])["verdict"], "VERIFIED_WIN")
        self.assertTrue(brute_force_oracle(s))
        self.assertFalse(brute_force_oracle(Scenario(s.exits, s.probes, 2)))

    def test_bounded_search_does_not_upgrade_no_known_win_to_unsat(self):
        s = families()["two_signal_mosaic"]
        self.assertEqual(find_guaranteed_policy(s, node_limit=1)["verdict"], "NO_KNOWN_WIN")

    def test_forged_action_does_not_count_as_novel_verified_win(self):
        s = families()["unobservable_fork"]
        claim = challenge_novel_rule(s, {"action": "ask minotaur for a third door",
                                           "claim": "There is an unlisted emergency passage"})
        self.assertEqual(claim["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(claim["independent_verification"], "NOT_PERFORMED")
        self.assertEqual(find_guaranteed_policy(s)["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_invalid_rule_proposal_remains_unknown(self):
        s = families()["unobservable_fork"]
        self.assertEqual(challenge_novel_rule(s, {"action": "commit", "claim": "trust me"})["verdict"],
                         "UNKNOWN")

    def test_novel_proposal_rejects_stateful_keys(self):
        s = families()["unobservable_fork"]
        proposal = {StateDependentBranchKey("first", True): "magic passage", "claim": "an exception"}
        self.assertEqual(challenge_novel_rule(s, proposal)["verdict"], "UNKNOWN")

    def test_strict_world_types_exclude_forged_identity(self):
        s = Scenario((0, True), (), 1)
        with self.assertRaisesRegex(ValueError, "invalid exit"):
            check_scenario(s)

    def test_strict_probe_types_exclude_forged_comparison(self):
        s = Scenario((0, 1), (Probe(MaliciousString("seen"), ("0", "1")),), 2)
        with self.assertRaisesRegex(ValueError, "invalid probe"):
            check_scenario(s)

    def test_probe_name_and_policy_freezer_agree_at_string_boundary(self):
        # An accepted solvable exam used to return UNKNOWN when its solver's
        # probe name exceeded the policy-freezer's 128-character value cap.
        valid = Scenario((0, 1), (Probe("p" * 128, ("left", "right")),), 2)
        check_scenario(valid)
        self.assertTrue(brute_force_oracle(valid))
        self.assertEqual(find_guaranteed_policy(valid)["verdict"], "VERIFIED_WIN")

        invalid = Scenario((0, 1), (Probe("p" * 129, ("left", "right")),), 2)
        with self.assertRaisesRegex(ValueError, "invalid probe identity"):
            check_scenario(invalid)

    def test_invalid_action_is_charged_and_not_retried(self):
        s = families()["identifiable_fork"]
        r = replay_policy(s, {"kind": "probe", "name": "nonsense", "branches": {}})
        self.assertEqual(r["verdict"], "VERIFIED_FAILURE")
        self.assertTrue(all(t["attempts"] == 1 for t in r["world_traces"]))

    def test_no_uncharged_excess_probes_before_commit(self):
        s = families()["identifiable_fork"]
        policy = {"kind": "probe", "name": "inspect", "branches": {
            "left": {"kind": "probe", "name": "inspect", "branches": {
                "left": {"kind": "commit", "exit": 0}}},
            "right": {"kind": "commit", "exit": 1},
        }}
        result = replay_policy(s, policy)
        self.assertEqual(result["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(result["world_traces"][0]["attempts"], 2)

    def test_malformed_subclass_values_cannot_bypass_bound(self):
        s = families()["identifiable_fork"]
        policy = {"kind": "probe", "name": "inspect", "branches": MaliciousList()}
        self.assertEqual(replay_policy(s, policy)["verdict"], "VERIFIED_FAILURE")

    def test_stateful_branch_key_cannot_spoof_omniscient_policy(self):
        # RED on prototype v1: world0 and world1 see the identical "same"
        # observation, yet Python dict equality changes the selected branch.
        k0 = StateDependentBranchKey("first", True)
        k1 = StateDependentBranchKey("second", False)
        s = families()["unobservable_fork"]
        policy = {"kind": "probe", "name": "echo", "branches": {
            k0: {"kind": "commit", "exit": 0},
            k1: {"kind": "commit", "exit": 1},
        }}
        self.assertEqual(find_guaranteed_policy(s)["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(replay_policy(s, policy)["verdict"], "VERIFIED_FAILURE")

    def test_cyclic_policy_terminates_and_fails_closed(self):
        s = families()["unobservable_fork"]
        cycle = {"kind": "probe", "name": "echo", "branches": {}}
        cycle["branches"]["same"] = cycle
        result = replay_policy(s, cycle)
        self.assertEqual(result["verdict"], "VERIFIED_FAILURE")
        self.assertEqual(result["input_error"], "cyclic_policy")

    def test_control_does_not_call_a_fixed_script_learning(self):
        report = control_trial()
        self.assertFalse(report["learning_claim"])
        self.assertEqual(report["neural_updates"], 0)
        self.assertTrue(all(row["agreement"] for row in report["families"].values()))
        self.assertFalse(report["families"]["two_signal_mosaic"]["fixed_exit0_robust_win"])

    def test_randomized_solver_vs_independent_boolean_oracle_1000_cases(self):
        rng = random.Random(20261109)
        for trial in range(1000):
            n = rng.randrange(2, 7)
            probes = tuple(Probe("probe" + str(i), tuple(
                str(rng.randrange(3)) for _ in range(n))) for i in range(rng.randrange(0, 5)))
            s = Scenario(tuple(rng.randrange(2) for _ in range(n)), probes, rng.randrange(1, 6))
            expected = brute_force_oracle(s)
            actual = find_guaranteed_policy(s)
            self.assertEqual(actual["verdict"] == "VERIFIED_WIN", expected,
                             f"trial={trial}, scenario={s!r}, solver={actual!r}")

    def test_json_round_trip_prevents_python_type_tricks_in_policy(self):
        s = families()["two_signal_mosaic"]
        policy = find_guaranteed_policy(s)["policy"]
        replica = json.loads(json.dumps(policy))
        self.assertEqual(replay_policy(s, replica)["verdict"], "VERIFIED_WIN")


if __name__ == "__main__":
    unittest.main()
