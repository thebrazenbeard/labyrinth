"""Independent hostile tests for order-sensitive temporal repair challenge."""

from dataclasses import replace
from functools import lru_cache
import random
import unittest
from unittest.mock import patch

from labyrinth.temporal_repair_tribunal import (
    Job, Scenario, adjudicate_unexpected, catalogue_control, check_scenario,
    families, fingerprint, fixed_script_control, independent_reachability,
    independent_witness_check, novel_rule_challenge, public_view,
    search_schedule, verify_schedule,
)


def independent_test_enumeration(s):
    """Test-owned exhaustive DFS, expressed without production transitions."""
    @lru_cache(None)
    def rec(t, fuel, completed, used):
        if completed == (1 << len(s.jobs)) - 1:
            return True
        if used >= s.action_budget or t >= s.horizon:
            return False
        # Waiting is a real charged action that restores energy.
        if rec(t + 1, min(s.capacity, fuel + 1), completed, used + 1):
            return True
        for i, job in enumerate(s.jobs):
            if completed & (1 << i):
                continue
            if any(not completed & (1 << p) for p in job.after):
                continue
            if t < job.release or fuel < job.energy:
                continue
            end = t + job.duration
            if end <= job.deadline and end <= s.horizon:
                if rec(end, fuel - job.energy, completed | (1 << i), used + 1):
                    return True
        return False

    return rec(0, s.capacity, 0, 0)


class TemporalRepairTribunalTests(unittest.TestCase):
    def test_order_energy_release_and_deadline_are_mechanically_active(self):
        cases = families()
        expected = {
            "recharge_window": "VERIFIED_WIN",
            "release_pivot": "VERIFIED_WIN",
            "dependency_cycle": "PROVEN_UNSAT_UNDER_DECLARED_RULES",
            "deadline_collision": "PROVEN_UNSAT_UNDER_DECLARED_RULES",
        }
        for name, s in cases.items():
            with self.subTest(case=name):
                self.assertEqual(search_schedule(s)["verdict"], expected[name])
                self.assertEqual(independent_reachability(s),
                                 expected[name] == "VERIFIED_WIN")
                self.assertEqual(independent_test_enumeration(s),
                                 expected[name] == "VERIFIED_WIN")

    def test_verified_winner_carries_exact_independent_witness(self):
        for name in ("recharge_window", "release_pivot"):
            s = families()[name]
            solved = search_schedule(s)
            witness = solved["schedule"]
            self.assertIs(type(witness), tuple)
            self.assertEqual(verify_schedule(s, witness)["verdict"], "VERIFIED_WIN")
            self.assertTrue(independent_witness_check(s, witness))
            self.assertLessEqual(len(witness), s.action_budget)
            self.assertIn(-1, witness, "waiting must matter in these fixtures")

    def test_attempt_budget_invalid_action_and_deadline_reject(self):
        s = families()["recharge_window"]
        candidates = ((), (0, 0, 0), (True, 0), (99,), [-1, 0],
                      tuple([-1] * 20), (0, -2, 1), (1, 0, 2))
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertEqual(verify_schedule(s, candidate)["verdict"],
                                 "VERIFIED_FAILURE")
                self.assertFalse(independent_witness_check(s, candidate))

    def test_incomplete_search_never_claims_impossibility(self):
        for case in families().values():
            self.assertEqual(search_schedule(case, node_limit=0)["verdict"],
                             "NO_KNOWN_WIN")
        self.assertEqual(search_schedule(case, node_limit=True)["verdict"],
                         "UNKNOWN")

    def test_valid_surprise_overturns_prior_unsat_but_fake_does_not(self):
        s = families()["recharge_window"]
        witness = search_schedule(s)["schedule"]
        r = adjudicate_unexpected(s, witness,
                                  "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(r["verdict"], "VERIFIED_WIN")
        self.assertTrue(r["overturned_prior"])
        bad = adjudicate_unexpected(s, (0, 1, 2), "VERIFIED_WIN")
        self.assertEqual(bad["verdict"], "VERIFIED_FAILURE")

    def test_valid_unexpected_witness_cannot_be_suppressed_by_lying_verifier(self):
        s = families()["recharge_window"]
        good = search_schedule(s)["schedule"]
        self.assertTrue(independent_witness_check(s, good))
        with patch("labyrinth.temporal_repair_tribunal.verify_schedule",
                   return_value={"verdict": "VERIFIED_FAILURE",
                                 "rules_fingerprint": fingerprint(s)}):
            r = adjudicate_unexpected(s, good,
                                      "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(r["verdict"], "UNKNOWN")
        self.assertFalse(r["independently_replayed"])

    def test_forged_winner_and_wrong_rules_fingerprint_fail_closed(self):
        s = families()["recharge_window"]
        valid = search_schedule(s)["schedule"]
        bad = (0, 1, 2)
        with patch("labyrinth.temporal_repair_tribunal.verify_schedule",
                   return_value={"verdict": "VERIFIED_WIN",
                                 "rules_fingerprint": fingerprint(s)}):
            r = adjudicate_unexpected(s, bad, "VERIFIED_FAILURE")
        self.assertEqual(r["verdict"], "UNKNOWN")
        with patch("labyrinth.temporal_repair_tribunal.verify_schedule",
                   return_value={"verdict": "VERIFIED_WIN",
                                 "rules_fingerprint": "another-ruleset"}):
            r = adjudicate_unexpected(s, valid, "VERIFIED_FAILURE")
        self.assertEqual(r["verdict"], "UNKNOWN")

    def test_declarative_rule_change_not_auto_win_or_auto_unsat(self):
        s = families()["dependency_cycle"]
        proposal = {"new_action": "borrow_generator",
                    "reason": "Add one independent emergency generator"}
        r = novel_rule_challenge(s, proposal)
        self.assertEqual(r["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(r["independent_verification"], "NOT_PERFORMED")
        self.assertEqual(search_schedule(s)["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        for already in ("RUN", "WAIT"):
            with self.subTest(action=already):
                known = novel_rule_challenge(s, {"new_action": already,
                                                 "reason": "Already available"})
                self.assertEqual(known["verdict"], "UNKNOWN")

    def test_source_public_catalogue_and_fixed_script_do_not_learn(self):
        unseen = Scenario((Job("u", 0, 8, 1, 1, ()),
                           Job("v", 0, 8, 1, 1, ()),
                           Job("w", 0, 8, 1, 1, ())), 2, 8, 7)
        self.assertNotIn(fingerprint(unseen),
                         {fingerprint(s) for s in families().values()})
        table = catalogue_control(unseen)
        self.assertEqual(table["public_fixtures"], 4)
        self.assertEqual(table["exact_matches"], 4)
        self.assertEqual(table["unseen_verdict"], "UNKNOWN")
        self.assertEqual(table["optimizer_updates"], 0)
        fixed = fixed_script_control()
        self.assertEqual(fixed["fixed_script_wins"], 0)
        self.assertEqual(fixed["symbolic_wins"], 2)
        self.assertEqual(fixed["optimizer_updates"], 0)

    def test_wording_variants_are_not_new_mechanics(self):
        s = families()["recharge_window"]
        views = [public_view(s, variant=v) for v in range(8)]
        self.assertEqual(len({x["prompt"] for x in views}), 8)
        self.assertEqual({x["mechanics_fingerprint"] for x in views},
                         {fingerprint(s)})

    def test_exact_schema_rejects_deceptive_values_and_duplicates(self):
        s = families()["recharge_window"]
        malformed = (replace(s, horizon=True),
                     replace(s, jobs=s.jobs + (s.jobs[0],)),
                     replace(s, jobs=(replace(s.jobs[0], duration=0),) + s.jobs[1:]),
                     replace(s, jobs=(replace(s.jobs[0], after=(True,)),) + s.jobs[1:]),
                     replace(s, capacity=10), replace(s, action_budget=0))
        for case in malformed:
            with self.subTest(case=case), self.assertRaises(ValueError):
                check_scenario(case)

    def test_tester_that_lies_about_success_is_not_a_witness(self):
        class Liar(str):
            def __eq__(self, other):
                return other == "VERIFIED_WIN"

            def __ne__(self, other):
                return not self.__eq__(other)

        s = families()["recharge_window"]
        with patch("labyrinth.temporal_repair_tribunal.verify_schedule",
                   return_value={"verdict": Liar("VERIFIED_FAILURE")}):
            r = search_schedule(s)
        self.assertEqual(r["verdict"], "UNKNOWN")
        self.assertNotIn("schedule", r)

    def test_tester_that_lies_about_reachability_cannot_prove_unsat(self):
        winning = families()["recharge_window"]
        failing = families()["dependency_cycle"]
        with patch("labyrinth.temporal_repair_tribunal.independent_reachability",
                   return_value=False):
            r = search_schedule(winning)
        self.assertEqual(r["verdict"], "UNKNOWN")
        with patch("labyrinth.temporal_repair_tribunal.independent_depth_enumeration",
                   return_value=True):
            r = search_schedule(failing)
        self.assertEqual(r["verdict"], "UNKNOWN")
        self.assertNotIn("certificate_scope", r)

    def test_malicious_rule_change_dict_key_fails_closed(self):
        class KeyTrap(str):
            def __eq__(self, other):
                raise RuntimeError("malicious equality callback")

            __hash__ = str.__hash__

        r = novel_rule_challenge(
            families()["dependency_cycle"],
            {KeyTrap("new_action"): "free_energy", "reason": "new power"})
        self.assertEqual(r["verdict"], "UNKNOWN")

    def test_randomized_independent_mechanical_differential(self):
        rng = random.Random(20261111)
        checks = 0
        for _ in range(160):
            horizon = rng.randint(5, 9)
            cap = rng.randint(1, 3)
            count = rng.randint(3, 4)
            jobs = []
            for i in range(count):
                release = rng.randint(0, horizon - 1)
                deadline = rng.randint(max(1, release), horizon)
                deps = tuple(p for p in range(i) if rng.random() < 0.3)
                jobs.append(Job(f"j{i}", release, deadline,
                                rng.randint(1, 3), rng.randint(1, cap), deps))
            s = Scenario(tuple(jobs), cap, horizon, rng.randint(4, 9))
            expected = independent_test_enumeration(s)
            self.assertEqual(independent_reachability(s), expected, s)
            actual = search_schedule(s)
            self.assertEqual(actual["verdict"] == "VERIFIED_WIN", expected, s)
            self.assertIn(actual["verdict"],
                          ("VERIFIED_WIN", "PROVEN_UNSAT_UNDER_DECLARED_RULES"))
            if expected:
                self.assertTrue(independent_witness_check(s, actual["schedule"]))
            checks += 1
        self.assertEqual(checks, 160)


if __name__ == "__main__":
    unittest.main()
