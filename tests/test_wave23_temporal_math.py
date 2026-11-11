"""Wave-23 independent temporal falsifiers, including ruleset-admission spoofs.

The test oracle deliberately does not import production transition helpers.
Its evidence is bounded to the declared finite single-machine rule system.
"""

from dataclasses import replace
from functools import lru_cache
import random
import unittest

from labyrinth.temporal_repair_tribunal import (
    Job, Scenario, adjudicate_unexpected, families, fingerprint,
    independent_depth_enumeration, independent_reachability,
    independent_witness_check, novel_rule_challenge, search_schedule,
    verify_schedule,
)


def outside_witness(s):
    """Return an independently searched exact plan, or None if none exists."""
    all_done = frozenset(range(len(s.jobs)))

    @lru_cache(None)
    def visit(t, battery, done, spent):
        if done == all_done:
            return ()
        if spent >= s.action_budget or t >= s.horizon:
            return None
        # Never call production solver, replay, or reachability here.
        for idx in range(len(s.jobs)):
            job = s.jobs[idx]
            if idx in done or t < job.release or battery < job.energy:
                continue
            if not set(job.after).issubset(done):
                continue
            if t + job.duration > min(job.deadline, s.horizon):
                continue
            suffix = visit(t + job.duration, battery - job.energy,
                           done | frozenset((idx,)), spent + 1)
            if suffix is not None:
                return (idx,) + suffix
        suffix = visit(t + 1, min(s.capacity, battery + 1), done, spent + 1)
        return (-1,) + suffix if suffix is not None else None

    return visit(0, s.capacity, frozenset(), 0)


def constructed_pair(rng):
    """Generate a real WIN and its structurally different cyclic UNSAT twin."""
    count = rng.randint(3, 6)
    capacity = rng.randint(1, 4)
    energy = capacity
    t = 0
    tasks = []
    actions = []
    for i in range(count):
        cost = rng.randint(1, capacity)
        while energy < cost:
            t += 1
            energy = min(capacity, energy + 1)
            actions.append(-1)
        duration = rng.randint(1, 2)
        end = t + duration
        deadline = min(18, end + rng.randint(0, 2))
        tasks.append(Job(f"run{i}", rng.randint(0, t), deadline,
                         duration, cost, (i - 1,) if i else ()))
        t = end
        energy -= cost
        actions.append(i)
    if t > 18 or len(actions) > 18:
        # The declared horizon is fixed at 18; reject overlong generated plans.
        return constructed_pair(rng)
    winning = Scenario(tuple(tasks), capacity, 18,
                       len(actions) + rng.randint(0, 18 - len(actions)))
    impossible = replace(winning, jobs=(replace(tasks[0], after=(count - 1,)),
                                         *tasks[1:]))
    return winning, impossible, tuple(actions)


class Wave23TemporalMathematicsTests(unittest.TestCase):
    def test_novel_rule_claim_must_have_actual_action_and_mechanism(self):
        s = families()["dependency_cycle"]
        aliases = ("   ", "\t\n", "WAIT\u200b", "RUN\u200c",
                   "ＷＡＩＴ", "ＲＵＮ", "\u2060WAIT")
        for action in aliases:
            with self.subTest(action=repr(action)):
                result = novel_rule_challenge(s, {
                    "new_action": action, "reason": "new mechanism"})
                self.assertEqual(result["verdict"], "UNKNOWN")
        for reason in ("   ", "\t", "\u200b", "\u2060"):
            with self.subTest(reason=repr(reason)):
                result = novel_rule_challenge(s, {
                    "new_action": "borrow_generator", "reason": reason})
                self.assertEqual(result["verdict"], "UNKNOWN")
        real = novel_rule_challenge(s, {
            "new_action": "borrow_generator",
            "reason": "Activate a generator absent from the declared rules"})
        self.assertEqual(real["verdict"], "NOVEL_RULESET_CHALLENGE")
        self.assertEqual(real["independent_verification"], "NOT_PERFORMED")

    def test_independent_solver_falsifies_false_unsat_and_false_wins(self):
        rng = random.Random(20261111)
        for case in range(256):
            horizon = rng.randint(4, 18)
            capacity = rng.randint(1, 4)
            count = rng.randint(3, 6)
            jobs = []
            for i in range(count):
                release = rng.randint(0, horizon - 1)
                deadline = rng.randint(1, horizon)
                prerequisites = tuple(j for j in range(count)
                                      if rng.random() < 0.24)
                jobs.append(Job(f"case{i}", release, deadline,
                                rng.randint(1, 4),
                                rng.randint(1, capacity), prerequisites))
            s = Scenario(tuple(jobs), capacity, horizon,
                         rng.randint(1, 18))
            independent = outside_witness(s)
            reached = independent_reachability(s)
            depth = independent_depth_enumeration(s)
            result = search_schedule(s)
            with self.subTest(case=case, fingerprint=fingerprint(s)):
                self.assertEqual(reached, independent is not None)
                self.assertEqual(depth, independent is not None)
                self.assertEqual(result["verdict"] == "VERIFIED_WIN",
                                 independent is not None)
                if independent is not None:
                    self.assertEqual(verify_schedule(s, independent)["verdict"],
                                     "VERIFIED_WIN")
                    self.assertTrue(independent_witness_check(s, independent))
                    self.assertEqual(adjudicate_unexpected(
                        s, independent, "PROVEN_UNSAT_UNDER_DECLARED_RULES"
                    )["verdict"], "VERIFIED_WIN")
                else:
                    self.assertEqual(result["verdict"],
                                     "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_constructive_wins_and_cyclic_twins_admit_no_fake_proof(self):
        rng = random.Random(230618)
        positives = negatives = 0
        waits = 0
        for case in range(128):
            win, fail, planted = constructed_pair(rng)
            with self.subTest(case=case):
                self.assertEqual(verify_schedule(win, planted)["verdict"],
                                 "VERIFIED_WIN")
                self.assertTrue(independent_witness_check(win, planted))
                self.assertIsNotNone(outside_witness(win))
                self.assertTrue(independent_reachability(win))
                self.assertTrue(independent_depth_enumeration(win))
                self.assertEqual(search_schedule(win)["verdict"], "VERIFIED_WIN")
                self.assertIsNone(outside_witness(fail))
                self.assertFalse(independent_reachability(fail))
                self.assertFalse(independent_depth_enumeration(fail))
                self.assertEqual(search_schedule(fail)["verdict"],
                                 "PROVEN_UNSAT_UNDER_DECLARED_RULES")
            positives += 1
            negatives += 1
            waits += -1 in planted
        self.assertEqual((positives, negatives), (128, 128))
        self.assertGreater(waits, 30, "recharge must be exercised substantially")

    def test_budget_caps_and_correlated_fault_boundaries(self):
        cases = families()
        for name, s in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(independent_reachability(s, state_cap=0))
                self.assertIsNone(independent_depth_enumeration(s, state_cap=0))
                for limit in (0, 1, 2, 3):
                    bounded = search_schedule(s, node_limit=limit)
                    self.assertNotEqual(bounded["verdict"], "VERIFIED_FAILURE")
                    if bounded["verdict"] == "PROVEN_UNSAT_UNDER_DECLARED_RULES":
                        self.assertIsNone(outside_witness(s))
                    if bounded["verdict"] == "VERIFIED_WIN":
                        self.assertTrue(independent_witness_check(
                            s, bounded["schedule"]))

    def test_two_independent_deadline_faults_require_both_repairs(self):
        # Minimal schedule uses two paid recharges before B and one before C:
        # RUN A, WAIT, WAIT, RUN B, WAIT, RUN C at ticks 0,1,2,3,4,5.
        valid = Scenario((
            Job("A", 0, 1, 1, 2, ()),
            Job("B", 1, 4, 1, 2, (0,)),
            Job("C", 2, 6, 1, 1, (1,)),
        ), 2, 6, 6)
        planted = (0, -1, -1, 1, -1, 2)
        self.assertEqual(verify_schedule(valid, planted)["verdict"],
                         "VERIFIED_WIN")
        altered_b = replace(valid, jobs=(valid.jobs[0],
                                          replace(valid.jobs[1], deadline=3),
                                          valid.jobs[2]))
        altered_c = replace(valid, jobs=(valid.jobs[0], valid.jobs[1],
                                          replace(valid.jobs[2], deadline=5)))
        both = replace(valid, jobs=(valid.jobs[0], altered_b.jobs[1],
                                    altered_c.jobs[2]))
        for scenario in (altered_b, altered_c, both):
            self.assertIsNone(outside_witness(scenario))
            self.assertFalse(independent_reachability(scenario))
            self.assertFalse(independent_depth_enumeration(scenario))
            self.assertEqual(search_schedule(scenario)["verdict"],
                             "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(search_schedule(valid)["verdict"], "VERIFIED_WIN")

    def test_semantically_equivalent_renamings_and_reindexing(self):
        s = families()["recharge_window"]
        baseline = outside_witness(s) is not None
        renamed = replace(s, jobs=tuple(replace(j, name="ren" + str(i))
                                        for i, j in enumerate(s.jobs)))
        reverse = tuple(reversed(range(len(s.jobs))))
        mapping = {old: new for new, old in enumerate(reverse)}
        permuted = replace(s, jobs=tuple(replace(
            s.jobs[old], after=tuple(mapping[d] for d in s.jobs[old].after))
            for old in reverse))
        for changed in (renamed, permuted):
            self.assertNotEqual(fingerprint(s), fingerprint(changed))
            self.assertEqual(outside_witness(changed) is not None, baseline)
            self.assertEqual(search_schedule(changed)["verdict"], "VERIFIED_WIN")


if __name__ == "__main__":
    unittest.main()
