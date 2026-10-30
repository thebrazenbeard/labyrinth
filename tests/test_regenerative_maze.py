"""Source-only tests for regenerative, strictly train-only maze planning."""

import random
import unittest

from labyrinth.regenerative_maze import MAX_ROOMS, MazePlan, MazeRoom, generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


def scope(**overrides):
    fields = {
        "objective_id": "reasoning.chain_transfer",
        "domain": "reasoning",
        "skills": ("arithmetic", "source_priority", "program_reasoning"),
        "specificity": 3,
        "scale": 3,
        "max_cycles": 4,
        "seed": 731,
    }
    fields.update(overrides)
    return TrainingScopeSpec(**fields)


class RegenerativeMazeTests(unittest.TestCase):
    def test_deterministic_plan_and_fair_skill_coverage(self):
        declared = scope()
        baseline = generate_maze_plan(declared, 0)
        self.assertIsInstance(baseline, MazePlan)
        self.assertEqual(baseline, generate_maze_plan(declared, 0))
        self.assertEqual(baseline.scope_fingerprint, declared.fingerprint)
        self.assertEqual(baseline.objective_id, declared.objective_id)
        self.assertEqual(len(baseline.rooms), len(declared.skills) * declared.chamber_count_per_skill)
        self.assertEqual(set(baseline.counts_by_skill), set(declared.skills))
        self.assertEqual(len(set(baseline.counts_by_skill.values())), 1)
        self.assertTrue(all(room.purpose == "train" for room in baseline.rooms))

    def test_regeneration_uses_new_room_ids_and_case_seeds(self):
        declared = scope()
        plans = [generate_maze_plan(declared, cycle) for cycle in range(declared.max_cycles)]
        ids = [room.room_id for plan in plans for room in plan.rooms]
        seeds = [room.case_seed for plan in plans for room in plan.rooms]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(type(seed) is int and 0 <= seed < 2**31 for seed in seeds))
        self.assertNotEqual(plans[0].rooms, plans[1].rooms)

    def test_failure_redistributes_fixed_budget_without_starvation(self):
        declared = scope()
        first = generate_maze_plan(declared, 0)
        retraining = generate_maze_plan(declared, 1, weak_skills=("source_priority",))
        self.assertEqual(len(first.rooms), len(retraining.rooms))
        self.assertGreater(
            retraining.counts_by_skill["source_priority"],
            first.counts_by_skill["source_priority"],
        )
        self.assertTrue(all(retraining.counts_by_skill[skill] >= 1 for skill in declared.skills))
        self.assertEqual(
            retraining,
            generate_maze_plan(declared, 1, weak_skills=("source_priority",)),
        )

    def test_weak_skill_set_order_does_not_change_plan(self):
        declared = scope()
        left = generate_maze_plan(declared, 1, weak_skills=("source_priority", "arithmetic"))
        right = generate_maze_plan(declared, 1, weak_skills=("arithmetic", "source_priority"))
        self.assertEqual(left, right)

    def test_all_skills_weak_remains_balanced(self):
        declared = scope()
        baseline = generate_maze_plan(declared, 0)
        same = generate_maze_plan(declared, 1, weak_skills=declared.skills)
        self.assertEqual(baseline.counts_by_skill, same.counts_by_skill)

    def test_invalid_cycle_and_malformed_feedback_fail_closed(self):
        declared = scope()
        for cycle in (-1, declared.max_cycles, True, 1.2, "1"):
            with self.subTest(cycle=cycle), self.assertRaises(ValueError):
                generate_maze_plan(declared, cycle)
        for feedback in (("unknown",), ("arithmetic", "arithmetic"), "arithmetic", iter(("arithmetic",)), (None,)):
            with self.subTest(feedback=feedback), self.assertRaises(ValueError):
                generate_maze_plan(declared, 1, weak_skills=feedback)

    def test_invalid_scope_and_post_construction_mutation_fail_closed(self):
        with self.assertRaises(ValueError):
            generate_maze_plan(object(), 0)
        declared = scope()
        object.__setattr__(declared, "scale", 1_000_000)
        with self.assertRaises(ValueError):
            generate_maze_plan(declared, 0)

    def test_planning_does_not_touch_global_random_state(self):
        before = random.getstate()
        generate_maze_plan(scope(), 0)
        self.assertEqual(random.getstate(), before)

    def test_scale_bounds_total_size(self):
        declared = scope(specificity=5, scale=5)
        plan = generate_maze_plan(declared, 0)
        self.assertLessEqual(len(plan.rooms), MAX_ROOMS)
        self.assertGreater(len(plan.rooms), 0)

    def test_manual_forged_exam_room_and_duplicate_plan_rejected(self):
        declared = scope()
        plan = generate_maze_plan(declared, 0)
        with self.assertRaises(ValueError):
            MazeRoom(plan.rooms[0].room_id, plan.rooms[0].skill, plan.rooms[0].case_seed, purpose="exam")
        with self.assertRaises(ValueError):
            MazePlan(plan.objective_id, plan.scope_fingerprint, plan.cycle_index, (plan.rooms[0], plan.rooms[0]))
        damaged = plan.rooms[0]
        object.__setattr__(damaged, "purpose", "exam")
        with self.assertRaises(ValueError):
            MazePlan(plan.objective_id, plan.scope_fingerprint, plan.cycle_index, plan.rooms)


if __name__ == "__main__":
    unittest.main()
