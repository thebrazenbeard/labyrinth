"""Boundary tests: Minotaur suggestions affect only trainer-side room content."""

from __future__ import annotations

import re
import unittest

from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.taskmaster_routing import route_training_rooms
from labyrinth.training_scope import TrainingScopeSpec


def _scope(*, skills=("arithmetic",), specificity=2, scale=2):
    return TrainingScopeSpec(
        objective_id="route-test", domain="arithmetic",
        skills=skills, specificity=specificity, scale=scale,
        max_cycles=2, seed=17,
    )


class FixedTaskmaster:
    def __init__(self, output):
        self.output = output
        self.received = []

    def propose_room(self, metadata):
        self.received.append(metadata)
        return self.output


class BrokenTaskmaster:
    def propose_room(self, metadata):
        raise RuntimeError("do not include internal secrets in fallback")


class TaskmasterRoutingTests(unittest.TestCase):
    def test_two_model_adapters_change_actual_arithmetic_tasks(self):
        scope = _scope()
        maze = generate_maze_plan(scope, 0)
        easy = FixedTaskmaster({"difficulty": 1, "tactic": "practice"})
        hard = FixedTaskmaster({"difficulty": 5, "tactic": "contrast"})
        first = route_training_rooms(scope, maze, taskmaster=easy)
        second = route_training_rooms(scope, maze, taskmaster=hard)
        self.assertEqual(len(first), scope.total_chambers)
        self.assertEqual(len(second), len(first))
        self.assertTrue(all(room.challenge.split == "train" for room in first + second))
        self.assertEqual({r.suggestion.tactic for r in first}, {"practice"})
        self.assertEqual({r.suggestion.tactic for r in second}, {"contrast"})
        self.assertTrue(all(f.challenge.prompt != s.challenge.prompt for f, s in zip(first, second)))
        self.assertTrue(all(f.challenge.id != s.challenge.id for f, s in zip(first, second)))

        for room in first:
            match = re.search(r"Calculate (\d+) ([+-]) (\d+)\.", room.challenge.prompt)
            self.assertIsNotNone(match)
            a, op, b = int(match[1]), match[2], int(match[3])
            self.assertTrue(1 <= a <= 9 and 1 <= b <= 9)
            self.assertEqual(int(room.challenge.answer), a + b if op == "+" else a - b)
        for room in second:
            match = re.search(r"Expression A is (\d+) \+ (\d+)\. Expression B", room.challenge.prompt)
            self.assertIsNotNone(match)
            a, b = int(match[1]), int(match[2])
            self.assertTrue(10000 <= a <= 99999 and 10000 <= b <= 99999)
            self.assertEqual(int(room.challenge.answer), 2 * b)

        for planner in (easy, hard):
            self.assertEqual(len(planner.received), len(first))
            for metadata in planner.received:
                self.assertNotIn("answer", metadata)
                self.assertNotIn("exam", metadata)
                self.assertNotIn("checkpoint", metadata)
                with self.assertRaises(TypeError):
                    metadata["graduation"] = "PASS"

    def test_transfer_is_a_new_word_problem_structure(self):
        scope = _scope(scale=1)
        maze = generate_maze_plan(scope, 0)
        transfer = route_training_rooms(
            scope, maze, taskmaster=FixedTaskmaster({"difficulty": 3, "tactic": "transfer"}),
        )
        self.assertTrue(all("workshop" in room.challenge.prompt for room in transfer))
        self.assertTrue(all("Calculate" not in room.challenge.prompt for room in transfer))
        self.assertTrue(all(room.suggestion.difficulty == 3 for room in transfer))
        self.assertEqual(len({room.trainee_view.prompt_sha256 for room in transfer}), len(transfer))

    def test_invalid_or_crashed_proposals_fall_back_to_valid_training(self):
        scope = _scope()
        maze = generate_maze_plan(scope, 0)
        default = route_training_rooms(scope, maze)
        invalid = route_training_rooms(
            scope, maze, taskmaster=FixedTaskmaster({"difficulty": 5, "tactic": "practice", "graduated": True}),
        )
        broken = route_training_rooms(scope, maze, taskmaster=BrokenTaskmaster())
        self.assertEqual([room.challenge for room in default], [room.challenge for room in invalid])
        self.assertEqual([room.challenge for room in default], [room.challenge for room in broken])
        self.assertEqual({r.suggestion.fallback_reason for r in invalid}, {"invalid_suggestion"})
        self.assertEqual({r.suggestion.fallback_reason for r in broken}, {"adapter_error"})
        self.assertNotIn("secret", repr(broken))

    def test_regeneration_changes_train_prompts_without_leaking_gold(self):
        scope = _scope()
        plan_a = generate_maze_plan(scope, 0)
        plan_b = generate_maze_plan(scope, 1)
        a = route_training_rooms(scope, plan_a)
        b = route_training_rooms(scope, plan_b)
        self.assertEqual(len(a), len(b))
        self.assertTrue(set(x.trainee_view.prompt_sha256 for x in a).isdisjoint(
            x.trainee_view.prompt_sha256 for x in b
        ))
        for room in a:
            self.assertFalse(hasattr(room.trainee_view, "answer"))
            self.assertFalse(hasattr(room.trainee_view, "evaluation"))

    def test_unapproved_skill_or_mismatched_scope_fails_closed(self):
        empathy = _scope(skills=("cognitive_empathy",))
        witness = FixedTaskmaster({"difficulty": 5, "tactic": "contrast"})
        with self.assertRaisesRegex(ValueError, "only arithmetic"):
            route_training_rooms(empathy, generate_maze_plan(empathy, 0), taskmaster=witness)
        self.assertEqual(witness.received, [])
        other = _scope(specificity=4)
        with self.assertRaisesRegex(ValueError, "does not match"):
            route_training_rooms(other, generate_maze_plan(_scope(), 0))


if __name__ == "__main__":
    unittest.main()
