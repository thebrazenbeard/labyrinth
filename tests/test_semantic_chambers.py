"""Behavioral checks for narrower program-reasoning training content."""

from __future__ import annotations

import re
import unittest

from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.semantic_chambers import generate_semantic_program_challenge, semantic_focus_profile
from labyrinth.training_scope import TrainingScopeSpec


def _scope(specificity: int, *, skill: str = "program_reasoning") -> TrainingScopeSpec:
    return TrainingScopeSpec(
        objective_id=f"program-focus-{specificity}", domain="reasoning", skills=(skill,),
        specificity=specificity, scale=2, max_cycles=2, seed=20261030,
    )


def _replay(source: str) -> tuple[int, int, set[str]]:
    """Independent small register interpreter for checking semantic profiles."""
    start = re.search(r"Initial registers: A=(\d+), B=(\d+)", source)
    if start is None:
        raise AssertionError("no initial state")
    values = {"A": int(start.group(1)), "B": int(start.group(2))}
    wraps = set()
    count = 0
    for match in re.finditer(r"(?m)^(\d+)\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", source):
        count += 1
        self_index, operation, target, operand = match.groups()
        if int(self_index) != count:
            raise AssertionError("nonsequential program")
        other = values[operand] if operand in values else int(operand)
        lhs = values[target]
        raw = ((lhs + other) if operation == "ADD" else
               (lhs - other) if operation == "SUB" else lhs * other)
        if not 0 <= raw < 97:
            wraps.add(target)
        values[target] = raw % 97
    return values["A"], count, wraps


class SemanticChambersTests(unittest.TestCase):
    def _case(self, specificity: int):
        scope = _scope(specificity)
        room = generate_maze_plan(scope, 0).rooms[0]
        return generate_semantic_program_challenge(scope, room)

    def test_each_specification_changes_task_constraints_not_just_room_count(self):
        cases = [self._case(level) for level in (1, 2, 3, 4, 5)]
        self.assertTrue(all(case.split == "train" for case in cases))
        self.assertTrue(all(case.tier == 4 for case in cases))
        self.assertTrue(all(case.kind == "register_program" for case in cases))
        self.assertEqual(len({case.prompt for case in cases}), 5)
        for level, case in enumerate(cases, 1):
            answer, count, wraps = _replay(case.prompt)
            self.assertEqual(int(case.answer), answer)
            self.assertEqual(dict(case.metadata)["specificity"], str(level))
            self.assertEqual(dict(case.metadata)["semantic_focus"], semantic_focus_profile(level))
            if level >= 2:
                self.assertEqual(count, 4)
            if level >= 3:
                self.assertIn("1. MUL A", case.prompt)
                self.assertIn("2. SUB B A", case.prompt)
                self.assertIn("3. ADD B", case.prompt)
                self.assertIn("4. ADD A B", case.prompt)
            if level >= 4:
                # At least two modular wrap events require cross-step residue tracking.
                self.assertGreaterEqual(int(dict(case.metadata)["wrap_events"]), 2)
            if level == 5:
                self.assertEqual(wraps, {"A", "B"})

    def test_train_only_and_new_cases_across_regenerated_cycles(self):
        scope = _scope(5)
        first = generate_maze_plan(scope, 0)
        second = generate_maze_plan(scope, 1)
        old = [generate_semantic_program_challenge(scope, room) for room in first.rooms]
        new = [generate_semantic_program_challenge(scope, room) for room in second.rooms]
        self.assertEqual(len({case.id for case in old + new}), len(old + new))
        self.assertEqual(len({case.prompt for case in old + new}), len(old + new))
        self.assertTrue(all(dict(case.metadata)["profile"] == "train" for case in old + new))

    def test_reproducible_selection_and_validated_training_contract(self):
        scope = _scope(4)
        room = generate_maze_plan(scope, 0).rooms[0]
        self.assertEqual(generate_semantic_program_challenge(scope, room),
                         generate_semantic_program_challenge(scope, room))
        forged = object.__new__(TrainingScopeSpec)
        for key, value in scope.__dict__.items():
            object.__setattr__(forged, key, value)
        object.__setattr__(forged, "specificity", 1000)
        with self.assertRaises(ValueError):
            generate_semantic_program_challenge(forged, room)

    def test_adaptive_difficulty_is_bounded_and_does_not_mutate_declared_scope(self):
        scope = _scope(2)
        room = generate_maze_plan(scope, 0).rooms[0]
        focused = generate_semantic_program_challenge(scope, room, difficulty=3)
        self.assertEqual(dict(focused.metadata)["adaptive_difficulty"], "3")
        self.assertEqual(dict(focused.metadata)["specificity"], "3")
        self.assertEqual(scope.specificity, 2)
        self.assertEqual(focused.split, "train")
        self.assertEqual(dict(generate_semantic_program_challenge(scope, room).metadata)["specificity"], "2")
        for invalid in (0, 6, True, 2.0, "3"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generate_semantic_program_challenge(scope, room, difficulty=invalid)

    def test_same_train_room_difficulty_alters_executable_structure(self):
        # Hold the scope, room, and seed fixed so novelty cannot explain
        # the difference. Only difficulty changes the selected structure.
        declared = TrainingScopeSpec(
            objective_id="abc", domain="reasoning",
            skills=("program_reasoning",), specificity=2, scale=2,
            max_cycles=2, seed=123,
        )
        room = generate_maze_plan(declared, 0).rooms[0]
        broad = generate_semantic_program_challenge(declared, room, difficulty=1)
        focused = generate_semantic_program_challenge(declared, room, difficulty=3)
        self.assertNotEqual(dict(broad.metadata)["structure"],
                            dict(focused.metadata)["structure"])
        self.assertEqual(_replay(broad.prompt)[0], int(broad.answer))
        self.assertEqual(_replay(focused.prompt)[0], int(focused.answer))
        self.assertEqual(_replay(broad.prompt)[1], 3)
        self.assertEqual(_replay(focused.prompt)[1], 4)

    def test_unsupported_skill_fails_closed(self):
        scope = _scope(3, skill="cognitive_empathy")
        room = generate_maze_plan(scope, 0).rooms[0]
        with self.assertRaisesRegex(ValueError, "program_reasoning"):
            generate_semantic_program_challenge(scope, room)


if __name__ == "__main__":
    unittest.main()
