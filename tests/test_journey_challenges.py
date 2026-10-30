import dataclasses
import unittest

from labyrinth.journey_challenges import prepare_training_rooms
from labyrinth.maze import generate_maze
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


def scope(skills=("arithmetic", "source_priority")):
    return TrainingScopeSpec(
        objective_id="demo-mechanism", domain="reasoning", skills=skills,
        specificity=2, scale=1, max_cycles=3, seed=77,
    )


class TrainingChambersTests(unittest.TestCase):
    def test_semantic_specificity_changes_program_training_content(self):
        def program_scope(specificity):
            return TrainingScopeSpec(
                objective_id="specific-program-journey", domain="program_reasoning",
                skills=("program_reasoning",), specificity=specificity,
                scale=2, max_cycles=2, seed=19,
            )

        broad = program_scope(1)
        narrow = program_scope(5)
        broad_rooms = prepare_training_rooms(generate_maze_plan(broad, 0), semantic_scope=broad)
        narrow_rooms = prepare_training_rooms(generate_maze_plan(narrow, 0), semantic_scope=narrow)
        self.assertTrue(all(dict(item.challenge.metadata)["semantic_focus"]
                            == "mixed_3_to_4_instruction_structures" for item in broad_rooms))
        self.assertTrue(all(dict(item.challenge.metadata)["semantic_focus"]
                            == "dependency_path_wrapping_both_registers" for item in narrow_rooms))
        self.assertTrue(all(dict(item.challenge.metadata)["wrap_events"].isdigit()
                            for item in narrow_rooms))
        self.assertTrue(all(item.challenge.split == "train" for item in narrow_rooms))
        self.assertTrue(all("answer" not in dataclasses.asdict(item.trainee_view)
                            for item in narrow_rooms))

    def test_semantic_provider_rejects_scope_mismatch_and_override(self):
        left = scope(("program_reasoning",))
        right = TrainingScopeSpec(
            objective_id=left.objective_id, domain=left.domain, skills=left.skills,
            specificity=5, scale=left.scale, max_cycles=left.max_cycles, seed=left.seed,
        )
        plan = generate_maze_plan(left, 0)
        with self.assertRaises(ValueError):
            prepare_training_rooms(plan, semantic_scope=right)
        with self.assertRaises(ValueError):
            prepare_training_rooms(plan, semantic_scope=left, providers={})

    def test_generated_rooms_have_real_train_tasks(self):
        plan = generate_maze_plan(scope(), 0)
        cases = prepare_training_rooms(plan)
        self.assertEqual(len(cases), len(plan.rooms))
        self.assertTrue(all(room.challenge.split == "train" for room in cases))
        self.assertEqual(len({room.challenge.id for room in cases}), len(cases))
        self.assertEqual(len({room.trainee_view.prompt_sha256 for room in cases}), len(cases))
        self.assertTrue(all("answer" not in dataclasses.asdict(room.trainee_view) for room in cases))

    def test_regeneration_changes_case_content_not_just_ids(self):
        first = prepare_training_rooms(generate_maze_plan(scope(), 0))
        second = prepare_training_rooms(generate_maze_plan(scope(), 1))
        self.assertTrue(set(r.trainee_view.prompt_sha256 for r in first).isdisjoint(
            r.trainee_view.prompt_sha256 for r in second
        ))

    def test_unregistered_empathy_skill_cannot_be_faked(self):
        plan = generate_maze_plan(scope(("cognitive_empathy",)), 0)
        with self.assertRaises(ValueError):
            prepare_training_rooms(plan)

    def test_external_provider_must_return_training_data(self):
        plan = generate_maze_plan(scope(("custom",)), 0)
        with self.assertRaises(ValueError):
            prepare_training_rooms(plan, providers={"custom": lambda seed: generate_maze(seed, (1,), 1, "eval")[0]})


if __name__ == "__main__":
    unittest.main()
