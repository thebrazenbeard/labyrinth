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
