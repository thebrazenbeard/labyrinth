import dataclasses
import hashlib
import unittest

from labyrinth.adaptive_corridors import (
    TrainingRoomFeedback, begin_adaptive_walk, offer_next_chamber,
    record_training_feedback,
)
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
    def test_adaptive_program_chambers_materialize_feedback_driven_semantic_difficulty(self):
        # One fixed declared scope: only *train* feedback changes the next
        # program's semantic profile, without changing learner identity.
        declared = TrainingScopeSpec(
            objective_id="adaptive-program-content", domain="reasoning",
            skills=("program_reasoning",), specificity=2, scale=3,
            max_cycles=2, seed=123,
        )
        initial = offer_next_chamber(begin_adaptive_walk(
            declared, generate_maze_plan(declared, 0), checkpoint_sha256="a" * 64,
        ))
        from labyrinth.journey_challenges import prepare_pending_adaptive_program_room

        first = prepare_pending_adaptive_program_room(initial)
        self.assertEqual(dict(first.challenge.metadata)["specificity"], "2")
        self.assertEqual(first.challenge.split, "train")
        self.assertNotIn("answer", dataclasses.asdict(first.trainee_view))

        for score, expected_difficulty in ((0, 1), (1, 3)):
            feedback = TrainingRoomFeedback(
                room_id=initial.pending.room_id,
                case_id="train-" + initial.pending.room_id,
                prompt_sha256=hashlib.sha256(first.challenge.prompt.encode()).hexdigest(),
                score=score, split="train",
                checkpoint_before_sha256="a" * 64,
                checkpoint_after_sha256="b" * 64,
            )
            routed = offer_next_chamber(record_training_feedback(initial, feedback))
            self.assertEqual(routed.pending.difficulty, expected_difficulty)
            actual = prepare_pending_adaptive_program_room(routed)
            self.assertEqual(dict(actual.challenge.metadata)["specificity"], str(expected_difficulty))
            self.assertEqual(actual.challenge.split, "train")
            self.assertNotEqual(actual.challenge.prompt, first.challenge.prompt)
            self.assertNotIn("answer", dataclasses.asdict(actual.trainee_view))
            # Level 3 has a fixed dependency structure; level 1 permits
            # broad training structures. This checks content, not a label.
            if expected_difficulty == 3:
                self.assertIn("1. MUL A", actual.challenge.prompt)
                self.assertIn("2. SUB B A", actual.challenge.prompt)
                self.assertIn("3. ADD B", actual.challenge.prompt)
                self.assertIn("4. ADD A B", actual.challenge.prompt)

    def test_adaptive_program_materialization_rejects_unapproved_or_stale_state(self):
        from labyrinth.journey_challenges import prepare_pending_adaptive_program_room

        declared = scope(("arithmetic",))
        start = begin_adaptive_walk(
            declared, generate_maze_plan(declared, 0), checkpoint_sha256="a" * 64,
        )
        with self.assertRaises(ValueError):
            prepare_pending_adaptive_program_room(start)
        with self.assertRaisesRegex(ValueError, "program_reasoning"):
            prepare_pending_adaptive_program_room(offer_next_chamber(start))

    def test_adaptive_program_materialization_seals_prior_exam_prompts(self):
        from labyrinth.journey_challenges import prepare_pending_adaptive_program_room

        declared = scope(("program_reasoning",))
        first = offer_next_chamber(begin_adaptive_walk(
            declared, generate_maze_plan(declared, 0), checkpoint_sha256="a" * 64,
        ))
        prepared = prepare_pending_adaptive_program_room(first)
        forged = dataclasses.replace(
            first, pending=dataclasses.replace(first.pending, difficulty=5),
        )
        with self.assertRaisesRegex(ValueError, "deterministic train-only route"):
            prepare_pending_adaptive_program_room(forged)
        prompt_hash = prepared.trainee_view.prompt_sha256
        # A prior development/examination prompt commitment cannot be
        # reintroduced merely because it has a new train-only room identity.
        sealed = begin_adaptive_walk(
            declared, generate_maze_plan(declared, 0), checkpoint_sha256="a" * 64,
            forbidden_case_ids=("sealed-development-case",),
            forbidden_prompt_sha256=(prompt_hash,),
        )
        with self.assertRaisesRegex(ValueError, "sealed exam"):
            prepare_pending_adaptive_program_room(offer_next_chamber(sealed))

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
