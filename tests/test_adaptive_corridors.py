"""Chamber-by-chamber feedback changes the train-only corridor route."""

from dataclasses import replace
import unittest

from labyrinth.adaptive_corridors import (
    begin_adaptive_walk, begin_adaptive_walk_from_journey,
    offer_next_chamber, record_training_feedback,
    TrainingRoomFeedback, AdaptiveWalkState,
)
from labyrinth.journey import (
    begin_journey, complete_maze, face_minotaur, TrainingCompletionReceipt,
)
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


BASE = "a" * 64
AFTER = "b" * 64


def scope():
    return TrainingScopeSpec(
        objective_id="adaptive-corridor-v1", domain="reasoning",
        skills=("arithmetic", "program_reasoning"),
        specificity=2, scale=3, max_cycles=2, seed=14,
    )


def start():
    manifest = scope()
    return begin_adaptive_walk(manifest, generate_maze_plan(manifest, 0),
                               checkpoint_sha256=BASE,
                               forbidden_case_ids=("eval-sealed-one",),
                               forbidden_prompt_sha256=("f" * 64,))


def feedback(state, *, score, after=AFTER):
    assert state.pending is not None
    return TrainingRoomFeedback(
        room_id=state.pending.room_id, case_id="train-" + state.pending.room_id,
        prompt_sha256=state.pending.room_id * 2,
        score=score, split="train",
        checkpoint_before_sha256=state.checkpoint_sha256,
        checkpoint_after_sha256=after,
    )


class AdaptiveCorridorTests(unittest.TestCase):
    def test_last_failure_focuses_next_room_while_success_changes_branch(self):
        initial = offer_next_chamber(start())
        failed = offer_next_chamber(record_training_feedback(initial, feedback(initial, score=0)))
        successful = offer_next_chamber(record_training_feedback(initial, feedback(initial, score=1)))
        # Both branches begin with an identical question and use only the
        # submitted training-only outcome to choose subsequent content.
        self.assertEqual(failed.pending.skill, initial.pending.skill)
        self.assertNotEqual(successful.pending.skill, initial.pending.skill)
        self.assertNotEqual(failed.pending.room_id, successful.pending.room_id)
        self.assertNotEqual(failed.pending.case_seed, successful.pending.case_seed)
        self.assertGreaterEqual(failed.pending.difficulty, 1)
        self.assertLessEqual(successful.pending.difficulty, 5)

    def test_checkpoint_continuity_and_unique_train_case_seeds(self):
        state = start()
        seen = set()
        while state.remaining_count:
            state = offer_next_chamber(state)
            self.assertNotIn(state.pending.case_seed, seen)
            seen.add(state.pending.case_seed)
            before = state.checkpoint_sha256
            after = ("b" if len(seen) % 2 else "c") * 64
            record = feedback(state, score=len(seen) % 2, after=after)
            self.assertEqual(record.checkpoint_before_sha256, before)
            state = record_training_feedback(state, record)
            self.assertEqual(state.checkpoint_sha256, after)
        self.assertIsNone(state.pending)
        self.assertEqual(len(state.history), scope().total_chambers)
        self.assertEqual(state.remaining_count, 0)
        with self.assertRaises(ValueError):
            offer_next_chamber(state)

    def test_eval_and_private_final_feedback_cannot_steer_training(self):
        pending = offer_next_chamber(start())
        for split in ("eval", "heldout", "transfer", "private_final"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                record_training_feedback(pending, replace(feedback(pending, score=0), split=split))
        with self.assertRaises(ValueError):
            record_training_feedback(pending, replace(feedback(pending, score=1),
                                                      case_id="eval-leak"))

    def test_rejects_duplicate_and_sealed_prompts_before_state_change(self):
        state = start()
        first = offer_next_chamber(state)
        record = feedback(first, score=0)
        with self.assertRaises(ValueError):
            record_training_feedback(first, replace(record, prompt_sha256="f" * 64))
        second = offer_next_chamber(record_training_feedback(first, record))
        with self.assertRaises(ValueError):
            record_training_feedback(second, replace(feedback(second, score=1),
                                                      prompt_sha256=record.prompt_sha256))
        with self.assertRaises(ValueError):
            record_training_feedback(second, replace(feedback(second, score=1),
                                                      case_id=record.case_id))

    def test_determinism_and_prevent_double_pending_or_double_feedback(self):
        left = offer_next_chamber(start())
        right = offer_next_chamber(start())
        self.assertEqual(left, right)
        with self.assertRaises(ValueError):
            offer_next_chamber(left)
        record = feedback(left, score=1)
        settled = record_training_feedback(left, record)
        self.assertEqual(settled, record_training_feedback(right, record))
        self.assertEqual(settled.trail_sha256, record_training_feedback(right, record).trail_sha256)
        self.assertNotEqual(left.trail_sha256, settled.trail_sha256)
        with self.assertRaises(ValueError):
            record_training_feedback(settled, record)

    def test_mutated_assignment_skill_and_sealed_budget_refused(self):
        state = offer_next_chamber(start())
        wrong_skill = "program_reasoning" if state.pending.skill == "arithmetic" else "arithmetic"
        with self.assertRaises(ValueError):
            record_training_feedback(
                replace(state, pending=replace(state.pending, skill=wrong_skill)),
                feedback(state, score=0),
            )
        s = scope()
        with self.assertRaises(ValueError):
            begin_adaptive_walk(
                s, generate_maze_plan(s, 0), checkpoint_sha256=BASE,
                forbidden_case_ids=tuple(f"eval-{i}" for i in range(4097)),
                forbidden_prompt_sha256=("c" * 64,) * 4097,
            )

    def test_mutated_valid_pending_difficulty_is_not_a_valid_route(self):
        state = offer_next_chamber(start())
        # Both values are syntactically valid; only the trained route decides
        # which one was actually assigned for this visit.
        changed = replace(state, pending=replace(state.pending, difficulty=5))
        self.assertNotEqual(state.pending.difficulty, changed.pending.difficulty)
        with self.assertRaises(ValueError):
            record_training_feedback(changed, feedback(changed, score=1))

    def test_mutated_prior_route_cannot_be_used_to_generate_next_chamber(self):
        pending = offer_next_chamber(start())
        visited = record_training_feedback(pending, feedback(pending, score=1))
        first = visited.history[0]
        changed = replace(visited, history=(replace(
            first, assignment=replace(first.assignment, difficulty=5)
        ),))
        with self.assertRaises(ValueError):
            offer_next_chamber(changed)

    def test_regeneration_requires_prior_exam_custody_even_with_explicit_empty_sets(self):
        s = scope()
        cycle1 = generate_maze_plan(s, 1)
        with self.assertRaises(ValueError):
            begin_adaptive_walk(s, cycle1, checkpoint_sha256=AFTER)
        with self.assertRaises(ValueError):
            begin_adaptive_walk(s, cycle1, checkpoint_sha256=AFTER,
                                forbidden_case_ids=(), forbidden_prompt_sha256=())
        with self.assertRaises(ValueError):
            begin_adaptive_walk(s, cycle1, checkpoint_sha256=AFTER,
                                forbidden_case_ids=("dev-case",))
        # A dummy nonempty seal is no better than no seal: direct callers
        # cannot attest that all prior development exams were included.
        with self.assertRaises(ValueError):
            begin_adaptive_walk(s, cycle1, checkpoint_sha256=AFTER,
                                forbidden_case_ids=("dummy-dev",),
                                forbidden_prompt_sha256=("f" * 64,))
        forged = AdaptiveWalkState(s, cycle1, AFTER, AFTER)
        with self.assertRaises(ValueError):
            offer_next_chamber(forged)

    def test_journey_binding_retains_old_train_and_exam_seals(self):
        s = scope()
        journey = begin_journey(s, checkpoint_sha256=BASE)
        train_pairs = tuple(
            (room.room_id, f"train-old-{i}", f"{i + 1:064x}")
            for i, room in enumerate(journey.maze.rooms)
        )
        journey = complete_maze(journey, TrainingCompletionReceipt(
            objective_id=s.objective_id, cycle_index=0,
            checkpoint_before_sha256=BASE, checkpoint_after_sha256=AFTER,
            room_case_ids=train_pairs,
        ))
        exam_cases = tuple(
            ExamCaseResult(
                case_id=("train-mislabel-old" if index == 0 else
                         f"dev-old-{skill}-{split}-{i}"),
                skill=skill, split=split,
                score=0, baseline_score=0,
                prompt_sha256=f"{100 + index:064x}",
            )
            for index, (skill, split, i) in enumerate(
                (skill, split, i)
                for skill in s.skills for split in ("heldout", "transfer") for i in range(4)
            )
        )
        journey = face_minotaur(journey, MinotaurExamReceipt(
            exam_id="dev-exam-zero", objective_id=s.objective_id, cycle_index=0,
            checkpoint_sha256=AFTER, baseline_sha256=BASE,
            evaluator_id="internal-dev", cases=exam_cases,
        ))
        self.assertEqual(journey.cycle_index, 1)
        walk = begin_adaptive_walk_from_journey(journey)
        self.assertEqual(walk.checkpoint_sha256, AFTER)
        self.assertEqual(set(walk.forbidden_case_ids),
                         {pair[1] for pair in train_pairs} | {case.case_id for case in exam_cases})
        self.assertEqual(set(walk.forbidden_prompt_sha256),
                         {pair[2] for pair in train_pairs} | {case.prompt_sha256 for case in exam_cases})
        walk = offer_next_chamber(walk)
        with self.assertRaises(ValueError):
            record_training_feedback(walk, replace(feedback(walk, score=0),
                                                   case_id=exam_cases[0].case_id))
        with self.assertRaises(ValueError):
            record_training_feedback(walk, replace(feedback(walk, score=0),
                                                   prompt_sha256=exam_cases[0].prompt_sha256))


if __name__ == "__main__":
    unittest.main()
