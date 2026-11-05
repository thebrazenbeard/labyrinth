"""Regenerative journey behavior, evidence isolation and no false graduation."""

from dataclasses import replace
import hashlib
import unittest

from labyrinth.training_scope import TrainingScopeSpec
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.journey import (
    AWAITING_EXAM, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION,
    EXHAUSTED, TRAINING, TrainingCompletionReceipt, begin_journey,
    complete_maze, face_minotaur,
)


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def scope(max_cycles=2):
    return TrainingScopeSpec(
        objective_id="empathy-task-v1", domain="empathy",
        skills=("cognitive", "compassionate"), specificity=2,
        scale=1, max_cycles=max_cycles, seed=17,
    )


def training_receipt(state, after, label="train", prompt_override=None):
    rooms = tuple(state.maze.rooms)
    triples = tuple(
        (room.room_id, f"{label}-{state.cycle_index}-{i}",
         prompt_override if i == 0 and prompt_override is not None else digest(f"{label}-{state.cycle_index}-{i}"))
        for i, room in enumerate(rooms)
    )
    return TrainingCompletionReceipt(
        objective_id=state.scope.objective_id,
        cycle_index=state.cycle_index,
        checkpoint_before_sha256=state.checkpoint_sha256,
        checkpoint_after_sha256=digest(after),
        room_case_ids=triples,
    )


def exam_receipt(state, status="fail", label="exam", count_per_section=4, prompt_override=None):
    rows = []
    for skill in state.scope.skills:
        for section in ("heldout", "transfer"):
            for index in range(count_per_section):
                case_id = f"{label}-{state.cycle_index}-{skill}-{section}-{index}"
                prompt = (prompt_override if skill == state.scope.skills[0]
                          and section == "heldout" and index == 0
                          and prompt_override is not None else digest(case_id))
                score = 0.95 if status == "pass" or skill != state.scope.skills[0] else 0.20
                rows.append(ExamCaseResult(case_id, skill, section, score, 0.20, prompt))
    return MinotaurExamReceipt(
        exam_id=f"{label}-{state.cycle_index}", objective_id=state.scope.objective_id,
        cycle_index=state.cycle_index, checkpoint_sha256=state.checkpoint_sha256,
        baseline_sha256=state.baseline_checkpoint_sha256,
        evaluator_id="independent-development-evaluator", cases=tuple(rows),
    )


class JourneyLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.scope = scope()
        self.initial = begin_journey(self.scope, checkpoint_sha256=digest("initial"))

    def complete(self, state=None, after="updated", label="train"):
        state = state or self.initial
        return complete_maze(state, training_receipt(state, after, label))

    def test_begin_has_frozen_goal_and_deterministic_entrance(self):
        state = self.initial
        self.assertEqual(state.phase, TRAINING)
        self.assertEqual(state.cycle_index, 0)
        self.assertEqual(state.checkpoint_sha256, state.baseline_checkpoint_sha256)
        self.assertEqual(state, begin_journey(self.scope, checkpoint_sha256=digest("initial")))

    def test_complete_maze_requires_all_train_rooms_and_changed_checkpoint(self):
        with self.assertRaises(ValueError):
            complete_maze(self.initial, training_receipt(self.initial, "initial"))
        receipt = training_receipt(self.initial, "updated")
        with self.assertRaises(ValueError):
            complete_maze(self.initial, replace(receipt, room_case_ids=receipt.room_case_ids[:-1]))
        with self.assertRaises(ValueError):
            complete_maze(self.initial, replace(receipt, split="eval"))
        next_state = complete_maze(self.initial, receipt)
        self.assertEqual(next_state.phase, AWAITING_EXAM)
        self.assertEqual(next_state.checkpoint_sha256, digest("updated"))

    def test_failed_minotaur_rerolls_maze_but_not_learner(self):
        trained = self.complete()
        rerolled = face_minotaur(trained, exam_receipt(trained, "fail"))
        self.assertEqual(rerolled.phase, TRAINING)
        self.assertEqual(rerolled.cycle_index, 1)
        self.assertEqual(rerolled.baseline_checkpoint_sha256, digest("initial"))
        self.assertEqual(rerolled.checkpoint_sha256, digest("updated"))
        self.assertTrue(set(room.room_id for room in rerolled.maze.rooms).isdisjoint(
            room.room_id for room in trained.maze.rooms
        ))
        self.assertEqual(rerolled.attempts[0].exam_status, "FAIL")
        self.assertIn("cognitive", rerolled.attempts[0].weak_skills)
        self.assertEqual(face_minotaur(trained, exam_receipt(trained, "fail")), rerolled)

    def test_development_pass_does_not_graduate(self):
        trained = self.complete()
        passed = face_minotaur(trained, exam_receipt(trained, "pass"))
        self.assertEqual(passed.phase, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION)
        self.assertNotEqual(passed.phase, "GRADUATED")
        self.assertEqual(passed.attempts[-1].exam_status, "PASS")
        with self.assertRaises(ValueError):
            complete_maze(passed, training_receipt(passed, "another"))
        with self.assertRaises(ValueError):
            face_minotaur(passed, exam_receipt(passed, "pass", label="another"))

    def test_failed_final_allowed_cycle_exhausts_without_graduation(self):
        initial = begin_journey(scope(max_cycles=1), checkpoint_sha256=digest("initial"))
        trained = self.complete(initial)
        exhausted = face_minotaur(trained, exam_receipt(trained, "fail"))
        self.assertEqual(exhausted.phase, EXHAUSTED)
        self.assertEqual(exhausted.checkpoint_sha256, digest("updated"))
        self.assertEqual(exhausted.attempts[-1].exam_status, "FAIL")

    def test_exam_prompt_reuse_different_ids_rejected(self):
        trained = self.complete()
        receipt = exam_receipt(trained, "fail")
        same_training_prompt = trained.used_training_prompt_sha256[0]
        with self.assertRaisesRegex(ValueError, "exam prompt overlaps"):
            face_minotaur(trained, exam_receipt(trained, "fail", prompt_override=same_training_prompt))
        rerolled = face_minotaur(trained, receipt)
        with self.assertRaisesRegex(ValueError, "training prompt was reused"):
            complete_maze(rerolled, training_receipt(rerolled, "second",
                                                    prompt_override=receipt.cases[0].prompt_sha256))

    def test_exam_ids_and_prompts_never_train_after_failure(self):
        trained = self.complete()
        exam = exam_receipt(trained, "fail")
        rerolled = face_minotaur(trained, exam)
        new_case_id = exam.cases[0].case_id
        receipt = training_receipt(rerolled, "second")
        triples = list(receipt.room_case_ids)
        triples[0] = (triples[0][0], new_case_id, triples[0][2])
        with self.assertRaisesRegex(ValueError, "training case was reused"):
            complete_maze(rerolled, replace(receipt, room_case_ids=tuple(triples)))

    def test_inconclusive_dev_exam_is_bounded_and_never_passes(self):
        trained = self.complete()
        one = face_minotaur(trained, exam_receipt(trained, "pass", count_per_section=1))
        self.assertEqual(one.phase, AWAITING_EXAM)
        self.assertEqual(one.inconclusive_exam_count, 1)
        self.assertEqual(one.exam_event_statuses, ("INCONCLUSIVE",))
        duplicated_event_id = exam_receipt(one, "pass", label="second", count_per_section=1)
        with self.assertRaisesRegex(ValueError, "reused Minotaur exam identity"):
            face_minotaur(one, replace(duplicated_event_id, exam_id=f"exam-{one.cycle_index}"))
        two = face_minotaur(one, exam_receipt(one, "pass", label="second", count_per_section=1))
        self.assertEqual(two.phase, EXHAUSTED)
        self.assertEqual(two.attempts[-1].exam_status, "INCONCLUSIVE")
        self.assertEqual(two.exam_event_statuses, ("INCONCLUSIVE", "INCONCLUSIVE"))
        with self.assertRaises(ValueError):
            face_minotaur(two, exam_receipt(two, "pass", label="third", count_per_section=1))

    def test_private_final_cannot_supply_weak_skill_hints(self):
        trained = self.complete()
        bad = exam_receipt(trained, "fail")
        object.__setattr__(bad, "exam_kind", "private_final")
        with self.assertRaisesRegex(ValueError, "only development"):
            face_minotaur(trained, bad)


if __name__ == "__main__":
    unittest.main()
