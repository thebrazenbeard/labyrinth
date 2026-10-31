"""Regression checks for the source-only journey's pending TRAIN audit claims.

These assertions do not authenticate a host, checkpoint bytes, or an evaluator.
"""

from dataclasses import replace
import unittest

from labyrinth.journey import (
    AWAITING_EXAM, EXHAUSTED, TRAINING, begin_journey, complete_maze,
    face_minotaur, _validate_state,
)
from tests.test_journey import digest, exam_receipt, scope, training_receipt


class PendingTrainingHistoryIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.initial = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        self.trained = complete_maze(
            self.initial, training_receipt(self.initial, "cycle-zero", label="train0")
        )

    def test_forged_pending_case_cannot_be_recorded_as_training_history(self):
        forged = replace(
            self.trained,
            pending_training_case_ids=("fabricated-train-case",)
            + self.trained.pending_training_case_ids[1:],
        )
        with self.assertRaisesRegex(ValueError, "pending training case history"):
            face_minotaur(forged, exam_receipt(forged, "fail"))

    def test_forged_pending_prompt_cannot_be_recorded_as_training_history(self):
        forged = replace(
            self.trained,
            pending_training_prompt_sha256=(digest("fabricated-train-prompt"),)
            + self.trained.pending_training_prompt_sha256[1:],
        )
        with self.assertRaisesRegex(ValueError, "pending training prompt history"):
            face_minotaur(forged, exam_receipt(forged, "fail"))

    def test_truncated_pending_train_batch_cannot_hide_completed_rooms(self):
        self.assertGreater(len(self.trained.maze.rooms), 1)
        forged = replace(
            self.trained,
            pending_training_case_ids=self.trained.pending_training_case_ids[-1:],
            pending_training_prompt_sha256=self.trained.pending_training_prompt_sha256[-1:],
        )
        with self.assertRaisesRegex(ValueError, "pending training room count"):
            face_minotaur(forged, exam_receipt(forged, "fail"))

    def test_full_length_reordered_pending_batch_is_rejected(self):
        forged = replace(
            self.trained,
            pending_training_case_ids=tuple(reversed(self.trained.pending_training_case_ids)),
            pending_training_prompt_sha256=tuple(
                reversed(self.trained.pending_training_prompt_sha256)
            ),
        )
        with self.assertRaisesRegex(ValueError, "pending training case history"):
            face_minotaur(forged, exam_receipt(forged, "fail"))

    def test_full_length_prior_cycle_pending_batch_is_not_accepted_as_current(self):
        rerolled = face_minotaur(self.trained, exam_receipt(self.trained, "fail"))
        trained_again = complete_maze(
            rerolled, training_receipt(rerolled, "cycle-one", label="train1")
        )
        self.assertEqual(len(trained_again.maze.rooms), len(self.trained.maze.rooms))
        forged = replace(
            trained_again,
            pending_training_case_ids=self.trained.pending_training_case_ids,
            pending_training_prompt_sha256=self.trained.pending_training_prompt_sha256,
        )
        with self.assertRaisesRegex(ValueError, "pending training case history"):
            face_minotaur(forged, exam_receipt(forged, "fail", label="exam1"))

    def test_inflated_pending_batch_cannot_include_prior_cycle_rooms(self):
        rerolled = face_minotaur(self.trained, exam_receipt(self.trained, "fail"))
        trained_again = complete_maze(
            rerolled, training_receipt(rerolled, "cycle-one", label="train1")
        )
        self.assertGreater(len(trained_again.used_training_case_ids),
                           len(trained_again.maze.rooms))
        forged = replace(
            trained_again,
            pending_training_case_ids=trained_again.used_training_case_ids,
            pending_training_prompt_sha256=trained_again.used_training_prompt_sha256,
        )
        with self.assertRaisesRegex(ValueError, "pending training room count"):
            face_minotaur(forged, exam_receipt(forged, "fail", label="exam1"))

    def test_cycle_one_cannot_drop_completed_prior_cycle_train_history(self):
        rerolled = face_minotaur(self.trained, exam_receipt(self.trained, "fail"))
        trained_again = complete_maze(
            rerolled, training_receipt(rerolled, "cycle-one", label="train1")
        )
        self.assertGreater(len(trained_again.used_training_case_ids),
                           len(trained_again.pending_training_case_ids))
        forged = replace(
            trained_again,
            used_training_case_ids=trained_again.pending_training_case_ids,
            used_training_prompt_sha256=trained_again.pending_training_prompt_sha256,
        )
        with self.assertRaisesRegex(ValueError, "accumulated training history"):
            face_minotaur(forged, exam_receipt(forged, "fail", label="exam1"))

    def test_training_phase_cannot_discard_completed_prior_cycle_history(self):
        rerolled = face_minotaur(self.trained, exam_receipt(self.trained, "fail"))
        forged = replace(
            rerolled, used_training_case_ids=(), used_training_prompt_sha256=(),
        )
        with self.assertRaisesRegex(ValueError, "accumulated training history"):
            complete_maze(forged, training_receipt(forged, "cycle-one", label="train1"))

    def test_prior_attempt_cannot_shrink_its_completed_maze_history(self):
        rerolled = face_minotaur(self.trained, exam_receipt(self.trained, "fail"))
        attempt = rerolled.attempts[0]
        shrunk_cases = attempt.training_case_ids[:-1]
        shrunk_prompts = attempt.training_prompt_sha256[:-1]
        forged = replace(
            rerolled,
            attempts=(replace(attempt, training_case_ids=shrunk_cases,
                              training_prompt_sha256=shrunk_prompts),),
            used_training_case_ids=shrunk_cases,
            used_training_prompt_sha256=shrunk_prompts,
        )
        with self.assertRaisesRegex(ValueError, "completed maze room count"):
            _validate_state(forged)

    def test_prior_attempt_cannot_rewrite_checkpoint_predecessor(self):
        rerolled = face_minotaur(self.trained, exam_receipt(self.trained, "fail"))
        forged = replace(
            rerolled,
            attempts=(replace(rerolled.attempts[0],
                              checkpoint_before_sha256=digest("unrelated-prior-checkpoint")),),
        )
        with self.assertRaisesRegex(ValueError, "completed checkpoint predecessor"):
            _validate_state(forged)

    def test_training_phase_current_checkpoint_matches_last_completed_attempt(self):
        rerolled = face_minotaur(self.trained, exam_receipt(self.trained, "fail"))
        forged = replace(rerolled, checkpoint_sha256=digest("unrelated-current-checkpoint"))
        with self.assertRaisesRegex(ValueError, "current checkpoint ancestry"):
            _validate_state(forged)

    def test_awaiting_exam_current_checkpoint_must_differ_from_predecessor(self):
        forged = replace(
            self.trained, checkpoint_sha256=self.trained.pending_checkpoint_before_sha256,
        )
        with self.assertRaisesRegex(ValueError, "unchanged pending training checkpoint"):
            _validate_state(forged)

    def test_terminal_phase_cannot_discard_completed_train_history(self):
        passed = face_minotaur(self.trained, exam_receipt(self.trained, "pass"))
        forged = replace(
            passed, used_training_case_ids=(), used_training_prompt_sha256=(),
        )
        with self.assertRaisesRegex(ValueError, "accumulated training history"):
            _validate_state(forged)

    def test_terminal_phase_current_checkpoint_matches_last_completed_attempt(self):
        passed = face_minotaur(self.trained, exam_receipt(self.trained, "pass"))
        forged = replace(passed, checkpoint_sha256=digest("unrelated-terminal-checkpoint"))
        with self.assertRaisesRegex(ValueError, "current checkpoint ancestry"):
            _validate_state(forged)

    def test_forged_cycle_zero_predecessor_cannot_be_recorded(self):
        forged = replace(
            self.trained,
            pending_checkpoint_before_sha256=digest("fabricated-checkpoint"),
        )
        with self.assertRaisesRegex(ValueError, "pending checkpoint predecessor"):
            face_minotaur(forged, exam_receipt(forged, "fail"))

    def test_forged_cycle_one_predecessor_cannot_be_recorded(self):
        rerolled = face_minotaur(self.trained, exam_receipt(self.trained, "fail"))
        self.assertEqual(rerolled.phase, TRAINING)
        trained_again = complete_maze(
            rerolled, training_receipt(rerolled, "cycle-one", label="train1")
        )
        forged = replace(
            trained_again,
            pending_checkpoint_before_sha256=digest("fabricated-checkpoint"),
        )
        with self.assertRaisesRegex(ValueError, "pending checkpoint predecessor"):
            face_minotaur(forged, exam_receipt(forged, "fail", label="exam1"))

    def test_normal_and_inconclusive_same_cycle_exam_keep_original_history(self):
        waiting = face_minotaur(
            self.trained,
            exam_receipt(self.trained, "pass", count_per_section=1),
        )
        self.assertEqual(waiting.phase, AWAITING_EXAM)
        self.assertEqual(waiting.pending_training_case_ids,
                         self.trained.pending_training_case_ids)
        self.assertEqual(waiting.pending_training_prompt_sha256,
                         self.trained.pending_training_prompt_sha256)
        self.assertEqual(waiting.pending_checkpoint_before_sha256,
                         self.initial.checkpoint_sha256)
        exhausted = face_minotaur(
            waiting, exam_receipt(waiting, "pass", label="retry", count_per_section=1)
        )
        self.assertEqual(exhausted.phase, EXHAUSTED)
        self.assertEqual(exhausted.attempts[0].training_case_ids,
                         self.trained.pending_training_case_ids)
        self.assertEqual(exhausted.attempts[0].training_prompt_sha256,
                         self.trained.pending_training_prompt_sha256)
        self.assertEqual(exhausted.attempts[0].checkpoint_before_sha256,
                         self.initial.checkpoint_sha256)


if __name__ == "__main__":
    unittest.main()
