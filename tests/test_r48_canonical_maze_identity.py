"""Canonical journey maze identity is source-only, not a signed host claim."""

from dataclasses import replace
import unittest

from labyrinth.journey import (
    AWAITING_EXAM, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION,
    EXHAUSTED, TRAINING, begin_journey, complete_maze, face_minotaur,
    _validate_state,
)
from tests.test_journey import digest, exam_receipt, scope, training_receipt


class CanonicalMazeIdentityTests(unittest.TestCase):
    @staticmethod
    def _forged_room_id(state):
        first = state.maze.rooms[0]
        changed = replace(first, room_id="f" * 32 if first.room_id != "f" * 32 else "e" * 32)
        return replace(state, maze=replace(state.maze, rooms=(changed,) + state.maze.rooms[1:]))

    def test_cycle_zero_matching_forged_room_and_receipt_rejected(self):
        original = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        forged = self._forged_room_id(original)
        with self.assertRaisesRegex(ValueError, "canonical generated plan"):
            complete_maze(forged, training_receipt(forged, "trained"))

    def test_adaptive_cycle_one_matching_forged_room_and_receipt_rejected(self):
        original = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        trained = complete_maze(original, training_receipt(original, "first"))
        rerolled = face_minotaur(trained, exam_receipt(trained, "fail"))
        self.assertEqual(rerolled.phase, TRAINING)
        self.assertTrue(rerolled.attempts[0].weak_skills)
        forged = self._forged_room_id(rerolled)
        with self.assertRaisesRegex(ValueError, "canonical generated plan"):
            complete_maze(forged, training_receipt(forged, "second"))

    def test_prior_pass_or_inconclusive_cannot_create_next_cycle(self):
        original = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        trained = complete_maze(original, training_receipt(original, "first"))
        rerolled = face_minotaur(trained, exam_receipt(trained, "fail"))
        for fake_status in ("PASS", "INCONCLUSIVE"):
            with self.subTest(fake_status=fake_status):
                forged_attempt = replace(rerolled.attempts[0], exam_status=fake_status)
                forged = replace(rerolled, attempts=(forged_attempt,))
                with self.assertRaisesRegex(ValueError, "prior cycle"):
                    complete_maze(forged, training_receipt(forged, "second"))

    def test_real_adaptive_reroll_training_and_terminal_development_pass(self):
        original = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        trained = complete_maze(original, training_receipt(original, "first"))
        rerolled = face_minotaur(trained, exam_receipt(trained, "fail"))
        self.assertEqual(rerolled.cycle_index, 1)
        completed = complete_maze(rerolled, training_receipt(rerolled, "second"))
        self.assertEqual(completed.phase, AWAITING_EXAM)
        terminal = face_minotaur(completed, exam_receipt(completed, "pass"))
        self.assertEqual(terminal.phase, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION)
        self.assertEqual(terminal.cycle_index, 1)

    def test_real_adaptive_reroll_can_exhaust_final_cycle(self):
        original = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        trained = complete_maze(original, training_receipt(original, "first"))
        rerolled = face_minotaur(trained, exam_receipt(trained, "fail"))
        completed = complete_maze(rerolled, training_receipt(rerolled, "second"))
        exhausted = face_minotaur(completed, exam_receipt(completed, "fail"))
        self.assertEqual(exhausted.phase, EXHAUSTED)
        self.assertEqual(exhausted.cycle_index, 1)

    def test_terminal_phase_cannot_contradict_last_exam_result(self):
        original = begin_journey(scope(max_cycles=1), checkpoint_sha256=digest("initial"))
        trained = complete_maze(original, training_receipt(original, "first"))
        passed = face_minotaur(trained, exam_receipt(trained, "pass"))
        exhausted = face_minotaur(trained, exam_receipt(trained, "fail"))
        for terminal, forged_status in ((passed, "FAIL"), (exhausted, "PASS")):
            with self.subTest(phase=terminal.phase, forged_status=forged_status):
                forged_attempt = replace(terminal.attempts[-1], exam_status=forged_status)
                forged = replace(terminal, attempts=terminal.attempts[:-1] + (forged_attempt,))
                with self.assertRaisesRegex(ValueError, "terminal exam"):
                    # Shared state validation applies to terminal states too.
                    _validate_state(forged)

    def test_inconclusive_keeps_current_canonical_maze(self):
        original = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        trained = complete_maze(original, training_receipt(original, "first"))
        waiting = face_minotaur(trained, exam_receipt(trained, "pass", count_per_section=1))
        self.assertEqual(waiting.phase, AWAITING_EXAM)
        self.assertEqual(waiting.maze, original.maze)
