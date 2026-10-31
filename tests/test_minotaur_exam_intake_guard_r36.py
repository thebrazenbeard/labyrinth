"""Narrow source-only exam-intake guard for a regenerated adaptive Journey."""

import unittest

from labyrinth.adaptive_corridors import (
    TrainingRoomFeedback, offer_next_chamber, record_training_feedback,
)
from labyrinth.journey import (
    AWAITING_EXAM, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION,
    TRAINING, begin_journey, complete_maze,
)
from labyrinth.minotaur_exam_intake_guard_r36 import admit_development_exam
from tests.test_journey import digest, exam_receipt, scope, training_receipt


def awaiting_exam():
    initial = begin_journey(scope(), checkpoint_sha256=digest("initial"))
    state = complete_maze(initial, training_receipt(initial, "updated"))
    assert state.phase == AWAITING_EXAM
    return state


class MinotaurExamIntakeGuardR36Tests(unittest.TestCase):
    def test_failed_exam_regenerates_train_only_adaptive_corridor(self):
        trained = awaiting_exam()
        development_claim = exam_receipt(trained, status="fail")
        transition = admit_development_exam(trained, development_claim)
        self.assertEqual(transition.journey.phase, TRAINING)
        self.assertEqual(transition.journey.cycle_index, 1)
        self.assertEqual(transition.adaptive_walk.plan.cycle_index, 1)
        self.assertFalse(transition.graduation_authorized)
        self.assertFalse(transition.evaluator_host_independence_verified)
        self.assertFalse(transition.neural_execution_verified)
        self.assertEqual(transition.adaptive_walk.forbidden_case_ids,
                         transition.journey.used_training_case_ids
                         + transition.journey.sealed_exam_case_ids)
        self.assertEqual(transition.adaptive_walk.forbidden_prompt_sha256,
                         transition.journey.used_training_prompt_sha256
                         + transition.journey.sealed_exam_prompt_sha256)

        assigned = offer_next_chamber(transition.adaptive_walk)
        self.assertIsNotNone(assigned.pending)
        exam_prompt = development_claim.cases[0].prompt_sha256
        injected = TrainingRoomFeedback(
            room_id=assigned.pending.room_id,
            case_id="train-new-case",
            prompt_sha256=exam_prompt,
            score=1,
            split="train",
            checkpoint_before_sha256=assigned.checkpoint_sha256,
            checkpoint_after_sha256=digest("later-train-checkpoint"),
        )
        with self.assertRaises(ValueError):
            record_training_feedback(assigned, injected)

    def test_development_pass_is_non_authorizing_and_has_no_train_handoff(self):
        trained = awaiting_exam()
        transition = admit_development_exam(trained, exam_receipt(trained, status="pass"))
        self.assertEqual(transition.journey.phase,
                         DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION)
        self.assertIsNone(transition.adaptive_walk)
        self.assertFalse(transition.graduation_authorized)

    def test_hostile_exam_kind_with_forged_equality_cannot_steer_training(self):
        class ForgedKind(str):
            def __ne__(self, other):
                return False

        trained = awaiting_exam()
        forged = exam_receipt(trained, status="fail")
        object.__setattr__(forged, "exam_kind", ForgedKind("private_final"))
        with self.assertRaisesRegex(ValueError, "exact.*development|development.*exact"):
            admit_development_exam(trained, forged)

    def test_hostile_phase_with_forged_equality_cannot_impersonate_exam_phase(self):
        class ForgedPhase(str):
            def __eq__(self, other):
                return other == AWAITING_EXAM

            def __ne__(self, other):
                return other != AWAITING_EXAM

        trained = awaiting_exam()
        receipt = exam_receipt(trained, status="fail")
        object.__setattr__(trained, "phase", ForgedPhase("TRAINING"))
        with self.assertRaisesRegex(ValueError, "exact.*phase|phase.*exact"):
            admit_development_exam(trained, receipt)

    def test_hostile_training_case_hash_cannot_hide_exam_reuse(self):
        class ForgedID(str):
            def __hash__(self):
                return 123456789

        trained = awaiting_exam()
        receipt = exam_receipt(trained, status="pass")
        train_id = trained.used_training_case_ids[0]
        object.__setattr__(trained, "used_training_case_ids",
                           (ForgedID(train_id),) + trained.used_training_case_ids[1:])
        object.__setattr__(receipt.cases[0], "case_id", train_id)
        with self.assertRaisesRegex(ValueError, "exact.*training|training.*exact"):
            admit_development_exam(trained, receipt)

    def test_hostile_maze_room_purpose_cannot_masquerade_as_train(self):
        class ForgedPurpose(str):
            def __ne__(self, other):
                return False

        trained = awaiting_exam()
        receipt = exam_receipt(trained, status="fail")
        object.__setattr__(trained.maze.rooms[0], "purpose", ForgedPurpose("heldout"))
        with self.assertRaisesRegex(ValueError, "exact.*maze|maze.*exact"):
            admit_development_exam(trained, receipt)

    def test_hostile_case_split_with_forged_equality_cannot_enter_exam(self):
        class ForgedSplit(str):
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        trained = awaiting_exam()
        for forged_split in (ForgedSplit("private_final"), ForgedSplit("train")):
            with self.subTest(split=forged_split):
                receipt = exam_receipt(trained, status="fail")
                object.__setattr__(receipt.cases[0], "split", forged_split)
                with self.assertRaisesRegex(ValueError, "exact.*split|split.*exact"):
                    admit_development_exam(trained, receipt)

    def test_mutated_exam_case_identity_rejects_overlap_and_no_pass_through(self):
        trained = awaiting_exam()
        receipt = exam_receipt(trained, status="fail")
        object.__setattr__(receipt.cases[0], "prompt_sha256",
                           trained.used_training_prompt_sha256[0])
        with self.assertRaises(ValueError):
            admit_development_exam(trained, receipt)

    def test_forged_heldout_feedback_cannot_steer_regenerated_train_corridor(self):
        class ForgedSplit(str):
            def __ne__(self, other):
                return False

        trained = awaiting_exam()
        transition = admit_development_exam(trained, exam_receipt(trained, status="fail"))
        assigned = offer_next_chamber(transition.adaptive_walk)
        feedback = TrainingRoomFeedback(
            room_id=assigned.pending.room_id,
            case_id="train-new-unique-case",
            prompt_sha256=digest("new-unique-train-only-prompt"),
            score=1,
            split="train",
            checkpoint_before_sha256=assigned.checkpoint_sha256,
            checkpoint_after_sha256=digest("new-unique-checkpoint"),
        )
        object.__setattr__(feedback, "split", ForgedSplit("heldout"))
        with self.assertRaisesRegex(ValueError, "only train feedback"):
            record_training_feedback(assigned, feedback)

    def test_wrong_phase_cannot_run_intake(self):
        started = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        completed = awaiting_exam()
        with self.assertRaises(ValueError):
            admit_development_exam(started, exam_receipt(completed))


if __name__ == "__main__":
    unittest.main()
