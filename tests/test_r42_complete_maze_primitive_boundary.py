"""Hostile primitive-admission tests for the direct TRAIN completion boundary.

These tests check internal routing integrity, not physical model updates or
independently authenticated host evidence.
"""

import unittest

from labyrinth.journey import AWAITING_EXAM, complete_maze, begin_journey
from tests.test_journey import digest, scope, training_receipt


class ForgedTrainingAlias(str):
    def __eq__(self, other):
        return other == "train"

    def __ne__(self, other):
        return other != "train"


class ForgedTrainingPhase(str):
    def __eq__(self, other):
        return other == "TRAINING"

    def __ne__(self, other):
        return other != "TRAINING"


class ForgedHistoryHash(str):
    def __hash__(self):
        return 0


class CompleteMazePrimitiveBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.state = begin_journey(scope(), checkpoint_sha256=digest("initial"))
        self.receipt = training_receipt(self.state, "updated")

    def test_valid_training_receipt_advances_without_graduation(self):
        result = complete_maze(self.state, self.receipt)
        self.assertEqual(result.phase, AWAITING_EXAM)

    def test_mutated_private_final_split_cannot_impersonate_train(self):
        object.__setattr__(self.receipt, "split", ForgedTrainingAlias("private_final"))
        with self.assertRaises(ValueError):
            complete_maze(self.state, self.receipt)

    def test_mutated_exam_phase_cannot_impersonate_training(self):
        object.__setattr__(self.state, "phase", ForgedTrainingPhase("AWAITING_EXAM"))
        with self.assertRaises(ValueError):
            complete_maze(self.state, self.receipt)

    def test_mutated_sealed_exam_case_hash_cannot_hide_training_reuse(self):
        case_id = self.receipt.room_case_ids[0][1]
        object.__setattr__(self.state, "sealed_exam_case_ids", (ForgedHistoryHash(case_id),))
        object.__setattr__(self.state, "sealed_exam_prompt_sha256", (digest("unrelated"),))
        with self.assertRaises(ValueError):
            complete_maze(self.state, self.receipt)

    def test_mutated_sealed_exam_prompt_hash_cannot_hide_training_reuse(self):
        prompt = self.receipt.room_case_ids[0][2]
        object.__setattr__(self.state, "sealed_exam_case_ids", ("other-exam-case",))
        object.__setattr__(self.state, "sealed_exam_prompt_sha256", (ForgedHistoryHash(prompt),))
        with self.assertRaises(ValueError):
            complete_maze(self.state, self.receipt)


if __name__ == "__main__":
    unittest.main()
