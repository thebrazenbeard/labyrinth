"""Hostile source-only checks on the public legacy Minotaur entry point.

These guard in-process claim integrity, never examiner custody or graduation.
"""

import unittest

from labyrinth.journey import (
    AWAITING_EXAM, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION,
    face_minotaur,
)
from tests.test_minotaur_exam_intake_guard_r36 import awaiting_exam
from tests.test_journey import exam_receipt


class FaceMinotaurPrimitiveBoundaryTests(unittest.TestCase):
    def test_valid_internal_development_exam_remains_supported(self):
        trained = awaiting_exam()
        next_state = face_minotaur(trained, exam_receipt(trained, status="pass"))
        self.assertEqual(next_state.phase, DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION)

    def test_forged_phase_cannot_impersonate_awaiting_exam(self):
        class ForgedPhase(str):
            def __eq__(self, other):
                return other == AWAITING_EXAM

            def __ne__(self, other):
                return other != AWAITING_EXAM

        trained = awaiting_exam()
        receipt = exam_receipt(trained, status="pass")
        object.__setattr__(trained, "phase", ForgedPhase("TRAINING"))
        with self.assertRaises(ValueError):
            face_minotaur(trained, receipt)

    def test_forged_history_hash_cannot_hide_reused_training_case(self):
        class ForgedID(str):
            def __hash__(self):
                return 123456789

        trained = awaiting_exam()
        receipt = exam_receipt(trained, status="pass")
        case_id = trained.used_training_case_ids[0]
        object.__setattr__(trained, "used_training_case_ids",
                           (ForgedID(case_id),) + trained.used_training_case_ids[1:])
        object.__setattr__(receipt.cases[0], "case_id", case_id)
        with self.assertRaises(ValueError):
            face_minotaur(trained, receipt)

    def test_forged_history_hash_cannot_hide_reused_training_prompt(self):
        class ForgedPrompt(str):
            def __hash__(self):
                return 123456789

        trained = awaiting_exam()
        receipt = exam_receipt(trained, status="pass")
        prompt = trained.used_training_prompt_sha256[0]
        object.__setattr__(trained, "used_training_prompt_sha256",
                           (ForgedPrompt(prompt),) + trained.used_training_prompt_sha256[1:])
        object.__setattr__(receipt.cases[0], "prompt_sha256", prompt)
        with self.assertRaises(ValueError):
            face_minotaur(trained, receipt)

    def test_mutated_private_final_receipt_cannot_masquerade_as_development(self):
        class ForgedKind(str):
            def __ne__(self, other):
                return False

        trained = awaiting_exam()
        receipt = exam_receipt(trained, status="pass")
        object.__setattr__(receipt, "exam_kind", ForgedKind("private_final"))
        with self.assertRaises(ValueError):
            face_minotaur(trained, receipt)

    def test_mutated_private_final_case_cannot_masquerade_as_heldout(self):
        class ForgedSplit(str):
            def __eq__(self, other):
                return other == "heldout"

            def __ne__(self, other):
                return other != "heldout"

        trained = awaiting_exam()
        receipt = exam_receipt(trained, status="pass")
        object.__setattr__(receipt.cases[0], "split", ForgedSplit("private_final"))
        with self.assertRaises(ValueError):
            face_minotaur(trained, receipt)


if __name__ == "__main__":
    unittest.main()
