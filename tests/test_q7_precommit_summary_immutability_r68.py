"""Hostile source-only regression: a Q7 summary is not an authority token."""

from __future__ import annotations

import unittest
from dataclasses import replace

from labyrinth.final_exam_precommit import FinalExamPrecommitSummary


class Q7PrecommitSummaryImmutabilityTests(unittest.TestCase):
    def summary(self):
        return FinalExamPrecommitSummary(
            exam_id="exam-fixture", event_id="event-fixture",
            objective_id="classification.fixture", scope_sha256="a" * 64,
            sealed_plan_sha256="b" * 64,
        )

    def test_low_level_setattr_cannot_forge_graduation_or_plan_identity(self):
        for field, value in (
            ("graduation_authorized", True),
            ("exam_id", "different-exam"),
            ("event_id", "different-event"),
            ("scope_sha256", "f" * 64),
            ("sealed_plan_sha256", "f" * 64),
            ("private_bank_verified", True),
            ("external_anchor_verified", True),
        ):
            result = self.summary()
            with self.subTest(field=field), self.assertRaises((AttributeError, TypeError)):
                object.__setattr__(result, field, value)
            self.assertFalse(result.graduation_authorized)
            self.assertEqual(result.exam_id, "exam-fixture")

    def test_constructor_or_replace_cannot_add_positive_authority(self):
        with self.assertRaises(TypeError):
            self.summary().__class__(
                "exam-fixture", "event-fixture", "classification.fixture",
                "a" * 64, "b" * 64, graduation_authorized=True)
        with self.assertRaises((TypeError, ValueError)):
            replace(self.summary(), graduation_authorized=True)
        self.assertFalse(self.summary().graduation_authorized)


if __name__ == "__main__":
    unittest.main()
