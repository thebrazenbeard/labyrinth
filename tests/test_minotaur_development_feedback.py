"""Signed-development diagnostics are never Minotaur graduation credentials."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
import json
import unittest

from labyrinth.measured_weakness_feedback import (
    TrainingExposureRef, training_roster_commitment,
)
from labyrinth.minotaur_development_feedback import (
    _tactic, derive_minotaur_development_feedback,
)
from labyrinth.minotaur_exam import MinotaurExamReceipt
from labyrinth.signed_exam_lineage import sign_exam_lineage_binding
from tests import test_signed_exam_lineage as lineage_fixtures


_SCORER_DOMAIN = b"labyrinth.blind-development-result.v1\0"


class MinotaurDevelopmentFeedbackTests(unittest.TestCase):
    def setUp(self):
        fixture = lineage_fixtures.SignedExamLineageTests(
            "test_exact_scored_exam_candidate_transition_and_nonce_verified")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.options = fixture.options.copy()
        self.binding = sign_exam_lineage_binding(**self.options)
        self.training = (
            TrainingExposureRef("train-1", "a" * 64),
            TrainingExposureRef("train-2", "b" * 64),
        )
        self.roster_sha = training_roster_commitment(self.training)

    def derive(self, **changes):
        options = {
            "binding": self.binding,
            "training_exposures": self.training,
            "expected_training_roster_sha256": self.roster_sha,
            **self.options,
        }
        options.update(changes)
        return derive_minotaur_development_feedback(**options)

    def test_real_signed_development_case_errors_drive_tactic(self):
        feedback = self.derive()
        self.assertEqual(feedback.exam_id, "dev-exam-0")
        self.assertEqual(feedback.event_id, "dev-event-1")
        self.assertEqual((feedback.from_cycle, feedback.to_cycle), (0, 1))
        self.assertEqual(feedback.candidate_artifact_sha256,
                         self.options["candidate_artifact_sha256"])
        self.assertFalse(hasattr(feedback, "candidate_checkpoint_sha256"))
        self.assertEqual(feedback.remediation.weak_skills, ("sentiment",))
        self.assertEqual(len(feedback.observations), 1)
        skill = feedback.observations[0]
        self.assertEqual(skill.skill, "sentiment")
        self.assertTrue(skill.weak)
        self.assertTrue(skill.uncertain)
        self.assertEqual((skill.heldout_total, skill.transfer_total), (2, 2))
        self.assertEqual(skill.heldout_errors, 2 - skill.heldout_correct)
        self.assertEqual(skill.transfer_errors, 2 - skill.transfer_correct)
        self.assertIn(skill.suggested_train_tactic, ("practice", "contrast", "transfer"))
        self.assertTrue(feedback.source_signature_verified)
        self.assertFalse(feedback.evaluator_host_independence_verified)
        self.assertFalse(feedback.matched_control_benefit_verified)
        self.assertFalse(feedback.model_execution_authenticated)
        self.assertFalse(feedback.graduation_authorized)

    def test_signed_split_rate_difference_changes_train_tactic(self):
        # These aggregate changes are independently re-signed and re-bound
        # through the real source signing contracts; no unsigned scores pass.
        for weak_split, expected in (("transfer", "transfer"),
                                     ("heldout", "practice")):
            with self.subTest(weak_split=weak_split):
                changed = json.loads(self.options["scored_receipt_bytes"])
                for group in changed["groups"]:
                    group["correct"] = (0 if group["split"] == weak_split
                                        else group["total"])
                raw = (json.dumps(changed, sort_keys=True, separators=(",", ":"))
                       + "\n").encode()
                signature = hmac.new(
                    lineage_fixtures.EVALUATOR_KEY, _SCORER_DOMAIN + raw, sha256
                ).hexdigest()
                transition = replace(
                    self.options["transition"],
                    scored_receipt_sha256=sha256(raw).hexdigest(),
                )
                signed = {
                    **self.options, "transition": transition,
                    "scored_receipt_bytes": raw, "scored_signature_hex": signature,
                }
                binding = sign_exam_lineage_binding(**signed)
                result = self.derive(binding=binding, **signed)
                self.assertEqual(result.observations[0].suggested_train_tactic,
                                 expected)
                self.assertEqual(result.observations[0].heldout_total, 2)
                self.assertFalse(result.graduation_authorized)

    def test_validly_resigned_wrong_skill_cannot_steer_weakness(self):
        # Authentication alone is insufficient: even an evaluator-signed
        # relabel must match the predeclared skill and exam roster.
        changed = json.loads(self.options["scored_receipt_bytes"])
        changed["groups"][0]["skill"] = "foreign_skill"
        raw = (json.dumps(changed, sort_keys=True, separators=(",", ":"))
               + "\n").encode()
        signature = hmac.new(
            lineage_fixtures.EVALUATOR_KEY, _SCORER_DOMAIN + raw, sha256
        ).hexdigest()
        transition = replace(
            self.options["transition"],
            scored_receipt_sha256=sha256(raw).hexdigest(),
        )
        changed_options = {
            **self.options, "transition": transition,
            "scored_receipt_bytes": raw, "scored_signature_hex": signature,
        }
        binding = sign_exam_lineage_binding(**changed_options)
        with self.assertRaises((ValueError, RuntimeError)):
            self.derive(binding=binding, **changed_options)

    def test_source_tactic_chooses_weaker_split_without_rounding(self):
        self.assertEqual(_tactic(3, 4, 1, 4), "transfer")
        self.assertEqual(_tactic(1, 4, 3, 4), "practice")
        self.assertEqual(_tactic(1, 4, 2, 8), "contrast")

    def test_authenticity_and_exact_candidate_identity_are_mandatory(self):
        for mutation in (
            {"scored_signature_hex": "f" * 64},
            {"expected_training_roster_sha256": "f" * 64},
            {"candidate_artifact_sha256": "f" * 64},
            {"independent_evaluator_key": b"z" * 32},
            {"expected_nonce_sha256": "f" * 64},
            {"training_exposures": (TrainingExposureRef("held-1", "a" * 64),)},
        ):
            with self.subTest(mutation=list(mutation)), self.assertRaises(
                    (ValueError, RuntimeError)):
                self.derive(**mutation)

    def test_no_unsigned_or_train_control_or_private_exam_relabel(self):
        for extra in (
            {"weak_skills": ("sentiment",)},
            {"claimed_graduation": True},
            {"control_arm": "zero_update"},
            {"control_arm": "corrupted"},
            {"exam_kind": "private_final"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.derive(**extra)
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            self.derive(binding=MinotaurExamReceipt)

    def test_no_prompt_gold_or_graduation_credential_exposed(self):
        feedback = self.derive()
        out = repr(feedback)
        self.assertNotIn("gold_answer", out)
        self.assertNotIn("private_final", out)
        self.assertNotIn("prompt_text", out)
        self.assertNotIn("signing_key", out)
        self.assertFalse(feedback.graduation_authorized)


if __name__ == "__main__":
    unittest.main()
