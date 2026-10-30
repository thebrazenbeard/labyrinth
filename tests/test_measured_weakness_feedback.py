"""Source custody tests for development-only measured Maze remediation."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
import json
import unittest

from labyrinth.measured_weakness_feedback import (
    TrainingExposureRef, derive_measured_remediation, training_roster_commitment,
)
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.signed_exam_lineage import sign_exam_lineage_binding
from tests import test_signed_exam_lineage as lineage_fixtures


_SCORER_DOMAIN = b"labyrinth.blind-development-result.v1\0"


class MeasuredWeaknessFeedbackTests(unittest.TestCase):
    def setUp(self):
        # Existing fixtures operate the real HMAC-sealed, one-shot SQLite
        # development scorer and four separately keyed signed custody layers.
        fixture = lineage_fixtures.SignedExamLineageTests(
            "test_exact_scored_exam_candidate_transition_and_nonce_verified")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.options = fixture.options.copy()
        self.binding = sign_exam_lineage_binding(**self.options)
        self.training = (TrainingExposureRef("train-1", "a" * 64),
                         TrainingExposureRef("train-2", "b" * 64))
        self.roster_sha = training_roster_commitment(self.training)

    def derive(self, **changes):
        bundle = {
            "binding": self.binding,
            "training_exposures": self.training,
            "expected_training_roster_sha256": self.roster_sha,
            **self.options,
        }
        bundle.update(changes)
        return derive_measured_remediation(**bundle)

    def test_signed_scored_weakness_drives_exact_next_maze(self):
        result = self.derive()
        self.assertEqual(result.weak_skills, ("sentiment",))
        self.assertEqual(result.uncertain_skills, ("sentiment",))
        self.assertEqual(result.from_cycle, 0)
        self.assertEqual(result.to_cycle, 1)
        self.assertEqual(result.exam_id, "dev-exam-0")
        self.assertEqual(result.event_id, "dev-event-1")
        self.assertEqual(result.roster_sha256, self.roster_sha)
        self.assertEqual(result.maze, generate_maze_plan(
            self.options["scope"], 1, weak_skills=("sentiment",)))
        self.assertEqual(result.coverage_by_skill,
                         (("sentiment", 2, 2),))
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.model_execution_authenticated)

    def test_host_pinned_training_roster_and_exposure_partition_required(self):
        for change in (
            {"training_exposures": ()},
            {"training_exposures": (self.training[0], self.training[0])},
            {"training_exposures": (self.training[0],
                                    TrainingExposureRef("train-3", "a" * 64))},
            {"expected_training_roster_sha256": "f" * 64},
            {"training_exposures": (TrainingExposureRef("held-1", "a" * 64),)},
            {"training_exposures": (TrainingExposureRef(
                "train-1", self.options["transition"].development_public_roster[0].prompt_sha256),)},
        ):
            with self.subTest(case=change), self.assertRaises(ValueError):
                self.derive(**change)

    def test_invalid_training_records_rejected(self):
        for value in (
            ("bad", "a" * 64),
            TrainingExposureRef("x", "A" * 64),
            TrainingExposureRef("train/unsafe", "a" * 64),
        ):
            with self.subTest(case=value), self.assertRaises(ValueError):
                training_roster_commitment((value,))

    def test_untrusted_weakness_override_and_unknown_option_rejected(self):
        with self.assertRaises(ValueError):
            self.derive(weak_skills=("sentiment",))
        with self.assertRaises(ValueError):
            self.derive(graduation_authorized=True)

    def test_signed_cycle_scope_candidate_and_exam_authenticity_required(self):
        for change in (
            {"binding": replace(self.binding, signature_hex="f" * 64)},
            {"candidate_run_id": "different-run"},
            {"candidate_artifact_sha256": "f" * 64},
            {"expected_evaluator_roster_sha256": "f" * 64},
            {"expected_nonce_sha256": "f" * 64},
            {"scored_signature_hex": "f" * 64},
            {"independent_evaluator_key": b"z" * 32},
            {"context": replace(self.options["context"], cycle_index=2)},
        ):
            with self.subTest(case=list(change)), self.assertRaises(ValueError):
                self.derive(**change)

    def test_signed_aggregate_totals_must_match_public_roster(self):
        mutated = json.loads(self.options["scored_receipt_bytes"])
        mutated["groups"][0]["total"] += 1
        raw = (json.dumps(mutated, sort_keys=True, separators=(",", ":"))
               + "\n").encode()
        signature = hmac.new(lineage_fixtures.EVALUATOR_KEY,
                             _SCORER_DOMAIN + raw, sha256).hexdigest()
        transition = replace(self.options["transition"],
                             scored_receipt_sha256=sha256(raw).hexdigest())
        options = {**self.options, "transition": transition,
                   "scored_receipt_bytes": raw, "scored_signature_hex": signature}
        binding = sign_exam_lineage_binding(**options)
        with self.assertRaisesRegex(ValueError, "aggregate totals"):
            self.derive(binding=binding, **options)

    def test_missing_signed_development_split_cannot_drive_regeneration(self):
        mutated = json.loads(self.options["scored_receipt_bytes"])
        mutated["groups"] = [g for g in mutated["groups"] if g["split"] == "heldout"]
        raw = (json.dumps(mutated, sort_keys=True, separators=(",", ":"))
               + "\n").encode()
        sig = hmac.new(lineage_fixtures.EVALUATOR_KEY,
                       _SCORER_DOMAIN + raw, sha256).hexdigest()
        modified = {**self.options, "scored_receipt_bytes": raw,
                    "scored_signature_hex": sig,
                    "transition": replace(self.options["transition"],
                        scored_receipt_sha256=sha256(raw).hexdigest())}
        with self.assertRaises(ValueError):
            sign_exam_lineage_binding(**modified)


if __name__ == "__main__":
    unittest.main()
