"""Real signed development-score → Journey regeneration source tests."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import unittest

from labyrinth.journey import (
    AWAITING_EXAM, TRAINING, TrainingCompletionReceipt,
    begin_journey, complete_maze,
)
from labyrinth.measured_corridor_handoff import (
    advance_from_measured_development_failure,
)
from labyrinth.measured_weakness_feedback import (
    TrainingExposureRef, training_roster_commitment,
)
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.signed_exam_lineage import sign_exam_lineage_binding
from tests import test_signed_exam_lineage as lineage_fixtures


class MeasuredCorridorHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = lineage_fixtures.SignedExamLineageTests(
            "test_exact_scored_exam_candidate_transition_and_nonce_verified")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.options = fixture.options.copy()
        self.binding = sign_exam_lineage_binding(**self.options)
        scope = self.options["scope"]
        self.initial = begin_journey(scope, checkpoint_sha256="0" * 64)
        triples = tuple((
            room.room_id,
            f"measured-train-{number}",
            sha256(f"new-TRAIN-only-prompt-{number}".encode()).hexdigest(),
        ) for number, room in enumerate(self.initial.maze.rooms))
        receipt = TrainingCompletionReceipt(
            objective_id=scope.objective_id, cycle_index=0,
            checkpoint_before_sha256=self.initial.checkpoint_sha256,
            # The Journey tracks LoRA tensor-value continuity, whereas the
            # signed candidate_artifact_sha256 pins adapter FILE bytes.
            checkpoint_after_sha256=self.options["context"].predecessor_lora_value_sha256,
            room_case_ids=triples,
        )
        self.waiting = complete_maze(self.initial, receipt)
        self.assertEqual(self.waiting.phase, AWAITING_EXAM)
        self.training = tuple(
            TrainingExposureRef(case_id, prompt_sha)
            for _, case_id, prompt_sha in triples
        )
        self.roster_sha = training_roster_commitment(self.training)

    def advance(self, **changes):
        arguments = {
            "state": self.waiting,
            "binding": self.binding,
            "training_exposures": self.training,
            "expected_training_roster_sha256": self.roster_sha,
            **self.options,
        }
        arguments.update(changes)
        return advance_from_measured_development_failure(**arguments)

    def test_authenticated_development_failure_advances_maze_without_resetting_learner(self):
        handoff = self.advance()
        state = handoff.journey
        self.assertEqual(state.phase, TRAINING)
        self.assertEqual(state.cycle_index, 1)
        self.assertEqual(state.maze, generate_maze_plan(
            self.waiting.scope, 1, weak_skills=handoff.remediation.weak_skills))
        self.assertEqual(handoff.remediation.weak_skills, ("sentiment",))
        self.assertEqual(state.checkpoint_sha256, self.waiting.checkpoint_sha256)
        self.assertEqual(state.checkpoint_sha256,
                         self.options["context"].predecessor_lora_value_sha256)
        self.assertNotEqual(state.checkpoint_sha256,
                            self.options["candidate_artifact_sha256"])
        self.assertEqual(state.baseline_checkpoint_sha256,
                         self.waiting.baseline_checkpoint_sha256)
        self.assertEqual(state.attempts[0].exam_status, "FAIL")
        self.assertEqual(state.attempts[0].exam_id, handoff.remediation.exam_id)
        self.assertEqual(state.attempts[0].checkpoint_after_sha256,
                         self.waiting.checkpoint_sha256)
        self.assertEqual(state.sealed_exam_ids, (handoff.remediation.exam_id,))
        self.assertEqual(len(state.sealed_exam_case_ids),
                         len(self.options["transition"].development_public_roster))
        self.assertFalse(state.pending_training_case_ids)
        self.assertFalse(handoff.graduation_authorized)
        self.assertFalse(handoff.independent_evaluator_host_verified)
        self.assertFalse(handoff.checkpoint_publication_fenced)
        self.assertEqual(self.waiting.phase, AWAITING_EXAM)  # immutable source state

    def test_wrong_journey_phase_and_checkpoint_cannot_advance(self):
        for changed in (
            {"state": self.initial},
            {"state": replace(self.waiting, checkpoint_sha256="f" * 64)},
            {"state": replace(self.waiting,
                              pending_checkpoint_before_sha256="e" * 64)},
            {"state": replace(self.waiting,
                              baseline_checkpoint_sha256="e" * 64)},
            {"context": replace(self.options["context"],
                                predecessor_lora_value_sha256="d" * 64)},
            {"state": replace(self.waiting, scope=replace(
                self.waiting.scope, objective_id="other-objective"))},
        ):
            with self.subTest(change=list(changed)), self.assertRaises(ValueError):
                self.advance(**changed)

    def test_training_roster_must_match_pending_journey_and_host_pin(self):
        for changed in (
            {"training_exposures": (replace(
                self.training[0], case_id="altered-train-case"),) + self.training[1:]},
            {"training_exposures": self.training[:-1]},
            {"training_exposures": (("invalid", "not-an-exposure"),)},
            {"expected_training_roster_sha256": "f" * 64},
            {"training_exposures": (TrainingExposureRef(
                self.training[0].case_id,
                self.options["transition"].development_public_roster[0].prompt_sha256),)
                + self.training[1:]},
        ):
            with self.subTest(change=list(changed)), self.assertRaises(ValueError):
                self.advance(**changed)

    def test_signed_candidate_exam_and_weakness_cannot_be_swapped(self):
        for changed in (
            {"binding": replace(self.binding, signature_hex="f" * 64)},
            {"candidate_artifact_sha256": "f" * 64},
            {"independent_binding_key": b"new-key-not-trusted" * 3},
            {"expected_nonce_sha256": "f" * 64},
            {"weak_skills": ("sentiment",)},
        ):
            with self.subTest(change=list(changed)), self.assertRaises(ValueError):
                self.advance(**changed)

    def test_historical_sealed_exam_replay_or_prompt_overlap_rejected(self):
        exam = self.options["transition"].development_public_roster[0]
        for state in (
            replace(self.waiting,
                    sealed_exam_ids=("dev-exam-0",),
                    exam_event_statuses=("FAIL",)),
            replace(self.waiting,
                    sealed_exam_case_ids=(exam.case_id,),
                    sealed_exam_prompt_sha256=(exam.prompt_sha256,)),
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.advance(state=state)


if __name__ == "__main__":
    unittest.main()
