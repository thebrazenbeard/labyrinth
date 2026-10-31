"""End-to-end source seam, not protected host custody or neural qualification."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import TestCase, mock

from labyrinth.measured_taskmaster_transition import (
    prepare_measured_taskmaster_transition, run_measured_taskmaster_successor,
)
from labyrinth.hf_pilot import PilotConfig
from labyrinth.journey import TRAINING
from tests import test_measured_corridor_handoff as handoff_fixtures


class MeasuredTaskmasterTransitionTests(TestCase):
    def setUp(self) -> None:
        # Reuse the real MAC-verified SQLite score fixture and signed four-key
        # binding, not a fabricated positive score or stub verifier.
        fixture = handoff_fixtures.MeasuredCorridorHandoffTests(
            "test_authenticated_development_failure_advances_maze_without_resetting_learner")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture

    def common(self) -> dict:
        return {
            "state": self.fixture.waiting,
            "binding": self.fixture.binding,
            "training_exposures": self.fixture.training,
            "expected_training_roster_sha256": self.fixture.roster_sha,
            **self.fixture.options,
        }

    def test_same_signed_score_yields_weakness_tactic_and_retained_learner(self):
        prepared = prepare_measured_taskmaster_transition(**self.common())
        handoff = prepared.handoff
        feedback = prepared.feedback
        self.assertEqual(handoff.journey.phase, TRAINING)
        self.assertEqual(handoff.journey.cycle_index, 1)
        self.assertEqual(handoff.remediation, feedback.remediation)
        self.assertEqual(feedback.event_id, handoff.remediation.event_id)
        self.assertEqual(feedback.exam_id, handoff.remediation.exam_id)
        self.assertEqual(handoff.remediation.weak_skills, ("sentiment",))
        self.assertTrue(feedback.observations[0].weak)
        self.assertIn(feedback.observations[0].suggested_train_tactic,
                      ("practice", "contrast", "transfer"))
        self.assertEqual(handoff.journey.checkpoint_sha256,
                         self.fixture.waiting.checkpoint_sha256)
        self.assertEqual(handoff.journey.checkpoint_sha256,
                         self.fixture.options["context"].predecessor_lora_value_sha256)
        self.assertEqual(feedback.candidate_artifact_sha256,
                         self.fixture.options["candidate_artifact_sha256"])
        self.assertNotEqual(feedback.candidate_artifact_sha256,
                            handoff.journey.checkpoint_sha256)
        self.assertFalse(prepared.graduation_authorized)
        self.assertFalse(prepared.model_execution_authenticated)
        self.assertFalse(prepared.host_event_consumption_verified)
        self.assertFalse(feedback.evaluator_host_independence_verified)
        self.assertFalse(feedback.model_execution_authenticated)
        self.assertFalse(handoff.checkpoint_publication_fenced)
        self.assertFalse(self.fixture.waiting.sealed_exam_ids)
        self.assertTrue(handoff.journey.sealed_exam_case_ids)

    def test_opt_in_delegate_passes_to_existing_real_journey_preflight(self):
        config = PilotConfig(model_path="/not/actually/loaded",
                             output_dir="/not/actually/written",
                             seed=33, train_count=4, eval_count=4,
                             tiers=(1,), steps=4)
        reserve = mock.Mock(return_value=True)
        with mock.patch(
            "labyrinth.hf_development_transition.run_exam_bound_successor",
            return_value={"pilot_delegated": True},
        ) as downstream:
            prepared, result = run_measured_taskmaster_successor(
                config, Path("/nonexistent/pilot-run"), reserve,
                **self.common(),
            )
        self.assertEqual(result, {"pilot_delegated": True})
        self.assertEqual(prepared.handoff.journey.phase, TRAINING)
        self.assertEqual(prepared.feedback.event_id, "dev-event-1")
        downstream.assert_called_once()
        self.assertIs(downstream.call_args.kwargs["reserve_development_event"],
                      reserve)
        # Deliberately not invoked by this test: it is host-owned state, not
        # our source's mock authority. No training, optimizer or side effects.
        reserve.assert_not_called()
        self.assertFalse(prepared.model_execution_authenticated)

    def test_forgery_and_operator_overrides_fail_before_downstream(self):
        config = PilotConfig(model_path="/nonexistent", output_dir="/unused",
                             seed=5, train_count=4, eval_count=4,
                             tiers=(1,), steps=4)
        bad = (
            {"candidate_artifact_sha256": "f" * 64},
            {"expected_training_roster_sha256": "f" * 64},
            {"binding": replace(self.fixture.binding, signature_hex="f" * 64)},
            {"state": replace(self.fixture.waiting, checkpoint_sha256="e" * 64)},
            {"state": replace(self.fixture.waiting,
                              pending_checkpoint_before_sha256="e" * 64)},
            {"weak_skills": ("sentiment",)},
            {"exam_kind": "private_final"},
        )
        with mock.patch(
            "labyrinth.measured_taskmaster_transition.run_journey_bound_successor",
        ) as downstream:
            for mutation in bad:
                with self.subTest(mutation=list(mutation)), self.assertRaises(
                        (ValueError, RuntimeError)):
                    run_measured_taskmaster_successor(
                        config, Path("/nonexistent"), lambda _: True,
                        **{**self.common(), **mutation},
                    )
            downstream.assert_not_called()

    def test_internal_minotaur_handoff_disagreement_rejected(self):
        from labyrinth import measured_taskmaster_transition as target

        real = target.derive_minotaur_development_feedback
        verified = real(
            binding=self.fixture.binding,
            training_exposures=self.fixture.training,
            expected_training_roster_sha256=self.fixture.roster_sha,
            **self.fixture.options,
        )
        mismatches = (
            replace(verified, event_id="other-event"),
            replace(verified, remediation=replace(
                verified.remediation, weak_skills=())),
            replace(verified, candidate_artifact_sha256="e" * 64),
            replace(verified, graduation_authorized=True),
            replace(verified, source_signature_verified=False),
        )
        for feedback in mismatches:
            with self.subTest(mismatch=feedback), mock.patch.object(
                target, "derive_minotaur_development_feedback",
                return_value=feedback,
            ), self.assertRaisesRegex(ValueError, "inconsistent"):
                self.assertIsNotNone(
                    prepare_measured_taskmaster_transition(**self.common()))

    def test_source_flags_remain_false_on_clone_state_replay(self):
        # Two in-memory replays can both derive a successor. That is a real
        # runtime limitation, not an in-process one-shot guarantee.
        first = prepare_measured_taskmaster_transition(**self.common())
        second = prepare_measured_taskmaster_transition(**self.common())
        self.assertEqual(first, second)
        self.assertFalse(first.host_event_consumption_verified)
        self.assertFalse(second.host_event_consumption_verified)
        exposed = repr(first)
        for forbidden in ("gold_answer", "private_final", "signing_key"):
            self.assertNotIn(forbidden, exposed)


if __name__ == "__main__":
    import unittest
    unittest.main()
