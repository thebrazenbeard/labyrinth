"""Signed development FAIL is the only accepted input to the opt-in HF bridge."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace
from unittest import TestCase, mock

from labyrinth.hf_development_transition import (
    SCHEMA, _DOMAIN, DevelopmentCaseRef, development_case_commitment,
    run_exam_bound_successor, run_journey_bound_successor, verify_development_transition,
)
from labyrinth.hf_resume_bridge import ResumeContext
from labyrinth.hf_pilot import (
    PilotConfig, _authorized_development_resume, run_pilot,
)
from labyrinth.training_scope import TrainingScopeSpec
from labyrinth.journey import JourneyAttempt, JourneyState, TRAINING
from labyrinth.regenerative_maze import generate_maze_plan


class SignedDevelopmentTransitionTests(TestCase):
    def setUp(self) -> None:
        self.scope = TrainingScopeSpec(
            "learn-arithmetic", "reasoning", ("arithmetic",), 1, 1, 3, 17,
        )
        self.context = ResumeContext(
            self.scope.fingerprint, 1, "a" * 64, "b" * 64,
            "trusted-development-cycle-one",
        )
        self.config = PilotConfig(model_path="unused-model", output_dir="unused-output")
        self.key = b"development-evaluator-secret-32-bytes!!"
        self.public_roster = (
            DevelopmentCaseRef("exam-case-0", "arithmetic", "f" * 64, "arithmetic-chain"),
        )
        self.fields = {
            "schema": SCHEMA,
            "event_id": "dev-event-1",
            "exam_id": "exam-failed-0",
            "exam_kind": "development",
            "exam_status": "FAIL",
            "objective_id": self.scope.objective_id,
            "scope_fingerprint": self.scope.fingerprint,
            "from_cycle": 0,
            "to_cycle": 1,
            "predecessor_report_sha256": self.context.predecessor_report_sha256,
            "predecessor_lora_value_sha256": self.context.predecessor_lora_value_sha256,
            "resume_authority_ref": self.context.authority_ref,
            "weak_skills": ["arithmetic"],
            "exam_case_commitment_sha256": development_case_commitment(self.public_roster),
        }

    def signed(self, changes: dict | None = None) -> tuple[bytes, str]:
        body = {**self.fields, **(changes or {})}
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sig = hmac.new(self.key, _DOMAIN + raw, hashlib.sha256).hexdigest()
        return raw, sig

    def verify(self, raw: bytes, sig: str):
        return verify_development_transition(
            self.scope, self.context, raw, sig,
            independent_development_key=self.key,
        )

    def _regenerated_journey(self) -> JourneyState:
        # This fixture adopts the LoRA-value digest as the Journey checkpoint
        # identity. It is source-only data, not an attested neural model.
        attempt = JourneyAttempt(
            cycle_index=0, exam_id=self.fields["exam_id"],
            checkpoint_before_sha256="d" * 64,
            checkpoint_after_sha256=self.context.predecessor_lora_value_sha256,
            training_case_ids=("train-case-0",),
            training_prompt_sha256=("e" * 64,),
            exam_case_ids=("exam-case-0",),
            exam_prompt_sha256=("f" * 64,),
            exam_status="FAIL", weak_skills=("arithmetic",),
        )
        return JourneyState(
            scope=self.scope, baseline_checkpoint_sha256="d" * 64,
            checkpoint_sha256=self.context.predecessor_lora_value_sha256,
            maze=generate_maze_plan(self.scope, 1, weak_skills=("arithmetic",)),
            phase=TRAINING, attempts=(attempt,),
            used_training_case_ids=("train-case-0",),
            used_training_prompt_sha256=("e" * 64,),
            sealed_exam_ids=(self.fields["exam_id"],),
            exam_event_statuses=("FAIL",),
            sealed_exam_case_ids=("exam-case-0",),
            sealed_exam_prompt_sha256=("f" * 64,),
        )

    def test_strict_journey_successor_matches_signed_development_evidence(self):
        raw, sig = self.signed()
        reserved = []
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot",
                         return_value={"trained": True}) as runner):
            result = run_journey_bound_successor(
                self.config, self.scope, "prior-run", self.context, raw, sig,
                journey=self._regenerated_journey(),
                evaluator_precommitment_sha256=self.fields["exam_case_commitment_sha256"],
                evaluator_public_roster=self.public_roster,
                independent_development_key=self.key,
                reserve_development_event=lambda event_id: reserved.append(event_id) or True,
            )
        self.assertEqual(result, {"trained": True})
        self.assertEqual(reserved, [self.fields["event_id"]])
        runner.assert_called_once()

    def test_strict_journey_successor_rejects_mismatched_lineage_before_reservation(self):
        raw, sig = self.signed()
        state = self._regenerated_journey()
        invalid_states = {
            "wrong checkpoint": replace(state, checkpoint_sha256="9" * 64),
            "wrong cycle": replace(state, maze=generate_maze_plan(self.scope, 0)),
            "wrong exam": replace(state, attempts=(replace(state.attempts[-1], exam_id="other-exam"),)),
            "wrong weaknesses": replace(state, attempts=(replace(state.attempts[-1], weak_skills=()),)),
            "wrong decision": replace(state, attempts=(replace(state.attempts[-1], exam_status="PASS"),)),
            "wrong sealed event": replace(state, sealed_exam_ids=("other-exam",)),
            "missing sealed event": replace(state, sealed_exam_ids=(), exam_event_statuses=()),
            "missing exam case": replace(state, attempts=(replace(state.attempts[-1], exam_case_ids=()),)),
            "discontinuous learner": replace(state, attempts=(replace(state.attempts[-1], checkpoint_before_sha256="8" * 64),)),
            "unrecorded train": replace(state, attempts=(replace(state.attempts[-1], training_case_ids=("unrecorded",)),)),
        }
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume") as preflight,
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            for label, bad_state in invalid_states.items():
                with self.subTest(case=label), self.assertRaises(ValueError):
                    run_journey_bound_successor(
                        self.config, self.scope, "prior-run", self.context, raw, sig,
                        journey=bad_state,
                        evaluator_precommitment_sha256=self.fields["exam_case_commitment_sha256"],
                        evaluator_public_roster=self.public_roster,
                        independent_development_key=self.key,
                        reserve_development_event=lambda _id: self.fail("event consumed"),
                    )
            with self.assertRaisesRegex(ValueError, "independently precommitted"):
                run_journey_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    journey=state, evaluator_precommitment_sha256="9" * 64,
                    evaluator_public_roster=self.public_roster,
                    independent_development_key=self.key,
                    reserve_development_event=lambda _id: self.fail("event consumed"),
                )
            preflight.assert_not_called()
            runner.assert_not_called()

    def test_strict_journey_successor_rejects_forged_public_roster(self):
        raw, sig = self.signed()
        state = self._regenerated_journey()
        forged_rosters = (
            (replace(self.public_roster[0], case_id="forged-id"),),
            (replace(self.public_roster[0], prompt_sha256="a" * 64),),
            (replace(self.public_roster[0], skill="other-skill"),),
            (replace(self.public_roster[0], structure_key="other-structure"),),
            self.public_roster + (DevelopmentCaseRef("extra", "arithmetic", "a" * 64, "extra-structure"),),
        )
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume") as preflight,
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            for roster in forged_rosters:
                with self.subTest(roster=roster), self.assertRaises(ValueError):
                    run_journey_bound_successor(
                        self.config, self.scope, "prior-run", self.context, raw, sig,
                        journey=state,
                        evaluator_precommitment_sha256=self.fields["exam_case_commitment_sha256"],
                        evaluator_public_roster=roster,
                        independent_development_key=self.key,
                        reserve_development_event=lambda _id: self.fail("event consumed"),
                    )
            preflight.assert_not_called()
            runner.assert_not_called()

    def test_signed_failure_controls_actual_successor_arguments(self):
        raw, sig = self.signed()
        seen = set()

        def reserve(event_id):
            if event_id in seen:
                return False
            seen.add(event_id)
            return True

        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot",
                         return_value={"ok": True}) as runner):
            result = run_exam_bound_successor(
                self.config, self.scope, "prior-run", self.context, raw, sig,
                independent_development_key=self.key,
                reserve_development_event=reserve,
            )
            self.assertEqual(result, {"ok": True})
            runner.assert_called_once_with(
                self.config, scope=self.scope, cycle_index=1,
                predecessor_run_dir="prior-run", resume_context=self.context,
                weak_skills=("arithmetic",),
            )
            # A replay cannot launch another training run.
            with self.assertRaisesRegex(ValueError, "not newly reserved"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    reserve_development_event=reserve,
                )
            self.assertEqual(runner.call_count, 1)

    def test_unsigned_direct_scoped_resume_fails_before_model_or_dependencies(self):
        with (mock.patch("labyrinth.hf_pilot.validate_config") as validate,
              mock.patch("labyrinth.hf_pilot._deps") as deps):
            with self.assertRaisesRegex(ValueError, "evaluator-signed development FAIL"):
                run_pilot(
                    self.config, scope=self.scope, cycle_index=1,
                    predecessor_run_dir="prior-run", resume_context=self.context,
                    weak_skills=("arithmetic",),
                )
            validate.assert_not_called()
            deps.assert_not_called()

    def test_untrusted_research_resume_is_explicit_and_not_a_signed_gate(self):
        with mock.patch("labyrinth.hf_pilot.validate_config",
                        side_effect=ValueError("after research gate")) as validate:
            with self.assertRaisesRegex(ValueError, "after research gate"):
                run_pilot(
                    self.config, scope=self.scope, cycle_index=1,
                    predecessor_run_dir="prior-run", resume_context=self.context,
                    weak_skills=("arithmetic",), unauthenticated_research_resume=True,
                )
            validate.assert_called_once()
        with self.assertRaisesRegex(ValueError, "requires a scoped successor"):
            run_pilot(self.config, unauthenticated_research_resume=True)

    def test_signed_authorization_context_is_scoped_and_cleared_on_exception(self):
        raw, sig = self.signed()
        self.assertIsNone(_authorized_development_resume.get())
        observed = []

        def capture(*_args, **_kwargs):
            observed.append(_authorized_development_resume.get())
            raise RuntimeError("test interruption")

        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot",
                         side_effect=capture)):
            with self.assertRaisesRegex(RuntimeError, "test interruption"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    reserve_development_event=lambda _id: True,
                )
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][1], self.scope.fingerprint)
        self.assertEqual(observed[0][-1], ("arithmetic",))
        self.assertIsNone(_authorized_development_resume.get())

    def test_signed_bridge_passes_pilot_guard_only_after_reservation(self):
        raw, sig = self.signed()
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_pilot.validate_config",
                         side_effect=ValueError("signed passed pilot guard")) as validate):
            with self.assertRaisesRegex(ValueError, "signed passed pilot guard"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    reserve_development_event=lambda _id: True,
                )
            validate.assert_called_once()
        self.assertIsNone(_authorized_development_resume.get())

    def test_unsigned_or_tampered_weakness_cannot_start_training(self):
        raw, sig = self.signed()
        tampered, _ = self.signed({"weak_skills": []})
        with mock.patch("labyrinth.hf_development_transition.run_pilot") as runner:
            for bad_raw, bad_sig in ((tampered, sig), (raw, "0" * 64)):
                with self.subTest(case=(len(bad_raw), bad_sig[:8])):
                    with self.assertRaisesRegex(ValueError, "not authenticated"):
                        run_exam_bound_successor(
                            self.config, self.scope, "prior-run", self.context,
                            bad_raw, bad_sig, independent_development_key=self.key,
                            reserve_development_event=lambda _id: True,
                        )
            runner.assert_not_called()

    def test_signed_nonfailure_and_private_exam_are_rejected(self):
        for changes in (
            {"exam_status": "PASS"},
            {"exam_status": "INCONCLUSIVE"},
            {"exam_kind": "private_final"},
            {"weak_skills": []},
        ):
            with self.subTest(changes=changes):
                raw, sig = self.signed(changes)
                with self.assertRaises(ValueError):
                    self.verify(raw, sig)

    def test_signed_cross_scope_cycle_or_predecessor_does_not_authorize(self):
        for changes in (
            {"objective_id": "other-objective"},
            {"scope_fingerprint": "d" * 64},
            {"from_cycle": 1},
            {"to_cycle": 2},
            {"predecessor_report_sha256": "e" * 64},
            {"predecessor_lora_value_sha256": "f" * 64},
            {"resume_authority_ref": "different-authority"},
            {"weak_skills": ["unauthorized_skill"]},
            {"exam_case_commitment_sha256": "not-a-digest"},
        ):
            with self.subTest(changes=changes):
                raw, sig = self.signed(changes)
                with self.assertRaises(ValueError):
                    self.verify(raw, sig)

    def test_duplicate_json_fields_and_wrong_key_rejected(self):
        raw, sig = self.signed()
        duplicate = raw[:-1] + b',"exam_status":"FAIL"}'
        duplicate_sig = hmac.new(self.key, _DOMAIN + duplicate, hashlib.sha256).hexdigest()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.verify(duplicate, duplicate_sig)
        with self.assertRaisesRegex(ValueError, r"32\+ byte"):
            verify_development_transition(
                self.scope, self.context, raw, sig,
                independent_development_key=b"short",
            )
        with self.assertRaisesRegex(ValueError, "not authenticated"):
            verify_development_transition(
                self.scope, self.context, raw, sig,
                independent_development_key=b"other-key-owned-by-someone-else-32",
            )

    def test_legacy_bootstrap_is_not_independently_failed_dev_cycle(self):
        raw, sig = self.signed()
        legacy = ResumeContext(
            self.scope.fingerprint, 1, "a" * 64, "b" * 64,
            "trusted-development-cycle-one", allow_legacy_unscoped=True,
        )
        with self.assertRaisesRegex(ValueError, "not legacy"):
            verify_development_transition(
                self.scope, legacy, raw, sig,
                independent_development_key=self.key,
            )

    def test_no_training_without_event_reservation(self):
        raw, sig = self.signed()
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            with self.assertRaisesRegex(ValueError, "not newly reserved"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context,
                    raw, sig, independent_development_key=self.key,
                    reserve_development_event=lambda _id: False,
                )
            runner.assert_not_called()

    def test_invalid_model_preflight_keeps_event_available_for_safe_retry(self):
        raw, sig = self.signed()
        consumed = set()

        def reserve(event_id):
            if event_id in consumed:
                return False
            consumed.add(event_id)
            return True

        bad_config = replace(
            self.config,
            model_path="/nonexistent-labyrinth-preflight-model-20261030",
        )
        with mock.patch("labyrinth.hf_development_transition.run_pilot",
                        side_effect=FileNotFoundError("model does not exist")):
            with self.assertRaises(FileNotFoundError):
                run_exam_bound_successor(
                    bad_config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    reserve_development_event=reserve,
                )
        self.assertEqual(consumed, set(), "invalid local model must not consume signed exam")

        # Once preflight succeeds, the same authentic event remains available.
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot",
                         return_value={"ok": True}) as runner):
            self.assertEqual(
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    reserve_development_event=reserve,
                ),
                {"ok": True},
            )
            runner.assert_called_once()
        self.assertEqual(consumed, {"dev-event-1"})

    def test_invalid_predecessor_preflight_does_not_consume_event(self):
        raw, sig = self.signed()
        consumed = []

        def reserve(event_id):
            consumed.append(event_id)
            return True

        with (mock.patch("labyrinth.hf_development_transition.preflight_resume",
                         side_effect=ValueError("predecessor checkpoint mismatch")),
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            with self.assertRaisesRegex(ValueError, "predecessor checkpoint mismatch"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    reserve_development_event=reserve,
                )
            runner.assert_not_called()
        self.assertEqual(consumed, [])

    def test_post_reservation_failure_stays_consumed_until_host_reconciles(self):
        raw, sig = self.signed()
        consumed = set()

        def reserve(event_id):
            if event_id in consumed:
                return False
            consumed.add(event_id)
            return True

        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot",
                         side_effect=RuntimeError("training interrupted")) as runner):
            with self.assertRaisesRegex(RuntimeError, "training interrupted"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    reserve_development_event=reserve,
                )
            with self.assertRaisesRegex(ValueError, "not newly reserved"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    reserve_development_event=reserve,
                )
            self.assertEqual(runner.call_count, 1)
        self.assertEqual(consumed, {"dev-event-1"})


if __name__ == "__main__":
    import unittest
    unittest.main()
