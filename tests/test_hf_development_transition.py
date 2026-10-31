"""Signed development FAIL is the only accepted input to the opt-in HF bridge."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
from dataclasses import replace
from unittest import TestCase, mock

from labyrinth.hf_development_transition import (
    SCHEMA, _DOMAIN, DevelopmentCaseRef, development_case_commitment,
    run_exam_bound_successor, run_journey_bound_successor, verify_development_transition,
    BoundReservationAuthority, BoundReservationReceipt,
    bound_reservation_receipt_payload,
    development_exam_lineage_commitment,
)
from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityConflict, ExternalMonotonicReservationClient,
    ExternalAuthorityOutcomeUnknown, ExternalReservationReceipt,
    ExternalReservationRequest, next_head_digest,
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

    def _bound_authority(self, callback):
        return BoundReservationAuthority(
            sink_id="isolated-reservation-sink",
            run_id="candidate-run-B",
            candidate_artifact_sha256="c" * 64,
            epoch=3, fence=5,
            receipt_verification_key=b"independent-reservation-key-for-tests-v1",
            reserve=callback,
        )

    def _signed_bound_receipt(self, request, *, change=None):
        from dataclasses import replace as dataclass_replace

        candidate = dataclass_replace(request, **(change or {}))
        receipt = BoundReservationReceipt(
            request=candidate,
            sink_id="isolated-reservation-sink",
            transaction_id="reservation-transaction-unique",
            signature_hex="0" * 64,
        )
        signature = hmac.new(
            b"independent-reservation-key-for-tests-v1",
            bound_reservation_receipt_payload(receipt), hashlib.sha256,
        ).hexdigest()
        return dataclass_replace(receipt, signature_hex=signature)

    def test_bound_receipt_exact_run_scope_and_fence_required_before_neural(self):
        raw, sig = self.signed()
        seen = []

        def reserve(request):
            seen.append(request)
            return self._signed_bound_receipt(request)

        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot",
                         return_value={"trained": True}) as runner):
            result = run_journey_bound_successor(
                self.config, self.scope, "prior-run", self.context, raw, sig,
                journey=self._regenerated_journey(),
                evaluator_precommitment_sha256=self.fields["exam_case_commitment_sha256"],
                evaluator_public_roster=self.public_roster,
                independent_development_key=self.key,
                bound_reservation_authority=self._bound_authority(reserve),
            )
        self.assertEqual(result, {"trained": True})
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].scope_fingerprint, self.scope.fingerprint)
        self.assertEqual(seen[0].predecessor_lora_value_sha256,
                         self.context.predecessor_lora_value_sha256)
        self.assertEqual((seen[0].epoch, seen[0].fence), (3, 5))
        self.assertEqual(seen[0].run_id, "candidate-run-B")
        runner.assert_called_once()

    def test_bound_receipt_from_foreign_run_same_event_rejected(self):
        raw, sig = self.signed()
        observed = []
        mutations = (
            {"run_id": "candidate-run-A"},
            {"scope_fingerprint": "f" * 64},
            {"candidate_artifact_sha256": "e" * 64},
            {"cycle_index": 2},
            {"predecessor_lora_value_sha256": "e" * 64},
            {"predecessor_report_sha256": "e" * 64},
            {"authority_ref": "different-authority"},
            {"epoch": 4},
            {"fence": 6},
            {"output_path_sha256": "e" * 64},
        )
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            for mutation in mutations:
                def wrong_run(request):
                    observed.append(request)
                    return self._signed_bound_receipt(request, change=mutation)

                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                        ValueError, "bound reservation"):
                    run_journey_bound_successor(
                        self.config, self.scope, "prior-run", self.context, raw, sig,
                        journey=self._regenerated_journey(),
                        evaluator_precommitment_sha256=self.fields[
                            "exam_case_commitment_sha256"],
                        evaluator_public_roster=self.public_roster,
                        independent_development_key=self.key,
                        bound_reservation_authority=self._bound_authority(wrong_run),
                    )
            runner.assert_not_called()
        self.assertEqual(len(observed), len(mutations))

    def test_bound_receipt_forged_signature_and_legacy_collision_refused(self):
        raw, sig = self.signed()
        def forged(request):
            return replace(self._signed_bound_receipt(request), signature_hex="f" * 64)
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            with self.assertRaisesRegex(ValueError, "bound reservation"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    bound_reservation_authority=self._bound_authority(forged),
                )
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    reserve_development_event=lambda _event: True,
                    bound_reservation_authority=self._bound_authority(forged),
                )
            runner.assert_not_called()

    def test_bound_authority_does_not_expose_symmetric_key_in_repr(self):
        key = b"high-sensitivity-test-only-reservation-key"
        authority = replace(self._bound_authority(lambda _request: None),
                            receipt_verification_key=key)
        displayed = repr(authority)
        self.assertNotIn(key.decode("ascii"), displayed)
        self.assertNotIn("<function", displayed)

    def _external_qualification_fixture(self, raw: bytes, sig: str):
        """A stateful signed FAKE host; source tests do not prove host custody."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        signing_key = Ed25519PrivateKey.generate()
        intent = ExternalReservationRequest(
            provider_id="external-authority", run_id="candidate-run-B",
            event_id=self.fields["event_id"], exam_id=self.fields["exam_id"],
            candidate_artifact_sha256="c" * 64,
            scope_fingerprint=self.scope.fingerprint,
            exam_lineage_sha256=development_exam_lineage_commitment(raw, sig),
            prior_checkpoint_sha256=self.context.predecessor_lora_value_sha256,
            to_cycle=self.context.cycle_index,
            expected_epoch=3, expected_fence=5, expected_commit_index=0,
            expected_head_sha256="0" * 64, operation_nonce="host-nonce-001",
        )

        class FakeExternalHost:
            def __init__(self):
                self.head = intent.expected_head()
                self.calls = 0
                self.consumed = set()

            def read_head(self, run_id):
                if run_id != intent.run_id:
                    raise ExternalAuthorityConflict("wrong run")
                return self.head

            def reserve_once(self, request):
                self.calls += 1
                if request != intent or request.expected_head() != self.head:
                    raise ExternalAuthorityConflict("external CAS conflict")
                if request.event_id in self.consumed:
                    raise ExternalAuthorityConflict("event already consumed")
                unsigned = ExternalReservationReceipt(
                    provider_id=request.provider_id, run_id=request.run_id,
                    event_id=request.event_id, reservation_id="reservation-one",
                    request_sha256=request.digest(), epoch=request.expected_epoch,
                    fence=request.expected_fence,
                    commit_index=request.expected_commit_index + 1,
                    previous_head_sha256=request.expected_head_sha256,
                    new_head_sha256=next_head_digest(request, "reservation-one"),
                    signature_hex="0" * 128,
                )
                receipt = replace(unsigned, signature_hex=signing_key.sign(
                    unsigned.signed_payload()).hex())
                self.consumed.add(request.event_id)
                self.head = receipt.post_head()
                return receipt

        host = FakeExternalHost()
        client = ExternalMonotonicReservationClient(
            port=host, provider_id=intent.provider_id,
            pinned_request=intent,
            pinned_ed25519_public_key=signing_key.public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw),
        )
        return host, client, intent

    def test_qualification_mode_rejects_legacy_and_hmac_and_missing_external_authority(self):
        raw, sig = self.signed()
        host, client, intent = self._external_qualification_fixture(raw, sig)
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume") as preflight,
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            for args in (
                {},
                {"reserve_development_event": lambda _event: True},
                {"bound_reservation_authority": self._bound_authority(
                    lambda request: self._signed_bound_receipt(request))},
                {"reserve_development_event": lambda _event: True,
                 "external_reservation_client": client,
                 "external_reservation_request": intent},
                {"bound_reservation_authority": self._bound_authority(
                    lambda request: self._signed_bound_receipt(request)),
                 "external_reservation_client": client,
                 "external_reservation_request": intent},
            ):
                with self.subTest(args=tuple(args)), self.assertRaisesRegex(
                    ValueError, "qualification"):
                    run_journey_bound_successor(
                        self.config, self.scope, "prior-run", self.context, raw, sig,
                        journey=self._regenerated_journey(),
                        evaluator_precommitment_sha256=self.fields[
                            "exam_case_commitment_sha256"],
                        evaluator_public_roster=self.public_roster,
                        independent_development_key=self.key,
                        qualification_mode=True, **args,
                    )
            preflight.assert_not_called()
            runner.assert_not_called()
            self.assertEqual(host.calls, 0)

    def test_qualification_mode_external_signed_one_shot_replay_denied_before_runner(self):
        raw, sig = self.signed()
        host, client, intent = self._external_qualification_fixture(raw, sig)
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot",
                         return_value={"source_neural_mock": True}) as runner):
            kwargs = dict(
                journey=self._regenerated_journey(),
                evaluator_precommitment_sha256=self.fields[
                    "exam_case_commitment_sha256"],
                evaluator_public_roster=self.public_roster,
                independent_development_key=self.key, qualification_mode=True,
                external_reservation_client=client,
                external_reservation_request=intent,
            )
            result = run_journey_bound_successor(
                self.config, self.scope, "prior-run", self.context, raw, sig, **kwargs)
            self.assertEqual(result, {"source_neural_mock": True})
            self.assertEqual(host.head.commit_index, 1)
            # Even with an authentic signed receipt from the first call,
            # the source must obtain a fresh host CAS decision, never replay it.
            with self.assertRaisesRegex(ExternalAuthorityConflict, "head mismatch"):
                run_journey_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig, **kwargs)
            self.assertEqual(host.calls, 1)
            runner.assert_called_once()

    def test_qualification_mode_wrong_lineage_denied_before_host_and_runner(self):
        raw, sig = self.signed()
        host, client, intent = self._external_qualification_fixture(raw, sig)
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume") as preflight,
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            for mutation in (
                {"exam_lineage_sha256": "e" * 64},
                {"prior_checkpoint_sha256": "e" * 64},
                {"scope_fingerprint": "e" * 64},
                {"event_id": "different-event"},
                {"to_cycle": 2},
            ):
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    ExternalAuthorityConflict, "signed predecessor or exam"):
                    run_journey_bound_successor(
                        self.config, self.scope, "prior-run", self.context, raw, sig,
                        journey=self._regenerated_journey(),
                        evaluator_precommitment_sha256=self.fields[
                            "exam_case_commitment_sha256"],
                        evaluator_public_roster=self.public_roster,
                        independent_development_key=self.key,
                        qualification_mode=True,
                        external_reservation_client=client,
                        external_reservation_request=replace(intent, **mutation),
                    )
            self.assertEqual(host.calls, 0)
            preflight.assert_not_called()
            runner.assert_not_called()

    def test_qualification_mode_cannot_bypass_journey_through_exam_only_api(self):
        raw, sig = self.signed()
        host, client, intent = self._external_qualification_fixture(raw, sig)
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume") as preflight,
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            with self.assertRaisesRegex(ValueError, "validated regenerated Journey"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    qualification_mode=True,
                    external_reservation_client=client,
                    external_reservation_request=intent,
                )
            self.assertEqual(host.calls, 0)
            preflight.assert_not_called()
            runner.assert_not_called()

    def test_direct_public_qualification_cannot_forge_internal_journey_token(self):
        """Regression for L28: exported module sentinel allowed Journey bypass."""
        raw, sig = self.signed()
        host, client, intent = self._external_qualification_fixture(raw, sig)
        self.assertNotIn("_journey_validation_token",
                         inspect.signature(run_exam_bound_successor).parameters)
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume") as preflight,
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            with self.assertRaises(TypeError):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    qualification_mode=True,
                    external_reservation_client=client,
                    external_reservation_request=intent,
                    _journey_validation_token=object(),
                )
            with self.assertRaisesRegex(ValueError, "validated regenerated Journey"):
                run_exam_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    independent_development_key=self.key,
                    qualification_mode=True,
                    external_reservation_client=client,
                    external_reservation_request=intent,
                )
            self.assertEqual(host.calls, 0)
            preflight.assert_not_called()
            runner.assert_not_called()

    def test_qualification_rejects_duck_typed_client_before_preflight(self):
        raw, sig = self.signed()
        host, client, intent = self._external_qualification_fixture(raw, sig)

        class ForgedClient:
            def reserve(self, _request):
                raise AssertionError("duck-typed client must not run")

        with (mock.patch("labyrinth.hf_development_transition.preflight_resume") as preflight,
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            with self.assertRaisesRegex(ValueError, "typed external reservation"):
                run_journey_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    journey=self._regenerated_journey(),
                    evaluator_precommitment_sha256=self.fields[
                        "exam_case_commitment_sha256"],
                    evaluator_public_roster=self.public_roster,
                    independent_development_key=self.key,
                    qualification_mode=True,
                    external_reservation_client=ForgedClient(),
                    external_reservation_request=intent,
                )
            preflight.assert_not_called()
            runner.assert_not_called()
            self.assertEqual(host.calls, 0)

    def test_qualification_invalid_ed25519_receipt_keeps_runner_fenced(self):
        raw, sig = self.signed()
        host, client, intent = self._external_qualification_fixture(raw, sig)
        signed_reserve_once = host.reserve_once

        def corrupt_after_commit(request):
            valid_receipt = signed_reserve_once(request)
            return replace(valid_receipt, signature_hex="f" * 128)

        host.reserve_once = corrupt_after_commit
        with (mock.patch("labyrinth.hf_development_transition.preflight_resume"),
              mock.patch("labyrinth.hf_development_transition.run_pilot") as runner):
            with self.assertRaisesRegex(ExternalAuthorityOutcomeUnknown, "unverified"):
                run_journey_bound_successor(
                    self.config, self.scope, "prior-run", self.context, raw, sig,
                    journey=self._regenerated_journey(),
                    evaluator_precommitment_sha256=self.fields[
                        "exam_case_commitment_sha256"],
                    evaluator_public_roster=self.public_roster,
                    independent_development_key=self.key,
                    qualification_mode=True,
                    external_reservation_client=client,
                    external_reservation_request=intent,
                )
            self.assertEqual(host.calls, 1)
            self.assertEqual(host.head.commit_index, 1)
            runner.assert_not_called()


if __name__ == "__main__":
    import unittest
    unittest.main()
