"""Signed-run / neural checkpoint linkage; checkpoint loading is MOCKED.

Real HMAC receipts and a real MAC-sealed journal establish source behavior,
not physical pretrained-model or host execution attestation.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from labyrinth.hf_development_transition import (
    SCHEMA as SIGNED_SCHEMA, _DOMAIN as SIGNED_DOMAIN,
    DevelopmentCaseRef, development_case_commitment,
)
from labyrinth.hf_development_event_journal import (
    initialize_development_event_journal, inspect_development_event,
)
from labyrinth.hf_pilot import PilotConfig
from labyrinth.hf_resume_bridge import ResumeContext, ResumeSource
from labyrinth.neural_resume_checkpoint import FullCheckpointContext, FullCheckpointReceipt
from labyrinth.recovery_composition import (
    FencedCheckpointBinding, RecoveryCompositionError,
)
from labyrinth.neural_run_binding import (
    ExamLineageCustody, NeuralRunBindingError, _pilot_identity,
    authenticate_signed_neural_run,
    run_bound_neural_successor, signed_neural_run_commitment,
)
from labyrinth.training_scope import TrainingScopeSpec
from labyrinth.signed_exam_lineage import sign_exam_lineage_binding
from tests import test_signed_exam_lineage as lineage_fixtures


class SignedNeuralRunBindingTests(TestCase):
    SIGN_KEY = b"t" * 32
    JOURNAL_KEY = b"separately-controlled-journal-state-key-32-bytes"
    BASE = "c" * 64
    PRIOR_FULL = "d" * 64
    CHECKPOINT = "a" * 64

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        # Preserve authentic four-key scorer/lineage/legacy transition evidence;
        # only the preflight HF model bytes/loader are mocked at this seam.
        self.exam_fixture = lineage_fixtures.SignedExamLineageTests(
            "test_exact_scored_exam_candidate_transition_and_nonce_verified")
        self.exam_fixture.setUp()
        self.addCleanup(self.exam_fixture.doCleanups)
        signed = self.exam_fixture.transition
        options = self.exam_fixture.options
        self.exam_custody = ExamLineageCustody(
            transition=signed,
            companion=sign_exam_lineage_binding(**options),
            scored_receipt_bytes=options["scored_receipt_bytes"],
            scored_signature_hex=options["scored_signature_hex"],
            candidate_lineage_bytes=options["candidate_lineage_bytes"],
            candidate_lineage_signature_hex=options["candidate_lineage_signature_hex"],
            candidate_run_id=options["candidate_run_id"],
            candidate_artifact_sha256=options["candidate_artifact_sha256"],
            expected_evaluator_roster_sha256=options["expected_evaluator_roster_sha256"],
            expected_development_roster_sha256=options["expected_development_roster_sha256"],
            failure_threshold_percent=options["failure_threshold_percent"],
            binding_signer_id=options["binding_signer_id"],
            expected_nonce_sha256=options["expected_nonce_sha256"],
            independent_evaluator_key=options["independent_evaluator_key"],
            independent_lineage_key=options["independent_lineage_key"],
            independent_binding_key=options["independent_binding_key"],
        )
        self.journal_path = root / "run-events.sqlite"
        initialize_development_event_journal(
            self.journal_path, state_key=self.JOURNAL_KEY,
        )
        self.scope = options["scope"]
        self.resume = options["context"]
        self.public_roster = signed.development_public_roster
        self.signed_fields = json.loads(signed.receipt_bytes)
        model_dir, predecessor = root / "local-model", root / "prior-run"
        model_dir.mkdir()
        predecessor.mkdir()
        self.config = PilotConfig(model_path=str(model_dir),
                                  output_dir=str(root / "fresh-output"))
        self.predecessor_dir = predecessor
        self.resume_source = ResumeSource(
            model_path=model_dir, output_dir=root / "fresh-output",
            base_weights=(model_dir / "model.safetensors",),
            base_manifest={"weights": {"model.safetensors": "1" * 64},
                           "metadata": {"config.json": "2" * 64}},
            predecessor_run=predecessor,
            predecessor_adapter=predecessor / "candidate_adapter",
            predecessor_adapter_manifest={
                "adapter_model.safetensors": options["candidate_artifact_sha256"]},
            predecessor_lora_value_sha256=self.resume.predecessor_lora_value_sha256,
            predecessor_frozen_sha256="3" * 64,
            predecessor_buffers_sha256="4" * 64,
            predecessor_report_sha256=self.resume.predecessor_report_sha256,
            predecessor_scope_status="MATCHED_SCOPED_PREDECESSOR_REPORT",
            same_objective_predecessor=True, predecessor_cycle_index=0,
        )
        self.preflight_patch = patch(
            "labyrinth.neural_run_binding.preflight_resume",
            return_value=self.resume_source,
        )
        self.preflight_patch.start()
        self.addCleanup(self.preflight_patch.stop)
        self.pilot_identity = _pilot_identity(
            self.config, predecessor, self.resume, {
                "candidate_artifact_sha256": options["candidate_artifact_sha256"]})
        neural = FullCheckpointContext(
            run_id="physical-neural-run-1",
            objective_id=self.scope.objective_id,
            base_checkpoint_sha256=self.BASE,
            prior_checkpoint_sha256=self.PRIOR_FULL,
            fence=0, cursor=1,
            exposures=(("occurrence-1", "2" * 64),),
        )
        self.binding = FencedCheckpointBinding(
            event_id=self.signed_fields["event_id"], run_id=neural.run_id,
            run_commitment_sha256="b" * 64,
            checkpoint_dir=root / "immutable-full-checkpoint",
            checkpoint_sha256=self.CHECKPOINT,
            checkpoint_context=neural, journal_fence=0,
        )
        self.raw, self.signature = signed.receipt_bytes, signed.signature_hex
        self.binding = replace(
            self.binding,
            run_commitment_sha256=signed_neural_run_commitment(
                self.scope, self.resume, self.raw, self.signature,
                self.binding, config=self.config,
                predecessor_run_dir=self.predecessor_dir,
                exam_custody=self.exam_custody,
                independent_development_key=self.SIGN_KEY,
            ),
        )

    def signed(self, changes=None):
        raw = json.dumps(
            {**self.signed_fields, **(changes or {})},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        signature = hmac.new(self.SIGN_KEY, SIGNED_DOMAIN + raw, sha256).hexdigest()
        return raw, signature

    def validate(self, **changes):
        return authenticate_signed_neural_run(
            self.scope, self.resume, changes.pop("raw", self.raw),
            changes.pop("signature", self.signature),
            changes.pop("binding", self.binding),
            config=changes.pop("config", self.config),
            predecessor_run_dir=changes.pop("predecessor_run_dir", self.predecessor_dir),
            exam_custody=changes.pop("exam_custody", self.exam_custody),
            independent_development_key=self.SIGN_KEY,
            pinned_run_id=changes.pop("pinned_run_id", self.binding.run_id),
            pinned_pretrained_base_sha256=changes.pop("pinned_base", self.BASE),
            pinned_prior_full_state_sha256=changes.pop("pinned_prior", self.PRIOR_FULL),
            pinned_checkpoint_sha256=changes.pop("pinned_checkpoint", self.CHECKPOINT),
            pinned_pilot_config_sha256=changes.pop(
                "pinned_pilot_config_sha256", self.pilot_identity.config_sha256),
            pinned_model_manifest_sha256=changes.pop(
                "pinned_model_manifest_sha256", self.pilot_identity.model_manifest_sha256),
            pinned_predecessor_manifest_sha256=changes.pop(
                "pinned_predecessor_manifest_sha256",
                self.pilot_identity.predecessor_manifest_sha256),
            **changes,
        )

    def state(self):
        return inspect_development_event(
            self.journal_path, self.binding.event_id, state_key=self.JOURNAL_KEY,
        )

    def test_exact_signed_bytes_and_neural_pins_validate_without_host_effect(self):
        verified = self.validate()
        self.assertEqual(verified.event_id, self.binding.event_id)
        self.assertEqual(verified.signed_receipt_sha256, sha256(self.raw).hexdigest())
        self.assertEqual(verified.run_commitment_sha256, self.binding.run_commitment_sha256)
        self.assertEqual(verified.pretrained_base_sha256, self.BASE)
        self.assertEqual(verified.model_manifest_sha256,
                         self.pilot_identity.model_manifest_sha256)
        self.assertEqual(verified.scored_exam_lineage_sha256,
                         sha256(self.exam_custody.companion.binding_bytes).hexdigest())
        self.assertFalse(verified.graduation_authorized)
        self.assertFalse(self.state()["consumed"])

    def test_signed_exam_swap_with_same_event_id_rejected_before_reservation(self):
        # Both receipts authenticate under the trusted signer, but the new
        # exam's exact bytes cannot reuse another signed run commitment.
        changed, signature = self.signed({"exam_id": "different-exam-id"})
        with self.assertRaisesRegex(NeuralRunBindingError, "exact scored-exam companion"):
            self.validate(raw=changed, signature=signature)
        self.assertFalse(self.state()["consumed"])

    def test_forged_or_tampered_hmac_rejected(self):
        with self.assertRaisesRegex(ValueError, "signature not authenticated"):
            self.validate(raw=self.raw + b" ")
        self.assertFalse(self.state()["consumed"])

    def test_mismatched_run_scope_base_prior_checkpoint_or_event_rejected(self):
        wrong = (
            replace(self.binding, event_id="different-event"),
            replace(self.binding, run_id="another-neural-run"),
            replace(self.binding, checkpoint_context=replace(
                self.binding.checkpoint_context, objective_id="other-objective")),
            replace(self.binding, checkpoint_context=replace(
                self.binding.checkpoint_context, base_checkpoint_sha256="3" * 64)),
            replace(self.binding, checkpoint_context=replace(
                self.binding.checkpoint_context, prior_checkpoint_sha256="4" * 64)),
            replace(self.binding, checkpoint_sha256="5" * 64),
        )
        for binding in wrong:
            with self.subTest(binding=binding):
                with self.assertRaises((ValueError, NeuralRunBindingError,
                                        RecoveryCompositionError)):
                    self.validate(binding=binding)
        for changed in (
            {"pinned_run_id": "other-run"},
            {"pinned_base": "6" * 64},
            {"pinned_prior": "7" * 64},
            {"pinned_checkpoint": "8" * 64},
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(NeuralRunBindingError, "pins differ"):
                    self.validate(**changed)
        self.assertFalse(self.state()["consumed"])

    def test_stale_or_nonzero_fence_rejected(self):
        for changed in (
            replace(self.binding, journal_fence=1),
            replace(self.binding, checkpoint_context=replace(
                self.binding.checkpoint_context, fence=1), journal_fence=1),
        ):
            with self.subTest(binding=changed):
                with self.assertRaisesRegex(NeuralRunBindingError, "pins differ"):
                    self.validate(binding=changed)

    def test_model_output_predecessor_and_training_parameter_swaps_refused(self):
        # Before the v2 fix both distinct model paths were accepted by the
        # same signed receipt, run commitment and checkpoint identity.
        other_model = self.predecessor_dir.parent / "different-base-model"
        other_prior = self.predecessor_dir.parent / "different-prior-run"
        other_model.mkdir()
        other_prior.mkdir()
        cases = (
            {"config": replace(self.config, model_path=str(other_model))},
            {"config": replace(self.config, output_dir=str(
                self.predecessor_dir.parent / "different-output"))},
            {"config": replace(self.config, learning_rate=0.0005)},
            {"predecessor_run_dir": other_prior},
            {"pinned_pilot_config_sha256": "9" * 64},
            {"pinned_model_manifest_sha256": "8" * 64},
            {"pinned_predecessor_manifest_sha256": "7" * 64},
        )
        for changed in cases:
            with self.subTest(changed=list(changed)), self.assertRaises(
                    (NeuralRunBindingError, ValueError)):
                self.validate(**changed)
        self.assertFalse(self.state()["consumed"])

    def test_companion_and_scored_candidate_are_required_before_any_reservation(self):
        for changed in (
            None,
            replace(self.exam_custody,
                    companion=replace(self.exam_custody.companion,
                                      signature_hex="0" * 64)),
            replace(self.exam_custody, candidate_run_id="different-run"),
            replace(self.exam_custody, candidate_artifact_sha256="a" * 64),
        ):
            with self.subTest(companion=changed), self.assertRaises(ValueError):
                self.validate(exam_custody=changed)
        self.assertFalse(self.state()["consumed"])

    def test_distinct_authenticated_scored_exam_cannot_reuse_run_commitment(self):
        other_raw, other_sig, other_transition = (
            self.exam_fixture.different_authenticated_scored_exam())
        self.assertEqual(other_transition.receipt_bytes, self.raw)
        other_opts = {**self.exam_fixture.options,
                      "transition": other_transition,
                      "scored_receipt_bytes": other_raw,
                      "scored_signature_hex": other_sig}
        other_custody = replace(
            self.exam_custody, transition=other_transition,
            companion=sign_exam_lineage_binding(**other_opts),
            scored_receipt_bytes=other_raw,
            scored_signature_hex=other_sig,
        )
        new_commitment = signed_neural_run_commitment(
            self.scope, self.resume, self.raw, self.signature, self.binding,
            config=self.config, predecessor_run_dir=self.predecessor_dir,
            exam_custody=other_custody,
            independent_development_key=self.SIGN_KEY,
        )
        self.assertNotEqual(new_commitment, self.binding.run_commitment_sha256)
        with self.assertRaisesRegex(NeuralRunBindingError, "commitment differs"):
            self.validate(exam_custody=other_custody)
        self.assertFalse(self.state()["consumed"])

    def test_wrapper_rejects_swapped_config_before_journal_or_runner(self):
        with patch("labyrinth.neural_run_binding.run_journey_bound_successor") as runner:
            with self.assertRaises(NeuralRunBindingError):
                run_bound_neural_successor(
                    replace(self.config, output_dir=str(
                        self.predecessor_dir.parent / "other-output")),
                    self.scope, self.predecessor_dir, self.resume,
                    self.raw, self.signature,
                    journey=object(),
                    evaluator_precommitment_sha256=self.signed_fields[
                        "exam_case_commitment_sha256"],
                    evaluator_public_roster=self.public_roster,
                    independent_development_key=self.SIGN_KEY,
                    binding=self.binding, exam_custody=self.exam_custody,
                    pinned_run_id=self.binding.run_id,
                    pinned_pretrained_base_sha256=self.BASE,
                    pinned_prior_full_state_sha256=self.PRIOR_FULL,
                    pinned_checkpoint_sha256=self.CHECKPOINT,
                    pinned_pilot_config_sha256=self.pilot_identity.config_sha256,
                    pinned_model_manifest_sha256=self.pilot_identity.model_manifest_sha256,
                    pinned_predecessor_manifest_sha256=self.pilot_identity.predecessor_manifest_sha256,
                    journal_path=self.journal_path, journal_state_key=self.JOURNAL_KEY,
                    checkpoint_loader=lambda *_: None,
                )
        runner.assert_not_called()
        self.assertFalse(self.state()["consumed"])

    def test_wrapper_owns_same_signed_bytes_and_consumes_exact_neural_run(self):
        calls = []

        def strict_receipt(path, digest, context):
            calls.append((path, digest, context))
            return FullCheckpointReceipt(digest, context, ("adapter.weight",))

        def observed_journey_runner(*args, **kwargs):
            self.assertEqual(args[4], self.raw)
            self.assertEqual(args[5], self.signature)
            self.assertTrue(kwargs["reserve_development_event"](self.binding.event_id))
            return {"executed": True, "graduation_authorized": False}

        with patch("labyrinth.neural_run_binding.run_journey_bound_successor",
                   side_effect=observed_journey_runner) as mocked:
            result = run_bound_neural_successor(
                self.config, self.scope, self.predecessor_dir, self.resume,
                self.raw, self.signature,
                journey=object(),
                evaluator_precommitment_sha256=self.signed_fields[
                    "exam_case_commitment_sha256"],
                evaluator_public_roster=self.public_roster,
                independent_development_key=self.SIGN_KEY,
                binding=self.binding,
                exam_custody=self.exam_custody,
                pinned_run_id=self.binding.run_id,
                pinned_pretrained_base_sha256=self.BASE,
                pinned_prior_full_state_sha256=self.PRIOR_FULL,
                pinned_checkpoint_sha256=self.CHECKPOINT,
                pinned_pilot_config_sha256=self.pilot_identity.config_sha256,
                pinned_model_manifest_sha256=self.pilot_identity.model_manifest_sha256,
                pinned_predecessor_manifest_sha256=self.pilot_identity.predecessor_manifest_sha256,
                journal_path=self.journal_path,
                journal_state_key=self.JOURNAL_KEY,
                checkpoint_loader=strict_receipt,
            )
        self.assertEqual(result, {"executed": True, "graduation_authorized": False})
        mocked.assert_called_once()
        self.assertEqual(calls, [(
            self.binding.checkpoint_dir, self.CHECKPOINT,
            self.binding.checkpoint_context,
        )])
        state = self.state()
        self.assertTrue(state["consumed"])
        self.assertEqual(state["run_id"], self.binding.run_id)
        self.assertEqual(state["run_commitment_sha256"], self.binding.run_commitment_sha256)
        self.assertEqual(state["checkpoint_sha256"], self.CHECKPOINT)
        self.assertEqual(state["fence"], 0)

    def test_invalid_matching_preflight_cannot_consume_event(self):
        with patch("labyrinth.neural_run_binding.run_journey_bound_successor") as runner:
            with self.assertRaisesRegex(NeuralRunBindingError, "pins differ"):
                run_bound_neural_successor(
                    self.config, self.scope, self.predecessor_dir, self.resume,
                    self.raw, self.signature,
                    journey=object(),
                    evaluator_precommitment_sha256=self.signed_fields[
                        "exam_case_commitment_sha256"],
                    evaluator_public_roster=self.public_roster,
                    independent_development_key=self.SIGN_KEY,
                    binding=self.binding,
                    exam_custody=self.exam_custody,
                    pinned_run_id=self.binding.run_id,
                    pinned_pretrained_base_sha256="6" * 64,
                    pinned_prior_full_state_sha256=self.PRIOR_FULL,
                    pinned_checkpoint_sha256=self.CHECKPOINT,
                    pinned_pilot_config_sha256=self.pilot_identity.config_sha256,
                    pinned_model_manifest_sha256=self.pilot_identity.model_manifest_sha256,
                    pinned_predecessor_manifest_sha256=self.pilot_identity.predecessor_manifest_sha256,
                    journal_path=self.journal_path,
                    journal_state_key=self.JOURNAL_KEY,
                    checkpoint_loader=lambda *_: None,
                )
        runner.assert_not_called()
        self.assertFalse(self.state()["consumed"])


if __name__ == "__main__":
    import unittest
    unittest.main()
