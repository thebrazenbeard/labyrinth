"""Source-level binding for signed development FAIL and a neural checkpoint run.

The legacy HF successor API passes only an event ID to a reservation callback.
Two otherwise authentic transitions can share that ID but name different
predecessor models or objectives. This module owns both the signed-transition
invocation and its run-bound callback, and requires a deterministic commitment
to the *exact signed receipt bytes*, resume lineage and full-state checkpoint.

No HMAC proves that a model really trained, a host used the strict loader, or
that a MAC-sealed journal cannot be rolled back. Host key custody, a pinned
frozen-base measurement and worker-fence enforcement remain separate gates.
This module cannot authorize graduation or independently attest inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
from pathlib import Path
from typing import Any

from .hf_development_transition import (
    DevelopmentCaseRef, VerifiedDevelopmentTransition,
    run_journey_bound_successor, verify_development_transition,
)
from .hf_pilot import PilotConfig
from .hf_resume_bridge import ResumeContext, ResumeSource, preflight_resume
from .blind_transition_bridge import SignedRegenerationTransition
from .signed_exam_lineage import (
    SignedExamLineageBinding, verify_exam_lineage_binding,
)
from .journey import JourneyState
from .recovery_composition import (
    FencedCheckpointBinding, TrustedCheckpointLoader,
    reservation_callback,
)
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.signed-neural-run-binding.v2"
_DOMAIN = b"labyrinth.signed-neural-run-binding.v2\0"


class NeuralRunBindingError(ValueError):
    """Signed objective and checkpoint cannot share one reserved neural run."""


@dataclass(frozen=True)
class VerifiedNeuralRunBinding:
    event_id: str
    exam_id: str
    signed_receipt_sha256: str
    run_commitment_sha256: str
    checkpoint_sha256: str
    pretrained_base_sha256: str
    prior_full_state_sha256: str
    scored_exam_lineage_sha256: str
    pilot_config_sha256: str
    model_manifest_sha256: str
    predecessor_manifest_sha256: str
    graduation_authorized: bool = False


@dataclass(frozen=True)
class ExamLineageCustody:
    """Actual scored-exam evidence, not a substitute for host key custody."""

    transition: SignedRegenerationTransition
    companion: SignedExamLineageBinding
    scored_receipt_bytes: bytes
    scored_signature_hex: str
    candidate_lineage_bytes: bytes
    candidate_lineage_signature_hex: str
    candidate_run_id: str
    candidate_artifact_sha256: str
    expected_evaluator_roster_sha256: str
    expected_development_roster_sha256: str
    failure_threshold_percent: int
    binding_signer_id: str
    expected_nonce_sha256: str
    independent_evaluator_key: bytes
    independent_lineage_key: bytes
    independent_binding_key: bytes


@dataclass(frozen=True)
class PilotByteIdentity:
    """Read-only actual preflight identity requiring separate trusted pins."""

    config_sha256: str
    model_manifest_sha256: str
    predecessor_manifest_sha256: str
    model_path: str
    output_path: str
    predecessor_path: str


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _digest(value: dict[str, Any]) -> str:
    return sha256(_canonical(value)).hexdigest()


def _exam_identity(
    scope: TrainingScopeSpec, context: ResumeContext,
    signed_receipt_bytes: bytes, signed_signature_hex: str,
    exam: ExamLineageCustody, *, independent_development_key: bytes,
) -> dict[str, Any]:
    if (type(exam) is not ExamLineageCustody
            or type(exam.transition) is not SignedRegenerationTransition
            or exam.transition.receipt_bytes != signed_receipt_bytes
            or exam.transition.signature_hex != signed_signature_hex):
        raise NeuralRunBindingError("exact scored-exam companion required")
    return verify_exam_lineage_binding(
        exam.companion,
        scope=scope, context=context, transition=exam.transition,
        scored_receipt_bytes=exam.scored_receipt_bytes,
        scored_signature_hex=exam.scored_signature_hex,
        candidate_lineage_bytes=exam.candidate_lineage_bytes,
        candidate_lineage_signature_hex=exam.candidate_lineage_signature_hex,
        candidate_run_id=exam.candidate_run_id,
        candidate_artifact_sha256=exam.candidate_artifact_sha256,
        expected_evaluator_roster_sha256=exam.expected_evaluator_roster_sha256,
        expected_development_roster_sha256=exam.expected_development_roster_sha256,
        failure_threshold_percent=exam.failure_threshold_percent,
        binding_signer_id=exam.binding_signer_id,
        expected_nonce_sha256=exam.expected_nonce_sha256,
        independent_evaluator_key=exam.independent_evaluator_key,
        independent_lineage_key=exam.independent_lineage_key,
        independent_transition_key=independent_development_key,
        independent_binding_key=exam.independent_binding_key,
    )


def _pilot_identity(
    config: PilotConfig, predecessor_run_dir: str | Path,
    context: ResumeContext, scored: dict[str, Any],
) -> PilotByteIdentity:
    """Measure pre-existing HF source bytes; caller must pin them separately.

    `preflight_resume` validates actual base/adapter/report file hashes and
    output isolation; a model loader is not invoked or attested here.
    """
    source = preflight_resume(config, predecessor_run_dir, context)
    if type(source) is not ResumeSource:
        raise NeuralRunBindingError("strict HF predecessor preflight required")
    # Do not trust an injected or stale preflight result naming other paths.
    # The real preflight also checks symlinks and data bytes at those paths.
    if (Path(config.model_path).expanduser().resolve(strict=True) != source.model_path
            or Path(config.output_dir).expanduser().resolve(strict=False) != source.output_dir
            or Path(predecessor_run_dir).expanduser().resolve(strict=True)
            != source.predecessor_run):
        raise NeuralRunBindingError("HF pilot/predecessor paths differ from preflight")
    candidate_sha = source.predecessor_adapter_manifest.get("adapter_model.safetensors")
    if (candidate_sha != scored["candidate_artifact_sha256"]
            or source.predecessor_report_sha256 != context.predecessor_report_sha256
            or source.predecessor_lora_value_sha256 != context.predecessor_lora_value_sha256):
        raise NeuralRunBindingError("scored candidate differs from pretrained predecessor bytes")
    model = str(source.model_path.resolve(strict=True))
    predecessor = str(source.predecessor_run.resolve(strict=True))
    output = str(source.output_dir.resolve(strict=False))
    config_fields = asdict(config)
    config_fields.update({"model_path": model, "output_dir": output})
    return PilotByteIdentity(
        _digest({"schema": SCHEMA, "config": config_fields}),
        _digest({"schema": SCHEMA, "base": source.base_manifest}),
        _digest({"schema": SCHEMA,
                 "adapter": source.predecessor_adapter_manifest,
                 "report_sha256": source.predecessor_report_sha256,
                 "lora_sha256": source.predecessor_lora_value_sha256}),
        model, output, predecessor,
    )


def _payload(
    scope: TrainingScopeSpec, context: ResumeContext,
    verified: VerifiedDevelopmentTransition,
    receipt_bytes: bytes, signature_hex: str,
    binding: FencedCheckpointBinding,
    exam: ExamLineageCustody, pilot: PilotByteIdentity,
) -> dict[str, Any]:
    neural = binding.checkpoint_context
    return {
        "schema": SCHEMA,
        "event_id": verified.event_id,
        "exam_id": verified.exam_id,
        "signed_receipt_sha256": sha256(receipt_bytes).hexdigest(),
        "signed_receipt_signature": signature_hex,
        "scored_exam_lineage_sha256": sha256(exam.companion.binding_bytes).hexdigest(),
        "scored_exam_lineage_signature": exam.companion.signature_hex,
        "candidate_run_id": exam.candidate_run_id,
        "candidate_artifact_sha256": exam.candidate_artifact_sha256,
        "pilot_config_sha256": pilot.config_sha256,
        "runnable_model_manifest_sha256": pilot.model_manifest_sha256,
        "predecessor_manifest_sha256": pilot.predecessor_manifest_sha256,
        "pilot_model_path": pilot.model_path,
        "pilot_output_path": pilot.output_path,
        "pilot_predecessor_path": pilot.predecessor_path,
        "objective_id": scope.objective_id,
        "scope_fingerprint": scope.fingerprint,
        "cycle_index": context.cycle_index,
        "weak_skills": list(verified.weak_skills),
        "exam_case_commitment_sha256": verified.exam_case_commitment_sha256,
        "predecessor_report_sha256": context.predecessor_report_sha256,
        "predecessor_lora_value_sha256": context.predecessor_lora_value_sha256,
        "resume_authority_ref": context.authority_ref,
        "neural_run_id": binding.run_id,
        "neural_checkpoint_sha256": binding.checkpoint_sha256,
        "pretrained_base_sha256": neural.base_checkpoint_sha256,
        "prior_full_state_sha256": neural.prior_checkpoint_sha256,
        "neural_checkpoint_fence": neural.fence,
        "journal_fence": binding.journal_fence,
        "training_cursor": neural.cursor,
        "training_exposures": [list(row) for row in neural.exposures],
        "graduation_authorized": False,
    }


def _commit(payload: dict[str, Any]) -> str:
    return sha256(_DOMAIN + _canonical(payload)).hexdigest()


def authenticate_signed_neural_run(
    scope: TrainingScopeSpec, resume_context: ResumeContext,
    signed_receipt_bytes: bytes, signed_signature_hex: str,
    binding: FencedCheckpointBinding, *, config: PilotConfig,
    predecessor_run_dir: str | Path, exam_custody: ExamLineageCustody,
    independent_development_key: bytes,
    pinned_run_id: str, pinned_pretrained_base_sha256: str,
    pinned_prior_full_state_sha256: str,
    pinned_checkpoint_sha256: str,
    pinned_pilot_config_sha256: str,
    pinned_model_manifest_sha256: str,
    pinned_predecessor_manifest_sha256: str,
) -> VerifiedNeuralRunBinding:
    """Verify exact signed development lineage and independently pinned run bytes.

    The *pinned* values must come from a distinct, trusted operator/host
    manifest. Repeating untrusted binding fields as the pins is not evidence
    of provenance. No journal mutation, loader execution, or inference occurs.
    """
    verified = verify_development_transition(
        scope, resume_context, signed_receipt_bytes, signed_signature_hex,
        independent_development_key=independent_development_key,
    )
    binding.validate()
    scored = _exam_identity(
        scope, resume_context, signed_receipt_bytes, signed_signature_hex,
        exam_custody, independent_development_key=independent_development_key,
    )
    pilot = _pilot_identity(config, predecessor_run_dir, resume_context, scored)
    neural = binding.checkpoint_context
    for name, value in (
        ("run", pinned_run_id),
        ("pretrained base", pinned_pretrained_base_sha256),
        ("prior full state", pinned_prior_full_state_sha256),
        ("initial checkpoint", pinned_checkpoint_sha256),
        ("pilot config", pinned_pilot_config_sha256),
        ("runnable model manifest", pinned_model_manifest_sha256),
        ("predecessor manifest", pinned_predecessor_manifest_sha256),
    ):
        if type(value) is not str or not value:
            raise NeuralRunBindingError(f"independent {name} pin required")
    if (binding.event_id != verified.event_id
            or neural.objective_id != scope.objective_id
            or neural.run_id != binding.run_id
            or neural.run_id != pinned_run_id
            or neural.base_checkpoint_sha256 != pinned_pretrained_base_sha256
            or neural.prior_checkpoint_sha256 != pinned_prior_full_state_sha256
            or binding.checkpoint_sha256 != pinned_checkpoint_sha256
            or pilot.config_sha256 != pinned_pilot_config_sha256
            or pilot.model_manifest_sha256 != pinned_model_manifest_sha256
            or pilot.predecessor_manifest_sha256 != pinned_predecessor_manifest_sha256
            or neural.fence != 0 or binding.journal_fence != 0):
        raise NeuralRunBindingError("signed successor and neural run pins differ")
    payload = _payload(
        scope, resume_context, verified, signed_receipt_bytes,
        signed_signature_hex, binding, exam_custody, pilot,
    )
    calculated = _commit(payload)
    if not hmac.compare_digest(binding.run_commitment_sha256, calculated):
        raise NeuralRunBindingError("neural run commitment differs from exact signed lineage")
    return VerifiedNeuralRunBinding(
        verified.event_id, verified.exam_id,
        sha256(signed_receipt_bytes).hexdigest(), calculated,
        binding.checkpoint_sha256, neural.base_checkpoint_sha256,
        neural.prior_checkpoint_sha256,
        sha256(exam_custody.companion.binding_bytes).hexdigest(),
        pilot.config_sha256, pilot.model_manifest_sha256,
        pilot.predecessor_manifest_sha256,
    )


def signed_neural_run_commitment(
    scope: TrainingScopeSpec, resume_context: ResumeContext,
    signed_receipt_bytes: bytes, signed_signature_hex: str,
    binding: FencedCheckpointBinding, *, config: PilotConfig,
    predecessor_run_dir: str | Path, exam_custody: ExamLineageCustody,
    independent_development_key: bytes,
) -> str:
    """Derive a commitment for pre-registration, not an authorization by itself.

    The caller must *independently* pin the resulting bytes before invoking
    :func:`authenticate_signed_neural_run` with all host-held identities.
    """
    verified = verify_development_transition(
        scope, resume_context, signed_receipt_bytes, signed_signature_hex,
        independent_development_key=independent_development_key,
    )
    binding.validate()
    scored = _exam_identity(
        scope, resume_context, signed_receipt_bytes, signed_signature_hex,
        exam_custody, independent_development_key=independent_development_key,
    )
    pilot = _pilot_identity(config, predecessor_run_dir, resume_context, scored)
    if (binding.event_id != verified.event_id
            or binding.checkpoint_context.objective_id != scope.objective_id
            or binding.checkpoint_context.fence != 0 or binding.journal_fence != 0):
        raise NeuralRunBindingError("cannot commit a mismatched or resumed neural run")
    return _commit(_payload(
        scope, resume_context, verified, signed_receipt_bytes,
        signed_signature_hex, binding, exam_custody, pilot,
    ))


def run_bound_neural_successor(
    config: PilotConfig, scope: TrainingScopeSpec,
    predecessor_run_dir: str | Path, resume_context: ResumeContext,
    signed_receipt_bytes: bytes, signed_signature_hex: str, *,
    journey: JourneyState,
    evaluator_precommitment_sha256: str,
    evaluator_public_roster: tuple[DevelopmentCaseRef, ...],
    independent_development_key: bytes,
    binding: FencedCheckpointBinding,
    exam_custody: ExamLineageCustody,
    pinned_run_id: str, pinned_pretrained_base_sha256: str,
    pinned_prior_full_state_sha256: str,
    pinned_checkpoint_sha256: str,
    pinned_pilot_config_sha256: str,
    pinned_model_manifest_sha256: str,
    pinned_predecessor_manifest_sha256: str,
    journal_path: Path, journal_state_key: bytes,
    checkpoint_loader: TrustedCheckpointLoader,
) -> dict[str, Any]:
    """Own both signed receipt and callback; fail before event consumption.

    A source-only adapter: the legacy successor still owns HF pilot execution.
    The checkpoint loader must be the real strict loader in a host-owned model
    process, and the host must independently enforce fences on every output.
    """
    authenticate_signed_neural_run(
        scope, resume_context, signed_receipt_bytes, signed_signature_hex,
        binding, config=config, predecessor_run_dir=predecessor_run_dir,
        exam_custody=exam_custody,
        independent_development_key=independent_development_key,
        pinned_run_id=pinned_run_id,
        pinned_pretrained_base_sha256=pinned_pretrained_base_sha256,
        pinned_prior_full_state_sha256=pinned_prior_full_state_sha256,
        pinned_checkpoint_sha256=pinned_checkpoint_sha256,
        pinned_pilot_config_sha256=pinned_pilot_config_sha256,
        pinned_model_manifest_sha256=pinned_model_manifest_sha256,
        pinned_predecessor_manifest_sha256=pinned_predecessor_manifest_sha256,
    )
    callback = reservation_callback(
        binding, journal_path=journal_path, state_key=journal_state_key,
        checkpoint_loader=checkpoint_loader,
    )
    return run_journey_bound_successor(
        config, scope, predecessor_run_dir, resume_context,
        signed_receipt_bytes, signed_signature_hex,
        journey=journey,
        evaluator_precommitment_sha256=evaluator_precommitment_sha256,
        evaluator_public_roster=evaluator_public_roster,
        independent_development_key=independent_development_key,
        reserve_development_event=callback,
    )
