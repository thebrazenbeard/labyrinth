"""Companion HMAC binding for a scored development exam and legacy HF FAIL.

The v1 HF development transition intentionally omits scored-candidate identity.
Two different authenticated scorings can therefore yield identical legacy
transition bytes. Consumers that need exact exam provenance must verify this
*additional* binding before relying on a transition; verifying the legacy
signature alone does not establish that provenance.

This module does not change legacy receipts or install a host enforcement hook.
Distinct keys, a previously pinned nonce, evaluator-controlled one-shot state,
and physically authentic model execution require external custody. Neither
signature authorizes graduation, resumes a worker, or proves neural learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
from typing import Any

from .blind_evaluation_custody import BlindCustodyError, verify_assessment, _canonical
from .blind_transition_bridge import (
    SignedRegenerationTransition, _digest, _identifier, _distinct_keys,
    _verify_lineage,
)
from .hf_development_transition import (
    development_case_commitment, verify_development_transition,
)
from .hf_resume_bridge import ResumeContext
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.signed-development-exam-lineage.v1"
_DOMAIN = b"labyrinth.signed-development-exam-lineage.v1\0"


@dataclass(frozen=True)
class SignedExamLineageBinding:
    """Development-only companion receipt; never a graduation credential."""

    binding_bytes: bytes
    signature_hex: str
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False


def _expected_body(
    *, scope: TrainingScopeSpec, context: ResumeContext,
    transition: SignedRegenerationTransition,
    scored_receipt_bytes: bytes, scored_signature_hex: str,
    candidate_lineage_bytes: bytes, candidate_lineage_signature_hex: str,
    candidate_run_id: str, candidate_artifact_sha256: str,
    expected_evaluator_roster_sha256: str,
    expected_development_roster_sha256: str,
    failure_threshold_percent: int,
    binding_signer_id: str, expected_nonce_sha256: str,
    independent_evaluator_key: bytes, independent_lineage_key: bytes,
    independent_transition_key: bytes, independent_binding_key: bytes,
) -> dict[str, Any]:
    _distinct_keys(independent_evaluator_key, independent_lineage_key,
                   independent_transition_key, independent_binding_key)
    _identifier(candidate_run_id, "candidate run")
    _identifier(binding_signer_id, "binding signer")
    _digest(candidate_artifact_sha256, "candidate artifact")
    _digest(expected_evaluator_roster_sha256, "evaluator precommitment")
    _digest(expected_development_roster_sha256, "development precommitment")
    _digest(expected_nonce_sha256, "precommitted binding nonce")
    if (type(failure_threshold_percent) is not int
            or not 1 <= failure_threshold_percent <= 100):
        raise ValueError("precommitted development failure threshold required")
    if (type(transition) is not SignedRegenerationTransition
            or transition.graduation_authorized is not False
            or transition.model_execution_authenticated is not False):
        raise ValueError("untrusted regeneration transition wrapper")
    _digest(transition.scored_receipt_sha256, "scored receipt")
    if (type(scored_receipt_bytes) is not bytes
            or sha256(scored_receipt_bytes).hexdigest() != transition.scored_receipt_sha256):
        raise ValueError("regeneration transition is not linked to this scored exam")

    verified = verify_development_transition(
        scope, context, transition.receipt_bytes, transition.signature_hex,
        independent_development_key=independent_transition_key,
    )
    try:
        scored = verify_assessment(
            scored_receipt_bytes, scored_signature_hex,
            evaluator_key=independent_evaluator_key,
            run_id=candidate_run_id,
            checkpoint_sha256=candidate_artifact_sha256,
        )
    except BlindCustodyError as exc:
        raise ValueError("candidate-scoring receipt did not authenticate") from exc
    if (scored["objective_id"] != scope.objective_id
            or scored["public_exam_sha256"] != expected_evaluator_roster_sha256):
        raise ValueError("scored exam differs from evaluator's independently pinned identity")
    groups = {(group["split"], group["skill"]): group
              for group in scored["groups"]}
    if len(groups) != len(scored["groups"]):
        raise ValueError("duplicate scored development group")
    weaknesses = tuple(skill for skill in scope.skills if any(
        groups[(split, skill)]["correct"] * 100
        < groups[(split, skill)]["total"] * failure_threshold_percent
        for split in ("heldout", "transfer") if (split, skill) in groups
    ))
    if weaknesses != verified.weak_skills:
        raise ValueError("signed transition weaknesses do not derive from this scored exam")
    observed_development = development_case_commitment(
        transition.development_public_roster,
    )
    if (observed_development != expected_development_roster_sha256
            or observed_development != verified.exam_case_commitment_sha256):
        raise ValueError("signed FAIL transition differs from public development roster")
    _verify_lineage(
        candidate_lineage_bytes, candidate_lineage_signature_hex,
        independent_lineage_key, scope=scope, context=context,
        run_id=candidate_run_id, exam_id=verified.exam_id,
        artifact_sha256=candidate_artifact_sha256,
        public_exam_sha256=expected_evaluator_roster_sha256,
    )
    return {
        "schema": SCHEMA,
        "objective_id": scope.objective_id,
        "scope_fingerprint": scope.fingerprint,
        "from_cycle": context.cycle_index - 1,
        "to_cycle": context.cycle_index,
        "exam_id": verified.exam_id,
        "event_id": verified.event_id,
        "candidate_run_id": candidate_run_id,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "scored_receipt_sha256": sha256(scored_receipt_bytes).hexdigest(),
        "scored_signature_hex": scored_signature_hex,
        "candidate_lineage_sha256": sha256(candidate_lineage_bytes).hexdigest(),
        "candidate_lineage_signature_hex": candidate_lineage_signature_hex,
        "legacy_transition_sha256": sha256(transition.receipt_bytes).hexdigest(),
        "legacy_transition_signature_hex": transition.signature_hex,
        "evaluator_roster_sha256": expected_evaluator_roster_sha256,
        "development_roster_sha256": expected_development_roster_sha256,
        "failure_threshold_percent": failure_threshold_percent,
        "binding_signer_id": binding_signer_id,
        "binding_nonce_sha256": expected_nonce_sha256,
        "graduation_authorized": False,
        "model_execution_authenticated": False,
    }


def sign_exam_lineage_binding(
    *, scope: TrainingScopeSpec, context: ResumeContext,
    transition: SignedRegenerationTransition,
    scored_receipt_bytes: bytes, scored_signature_hex: str,
    candidate_lineage_bytes: bytes, candidate_lineage_signature_hex: str,
    candidate_run_id: str, candidate_artifact_sha256: str,
    expected_evaluator_roster_sha256: str,
    expected_development_roster_sha256: str,
    failure_threshold_percent: int,
    binding_signer_id: str, expected_nonce_sha256: str,
    independent_evaluator_key: bytes, independent_lineage_key: bytes,
    independent_transition_key: bytes, independent_binding_key: bytes,
) -> SignedExamLineageBinding:
    """Sign only after all independent legacy/scorer/lineage evidence verifies."""
    body = _expected_body(**locals())
    raw = _canonical(body)
    if len(raw) > 8192:
        raise ValueError("development-exam lineage binding is oversized")
    signature = hmac.new(independent_binding_key, _DOMAIN + raw, sha256).hexdigest()
    return SignedExamLineageBinding(raw, signature)


def verify_exam_lineage_binding(
    binding: SignedExamLineageBinding, *,
    scope: TrainingScopeSpec, context: ResumeContext,
    transition: SignedRegenerationTransition,
    scored_receipt_bytes: bytes, scored_signature_hex: str,
    candidate_lineage_bytes: bytes, candidate_lineage_signature_hex: str,
    candidate_run_id: str, candidate_artifact_sha256: str,
    expected_evaluator_roster_sha256: str,
    expected_development_roster_sha256: str,
    failure_threshold_percent: int,
    binding_signer_id: str, expected_nonce_sha256: str,
    independent_evaluator_key: bytes, independent_lineage_key: bytes,
    independent_transition_key: bytes, independent_binding_key: bytes,
) -> dict[str, Any]:
    """Fail closed unless this exact scored exam produced this signed FAIL.

    A host MUST call this verifier before its legacy successor reservation
    to obtain the provenance guarantee. The older HF entry point by itself
    does not know about this companion receipt and is still unqualified.
    """
    if (type(binding) is not SignedExamLineageBinding
            or binding.graduation_authorized is not False
            or binding.model_execution_authenticated is not False
            or type(binding.binding_bytes) is not bytes
            or not 0 < len(binding.binding_bytes) <= 8192):
        raise ValueError("invalid or over-authorized lineage binding")
    _digest(binding.signature_hex, "binding signature")
    expected = hmac.new(independent_binding_key, _DOMAIN + binding.binding_bytes,
                        sha256).hexdigest()
    if not hmac.compare_digest(expected, binding.signature_hex):
        raise ValueError("exam lineage binding signature not authenticated")
    body = _expected_body(
        scope=scope, context=context, transition=transition,
        scored_receipt_bytes=scored_receipt_bytes,
        scored_signature_hex=scored_signature_hex,
        candidate_lineage_bytes=candidate_lineage_bytes,
        candidate_lineage_signature_hex=candidate_lineage_signature_hex,
        candidate_run_id=candidate_run_id,
        candidate_artifact_sha256=candidate_artifact_sha256,
        expected_evaluator_roster_sha256=expected_evaluator_roster_sha256,
        expected_development_roster_sha256=expected_development_roster_sha256,
        failure_threshold_percent=failure_threshold_percent,
        binding_signer_id=binding_signer_id,
        expected_nonce_sha256=expected_nonce_sha256,
        independent_evaluator_key=independent_evaluator_key,
        independent_lineage_key=independent_lineage_key,
        independent_transition_key=independent_transition_key,
        independent_binding_key=independent_binding_key,
    )
    if not hmac.compare_digest(binding.binding_bytes, _canonical(body)):
        raise ValueError("binding does not match exact independently signed exam and transition")
    return body
