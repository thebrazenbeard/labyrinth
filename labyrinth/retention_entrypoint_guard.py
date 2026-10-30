"""Opt-in evaluator-side retention admission before private qualification.

This is a development-only *veto* seam.  It binds the evaluator's independently
precommitted journey and prior-skill checkpoint identities to a signed retention
receipt and a distinct signed family taxonomy.  A PASS is permission to *seek*
the separately governed private exam, never proof of model execution or a
graduation credential.

An operator must explicitly route all evaluator admission through this seam;
the older ``decide_retention`` and ``decide_in_evaluator_custody`` functions
remain directly callable and are not made unbypassable by this module.  No
external custodian, key vault, or durable state store is installed here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .journey_qualification import (
    JourneyQualificationPrecommit, PublicQualificationRequest,
)
from .retention_family_registry import (
    FamilyTaxonomyError, SignedFamilyTaxonomy,
    decide_retention_with_family_custody,
)
from .retention_qualification import (
    RetentionDecision, RetentionError, RetentionPair, RetentionPolicy,
    SignedRetentionEvidence,
)


@dataclass(frozen=True)
class RetentionAdmission:
    status: str  # PASS, HOLD, BLOCKED.  None means no verified retention result.
    reason: str
    retention: RetentionDecision | None
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False


def evaluate_retention_admission(
    request: PublicQualificationRequest,
    policy: RetentionPolicy,
    roster: tuple[RetentionPair, ...],
    evidence: SignedRetentionEvidence,
    signed_taxonomy: SignedFamilyTaxonomy,
    *,
    trusted_journey_precommit: JourneyQualificationPrecommit,
    trusted_prior_checkpoint_sha256: str,
    expected_taxonomy_signer_id: str,
    expected_taxonomy_nonce_sha256: str,
    independent_taxonomy_key: bytes,
    independent_evaluator_key: bytes,
) -> RetentionAdmission:
    """Fail closed on absent/mismatched precommit and distinct signed custody.

    The independent evaluator must freeze these expectations separately from
    the trainee request; copying them from the request is NOT independence.
    This does not call the neural learner or authorize ``decide_graduation``.
    """
    if (type(request) is not PublicQualificationRequest
            or type(policy) is not RetentionPolicy
            or type(trusted_journey_precommit) is not JourneyQualificationPrecommit):
        return RetentionAdmission("BLOCKED", "missing typed evaluator precommit", None)
    try:
        request.__post_init__()
        policy.__post_init__()
        trusted_journey_precommit.__post_init__()
    except (TypeError, ValueError):
        return RetentionAdmission("BLOCKED", "invalid evaluator-bound identities", None)

    if (request.scope_fingerprint != trusted_journey_precommit.scope_fingerprint
            or request.development_exam_id != trusted_journey_precommit.development_exam_id
            or request.entrant_checkpoint_sha256 != trusted_journey_precommit.entrant_checkpoint_sha256
            or request.development_cycle_index != trusted_journey_precommit.development_cycle_index
            or request.objective_id != policy.objective_id
            or request.scope_fingerprint != policy.scope_sha256
            or request.candidate_checkpoint_sha256 != policy.after_checkpoint_sha256
            or type(trusted_prior_checkpoint_sha256) is not str
            or trusted_prior_checkpoint_sha256 != policy.before_checkpoint_sha256):
        return RetentionAdmission("BLOCKED", "unbound journey or prior-skill checkpoint", None)

    try:
        decision = decide_retention_with_family_custody(
            policy, roster, evidence, signed_taxonomy,
            expected_signer_id=expected_taxonomy_signer_id,
            expected_nonce_sha256=expected_taxonomy_nonce_sha256,
            independent_taxonomy_key=independent_taxonomy_key,
            independent_evaluator_key=independent_evaluator_key,
        )
    except (RetentionError, FamilyTaxonomyError, TypeError, ValueError):
        return RetentionAdmission("BLOCKED", "retention or taxonomy custody invalid", None)

    if decision.disposition != "PASS":
        return RetentionAdmission("HOLD", "prior-skill retention veto", decision)
    return RetentionAdmission("PASS", "signed development retention passed; private exam still required", decision)
