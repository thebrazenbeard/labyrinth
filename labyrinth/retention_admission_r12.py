"""Strict arithmetic retention **development** admission for an evaluator.

Unlike the legacy admission path, this callable makes the restrictive
arithmetic semantic-roster test mandatory *within this entrypoint*, before a
signed development result can be returned as PASS.  The policy, roster,
semantic inventory, evaluator identity and taxonomy expectations must come
from an independently protected operator precommit, not the learner request.

No code in this module calls graduation_state.decide_campaign or
journey_qualification.decide_in_evaluator_custody.  A development PASS never
consumes a private exam attempt, authorizes its release, authenticates model
execution, or graduates a learner.  Legacy APIs remain callable; deployment
must replace their admission routing to make this requirement universal.

The grammar supports only single, nonnegative arithmetic addition.  It is
not a general semantic independence classifier, and synthetic HMAC keys do
not establish independent examiner custody.
"""

from __future__ import annotations

from dataclasses import dataclass

from .journey_qualification import JourneyQualificationPrecommit, PublicQualificationRequest
from .retention_entrypoint_guard import RetentionAdmission, evaluate_retention_admission
from .retention_family_registry import SignedFamilyTaxonomy
from .retention_qualification import (
    RetentionDecision, RetentionPair, RetentionPolicy, SignedRetentionEvidence,
)
from .strict_retention_development import (
    StrictRetentionDevelopmentResult, evaluate_strict_arithmetic_retention,
)


@dataclass(frozen=True)
class StrictRetentionAdmissionResult:
    status: str  # PASS is only a development-retention veto result.
    reason: str
    retention: RetentionDecision | None
    strict_evidence: StrictRetentionDevelopmentResult | None
    graduation_authorized: bool = False
    private_exam_authorized: bool = False
    model_execution_authenticated: bool = False
    examiner_custody_verified: bool = False
    graduation_state_transitioned: bool = False


def _blocked(reason: str) -> StrictRetentionAdmissionResult:
    return StrictRetentionAdmissionResult("BLOCKED", reason, None, None)


def evaluate_strict_retention_admission(
    request: PublicQualificationRequest,
    policy: RetentionPolicy,
    roster: tuple[RetentionPair, ...],
    evidence: SignedRetentionEvidence,
    signed_taxonomy: SignedFamilyTaxonomy,
    *,
    trusted_journey_precommit: JourneyQualificationPrecommit,
    trusted_prior_checkpoint_sha256: str,
    expected_policy_sha256: str,
    expected_roster_sha256: str,
    expected_semantic_inventory_sha256: str,
    expected_taxonomy_signer_id: str,
    expected_taxonomy_nonce_sha256: str,
    independent_taxonomy_key: bytes,
    independent_evaluator_key: bytes,
) -> StrictRetentionAdmissionResult:
    """Fail closed unless strict semantic, signed and journey gates all pass.

    A PASS allows *only* acknowledging a public-development retention veto.
    It must not be fed to `graduation_state.decide_campaign` as a private exam
    receipt.  The independent evaluator, frozen precommit, inference execution
    and actually separate private examination remain external requirements.
    """
    if (type(request) is not PublicQualificationRequest
            or type(policy) is not RetentionPolicy
            or type(roster) is not tuple
            or type(evidence) is not SignedRetentionEvidence
            or type(signed_taxonomy) is not SignedFamilyTaxonomy
            or type(trusted_journey_precommit) is not JourneyQualificationPrecommit):
        return _blocked("strict typed retention and evaluator precommit required")

    # Trusting request-derived expected values would defeat the precommit.
    # Revalidation matters even for frozen dataclasses: object.__setattr__ can
    # mutate them after construction.
    try:
        PublicQualificationRequest.__post_init__(request)
        JourneyQualificationPrecommit.__post_init__(trusted_journey_precommit)
        RetentionPolicy.__post_init__(policy)
    except (ValueError, TypeError, AttributeError, OverflowError):
        return _blocked("invalid request or operator-bound retention identity")

    if (request.scope_fingerprint != trusted_journey_precommit.scope_fingerprint
            or request.development_exam_id != trusted_journey_precommit.development_exam_id
            or request.entrant_checkpoint_sha256 != trusted_journey_precommit.entrant_checkpoint_sha256
            or request.development_cycle_index != trusted_journey_precommit.development_cycle_index
            or request.objective_id != policy.objective_id
            or request.scope_fingerprint != policy.scope_sha256
            or request.candidate_checkpoint_sha256 != policy.after_checkpoint_sha256
            or type(trusted_prior_checkpoint_sha256) is not str
            or trusted_prior_checkpoint_sha256 != policy.before_checkpoint_sha256):
        return _blocked("retention policy, candidate or journey precommit mismatch")

    try:
        strict = evaluate_strict_arithmetic_retention(
            policy, roster, evidence, signed_taxonomy,
            expected_policy_sha256=expected_policy_sha256,
            expected_roster_sha256=expected_roster_sha256,
            expected_semantic_inventory_sha256=expected_semantic_inventory_sha256,
            expected_signer_id=expected_taxonomy_signer_id,
            expected_nonce_sha256=expected_taxonomy_nonce_sha256,
            independent_taxonomy_key=independent_taxonomy_key,
            independent_evaluator_key=independent_evaluator_key,
        )
    except (ValueError, TypeError, AttributeError, OverflowError):
        return _blocked("semantic roster, signed custody or precommit invalid")

    if (type(strict) is not StrictRetentionDevelopmentResult
            or type(strict.development_decision) is not RetentionDecision
            or strict.development_decision.disposition not in {"PASS", "HOLD"}
            or strict.graduation_authorized is not False
            or strict.model_execution_authenticated is not False
            or strict.examiner_custody_verified is not False
            or strict.independent_semantics_authenticated is not False):
        return _blocked("strict evidence exceeded development-only authority")

    try:
        admission: RetentionAdmission = evaluate_retention_admission(
            request, policy, roster, evidence, signed_taxonomy,
            trusted_journey_precommit=trusted_journey_precommit,
            trusted_prior_checkpoint_sha256=trusted_prior_checkpoint_sha256,
            expected_taxonomy_signer_id=expected_taxonomy_signer_id,
            expected_taxonomy_nonce_sha256=expected_taxonomy_nonce_sha256,
            independent_taxonomy_key=independent_taxonomy_key,
            independent_evaluator_key=independent_evaluator_key,
        )
    except (ValueError, TypeError, AttributeError, OverflowError):
        return _blocked("journey-bound retention admission rejected")

    if (type(admission) is not RetentionAdmission
            or admission.status not in {"PASS", "HOLD"}
            or type(admission.retention) is not RetentionDecision
            or admission.retention != strict.development_decision
            or admission.status != strict.development_decision.disposition
            or admission.graduation_authorized is not False
            or admission.model_execution_authenticated is not False):
        return _blocked("journey admission differed from strict signed result")

    reason = ("strict signed arithmetic development retention passed; "
              "private exam still separately required" if admission.status == "PASS"
              else "prior-skill retention deterioration veto")
    return StrictRetentionAdmissionResult(admission.status, reason,
                                          admission.retention, strict)
