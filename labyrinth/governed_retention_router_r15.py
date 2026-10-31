"""Opt-in signed host route for *development-only* arithmetic retention.

This entrypoint deliberately has no legacy fallback. It admits only the
restricted single-addition grammar after the existing strict semantic and
signed-evaluator checks. An operator-provisioned Ed25519 public key binds the
classification, policy, current candidate, roster and declared TRAIN inventory.
The separately supplied host route port must read the same signed record both
before and after evaluation. No private signing key belongs in this module.

The port and public key must actually be controlled outside the learner's
rollback/administrative domain for an operational trust claim. In-process test
ports or signatures do not establish such custody, pre-outcome timing, complete
training inventory, actual model inference, private exams, or graduation.
Legacy APIs remain callable; only a host enforcing this path can make routing
mandatory. A local PASS is not a graduate credential.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hmac
import json
import re
from typing import Callable

from .arithmetic_retention_admission import arithmetic_training_inventory_sha256
from .journey_qualification import JourneyQualificationPrecommit, PublicQualificationRequest
from .retention_admission_r12 import (
    StrictRetentionAdmissionResult, evaluate_strict_retention_admission,
)
from .retention_family_registry import SignedFamilyTaxonomy
from .retention_qualification import (
    RetentionPair, RetentionPolicy, SignedRetentionEvidence, _policy_sha256,
    retention_roster_sha256,
)


SCHEMA = "labyrinth.governed-retention-route.v1"
_DOMAIN = b"labyrinth.governed-retention-route.v1\0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_SIG = re.compile(r"[a-f0-9]{128}\Z", re.ASCII)
_ONLY_CLASS = "single_addition_arithmetic"


@dataclass(frozen=True)
class SignedRetentionRoute:
    """Exact independently pinned route identity, not a learner request."""

    provider_id: str
    route_id: str
    classification: str
    objective_id: str
    scope_sha256: str
    development_exam_id: str
    prior_checkpoint_sha256: str
    candidate_checkpoint_sha256: str
    policy_sha256: str
    roster_sha256: str
    semantic_inventory_sha256: str
    training_inventory_sha256: str
    taxonomy_signer_id: str
    taxonomy_nonce_sha256: str
    host_epoch: int
    signature_hex: str
    schema: str = SCHEMA

    def signed_bytes(self) -> bytes:
        """Domain-separated deterministic payload for an external signer."""
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__ if name != "signature_hex"
        }
        return _DOMAIN + (json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ) + "\n").encode("ascii")

    def validate(self) -> None:
        for field in ("provider_id", "route_id", "objective_id",
                      "development_exam_id", "taxonomy_signer_id"):
            value = getattr(self, field)
            if type(value) is not str or _ID.fullmatch(value) is None:
                raise ValueError(f"invalid signed route {field}")
        for field in ("scope_sha256", "prior_checkpoint_sha256",
                      "candidate_checkpoint_sha256", "policy_sha256",
                      "roster_sha256", "semantic_inventory_sha256",
                      "training_inventory_sha256", "taxonomy_nonce_sha256"):
            value = getattr(self, field)
            if type(value) is not str or _SHA.fullmatch(value) is None:
                raise ValueError(f"invalid signed route {field}")
        if (self.schema != SCHEMA
                or type(self.classification) is not str
                or self.classification != _ONLY_CLASS
                or self.objective_id != "arithmetic"
                or type(self.host_epoch) is not int
                or not 1 <= self.host_epoch < 2**63
                or type(self.signature_hex) is not str
                or _SIG.fullmatch(self.signature_hex) is None):
            raise ValueError("route classification, epoch or signature is invalid")


@dataclass(frozen=True)
class GovernedRetentionResult:
    status: str
    reason: str
    admission: StrictRetentionAdmissionResult | None
    provenance: str = "SIGNED_ROUTE_SOURCE_ONLY"
    graduation_authorized: bool = False
    private_exam_authorized: bool = False
    model_execution_authenticated: bool = False
    examiner_custody_verified: bool = False


def _blocked(reason: str) -> GovernedRetentionResult:
    return GovernedRetentionResult("BLOCKED", reason, None)


def _verified_signature(route: SignedRetentionRoute, public_key: bytes) -> bool:
    if type(public_key) is not bytes or len(public_key) != 32:
        return False
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(route.signature_hex), route.signed_bytes())
        return True
    except (ImportError, InvalidSignature, ValueError, TypeError, UnicodeError):
        return False


def route_governed_retention(
    request: PublicQualificationRequest,
    policy: RetentionPolicy,
    roster: tuple[RetentionPair, ...],
    evidence: SignedRetentionEvidence,
    signed_taxonomy: SignedFamilyTaxonomy,
    *,
    route: SignedRetentionRoute,
    pinned_provider_id: str,
    pinned_ed25519_public_key: bytes,
    read_host_route: Callable[[str], SignedRetentionRoute],
    trusted_journey_precommit: JourneyQualificationPrecommit,
    independent_taxonomy_key: bytes,
    independent_evaluator_key: bytes,
) -> GovernedRetentionResult:
    """Require a verified, live, exact arithmetic route before local PASS.

    Neither the signature nor a user-injected port proves the host is actually
    independent. Integrators must provision and enforce that authority; this
    module never calls private graduation or returns a release permit.
    """
    if (type(route) is not SignedRetentionRoute
            or type(pinned_provider_id) is not str
            or _ID.fullmatch(pinned_provider_id) is None
            or not callable(read_host_route)
            or type(request) is not PublicQualificationRequest
            or type(policy) is not RetentionPolicy
            or type(roster) is not tuple
            or type(evidence) is not SignedRetentionEvidence
            or type(signed_taxonomy) is not SignedFamilyTaxonomy
            or type(trusted_journey_precommit) is not JourneyQualificationPrecommit):
        return _blocked("typed host route and independent evaluator inputs required")
    try:
        route.validate()
        request.__post_init__()
        policy.__post_init__()
        trusted_journey_precommit.__post_init__()
        if (route.provider_id != pinned_provider_id
                or not _verified_signature(route, pinned_ed25519_public_key)):
            return _blocked("host-signed classification or provider invalid")
        if (route.objective_id != policy.objective_id
                or request.objective_id != route.objective_id
                or policy.prior_skill_ids != ("arithmetic",)
                or route.scope_sha256 != policy.scope_sha256
                or route.scope_sha256 != request.scope_fingerprint
                or route.development_exam_id != request.development_exam_id
                or route.candidate_checkpoint_sha256 != request.candidate_checkpoint_sha256
                or route.candidate_checkpoint_sha256 != policy.after_checkpoint_sha256
                or not hmac.compare_digest(route.policy_sha256, _policy_sha256(policy))
                or not hmac.compare_digest(route.roster_sha256,
                                           retention_roster_sha256(roster))
                or not hmac.compare_digest(route.training_inventory_sha256,
                                           arithmetic_training_inventory_sha256(
                                               policy.excluded_training_prompts))):
            return _blocked("signed operator route mismatches candidate or inventory")

        def live_route_matches() -> bool:
            observed = read_host_route(route.route_id)
            if type(observed) is not SignedRetentionRoute:
                return False
            observed.validate()
            return (observed == route
                    and _verified_signature(observed, pinned_ed25519_public_key))

        if not live_route_matches():
            return _blocked("external host route readback missing or stale")
        # The learner and evaluator share an in-process object graph here.
        # Freeze a value snapshot before invoking the local evaluation hook:
        # a frozen dataclass can still be mutated through object.__setattr__.
        # This does not replace host-owned immutable input custody.
        admission_inputs = (route, request, policy, roster, evidence,
                            signed_taxonomy, trusted_journey_precommit)
        admission_snapshot = deepcopy(admission_inputs)
        admission = evaluate_strict_retention_admission(
            request, policy, roster, evidence, signed_taxonomy,
            trusted_journey_precommit=trusted_journey_precommit,
            trusted_prior_checkpoint_sha256=route.prior_checkpoint_sha256,
            expected_policy_sha256=route.policy_sha256,
            expected_roster_sha256=route.roster_sha256,
            expected_semantic_inventory_sha256=route.semantic_inventory_sha256,
            expected_taxonomy_signer_id=route.taxonomy_signer_id,
            expected_taxonomy_nonce_sha256=route.taxonomy_nonce_sha256,
            independent_taxonomy_key=independent_taxonomy_key,
            independent_evaluator_key=independent_evaluator_key,
        )
        if not live_route_matches():
            return _blocked("external host route changed during retention review")
        if admission_inputs != admission_snapshot:
            return _blocked("retention candidate or evidence changed during admission")
        if (type(admission) is not StrictRetentionAdmissionResult
                or admission.status not in {"PASS", "HOLD"}
                or admission.graduation_authorized is not False
                or admission.private_exam_authorized is not False
                or admission.model_execution_authenticated is not False
                or admission.examiner_custody_verified is not False):
            return _blocked("strict local semantic retention admission rejected")
        return GovernedRetentionResult(admission.status, admission.reason,
                                       admission)
    except (ValueError, TypeError, AttributeError, OverflowError):
        return _blocked("invalid or unsupported signed arithmetic retention route")
    except Exception:
        # A host readback transport failure is a hard denial. The caller may
        # need to reconcile host state before attempting a new assessment.
        return _blocked("external host route readback unavailable")
