"""Explicit strict arithmetic retention *development* entrypoint.

This optional composition requires three separate source checks before a
development retention verdict can be returned:

1. An operator-pinned roster, policy and arithmetic semantic-key inventory;
2. A signed structure-family taxonomy bound to an expected signer and nonce;
3. Evaluator-keyed per-situation before/after observations.

All of these remain declarations until an independent examiner controls the
keys, roster, model processes and immutable precommit. This module neither
launches inference nor authorizes graduation. It supports only recognized
single-addition arithmetic, not general semantic independence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import re

from .retention_family_registry import (
    SignedFamilyTaxonomy, decide_retention_with_family_custody,
)
from .retention_qualification import (
    RetentionDecision, RetentionError, RetentionPair, RetentionPolicy,
    SignedRetentionEvidence, _policy_sha256, retention_roster_sha256,
)
from .retention_semantic_guard import (
    StrictArithmeticRetentionError, StrictArithmeticPreflight,
    require_distinct_arithmetic_retention,
)


SCHEMA = "labyrinth.strict-arithmetic-retention-development.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class StrictRetentionDevelopmentError(RetentionError):
    """A required source-level precommit or arithmetic invariant is absent."""


@dataclass(frozen=True)
class StrictRetentionDevelopmentResult:
    development_decision: RetentionDecision
    policy_sha256: str
    roster_sha256: str
    semantic_inventory_sha256: str
    semantic_case_count: int
    taxonomy_receipt_sha256: str
    schema: str = SCHEMA
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    independent_semantics_authenticated: bool = False
    examiner_custody_verified: bool = False


def _pin(value: str, name: str) -> None:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise StrictRetentionDevelopmentError(f"invalid operator-pinned {name} digest")


def evaluate_strict_arithmetic_retention(
    policy: RetentionPolicy,
    roster: tuple[RetentionPair, ...],
    evidence: SignedRetentionEvidence,
    signed_taxonomy: SignedFamilyTaxonomy, *,
    expected_policy_sha256: str,
    expected_roster_sha256: str,
    expected_semantic_inventory_sha256: str,
    expected_signer_id: str,
    expected_nonce_sha256: str,
    independent_taxonomy_key: bytes,
    independent_evaluator_key: bytes,
) -> StrictRetentionDevelopmentResult:
    """Return only a development decision after all declared guards agree.

    `expected_*` digests, signer, nonce and keys must be independently pinned
    before observation. The function cannot prove when or by whom that happened.
    It deliberately does not accept a Boolean claiming physical authority.
    """
    for name, expected in (
        ("policy", expected_policy_sha256),
        ("roster", expected_roster_sha256),
        ("semantic", expected_semantic_inventory_sha256),
    ):
        _pin(expected, name)

    try:
        preflight: StrictArithmeticPreflight = require_distinct_arithmetic_retention(
            policy, roster)
    except (StrictArithmeticRetentionError, TypeError, AttributeError) as exc:
        raise StrictRetentionDevelopmentError(
            "strict arithmetic semantic preflight rejected retention roster") from exc

    current_policy = _policy_sha256(policy)
    current_roster = retention_roster_sha256(roster)
    for name, actual, expected in (
        ("policy", current_policy, expected_policy_sha256),
        ("roster", current_roster, expected_roster_sha256),
        ("semantic", preflight.semantic_inventory_sha256,
         expected_semantic_inventory_sha256),
    ):
        if not hmac.compare_digest(actual, expected):
            raise StrictRetentionDevelopmentError(
                f"operator-pinned {name} commitment mismatch")
    if (type(preflight) is not StrictArithmeticPreflight
            or preflight.graduation_authorized is not False
            or preflight.model_execution_authenticated is not False
            or preflight.independent_semantics_authenticated is not False
            or preflight.roster_sha256 != current_roster
            or preflight.objective_id != policy.objective_id):
        raise StrictRetentionDevelopmentError("semantic preflight asserted invalid authority")

    # Existing signed taxonomic and evaluator proofs are individually checked
    # here; a self-appointed caller holding both keys is still not independent.
    decision = decide_retention_with_family_custody(
        policy, roster, evidence, signed_taxonomy,
        expected_signer_id=expected_signer_id,
        expected_nonce_sha256=expected_nonce_sha256,
        independent_taxonomy_key=independent_taxonomy_key,
        independent_evaluator_key=independent_evaluator_key,
    )
    if (type(decision) is not RetentionDecision
            or decision.graduation_authorized is not False
            or decision.model_execution_authenticated is not False):
        raise StrictRetentionDevelopmentError("retention decision asserted forbidden authority")

    return StrictRetentionDevelopmentResult(
        development_decision=decision,
        policy_sha256=current_policy,
        roster_sha256=current_roster,
        semantic_inventory_sha256=preflight.semantic_inventory_sha256,
        semantic_case_count=preflight.semantic_case_count,
        taxonomy_receipt_sha256=sha256(signed_taxonomy.receipt_bytes).hexdigest(),
    )
