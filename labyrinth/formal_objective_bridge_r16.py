"""Explicit, signed co-binding for two otherwise incompatible objective models.

The TrainingDay FormalObjectiveContract does NOT contain the operational
ObjectiveContract's curriculum focus, gain target, scope pacing, or forbidden
claims. Conversely ObjectiveContract does NOT commit the formal evaluator,
case thresholds, public observation conditions, or excluded skill taxonomy.
There is no lossless automatic converter between them.

This opt-in adapter admits a separately authored pair only as an exact source
identity *binding*. It never synthesizes a contract, approves training, treats
public development as an independent exam, or grants graduation. A locally
generated test signature is NOT proof of independent host key custody.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .objective_contract import ObjectiveContract
from .trainingday_objective_metamorphic import (
    FormalObjectiveContract, _formal_fingerprint,
)


SCHEMA = "labyrinth.dual-objective-source-binding.v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_DOMAIN = b"labyrinth.dual-objective-operator-approval.v1\0"


class FormalObjectiveBridgeError(ValueError):
    """Cannot establish this exact dual-objective source binding."""


@dataclass(frozen=True)
class DualObjectivePins:
    """Pinned by a separate operator; content and key provenance are external."""

    authority_id: str
    candidate_id: str
    formal_sha256: str
    operational_sha256: str
    scope_sha256: str
    host_binding_sha256: str
    policy_sha256: str
    source_tree_sha256: str
    public_key: bytes

    def validate(self) -> None:
        if (any(type(v) is not str or _ID.fullmatch(v) is None
                for v in (self.authority_id, self.candidate_id))
                or any(type(v) is not str or _SHA.fullmatch(v) is None
                       for v in (self.formal_sha256, self.operational_sha256,
                                 self.scope_sha256, self.host_binding_sha256,
                                 self.policy_sha256, self.source_tree_sha256))
                or type(self.public_key) is not bytes
                or len(self.public_key) != 32):
            raise FormalObjectiveBridgeError("invalid independently pinned operator identity")


@dataclass(frozen=True)
class DualObjectiveSourceBinding:
    status: str
    authority_id: str
    candidate_id: str
    formal_sha256: str
    operational_sha256: str
    host_binding_sha256: str
    policy_sha256: str
    source_tree_sha256: str
    semantic_equivalence_proven: bool = False
    training_authorized: bool = False
    neural_execution_authenticated: bool = False
    graduation_authorized: bool = False


def _contract_pair(formal: FormalObjectiveContract,
                   operational: ObjectiveContract,
                   pins: DualObjectivePins) -> dict[str, object]:
    if type(pins) is not DualObjectivePins:
        raise FormalObjectiveBridgeError("typed independent operator pins required")
    pins.validate()
    formal_hash = _formal_fingerprint(formal)
    if type(operational) is not ObjectiveContract or formal_hash is None:
        raise FormalObjectiveBridgeError("valid typed contracts required")
    try:
        operational.__post_init__()
        scope = operational.scope
        original = operational.fingerprint
        scope_hash = scope.fingerprint
    except (ValueError, TypeError, AttributeError) as exc:
        raise FormalObjectiveBridgeError("invalid operational scope") from exc
    if (not hmac.compare_digest(formal_hash, pins.formal_sha256)
            or not hmac.compare_digest(original, pins.operational_sha256)
            or not hmac.compare_digest(scope_hash, pins.scope_sha256)):
        raise FormalObjectiveBridgeError("contracts differ from independent exact pins")

    # This strict intersection is deliberately narrower than either model.
    # An identifier with distinct names/meaning cannot be mapped by similarity.
    formal_skills = {item.skill_id: item for item in formal.skills}
    operational_targets = {item.skill: item for item in operational.targets}
    if (formal.objective_id != scope.objective_id
            or set(formal_skills) != set(scope.skills)
            or set(formal_skills) != set(operational_targets)
            or set(formal.excluded_skill_ids) != set(operational.excluded_scopes)
            or any(formal_skills[skill].metric_id != operational_targets[skill].rubric_id
                   for skill in formal_skills)):
        raise FormalObjectiveBridgeError("formal skill/metric/exclusion is not explicitly identical")
    # Formal min_cases/min_correct and operational score/gain percentages
    # are DISTINCT policies. Sign both hashes; never convert one into another.
    return {
        "schema": SCHEMA,
        "authority_id": pins.authority_id,
        "candidate_id": pins.candidate_id,
        "formal_sha256": pins.formal_sha256,
        "operational_sha256": pins.operational_sha256,
        "scope_sha256": pins.scope_sha256,
        "host_binding_sha256": pins.host_binding_sha256,
        "policy_sha256": pins.policy_sha256,
        "source_tree_sha256": pins.source_tree_sha256,
        "metric_to_rubric": [
            {"skill_id": skill, "formal_metric": formal_skills[skill].metric_id,
             "operational_rubric": operational_targets[skill].rubric_id}
            for skill in sorted(formal_skills)
        ],
        "semantic_equivalence": "NOT_CLAIMED",
        "training_authorized": False,
        "graduation_authorized": False,
    }


def approved_dual_objective_payload(
    formal: FormalObjectiveContract,
    operational: ObjectiveContract,
    pins: DualObjectivePins,
) -> bytes:
    """Build exact domain-separated signing bytes; signer must be host-owned."""
    body = json.dumps(_contract_pair(formal, operational, pins),
                      sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")
    return _DOMAIN + body


def verify_dual_objective_binding(
    formal: FormalObjectiveContract,
    operational: ObjectiveContract,
    pins: DualObjectivePins,
    signed_payload: bytes,
    signature: bytes,
) -> DualObjectiveSourceBinding:
    """Verify exact pair against independently pinned operator public key.

    No host readback, no semantic equivalence, and no execution credential is
    produced. A caller may not treat this SOURCE_BINDING_ONLY receipt as an
    authority capability or evaluator result.
    """
    if (type(signed_payload) is not bytes or len(signed_payload) > 16_384
            or type(signature) is not bytes or len(signature) != 64):
        raise FormalObjectiveBridgeError("bounded signed operator pair required")
    expected = approved_dual_objective_payload(formal, operational, pins)
    if not hmac.compare_digest(expected, signed_payload):
        raise FormalObjectiveBridgeError("operator pair bytes do not match exact contracts")
    try:
        Ed25519PublicKey.from_public_bytes(pins.public_key).verify(signature, signed_payload)
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise FormalObjectiveBridgeError("independent operator signature refused") from exc
    return DualObjectiveSourceBinding(
        status="DUAL_OBJECTIVE_SOURCE_BINDING_ONLY",
        authority_id=pins.authority_id,
        candidate_id=pins.candidate_id,
        formal_sha256=pins.formal_sha256,
        operational_sha256=pins.operational_sha256,
        host_binding_sha256=pins.host_binding_sha256,
        policy_sha256=pins.policy_sha256,
        source_tree_sha256=pins.source_tree_sha256,
    )
