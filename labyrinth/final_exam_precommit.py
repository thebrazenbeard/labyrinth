"""Evaluator-side, source-only pre-training commitment for a sealed final exam.

Only this *projection*, never the sealed plan or its private question bank,
may be disclosed to the trainee/Minotaur before training. A pinned Ed25519
signer authenticates bytes, not when they were signed, whether custody was
independent, or whether an exam was physically performed. An external owner
must anchor these bytes before training and keep a rollback-resistant spent
event register. This module cannot authorize graduation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from .sealed_semantic_exam_protocol import SignedSealedPlan, _verified_plan


SCHEMA = "labyrinth.final-exam-pretraining-precommit.v1"
_DOMAIN = b"labyrinth.final-exam-pretraining-precommit.v1\0"
_HEX = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_MAX_BYTES = 4096

# Exact schema of build_sealed_plan's *public* source proposal. An unexpected
# new field, including a private case/seed/label accidentally added upstream,
# must be reviewed before the pre-training projection can be published.
_SEALED_PLAN_FIELDS = frozenset({
    "schema", "exam_id", "event_id", "objective_id", "scope_sha256",
    "pretrained_model_sha256", "tokenizer_sha256",
    "optimizer_config_sha256", "source_sha256",
    "prior_skill_checkpoint_sha256", "training_inventory_sha256",
    "trainer_key_commitment", "private_bank_commitment", "family_count",
    "situations_per_family", "transfer_situations", "retention_situations",
    "case_count", "seeds", "step_budgets", "arms",
    "decision_policy_status", "proposal_primary_margin_pp",
    "proposal_retention_aggregate_loss_pp", "proposal_retention_family_loss_pp",
    "proposal_family_stratified_method", "decision_policy_qualified",
    "status", "external_timestamp_verified", "host_custody_verified",
    "independent_model_execution_verified", "graduation_authorized",
})

_PRECOMMIT_FIELDS = frozenset({
    "schema", "exam_id", "event_id", "objective_id", "scope_sha256",
    "training_inventory_sha256", "source_sha256", "sealed_plan_sha256",
    "signer_public_key_sha256", "external_anchor_verified",
    "host_custody_verified", "physical_model_execution_verified",
    "graduation_authorized",
})


class FinalExamPrecommitError(ValueError):
    """A pre-training source commitment cannot be authenticated or released."""


@dataclass(frozen=True)
class SignedFinalExamPrecommit:
    public_bytes: bytes
    signature_hex: str
    graduation_authorized: bool = False


@dataclass(frozen=True)
class FinalExamPrecommitSummary:
    exam_id: str
    event_id: str
    objective_id: str
    scope_sha256: str
    sealed_plan_sha256: str
    graduation_authorized: bool = False


def _canonical(body: dict[str, Any]) -> bytes:
    return (json.dumps(body, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for key, value in pairs:
        if key in body:
            raise FinalExamPrecommitError("duplicate precommit field")
        body[key] = value
    return body


def _digest(value: Any, label: str) -> None:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise FinalExamPrecommitError(f"invalid {label} SHA-256")


def _identifier(value: Any, label: str) -> None:
    if type(value) is not str or _IDENT.fullmatch(value) is None:
        raise FinalExamPrecommitError(f"invalid {label}")


def create_pretraining_precommit(
    plan: SignedSealedPlan, *, evaluator_key: bytes,
    signing_key: Ed25519PrivateKey,
) -> SignedFinalExamPrecommit:
    """Create an answer-free commitment on the independent evaluator side.

    This is not a clock, external timestamp, one-shot ledger, or host-bound
    signing service. None of those claims may be inferred from this result.
    """
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise FinalExamPrecommitError("external Ed25519 signing key required")
    try:
        source = _verified_plan(plan, evaluator_key=evaluator_key)
    except (ValueError, TypeError) as exc:
        raise FinalExamPrecommitError("sealed plan is not authenticated") from exc
    if source.keys() != _SEALED_PLAN_FIELDS:
        raise FinalExamPrecommitError("sealed plan schema changed; review required")
    if any(source[field] is not False for field in (
            "external_timestamp_verified", "host_custody_verified",
            "independent_model_execution_verified", "graduation_authorized",
            "decision_policy_qualified")):
        raise FinalExamPrecommitError("unqualified source-only plan required")
    key_bytes = signing_key.public_key().public_bytes(
        encoding=Encoding.Raw, format=PublicFormat.Raw)
    body = {
        "schema": SCHEMA,
        "exam_id": source["exam_id"],
        "event_id": source["event_id"],
        "objective_id": source["objective_id"],
        "scope_sha256": source["scope_sha256"],
        "training_inventory_sha256": source["training_inventory_sha256"],
        "source_sha256": source["source_sha256"],
        "sealed_plan_sha256": sha256(plan.public_bytes).hexdigest(),
        "signer_public_key_sha256": sha256(key_bytes).hexdigest(),
        "external_anchor_verified": False,
        "host_custody_verified": False,
        "physical_model_execution_verified": False,
        "graduation_authorized": False,
    }
    for field in ("exam_id", "event_id", "objective_id"):
        _identifier(body[field], field)
    for field in ("scope_sha256", "training_inventory_sha256", "source_sha256"):
        _digest(body[field], field)
    public_bytes = _canonical(body)
    if len(public_bytes) > _MAX_BYTES:
        raise FinalExamPrecommitError("precommit size limit exceeded")
    return SignedFinalExamPrecommit(
        public_bytes, signing_key.sign(_DOMAIN + public_bytes).hex())


def verify_pretraining_precommit(
    precommit: SignedFinalExamPrecommit, *,
    pinned_signer_public_key: bytes,
    expected_exam_id: str,
    expected_event_id: str,
    expected_objective_id: str,
    expected_scope_sha256: str,
    expected_sealed_plan_sha256: str,
    consumed_event_sha256s: frozenset[str],
    expected_source_sha256: str | None = None,
    expected_training_inventory_sha256: str | None = None,
) -> FinalExamPrecommitSummary:
    """Validate a source-only public pin against EXTERNALLY owned expectations.

    The caller must commit the returned bytes before training and independently
    maintain the spent-event set; passing a fresh empty set is NOT replay safety.
    Source and training-inventory SHA expectations must also come from the
    external owner, never from the signed bytes being checked. Both are
    mandatory in effect: omitted, malformed or mismatched values fail closed.
    This never consumes an event or grants graduation.
    """
    if (type(precommit) is not SignedFinalExamPrecommit
            or precommit.graduation_authorized is not False
            or type(precommit.public_bytes) is not bytes
            or not 0 < len(precommit.public_bytes) <= _MAX_BYTES
            or type(precommit.signature_hex) is not str
            or re.fullmatch(r"[0-9a-f]{128}", precommit.signature_hex) is None
            or type(pinned_signer_public_key) is not bytes
            or len(pinned_signer_public_key) != 32):
        raise FinalExamPrecommitError("invalid signed precommit")
    for label, value in (("exam", expected_exam_id), ("event", expected_event_id),
                         ("objective", expected_objective_id)):
        _identifier(value, label)
    for label, value in (("scope", expected_scope_sha256),
                         ("sealed plan", expected_sealed_plan_sha256),
                         ("source", expected_source_sha256),
                         ("training inventory", expected_training_inventory_sha256)):
        _digest(value, label)
    if (type(consumed_event_sha256s) is not frozenset
            or any(type(value) is not str or _HEX.fullmatch(value) is None
                   for value in consumed_event_sha256s)):
        raise FinalExamPrecommitError("external spent-event roster required")
    try:
        Ed25519PublicKey.from_public_bytes(pinned_signer_public_key).verify(
            bytes.fromhex(precommit.signature_hex), _DOMAIN + precommit.public_bytes)
        body = json.loads(precommit.public_bytes, object_pairs_hook=_unique)
    except (InvalidSignature, ValueError, TypeError, UnicodeError) as exc:
        raise FinalExamPrecommitError("precommit signature or encoding invalid") from exc
    if (type(body) is not dict or body.keys() != _PRECOMMIT_FIELDS
            or _canonical(body) != precommit.public_bytes
            or body["schema"] != SCHEMA):
        raise FinalExamPrecommitError("precommit public schema changed")
    for field in ("exam_id", "event_id", "objective_id"):
        _identifier(body[field], field)
    for field in ("scope_sha256", "training_inventory_sha256", "source_sha256",
                  "sealed_plan_sha256", "signer_public_key_sha256"):
        _digest(body[field], field)
    if (body["exam_id"] != expected_exam_id
            or body["event_id"] != expected_event_id
            or body["objective_id"] != expected_objective_id
            or body["scope_sha256"] != expected_scope_sha256
            or body["sealed_plan_sha256"] != expected_sealed_plan_sha256
            or body["source_sha256"] != expected_source_sha256
            or body["training_inventory_sha256"] != expected_training_inventory_sha256
            or body["signer_public_key_sha256"] != sha256(pinned_signer_public_key).hexdigest()
            or any(body[field] is not False for field in (
                "external_anchor_verified", "host_custody_verified",
                "physical_model_execution_verified", "graduation_authorized"))):
        raise FinalExamPrecommitError("precommit pins or authority flags invalid")
    event_digest = sha256((SCHEMA + "\0" + expected_event_id).encode("ascii")).hexdigest()
    if event_digest in consumed_event_sha256s:
        raise FinalExamPrecommitError("exam event already consumed externally")
    return FinalExamPrecommitSummary(
        body["exam_id"], body["event_id"], body["objective_id"],
        body["scope_sha256"], body["sealed_plan_sha256"])
