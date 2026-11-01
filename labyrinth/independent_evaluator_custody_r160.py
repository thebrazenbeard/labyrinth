"""Fail-closed Q5–Q9/Q6 protected-evaluator admission *source* boundary.

There is no independently authenticated examiner issuer, key registry,
protected bank service, transport principal, revocation register, or external
monotonic Q6 witness installed here. A trainer's claim, signature and hash can
never turn into custody authority merely because local parsing succeeds.

This module accepts only canonical, bounded, **metadata-only** claims. Do not
pass exam prompts, private gold, data files, signer secrets, or answer keys.
It does not open a protected bank, perform any network I/O, or grant admission.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import NoReturn


SCHEMA = "labyrinth.independent-evaluator-custody-claim.v1"
RESULT_SCHEMA = "labyrinth.independent-evaluator-custody-source-result.v1"
UNVERIFIED_REJECTED = "UNVERIFIED_REJECTED"
MAX_CLAIM_BYTES = 8192
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEX128 = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
_OPAQUE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{7,127}\Z", re.ASCII)
_ROLE_FIELDS = (
    "examiner_id", "bank_custodian_id", "rights_steward_id",
    "q6_witness_owner_id", "transport_principal_id",
)
_DIGEST_FIELDS = (
    "private_bank_commitment_sha256", "train_exclusion_sha256",
    "pretraining_precommit_sha256", "model_freeze_sha256",
    "q6_witness_head_sha256", "rights_receipt_sha256", "exam_policy_sha256",
)
_CLAIM_FIELDS = frozenset({
    "schema", "protected_store_id", "issuer_key_generation",
    "issuer_public_key_hex", "receipt_signature_hex",
    *_ROLE_FIELDS, *_DIGEST_FIELDS,
})
_FALSE_FLAGS = (
    "issuer_registry_verified", "signing_key_custody_verified",
    "transport_identity_verified", "protected_store_verified",
    "receipt_signature_verified", "revocation_verified", "rights_admitted",
    "train_exclusions_verified", "pretraining_chronology_verified",
    "q6_monotonic_witness_verified", "model_freeze_verified",
    "authorizes_private_exam", "authorizes_training", "authorizes_graduation",
)


class InvalidEvaluatorCustodyClaim(ValueError):
    """Malformed source metadata; private data is never an accepted field."""


class IndependentEvaluatorIssuerUnavailable(RuntimeError):
    """The independent protected-evaluator admission trust root is absent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidEvaluatorCustodyClaim(message)


def _reject_duplicate_members(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON custody member")
        result[key] = value
    return result


def _reject_nonfinite(_: str) -> NoReturn:
    raise InvalidEvaluatorCustodyClaim("nonfinite JSON custody value")


def _parse_claim(raw: bytes) -> dict:
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_CLAIM_BYTES,
             "bounded canonical custody claim bytes required")
    try:
        body = json.loads(raw.decode("utf-8"),
                          object_pairs_hook=_reject_duplicate_members,
                          parse_constant=_reject_nonfinite)
    except InvalidEvaluatorCustodyClaim:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise InvalidEvaluatorCustodyClaim("invalid custody claim JSON") from exc
    _require(type(body) is dict and set(body) == _CLAIM_FIELDS,
             "custody claim requires exact metadata fields; no gold or grant flags")
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=True,
                           separators=(",", ":")).encode("ascii")
    _require(raw == canonical, "noncanonical custody claim JSON")
    _require(body["schema"] == SCHEMA, "unexpected custody claim schema")
    role_ids = []
    for field in (*_ROLE_FIELDS, "protected_store_id"):
        value = body[field]
        _require(type(value) is str and _OPAQUE_ID.fullmatch(value) is not None,
                 "custody identity/store must be an opaque bounded ID")
        if field != "protected_store_id":
            role_ids.append(value)
    _require(len(set(role_ids)) == len(role_ids),
             "source role names must be distinct (not independent proof)")
    _require(type(body["issuer_key_generation"]) is int
             and 1 <= body["issuer_key_generation"] <= 2**31 - 1,
             "invalid claimed signing key generation")
    for field in (*_DIGEST_FIELDS, "issuer_public_key_hex"):
        value = body[field]
        _require(type(value) is str and _HEX64.fullmatch(value) is not None,
                 "invalid claimed SHA-256 or Ed25519 public-key encoding")
    signature = body["receipt_signature_hex"]
    _require(type(signature) is str and _HEX128.fullmatch(signature) is not None,
             "invalid claimed Ed25519 receipt signature encoding")
    return body


def inspect_evaluator_custody_claim(claim_json: bytes) -> dict[str, str | bool]:
    """Check metadata shape, *reject* every self-issued custody claim.

    Even a correctly formatted Ed25519-looking signature is not validated as
    an externally issued receipt. This module has no independently installed
    issuer registry, revocation state, protected transport or prompt store.
    Hashes and role names are caller-owned claims, not evidence of separation.
    """
    _parse_claim(claim_json)
    report: dict[str, str | bool] = {
        "schema": RESULT_SCHEMA,
        "status": UNVERIFIED_REJECTED,
        "claim_sha256": hashlib.sha256(claim_json).hexdigest(),
        "metadata_structure_checked": True,
    }
    for flag in _FALSE_FLAGS:
        report[flag] = False
    return report


def require_independent_evaluator_admission(claim_json: bytes) -> NoReturn:
    """Physical Q5–Q9/Q6 callers must never treat a local claim as ADMIT."""
    inspect_evaluator_custody_claim(claim_json)
    raise IndependentEvaluatorIssuerUnavailable(
        "Q5–Q9/Q6 independent protected evaluator HARD HOLD: separately "
        "authenticated issuer/key registry, protected prompt store, signed "
        "receipts, transport identity and external Q6 witness unavailable")
