"""Nonauthorizing source check of an externally signed graduation-state snapshot.

``GateState(graduated=True)`` is only a Python value.  It is not evidence that
an independently administered examiner ever granted graduation.  This adapter
binds a *reported* state to a signed, canonical source envelope and separately
supplied pins.  The result deliberately cannot be passed to the graduation
gate as an authorization or used for a production cutover.

The caller must obtain the pins/public key from a genuinely independent host.
This Python check cannot attest that independence, monotonic custody, or that
the signed state was the latest.  Q8 and independent final examination HOLD.
"""

from __future__ import annotations

from hashlib import sha256
import hmac
import json
import re
from typing import Any, NamedTuple

from .graduation_gate import GateState, GraduationPolicy, _policy_fingerprint
from .graduation_state import _serialize


_DOMAIN = b"labyrinth.external-gate-state-source.w4\x00"
_SCHEMA = "labyrinth.external-gate-state-source.w4"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEX128 = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_KEYS = frozenset({"schema", "provider_id", "policy_fingerprint", "state_json",
                   "state_sha256", "version", "external_head_sha256"})


class ExternalGateStateSourceError(ValueError):
    """An untrusted or mismatched source statement must never become authority."""


class GateStateSourceObservation(NamedTuple):
    provider_id: str
    version: int
    state_sha256: str
    external_head_sha256: str
    reported_prior_graduated: bool
    source_bytes_and_signature_match: bool = True
    independent_host_custody_verified: bool = False
    fresh_final_exam_verified: bool = False
    graduation_authorized: bool = False
    deployment_authorized: bool = False


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ExternalGateStateSourceError(f"invalid pinned {name}")
    return value


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ExternalGateStateSourceError("duplicate signed source key")
        out[key] = value
    return out


def _reject_constant(value: str) -> None:
    raise ExternalGateStateSourceError("nonfinite signed number")


def _canonical(record: dict[str, Any]) -> bytes:
    try:
        return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    except (ValueError, OverflowError, TypeError, RecursionError) as exc:
        raise ExternalGateStateSourceError("invalid canonical signed source") from exc


def _check_state_shape(state: GateState) -> None:
    if type(state.next_attempt) is not int or not 1 <= state.next_attempt < 2**63:
        raise ExternalGateStateSourceError("invalid attempt state")
    for field in ("consumed_run_ids", "consumed_suite_sha256s",
                  "consumed_eval_case_ids", "consumed_eval_prompt_commitments"):
        entries = getattr(state, field)
        pattern = (_HEX64 if field in ("consumed_suite_sha256s",
                                       "consumed_eval_prompt_commitments") else _IDENTIFIER)
        if (type(entries) is not frozenset
                or any(type(item) is not str or pattern.fullmatch(item) is None
                       for item in entries)):
            raise ExternalGateStateSourceError("invalid source state entry")
    if type(state.graduated) is not bool:
        raise ExternalGateStateSourceError("invalid source graduated flag")


def inspect_signed_source_state(
    *, policy: GraduationPolicy, candidate_state: GateState,
    signed_source_bytes: bytes, signature_hex: str,
    pinned_public_key: bytes, pinned_provider_id: str,
    expected_state_sha256: str, expected_version: int,
    expected_external_head_sha256: str,
) -> GateStateSourceObservation:
    """Check source consistency only; never return a trusted ``GateState``.

    Signed bytes have the form ``_DOMAIN + canonical JSON`` under Ed25519.
    The independently custodied host must prove actual nonrollback freshness;
    an attacker may sign an old state and present matching old pins in-process.
    """
    if type(policy) is not GraduationPolicy or type(candidate_state) is not GateState:
        raise ExternalGateStateSourceError("exact policy and state records required")
    # A frozen dataclass is not immutable against object.__setattr__. Reapply
    # the exact constructor invariants to the *current* fields rather than
    # treating the object's successful historical construction as evidence.
    try:
        GraduationPolicy.__post_init__(policy)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ExternalGateStateSourceError("mutated or invalid graduation policy") from exc
    _check_state_shape(candidate_state)
    if (type(pinned_provider_id) is not str
            or _IDENTIFIER.fullmatch(pinned_provider_id) is None):
        raise ExternalGateStateSourceError("invalid pinned provider")
    expected_state_sha256 = _digest(expected_state_sha256, "state")
    expected_external_head_sha256 = _digest(expected_external_head_sha256, "head")
    if type(expected_version) is not int or not 0 <= expected_version < 2**63:
        raise ExternalGateStateSourceError("invalid pinned version")
    if (type(pinned_public_key) is not bytes or len(pinned_public_key) != 32
            or type(signed_source_bytes) is not bytes
            or not 1 <= len(signed_source_bytes) <= 1_048_576
            or type(signature_hex) is not str or _HEX128.fullmatch(signature_hex) is None):
        raise ExternalGateStateSourceError("missing exact source signature or payload")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
    except ImportError as exc:
        raise ExternalGateStateSourceError("Ed25519 source verifier unavailable") from exc
    try:
        Ed25519PublicKey.from_public_bytes(pinned_public_key).verify(
            bytes.fromhex(signature_hex), _DOMAIN + signed_source_bytes)
    except (ValueError, TypeError, InvalidSignature) as exc:
        raise ExternalGateStateSourceError("untrusted source signature") from exc
    try:
        receipt = json.loads(signed_source_bytes.decode("ascii"),
                             object_pairs_hook=_unique, parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ExternalGateStateSourceError("invalid signed source JSON") from exc
    if type(receipt) is not dict or receipt.keys() != _KEYS:
        raise ExternalGateStateSourceError("unexpected signed source fields")
    if not hmac.compare_digest(signed_source_bytes, _canonical(receipt)):
        raise ExternalGateStateSourceError("noncanonical signed source bytes")
    fingerprint = _policy_fingerprint(policy)
    if candidate_state.policy_fingerprint != fingerprint:
        raise ExternalGateStateSourceError("candidate state unbound to policy")
    serialized_state = _serialize(candidate_state)
    state_sha = sha256(serialized_state.encode("utf-8")).hexdigest()
    if (receipt["schema"] != _SCHEMA
            or receipt["provider_id"] != pinned_provider_id
            or receipt["policy_fingerprint"] != fingerprint
            or receipt["state_json"] != serialized_state
            or receipt["state_sha256"] != state_sha
            or not hmac.compare_digest(state_sha, expected_state_sha256)
            or type(receipt["version"]) is not int
            or receipt["version"] != expected_version
            or receipt["external_head_sha256"] != expected_external_head_sha256):
        raise ExternalGateStateSourceError("signed state or independently pinned source mismatched")
    return GateStateSourceObservation(
        pinned_provider_id, expected_version, state_sha,
        expected_external_head_sha256, candidate_state.graduated)
