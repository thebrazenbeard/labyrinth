"""Verify externally pinned Q8 operator/witness evidence envelopes.

This is an *intake verifier*, not an operator, signer, monotonic store, or
graduation authority. A valid envelope attests that the two configured keys
signed the claimed observations; it does not prove private-key custody, host
isolation, nonrollback hardware, truthful observations or a physical Q8 PASS.

No signing keys, host endpoints or fallback authority are implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import NamedTuple


_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEX128 = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_DOMAIN_OPERATOR = b"labyrinth.q8.operator-observation.v1\x00"
_DOMAIN_WITNESS = b"labyrinth.q8.witness-observation.v1\x00"
SCHEMA = "labyrinth.q8.external-observation.v1"
REQUIRED_CASES = tuple(f"Q8-A{i}" for i in range(1, 11))
_REQUIRED_OUTCOMES = {
    "Q8-A1": frozenset({"refused"}),
    "Q8-A2": frozenset({"refused"}),
    "Q8-A3": frozenset({"refused"}),
    "Q8-A4": frozenset({"refused"}),
    "Q8-A5": frozenset({"reconciled", "unknown"}),
    "Q8-A6": frozenset({"refused"}),
    "Q8-A7": frozenset({"refused"}),
    "Q8-A8": frozenset({"refused"}),
    "Q8-A9": frozenset({"one_winner"}),
    "Q8-A10": frozenset({"fail_closed"}),
}
_MAX_BYTES = 128 * 1024
_MAX_COUNTER = 2**63 - 1


class Q8EvidenceError(ValueError):
    """Unsigned, malformed, incorrectly scoped or unauthenticated evidence."""


@dataclass(frozen=True)
class PinnedQ8Authorities:
    """Pins must come from the independent operator, not the supplied blob."""

    host_id: str
    source_sha256: str
    audit_bundle_sha256: str
    operator_public_key: bytes
    witness_public_key: bytes

    def __post_init__(self) -> None:
        if type(self.host_id) is not str or not _ID.fullmatch(self.host_id):
            raise Q8EvidenceError("invalid independently pinned host ID")
        for name in ("source_sha256", "audit_bundle_sha256"):
            value = getattr(self, name)
            if type(value) is not str or not _HEX64.fullmatch(value):
                raise Q8EvidenceError(f"invalid independently pinned {name}")
        if (type(self.operator_public_key) is not bytes or
                len(self.operator_public_key) != 32 or
                type(self.witness_public_key) is not bytes or
                len(self.witness_public_key) != 32 or
                self.operator_public_key == self.witness_public_key):
            raise Q8EvidenceError("operator and witness require distinct pinned Ed25519 keys")


class Q8EnvelopeInspection(NamedTuple):
    """Cryptographic evidence status only; cannot authorize any effect."""

    evidence_sha256: str
    observed_cases: tuple[str, ...]
    classification: str = "SIGNED_SOURCE_ENVELOPE_NOT_PHYSICAL_Q8"


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Q8EvidenceError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise Q8EvidenceError("nonfinite JSON number")


def _keys(item: object, names: frozenset[str]) -> bool:
    return type(item) is dict and item.keys() == names


def _hex64(item: object) -> bool:
    return type(item) is str and _HEX64.fullmatch(item) is not None


def _counter(item: object) -> bool:
    return type(item) is int and 0 <= item < _MAX_COUNTER


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _verify_signature(public_key: bytes, signature: bytes, message: bytes) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise Q8EvidenceError("cryptography dependency required; fail closed") from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except Exception as exc:
        # cryptography.InvalidSignature is a separate exception type.
        raise Q8EvidenceError("Q8 independent signature verification failed") from exc


def inspect_signed_q8_evidence(raw: bytes, pins: PinnedQ8Authorities) -> Q8EnvelopeInspection:
    """Check exact external envelope signatures, pins and ten claimed outcomes.

    The result is ALWAYS nonauthorizing. Operator/human host custody and the
    independently run A1-A10 acceptance tests are outside this function.
    """
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_BYTES:
        raise Q8EvidenceError("invalid or oversized Q8 envelope")
    if type(pins) is not PinnedQ8Authorities:
        raise Q8EvidenceError("independent pins are mandatory")
    # Reconstruct a verified, stable local authority snapshot. Frozen
    # dataclass inputs can still be mutated with object.__setattr__, so
    # constructor-time validation cannot be the sole trust boundary.
    pins = PinnedQ8Authorities(
        host_id=pins.host_id,
        source_sha256=pins.source_sha256,
        audit_bundle_sha256=pins.audit_bundle_sha256,
        operator_public_key=pins.operator_public_key,
        witness_public_key=pins.witness_public_key,
    )
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys,
                            parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise Q8EvidenceError("malformed Q8 evidence JSON") from exc
    root_keys = frozenset({"schema", "host_id", "source_sha256",
                           "audit_bundle_sha256", "observations",
                           "operator_signature_hex", "witness_signature_hex"})
    if not _keys(parsed, root_keys):
        raise Q8EvidenceError("invalid Q8 evidence root shape")
    try:
        canonical_raw = _canonical(parsed)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise Q8EvidenceError("invalid Q8 evidence canonical JSON") from exc
    if raw != canonical_raw:
        raise Q8EvidenceError("Q8 evidence envelope must be canonical wire JSON")
    for name, expected in (("schema", SCHEMA), ("host_id", pins.host_id),
                           ("source_sha256", pins.source_sha256),
                           ("audit_bundle_sha256", pins.audit_bundle_sha256)):
        if type(parsed[name]) is not str or parsed[name] != expected:
            raise Q8EvidenceError(f"independent Q8 pin mismatch: {name}")
    observations = parsed["observations"]
    if type(observations) is not list or len(observations) != len(REQUIRED_CASES):
        raise Q8EvidenceError("exactly ten independent Q8 observations required")
    observed: list[str] = []
    seen_traces: set[str] = set()
    case_keys = frozenset({"case_id", "outcome", "before_index", "after_index",
                           "before_head_sha256", "after_head_sha256", "trace_sha256"})
    for row in observations:
        if not _keys(row, case_keys):
            raise Q8EvidenceError("invalid Q8 observation shape")
        case = row["case_id"]
        if type(case) is not str or case not in _REQUIRED_OUTCOMES or case in observed:
            raise Q8EvidenceError("unknown or duplicate Q8 acceptance case")
        if type(row["outcome"]) is not str or row["outcome"] not in _REQUIRED_OUTCOMES[case]:
            raise Q8EvidenceError(f"unacceptable claimed outcome for {case}")
        if not all(_hex64(row[k]) for k in ("before_head_sha256", "after_head_sha256", "trace_sha256")):
            raise Q8EvidenceError("missing exact observed head or trace digest")
        # Each independently executed acceptance case needs its own reviewed
        # trace. Do not equate witness indices/heads across cases: the Q8
        # contract permits separate experiments starting at reset snapshots.
        trace_sha256 = row["trace_sha256"]
        if trace_sha256 in seen_traces:
            raise Q8EvidenceError("duplicate Q8 trace_sha256 across independent cases")
        seen_traces.add(trace_sha256)
        before, after = row["before_index"], row["after_index"]
        if not _counter(before) or not _counter(after) or after < before:
            raise Q8EvidenceError("nonmonotonic or malformed witness index")
        if row["outcome"] in ("refused", "fail_closed", "unknown"):
            if before != after or row["before_head_sha256"] != row["after_head_sha256"]:
                raise Q8EvidenceError("rejected/unknown operation changed claimed witness head")
        elif row["outcome"] == "one_winner":
            if after != before + 1 or row["before_head_sha256"] == row["after_head_sha256"]:
                raise Q8EvidenceError("one-winner race requires one claimed CAS advance")
        elif row["outcome"] == "reconciled":
            if after not in (before, before + 1):
                raise Q8EvidenceError("reconciliation claims multiple state advances")
            if (after == before) != (row["before_head_sha256"] == row["after_head_sha256"]):
                raise Q8EvidenceError("reconciliation head mismatch")
        observed.append(case)
    if set(observed) != set(REQUIRED_CASES):
        raise Q8EvidenceError("missing Q8 acceptance cases")
    for name in ("operator_signature_hex", "witness_signature_hex"):
        if type(parsed[name]) is not str or not _HEX128.fullmatch(parsed[name]):
            raise Q8EvidenceError(f"invalid {name}")
    unsigned = {k: v for k, v in parsed.items() if not k.endswith("_signature_hex")}
    payload = _canonical(unsigned)
    _verify_signature(pins.operator_public_key,
                      bytes.fromhex(parsed["operator_signature_hex"]), _DOMAIN_OPERATOR + payload)
    _verify_signature(pins.witness_public_key,
                      bytes.fromhex(parsed["witness_signature_hex"]), _DOMAIN_WITNESS + payload)
    return Q8EnvelopeInspection(hashlib.sha256(raw).hexdigest(), tuple(observed))
