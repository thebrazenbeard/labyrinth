"""Fail-closed SOURCE-ONLY validation of a Q8-A1..A10 external test ledger.

The caller supplies independently obtained, immutable verification *pins*.
Neither this parser nor a valid pair of signatures attests host key custody,
physical CAS isolation, observed behavior, permission, or Q8 qualification.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import NamedTuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class Q8AcceptanceSourceError(ValueError):
    """The source ledger does not match the independently pinned contract."""


class Q8ExpectedWitnessPins(NamedTuple):
    provider_id: str
    run_id: str
    host_sha256: str
    service_sha256: str
    source_commit_sha: str
    source_tree_sha: str
    roster_sha256: str
    policy_sha256: str
    signer_generation: int
    witness_generation: int
    operator_public_key: bytes
    witness_public_key: bytes


class Q8SourceObservation(NamedTuple):
    """A non-authoritative immutable parser result; never a release permit."""

    bundle_sha256: str
    case_ids: tuple[str, ...]

    @property
    def status(self) -> str:
        return "SOURCE_ONLY_UNATTESTED"

    @property
    def q8_physically_qualified(self) -> bool:
        return False

    @property
    def can_graduate(self) -> bool:
        return False

    @property
    def can_merge(self) -> bool:
        return False

    @property
    def can_deploy(self) -> bool:
        return False


_SCHEMA = "labyrinth.q8-external-acceptance-source.v1"
_DOMAIN = b"labyrinth.q8-external-acceptance-case.v1\0"
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_HEX_160 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_512 = re.compile(r"[0-9a-f]{128}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}\Z")
_CASES = {
    "Q8-A1": "RESTORED_TRAINER_REFUSED",
    "Q8-A2": "RESTORED_SERVICE_REFUSED",
    "Q8-A3": "CLONE_REFUSED",
    "Q8-A4": "REPLAY_REFUSED",
    "Q8-A5": "RECONCILED_COMMIT",
    "Q8-A6": "BINDING_MISMATCH",
    "Q8-A7": "PEER_UNAUTHORIZED",
    "Q8-A8": "ROTATION_STALE",
    "Q8-A9": "SINGLE_CAS_WINNER",
    "Q8-A10": "FAIL_CLOSED_UNAVAILABLE",
}
_STATEMENT_KEYS = frozenset({
    "schema", "provider_id", "run_id", "host_sha256", "service_sha256",
    "source_commit_sha", "source_tree_sha", "roster_sha256", "policy_sha256",
    "signer_generation", "witness_generation",
})
_CASE_KEYS = frozenset({
    "test_id", "operation_nonce", "before_index", "before_head_sha256",
    "after_index", "after_head_sha256", "refusal_code", "audit_sha256",
    "trace_sha256", "operator_signature_hex", "witness_signature_hex",
})
_UNSIGNED_CASE_KEYS = _CASE_KEYS - {
    "operator_signature_hex", "witness_signature_hex",
}


def _fail(message: str) -> None:
    raise Q8AcceptanceSourceError(message)


def _dict_exact(value: object, keys: frozenset[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(f"{label}: wrong or unrecognized fields")
    return value


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{name}: invalid identifier")


def _digest(value: object, name: str, pattern: re.Pattern[str] = _HEX_256) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(f"{name}: invalid lower-case hex digest")


def _integer(value: object, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= (2**63 - 1):
        _fail(f"{name}: invalid monotonic integer")


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    unique = {}
    for key, value in pairs:
        if key in unique:
            _fail(f"duplicate JSON key: {key}")
        unique[key] = value
    return unique


def q8_case_signed_payload(statement: dict, case: dict) -> bytes:
    """Domain-separated canonical signer input; contains no signatures or keys."""
    return _DOMAIN + json.dumps(
        {"statement": statement, "case": case},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def _check_pins(pins: Q8ExpectedWitnessPins) -> None:
    if type(pins) is not Q8ExpectedWitnessPins:
        _fail("expected pins must be an independently supplied immutable tuple")
    _identifier(pins.provider_id, "pinned provider")
    _identifier(pins.run_id, "pinned run")
    for name in ("host_sha256", "service_sha256", "roster_sha256", "policy_sha256"):
        _digest(getattr(pins, name), f"pinned {name}")
    for name in ("source_commit_sha", "source_tree_sha"):
        _digest(getattr(pins, name), f"pinned {name}", _HEX_160)
    for name in ("signer_generation", "witness_generation"):
        _integer(getattr(pins, name), f"pinned {name}", minimum=1)
    for name in ("operator_public_key", "witness_public_key"):
        key = getattr(pins, name)
        if type(key) is not bytes or len(key) != 32:
            _fail(f"pinned {name}: expected 32 immutable public-key bytes")
    if pins.operator_public_key == pins.witness_public_key:
        _fail("operator and witness verification keys must be distinct")


def _check_statement(statement: dict, pins: Q8ExpectedWitnessPins) -> None:
    _dict_exact(statement, _STATEMENT_KEYS, "bundle statement")
    if statement["schema"] != _SCHEMA:
        _fail("unsupported source schema")
    for key in _STATEMENT_KEYS - {"schema"}:
        if statement[key] != getattr(pins, key) or type(statement[key]) is not type(getattr(pins, key)):
            _fail(f"statement is not independently pinned: {key}")


def _check_case(case: dict, expected_test_id: str) -> None:
    _dict_exact(case, _CASE_KEYS, expected_test_id)
    if case["test_id"] != expected_test_id:
        _fail("missing, duplicate or out-of-order Q8 acceptance case")
    _identifier(case["operation_nonce"], "operation nonce")
    for key in ("before_index", "after_index"):
        _integer(case[key], key)
    for key in ("before_head_sha256", "after_head_sha256", "audit_sha256", "trace_sha256"):
        _digest(case[key], key)
    for key in ("operator_signature_hex", "witness_signature_hex"):
        _digest(case[key], key, _HEX_512)
    before = (case["before_index"], case["before_head_sha256"])
    after = (case["after_index"], case["after_head_sha256"])
    if expected_test_id == "Q8-A5":
        allowed = (
            case["refusal_code"] == "REMAINS_UNKNOWN" and after == before
        ) or (
            case["refusal_code"] == _CASES[expected_test_id]
            and case["after_index"] == case["before_index"] + 1
            and case["after_head_sha256"] != case["before_head_sha256"]
        )
    elif expected_test_id == "Q8-A9":
        allowed = (
            case["refusal_code"] == _CASES[expected_test_id]
            and case["after_index"] == case["before_index"] + 1
            and case["after_head_sha256"] != case["before_head_sha256"]
        )
    else:
        allowed = case["refusal_code"] == _CASES[expected_test_id] and after == before
    if not allowed:
        _fail(f"{expected_test_id}: invalid declared refusal/CAS-head transition")
    if case["audit_sha256"] == case["trace_sha256"]:
        _fail(f"{expected_test_id}: audit and trace evidence must be distinct")


def validate_q8_acceptance_bundle(
    bundle_bytes: bytes, expected_pins: Q8ExpectedWitnessPins,
) -> Q8SourceObservation:
    """Check source syntax, pinned bindings, signer proofs, and declared Q8 rows.

    This cannot establish that protected heads are genuine or that adversarial
    tests ran. Even complete, valid signatures return SOURCE_ONLY_UNATTESTED.
    """
    _check_pins(expected_pins)
    if type(bundle_bytes) is not bytes or not 1 <= len(bundle_bytes) <= 131072:
        _fail("source bundle must be bytes within size limit")
    try:
        bundle = json.loads(bundle_bytes.decode("utf-8"), object_pairs_hook=_pairs_no_duplicates)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise Q8AcceptanceSourceError("malformed/ambiguous source JSON") from exc
    if not isinstance(bundle, dict) or set(bundle) != _STATEMENT_KEYS | {"cases"}:
        _fail("bundle has missing or unrecognized root fields")
    statement = {key: value for key, value in bundle.items() if key != "cases"}
    _check_statement(statement, expected_pins)
    cases = bundle["cases"]
    if not isinstance(cases, list) or len(cases) != len(_CASES):
        _fail("source evidence must contain exactly Q8-A1 through Q8-A10")
    operator = Ed25519PublicKey.from_public_bytes(expected_pins.operator_public_key)
    witness = Ed25519PublicKey.from_public_bytes(expected_pins.witness_public_key)
    # One digest must not be recycled across either evidence class, even in
    # a different case: an audit blob cannot masquerade as a replay trace.
    seen_nonce, seen_evidence_sha256 = set(), set()
    for expected_test_id, case in zip(_CASES, cases, strict=True):
        _check_case(case, expected_test_id)
        for field, seen in (("operation_nonce", seen_nonce),
                            ("audit_sha256", seen_evidence_sha256),
                            ("trace_sha256", seen_evidence_sha256)):
            value = case[field]
            if value in seen:
                _fail(f"{expected_test_id}: duplicate {field}")
            seen.add(value)
        payload = q8_case_signed_payload(
            statement, {key: case[key] for key in _UNSIGNED_CASE_KEYS}
        )
        try:
            operator.verify(bytes.fromhex(case["operator_signature_hex"]), payload)
            witness.verify(bytes.fromhex(case["witness_signature_hex"]), payload)
        except InvalidSignature as exc:
            raise Q8AcceptanceSourceError(f"{expected_test_id}: signer mismatch") from exc
    return Q8SourceObservation(
        bundle_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        case_ids=tuple(_CASES),
    )
