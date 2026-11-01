"""Source-only Q8 host/witness CAS and restart readback contract.

The CAS implementation, signing keys, journal, exclusive host lease and durable
high-watermark MUST reside in independent operator-controlled services. This
module supplies neither a signer nor a backend. Tests use an explicitly toy
backend; a successful result NEVER attests host isolation or physical Q8 PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import re
from typing import NamedTuple, Protocol

from .external_monotonic_authority_contract import ExternalAuthorityHead


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEX128 = re.compile(r"[0-9a-f]{128}\Z", re.ASCII)
_SIGN_SERVICE = b"labyrinth.q8.r93.service-head.v1\0"
_SIGN_WITNESS = b"labyrinth.q8.r93.witness-head.v1\0"
_CHANGE_DOMAIN = b"labyrinth.q8.r93.plastic-state-change.v1\0"
_MAX = 2**63 - 1
_GENESIS = "0" * 64


class Q8SourceRejection(ValueError):
    """The supplied source/pins do not satisfy the declared contract."""


class Q8OutcomeUnknown(Q8SourceRejection):
    """CAS may have committed; reconcile externally, NEVER retry blindly."""


def _id(value: object, label: str) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise Q8SourceRejection(f"invalid {label}")
    return value


def _hex(value: object, label: str) -> str:
    if type(value) is not str or not _HEX64.fullmatch(value):
        raise Q8SourceRejection(f"invalid {label}")
    return value


def _num(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < _MAX:
        raise Q8SourceRejection(f"invalid {label}")
    return value


def _canonical(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


@dataclass(frozen=True)
class Q8PlasticChange:
    """Typed STATE_CHANGE, deliberately restricted to plastic model weights."""

    run_id: str
    event_id: str
    target_id: str
    prior_sha256: str
    successor_sha256: str
    evidence_sha256: str
    authority_lease_sha256: str
    state_family: str = "plastic_model"
    change_class: str = "model_component_update"
    reversible: bool = True

    def validate(self) -> None:
        for n in ("run_id", "event_id", "target_id"):
            _id(getattr(self, n), n)
        for n in ("prior_sha256", "successor_sha256", "evidence_sha256",
                  "authority_lease_sha256"):
            _hex(getattr(self, n), n)
        if (self.state_family != "plastic_model"
                or self.change_class != "model_component_update"
                or self.reversible is not True):
            raise Q8SourceRejection("nonplastic or nonreversible state mutation is out of Q8 scope")
        if self.prior_sha256 == self.successor_sha256:
            raise Q8SourceRejection("state change has no successor update")

    def digest(self) -> str:
        self.validate()
        return sha256(_CHANGE_DOMAIN + _canonical({
            n: getattr(self, n) for n in self.__dataclass_fields__
        })).hexdigest()


@dataclass(frozen=True)
class Q8WitnessPins:
    """Host/operator supplied; NOT discoverable from the witness response."""

    host_id: str
    provider_id: str
    run_id: str
    service_public_key: bytes
    witness_public_key: bytes
    minimum_witness_counter: int
    minimum_witness_digest: str

    def validate(self) -> None:
        _id(self.host_id, "host")
        _id(self.provider_id, "provider")
        _id(self.run_id, "run")
        if (type(self.service_public_key) is not bytes
                or len(self.service_public_key) != 32
                or type(self.witness_public_key) is not bytes
                or len(self.witness_public_key) != 32
                or self.service_public_key == self.witness_public_key):
            raise Q8SourceRejection("distinct externally pinned Ed25519 role keys required")
        _num(self.minimum_witness_counter, "operator-pinned witness counter", 1)
        _hex(self.minimum_witness_digest, "operator-pinned witness digest")


@dataclass(frozen=True)
class Q8WitnessRecord:
    """Two-role signature of one exact service+journal high-watermark."""

    host_id: str
    run_id: str
    epoch: int
    fence: int
    commit_index: int
    head_sha256: str
    journal_generation: int
    journal_sha256: str
    witness_counter: int
    previous_witness_sha256: str
    event_id: str
    state_change_sha256: str
    service_signature_hex: str
    witness_signature_hex: str

    def signed_payload(self) -> bytes:
        _id(self.host_id, "host")
        _id(self.run_id, "run")
        _id(self.event_id, "event")
        for n in ("epoch", "journal_generation", "witness_counter"):
            _num(getattr(self, n), n, 1)
        for n in ("fence", "commit_index"):
            _num(getattr(self, n), n)
        for n in ("head_sha256", "journal_sha256", "previous_witness_sha256",
                  "state_change_sha256"):
            _hex(getattr(self, n), n)
        return _canonical({
            n: getattr(self, n) for n in self.__dataclass_fields__
            if n not in {"service_signature_hex", "witness_signature_hex"}
        })

    def digest(self) -> str:
        payload = self.signed_payload()
        for name in ("service_signature_hex", "witness_signature_hex"):
            value = getattr(self, name)
            if type(value) is not str or not _HEX128.fullmatch(value):
                raise Q8SourceRejection(f"invalid {name}")
        return sha256(payload + bytes.fromhex(self.service_signature_hex)
                      + bytes.fromhex(self.witness_signature_hex)).hexdigest()

class Q8CASPort(Protocol):
    """Must be implemented by independently owned host *and* witness services."""

    def read_service_head(self, run_id: str) -> ExternalAuthorityHead: ...

    def read_witness_record(self, run_id: str) -> Q8WitnessRecord: ...

    def compare_and_swap_once(self, change: Q8PlasticChange,
                              expected: Q8WitnessRecord) -> Q8WitnessRecord: ...


class Q8SourceCASObservation(NamedTuple):
    witness_record_sha256: str
    service_commit_index: int

    @property
    def classification(self) -> str:
        return "SOURCE_ONLY_HOST_CAS_CONTRACT_NOT_ATTESTED"

    @property
    def physical_q8_verified(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False

    @property
    def deployment_authorized(self) -> bool:
        return False


def _verify_signature(key: bytes, signed: bytes, signature_hex: str) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise Q8SourceRejection("cryptography dependency unavailable") from exc
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(bytes.fromhex(signature_hex), signed)
    except Exception as exc:
        raise Q8SourceRejection("independent Q8 signer verification failed") from exc


def _check_record(record: object, pins: Q8WitnessPins) -> Q8WitnessRecord:
    if type(record) is not Q8WitnessRecord:
        raise Q8SourceRejection("exact signed witness record required")
    # frozen=True is bypassable by object.__setattr__; copy once before any
    # validation or use, including post-CAS comparisons.
    record = replace(record)
    payload = record.signed_payload()
    record.digest()
    if record.host_id != pins.host_id or record.run_id != pins.run_id:
        raise Q8SourceRejection("witness host/run pin mismatch")
    _verify_signature(pins.service_public_key, _SIGN_SERVICE + payload,
                      record.service_signature_hex)
    _verify_signature(pins.witness_public_key, _SIGN_WITNESS + payload,
                      record.witness_signature_hex)
    return record


def probe_q8_source_cas(*, port: Q8CASPort, pins: Q8WitnessPins,
                        change: Q8PlasticChange) -> Q8SourceCASObservation:
    """Run one claimed CAS; post-attempt defects are UNKNOWN, never retryable.

    The external host/witness must independently ensure atomic event uniqueness,
    protected signer custody, real persistence, nonrollback and lease ownership.
    The client cannot assess those properties from a Python object or signatures.
    """
    if type(pins) is not Q8WitnessPins or type(change) is not Q8PlasticChange:
        raise Q8SourceRejection("typed externally pinned Q8 inputs required")
    pins = replace(pins)
    change = replace(change)
    pins.validate()
    change.validate()
    if change.run_id != pins.run_id:
        raise Q8SourceRejection("state change run differs from host pin")
    # Freeze the exact caller-authorized intent *before* invoking an
    # adversarial external port. Passing even a frozen dataclass across that
    # boundary is not immutable: object.__setattr__ can rewrite it. The port
    # receives a separate transport copy, never our retained comparison input.
    expected_change_sha256 = change.digest()
    transport_change = replace(change)
    try:
        before = _check_record(port.read_witness_record(pins.run_id), pins)
        service_before = port.read_service_head(pins.run_id)
    except Exception as exc:
        raise Q8SourceRejection("Q8 preflight host/witness unavailable") from exc
    if (type(service_before) is not ExternalAuthorityHead
            or service_before.run_id != pins.run_id
            or service_before.provider_id != pins.provider_id):
        raise Q8SourceRejection("invalid service head")
    service_before = replace(service_before)
    service_before.validate()
    if (before.witness_counter != pins.minimum_witness_counter
            or before.digest() != pins.minimum_witness_digest):
        raise Q8SourceRejection("witness rollback or unpinned predecessor")
    if (before.epoch != service_before.epoch or before.fence != service_before.fence
            or before.commit_index != service_before.commit_index
            or before.head_sha256 != service_before.head_sha256):
        raise Q8SourceRejection("service/witness preflight disagreement")
    # Preserve the validated, signed predecessor for every post-CAS check.
    # The external port receives a separate copy, so it cannot rewrite our
    # expected index/counter/journal generation via object.__setattr__.
    transport_before = replace(before)
    expected_witness_sha256 = before.digest()
    # A timeout or malformed receipt after this line may hide a real commit.
    try:
        after = port.compare_and_swap_once(transport_change, transport_before)
        after = _check_record(after, pins)
        final_witness = _check_record(port.read_witness_record(pins.run_id), pins)
        final_service = port.read_service_head(pins.run_id)
        if type(final_service) is not ExternalAuthorityHead:
            raise Q8SourceRejection("invalid final service head")
        final_service = replace(final_service)
        final_service.validate()
        if (after != final_witness or after.epoch != before.epoch
                or after.fence != before.fence
                or after.commit_index != before.commit_index + 1
                or after.witness_counter != before.witness_counter + 1
                or after.previous_witness_sha256 != expected_witness_sha256
                or after.state_change_sha256 != expected_change_sha256
                or after.event_id != change.event_id
                or after.journal_generation != before.journal_generation + 1
                or after.journal_sha256 == before.journal_sha256
                or after.head_sha256 == before.head_sha256
                or after.run_id != final_service.run_id
                or final_service.provider_id != pins.provider_id
                or after.epoch != final_service.epoch
                or after.fence != final_service.fence
                or after.commit_index != final_service.commit_index
                or after.head_sha256 != final_service.head_sha256):
            raise Q8SourceRejection("CAS postcondition or external readback mismatch")
        return Q8SourceCASObservation(after.digest(), after.commit_index)
    except Exception as exc:
        raise Q8OutcomeUnknown("Q8 CAS may have committed; reconcile externally") from exc
