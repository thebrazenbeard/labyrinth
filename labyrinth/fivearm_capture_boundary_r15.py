"""Optional Ed25519 permit/capture chain for the FIVE-arm public source auditor.

This is a consumer-side verification boundary, NOT a capture service, clock,
append-only log, pretrained model probe, or independent examiner. Only a
separately hosted authority enforcing one permit and one capture per trial,
holding signing keys outside the trainer rollback domain, can establish
one-shot custody. The injected port cannot authenticate its own deployment.

In particular, cryptographically ordering a signed permit before a signed
report is not proof that the permit existed before the physical run. That
requires a host-owned real-time append-only witness, not this source module.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import re
from pathlib import Path
from typing import Callable, Protocol, TypeAlias

from scripts import qualify_maze_multiseed_protocol_cpu as protocol

from .fivearm_campaign_admission_r13 import (
    FiveArmCampaignEvidence, FiveArmCampaignError,
    audit_fivearm_campaign_from_disk,
)


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_HEX64 = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_HEX128 = re.compile(r"[a-f0-9]{128}\Z", re.ASCII)
_MAX_SEQ = 2**63 - 1
_MAX_GRID = 4096
_CHAIN_DOMAIN = b"labyrinth.fivearm-capture-chain.v1\0"
_PERMIT_DOMAIN = b"labyrinth.fivearm-trial-permit.v1\0"
_CAPTURE_DOMAIN = b"labyrinth.fivearm-trial-capture.v1\0"


class CaptureBoundaryError(ValueError):
    """Capture receipts were absent, contradictory, or unverified."""


def _id(value: object, name: str) -> None:
    if type(value) is not str or not _ID.fullmatch(value):
        raise CaptureBoundaryError(f"invalid {name}")


def _hex(value: object, name: str, *, signature: bool = False) -> None:
    if type(value) is not str or not (
        _HEX128 if signature else _HEX64
    ).fullmatch(value):
        raise CaptureBoundaryError(f"invalid {name}")


def _seq(value: object, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value < _MAX_SEQ:
        raise CaptureBoundaryError(f"invalid {name}")


def _canonical(value: dict[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


@dataclass(frozen=True)
class CaptureHead:
    provider_id: str
    campaign_id: str
    sequence: int
    head_sha256: str

    def validate(self) -> None:
        _id(self.provider_id, "provider")
        _id(self.campaign_id, "campaign")
        _seq(self.sequence, "capture head sequence")
        _hex(self.head_sha256, "capture head")


@dataclass(frozen=True)
class CapturePins:
    """Provisioned by an independent host, never by the learner/provider."""

    provider_id: str
    campaign_id: str
    plan_sha256: str
    initial_sequence: int
    initial_head_sha256: str
    permit_public_key: bytes
    capture_public_key: bytes

    def validate(self) -> None:
        _id(self.provider_id, "pinned provider")
        _id(self.campaign_id, "pinned campaign")
        _hex(self.plan_sha256, "pinned pre-outcome plan")
        _hex(self.initial_head_sha256, "initial head")
        _seq(self.initial_sequence, "initial sequence")
        if (type(self.permit_public_key) is not bytes
                or type(self.capture_public_key) is not bytes
                or len(self.permit_public_key) != 32
                or len(self.capture_public_key) != 32
                or hmac.compare_digest(self.permit_public_key,
                                       self.capture_public_key)):
            raise CaptureBoundaryError("distinct pinned issuer and capture Ed25519 keys required")


@dataclass(frozen=True)
class PermitReceipt:
    provider_id: str
    campaign_id: str
    run_name: str
    plan_sha256: str
    attempt_id: str
    attempt_number: int
    sequence: int
    previous_head_sha256: str
    new_head_sha256: str
    signature_hex: str

    def validate(self) -> None:
        for name in ("provider_id", "campaign_id", "run_name", "attempt_id"):
            _id(getattr(self, name), name)
        for name in ("plan_sha256", "previous_head_sha256", "new_head_sha256"):
            _hex(getattr(self, name), name)
        if type(self.attempt_number) is not int or self.attempt_number != 1:
            raise CaptureBoundaryError("exactly one initial attempt permitted")
        _seq(self.sequence, "permit sequence", minimum=1)
        _hex(self.signature_hex, "permit signature", signature=True)

    def signed_payload(self) -> bytes:
        self.validate()
        return _PERMIT_DOMAIN + _canonical({
            key: getattr(self, key) for key in self.__dataclass_fields__
            if key != "signature_hex"
        })


@dataclass(frozen=True)
class CaptureReceipt:
    provider_id: str
    campaign_id: str
    run_name: str
    attempt_id: str
    permit_sha256: str
    report_sha256: str
    sequence: int
    previous_head_sha256: str
    new_head_sha256: str
    signature_hex: str

    def validate(self) -> None:
        for name in ("provider_id", "campaign_id", "run_name", "attempt_id"):
            _id(getattr(self, name), name)
        for name in ("permit_sha256", "report_sha256", "previous_head_sha256",
                     "new_head_sha256"):
            _hex(getattr(self, name), name)
        _seq(self.sequence, "capture sequence", minimum=1)
        _hex(self.signature_hex, "capture signature", signature=True)

    def signed_payload(self) -> bytes:
        self.validate()
        return _CAPTURE_DOMAIN + _canonical({
            key: getattr(self, key) for key in self.__dataclass_fields__
            if key != "signature_hex"
        })


TrialRecord: TypeAlias = PermitReceipt | CaptureReceipt


def chained_head_sha256(record: TrialRecord) -> str:
    """Public deterministic record-chain digest (NOT a timestamp)."""
    if type(record) not in (PermitReceipt, CaptureReceipt):
        raise CaptureBoundaryError("typed capture-chain record required")
    record.validate()
    kind = b"permit" if type(record) is PermitReceipt else b"capture"
    body = _canonical({
        key: getattr(record, key) for key in record.__dataclass_fields__
        if key not in ("new_head_sha256", "signature_hex")
    })
    return sha256(_CHAIN_DOMAIN + kind + b"\0" + body).hexdigest()


class CaptureAuthorityPort(Protocol):
    """Read-only facade to an independently maintained immutable host log."""

    def read_head(self, campaign_id: str) -> CaptureHead: ...

    def list_run_names(self, campaign_id: str) -> tuple[str, ...]: ...

    def read_run_receipts(self, campaign_id: str, run_name: str) -> tuple[PermitReceipt, CaptureReceipt]: ...


@dataclass(frozen=True)
class CaptureSourceEvidence:
    status: str
    captured_runs: int
    final_head: CaptureHead
    audit: FiveArmCampaignEvidence
    limitations: tuple[str, ...]
    independent_custody_verified: bool = False
    physical_model_verified: bool = False
    graduation_authorized: bool = False


def _verify_ed25519(public_key: bytes, payload: bytes, signature_hex: str) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
    except ImportError as exc:
        raise CaptureBoundaryError("cryptography Ed25519 verification unavailable") from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(signature_hex), payload)
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise CaptureBoundaryError("external capture signer signature invalid") from exc


def verify_captured_fivearm_campaign(
    *,
    spec: protocol.ProtocolSpec,
    campaign_root: Path,
    pins: CapturePins,
    port: CaptureAuthorityPort,
    expected_runner_sha256: str,
    expected_protocol_sha256: str,
    expected_pretrained_base_sha256: str,
    preoutcome_witness: bytes,
    verify_external_preoutcome_witness: Callable[[bytes, str], bool],
) -> CaptureSourceEvidence:
    """Require complete signed pre-permits and captured reports before source audit.

    Source verification only. No callback, signature or synthetic port establishes
    actual external custody or timing; host deployment must be qualified apart.
    """
    if type(pins) is not CapturePins:
        raise CaptureBoundaryError("typed host capture pins required")
    pins.validate()
    if type(spec) is not protocol.ProtocolSpec or spec.preregistered_controls is not True:
        raise CaptureBoundaryError("five-arm preregistered protocol required")
    try:
        plan = protocol.build_plan(spec)
        if sha256(protocol._canonical(plan)).hexdigest() != pins.plan_sha256:
            raise CaptureBoundaryError("capture pins do not match precommitted plan")
        jobs = plan["manifest"]["jobs"]
        expected_runs = {job["run_name"] for job in jobs}
        if not 1 <= len(jobs) <= _MAX_GRID or len(expected_runs) != len(jobs):
            raise CaptureBoundaryError("duplicate or excessive campaign grid")
        initial_read = port.read_head(pins.campaign_id)
        listed = port.list_run_names(pins.campaign_id)
        if (type(initial_read) is not CaptureHead
                or type(listed) is not tuple or set(listed) != expected_runs
                or len(listed) != len(expected_runs)):
            raise CaptureBoundaryError("incomplete host-owned capture registry")
        initial_read.validate()
        records: list[TrialRecord] = []
        report_hashes: dict[str, str] = {}
        attempts: set[str] = set()
        for run_name in sorted(expected_runs):
            pair = port.read_run_receipts(pins.campaign_id, run_name)
            if (type(pair) is not tuple or len(pair) != 2
                    or type(pair[0]) is not PermitReceipt
                    or type(pair[1]) is not CaptureReceipt):
                raise CaptureBoundaryError("one signed permit and capture per run required")
            permit, capture = pair
            permit.validate()
            capture.validate()
            if (permit.provider_id != pins.provider_id
                    or capture.provider_id != pins.provider_id
                    or permit.campaign_id != pins.campaign_id
                    or capture.campaign_id != pins.campaign_id
                    or permit.run_name != run_name or capture.run_name != run_name
                    or permit.plan_sha256 != pins.plan_sha256
                    or permit.attempt_id != capture.attempt_id
                    or permit.attempt_id in attempts
                    or capture.sequence <= permit.sequence
                    or not hmac.compare_digest(
                        capture.permit_sha256,
                        sha256(permit.signed_payload()).hexdigest())):
                raise CaptureBoundaryError("run, permit, attempt or report lineage changed")
            attempts.add(permit.attempt_id)
            records.extend((permit, capture))
            report_hashes[run_name] = capture.report_sha256
        if len({record.sequence for record in records}) != 2 * len(jobs):
            raise CaptureBoundaryError("duplicate external capture sequence")
        previous = pins.initial_head_sha256
        for index, record in enumerate(sorted(records, key=lambda x: x.sequence),
                                       start=pins.initial_sequence + 1):
            if (record.sequence != index
                    or not hmac.compare_digest(record.previous_head_sha256, previous)
                    or not hmac.compare_digest(record.new_head_sha256,
                                               chained_head_sha256(record))):
                raise CaptureBoundaryError("missing, reordered or rolled-back host chain")
            _verify_ed25519(pins.permit_public_key if type(record) is PermitReceipt
                            else pins.capture_public_key,
                            record.signed_payload(), record.signature_hex)
            previous = record.new_head_sha256
        expected_head = CaptureHead(pins.provider_id, pins.campaign_id,
                                    pins.initial_sequence + len(records), previous)
        final_read = port.read_head(pins.campaign_id)
        if (type(final_read) is not CaptureHead
                or initial_read != expected_head or final_read != expected_head):
            raise CaptureBoundaryError("external capture readback unavailable or stale")
    except CaptureBoundaryError:
        raise
    except Exception as exc:
        # A failed remote read may be a torn ledger/snapshot. It is never
        # permission to use a locally supplied report digest as fallback.
        raise CaptureBoundaryError("external capture authority readback unavailable") from exc

    try:
        audited = audit_fivearm_campaign_from_disk(
            spec=spec, campaign_root=campaign_root,
            expected_plan_sha256=pins.plan_sha256,
            expected_runner_sha256=expected_runner_sha256,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_pretrained_base_sha256=expected_pretrained_base_sha256,
            expected_report_sha256_by_run=report_hashes,
            preoutcome_witness=preoutcome_witness,
            verify_external_preoutcome_witness=verify_external_preoutcome_witness,
        )
    except FiveArmCampaignError as exc:
        raise CaptureBoundaryError("captured reports fail offline five-arm audit") from exc
    except Exception as exc:
        # The independently supplied pre-outcome witness callback may lose
        # transport or raise an unexpected runtime error. Never leak a raw
        # callback exception as an alternate audit outcome or bypass custody.
        raise CaptureBoundaryError("external witness or source audit unavailable") from exc
    try:
        after_audit = port.read_head(pins.campaign_id)
    except Exception as exc:
        raise CaptureBoundaryError("external capture head unavailable after audit") from exc
    if type(after_audit) is not CaptureHead or after_audit != expected_head:
        raise CaptureBoundaryError("external capture head changed during source audit")
    return CaptureSourceEvidence(
        status="FIVE_ARM_CAPTURE_SOURCE_ONLY",
        captured_runs=len(jobs),
        final_head=expected_head,
        audit=audited,
        limitations=(
            "External port and role-key independence are not attested by this client",
            "Signature order does not prove real-time permit-before-training custody",
            "No physical neural execution, blinded evaluator or final examination verified",
            "No host deployment/rollback-domain proof; no graduation authorization",
        ),
    )
