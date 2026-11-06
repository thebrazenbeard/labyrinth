"""Source-only TRAIN exposure completeness contract for sealed examination.

The trainer must not hand an examiner a conveniently incomplete
TRAIN overlap inventory. This module compares an ordered observed visit stream
against a separately signed, *previously pinned* full TRAIN schedule before
constructing ``TrainingOverlapInventory`` for the sealed exam's alias guards.

The caller must obtain both an independently held schedule commitment and a
recorder-signed capture from outside trainer control. Passing keys to these
Python functions does NOT establish such custody, prove neural optimizer
execution, prove an inventory covers undeclared training, or authorize
graduation. Those facts require a separately qualified physical service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import re
from typing import Any

from .arithmetic_semantic_case_key import (
    UnsupportedArithmeticCase, arithmetic_semantic_case_key,
)
from .retention_family_registry import (
    canonical_family_id, canonical_prompt_alias_sha256,
)
from .sealed_semantic_exam_protocol import (
    TrainingOverlapInventory, training_inventory_sha256,
)


SCHEMA = "labyrinth.train-exposure-attestation.v1"
_PLAN_DOMAIN = b"labyrinth.train-exposure-plan.v1\0"
_CAPTURE_DOMAIN = b"labyrinth.train-exposure-capture.v1\0"
_STREAM_DOMAIN = b"labyrinth.train-exposure-stream.v1\0"
_FAMILY_DOMAIN = b"labyrinth.train-exposure-family-roster.v1\0"
_EVENT_DOMAIN = b"labyrinth.train-exposure-event.v1\0"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_EXPOSURES = 100_000


class TrainingExposureAttestationError(ValueError):
    """A precommit, capture or complete TRAIN roster requirement failed."""


@dataclass(frozen=True)
class TrainingWitnessContext:
    logical_run_id: str
    objective_id: str
    scope_sha256: str
    base_checkpoint_sha256: str
    source_sha256: str
    fence: int
    cycles: int
    steps_per_cycle: int
    visits_per_step: int


@dataclass(frozen=True)
class TrainExposureEvent:
    ordinal: int
    cycle: int
    step: int
    visit: int
    event_id: str
    case_id: str
    family_id: str
    semantic_key: str
    prompt: str


@dataclass(frozen=True)
class SignedTrainSchedule:
    receipt_bytes: bytes
    signature_hex: str
    independently_hosted: bool = False


@dataclass(frozen=True)
class SignedTrainCapture:
    receipt_bytes: bytes
    signature_hex: str
    physically_authenticated: bool = False


@dataclass(frozen=True)
class VerifiedTrainInventory:
    inventory: TrainingOverlapInventory
    inventory_sha256: str
    schedule_sha256: str
    capture_sha256: str
    checkpoint_sha256: str
    independent_host_verified: bool = False
    physical_execution_authenticated: bool = False
    graduation_authorized: bool = False


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise TrainingExposureAttestationError(f"invalid {name} SHA-256")
    return value


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or _IDENT.fullmatch(value) is None:
        raise TrainingExposureAttestationError(f"invalid {name} identifier")
    return value


def _key(value: Any, name: str) -> bytes:
    if type(value) is not bytes or not 32 <= len(value) <= 4096:
        raise TrainingExposureAttestationError(f"independently held {name} key required")
    return value


def _separate_keys(schedule_key: bytes, recorder_key: bytes) -> None:
    _key(schedule_key, "schedule")
    _key(recorder_key, "recorder")
    if hmac.compare_digest(schedule_key, recorder_key):
        raise TrainingExposureAttestationError("schedule and recorder keys must differ")


def _validate_context(context: TrainingWitnessContext) -> None:
    if type(context) is not TrainingWitnessContext:
        raise TrainingExposureAttestationError("typed TRAIN witness context required")
    _identifier(context.logical_run_id, "run")
    _identifier(context.objective_id, "objective")
    for name in ("scope_sha256", "base_checkpoint_sha256", "source_sha256"):
        _digest(getattr(context, name), name)
    if type(context.fence) is not int or not 0 < context.fence < 2**63:
        raise TrainingExposureAttestationError("positive monotonic fence required")
    for name in ("cycles", "steps_per_cycle", "visits_per_step"):
        value = getattr(context, name)
        if type(value) is not int or not 0 < value <= _MAX_EXPOSURES:
            raise TrainingExposureAttestationError(f"invalid scheduled {name}")
    expected = context.cycles * context.steps_per_cycle * context.visits_per_step
    if expected > _MAX_EXPOSURES:
        raise TrainingExposureAttestationError("TRAIN schedule exceeds exposure bound")


def _inventory_from_complete_stream(
    context: TrainingWitnessContext, events: tuple[TrainExposureEvent, ...],
) -> TrainingOverlapInventory:
    _validate_context(context)
    expected = context.cycles * context.steps_per_cycle * context.visits_per_step
    if (type(events) is not tuple or len(events) != expected
            or any(type(row) is not TrainExposureEvent for row in events)):
        raise TrainingExposureAttestationError("incomplete scheduled TRAIN visit roster")
    case_inventory: dict[str, tuple[str, str, str]] = {}
    event_ids: set[str] = set()
    aliases: dict[str, str] = {}
    semantic_cases: dict[str, str] = {}
    arithmetic_cases: dict[str, str] = {}
    families: set[str] = set()
    for ordinal, row in enumerate(events):
        position = (ordinal // (context.steps_per_cycle * context.visits_per_step),
                    (ordinal // context.visits_per_step) % context.steps_per_cycle,
                    ordinal % context.visits_per_step)
        if ((type(row.ordinal) is not int or row.ordinal != ordinal)
                or any(type(value) is not int for value in
                       (row.cycle, row.step, row.visit))
                or (row.cycle, row.step, row.visit) != position):
            raise TrainingExposureAttestationError("missing, reordered or duplicate TRAIN visit")
        for name in ("event_id", "case_id", "semantic_key"):
            _identifier(getattr(row, name), name)
        if row.event_id in event_ids:
            raise TrainingExposureAttestationError("duplicate TRAIN event identity")
        event_ids.add(row.event_id)
        try:
            family = canonical_family_id(row.family_id)
            alias = canonical_prompt_alias_sha256(row.prompt)
        except ValueError as exc:
            raise TrainingExposureAttestationError("invalid TRAIN family or prompt") from exc
        if family != row.family_id:
            raise TrainingExposureAttestationError("noncanonical TRAIN family")
        spec = (row.family_id, row.semantic_key, alias)
        if row.case_id in case_inventory and case_inventory[row.case_id] != spec:
            raise TrainingExposureAttestationError("TRAIN case identity changed between visits")
        case_inventory[row.case_id] = spec
        if (alias in aliases and aliases[alias] != row.case_id
                or row.semantic_key in semantic_cases
                and semantic_cases[row.semantic_key] != row.case_id):
            raise TrainingExposureAttestationError("TRAIN alias reused under distinct case IDs")
        aliases[alias] = row.case_id
        semantic_cases[row.semantic_key] = row.case_id
        families.add(family)
        try:
            arithmetic = arithmetic_semantic_case_key(row.prompt)
        except UnsupportedArithmeticCase:
            arithmetic = None
        if arithmetic is not None:
            if arithmetic in arithmetic_cases and arithmetic_cases[arithmetic] != row.case_id:
                raise TrainingExposureAttestationError(
                    "TRAIN arithmetic alias reused under distinct case IDs")
            arithmetic_cases[arithmetic] = row.case_id
    return TrainingOverlapInventory(
        family_ids=tuple(sorted(families)),
        prompt_alias_sha256s=tuple(sorted(aliases)),
        semantic_keys=tuple(sorted(semantic_cases)),
        arithmetic_semantic_keys=tuple(sorted(arithmetic_cases)),
    )


def _stream_sha256(events: tuple[TrainExposureEvent, ...]) -> str:
    return sha256(_STREAM_DOMAIN + _canonical([asdict(row) for row in events])).hexdigest()


def _family_roster_sha256(events: tuple[TrainExposureEvent, ...]) -> str:
    """Bind family at *every* ordered visit, not just the unique set."""
    return sha256(_FAMILY_DOMAIN + _canonical([
        (row.ordinal, row.family_id) for row in events
    ])).hexdigest()


def _chain_sha256(events: tuple[TrainExposureEvent, ...]) -> str:
    head = "0" * 64
    for row in events:
        head = sha256(_EVENT_DOMAIN + head.encode("ascii") +
                      _canonical(asdict(row))).hexdigest()
    return head


def _signed_body(receipt: bytes, signature: str, *, key: bytes, domain: bytes,
                 name: str) -> dict[str, Any]:
    _key(key, name)
    if (type(receipt) is not bytes or not 0 < len(receipt) < 1024 * 1024
            or type(signature) is not str or _HEX.fullmatch(signature) is None
            or not hmac.compare_digest(hmac.new(key, domain + receipt, sha256).hexdigest(),
                                       signature)):
        raise TrainingExposureAttestationError(f"{name} signature or receipt invalid")
    try:
        body = json.loads(receipt)
        if type(body) is not dict or _canonical(body) != receipt:
            raise ValueError("noncanonical receipt")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrainingExposureAttestationError(f"{name} receipt is noncanonical") from exc
    return body


def precommit_training_schedule(
    context: TrainingWitnessContext, scheduled: tuple[TrainExposureEvent, ...],
    *, schedule_key: bytes,
) -> SignedTrainSchedule:
    """Signing is a source operation, NOT evidence that a host owns the key."""
    _key(schedule_key, "schedule")
    inventory = _inventory_from_complete_stream(context, scheduled)
    body = {
        "schema": SCHEMA, "type": "schedule",
        "context": asdict(context), "count": len(scheduled),
        "ordered_stream_sha256": _stream_sha256(scheduled),
        "ordered_family_roster_sha256": _family_roster_sha256(scheduled),
        "train_inventory_sha256": training_inventory_sha256(inventory),
        "host_custody_qualified": False,
    }
    receipt = _canonical(body)
    return SignedTrainSchedule(receipt, hmac.new(
        schedule_key, _PLAN_DOMAIN + receipt, sha256).hexdigest())


def capture_training_visits(
    context: TrainingWitnessContext, observed: tuple[TrainExposureEvent, ...],
    *, schedule_sha256: str, checkpoint_sha256: str, recorder_key: bytes,
) -> SignedTrainCapture:
    """Seal a claimed visit record; verified custody remains external work."""
    _key(recorder_key, "recorder")
    _digest(schedule_sha256, "schedule")
    _digest(checkpoint_sha256, "checkpoint")
    inventory = _inventory_from_complete_stream(context, observed)
    body = {
        "schema": SCHEMA, "type": "capture",
        "context": asdict(context), "count": len(observed),
        "schedule_sha256": schedule_sha256,
        "ordered_stream_sha256": _stream_sha256(observed),
        "ordered_family_roster_sha256": _family_roster_sha256(observed),
        "chain_head_sha256": _chain_sha256(observed),
        "train_inventory_sha256": training_inventory_sha256(inventory),
        "checkpoint_sha256": checkpoint_sha256,
        "physical_execution_authenticated": False,
    }
    receipt = _canonical(body)
    return SignedTrainCapture(receipt, hmac.new(
        recorder_key, _CAPTURE_DOMAIN + receipt, sha256).hexdigest())


def verify_complete_train_inventory(
    *, context: TrainingWitnessContext,
    scheduled: tuple[TrainExposureEvent, ...],
    observed: tuple[TrainExposureEvent, ...],
    signed_schedule: SignedTrainSchedule,
    signed_capture: SignedTrainCapture,
    externally_pinned_schedule_sha256: str,
    expected_checkpoint_sha256: str,
    schedule_key: bytes, recorder_key: bytes,
) -> VerifiedTrainInventory:
    """Return a complete *source-matched* TRAIN roster for the sealed exam.

    External custody must verify that ``externally_pinned_schedule_sha256``
    predates training and that recorder observations came from actual updates.
    Neither property can be checked inside this trainer-capable process.
    """
    _separate_keys(schedule_key, recorder_key)
    if (type(signed_schedule) is not SignedTrainSchedule
            or type(signed_capture) is not SignedTrainCapture
            or signed_schedule.independently_hosted
            or signed_capture.physically_authenticated):
        raise TrainingExposureAttestationError("unqualified source receipt flags")
    pinned = _digest(externally_pinned_schedule_sha256, "pinned schedule")
    checkpoint = _digest(expected_checkpoint_sha256, "expected checkpoint")
    if sha256(signed_schedule.receipt_bytes).hexdigest() != pinned:
        raise TrainingExposureAttestationError("TRAIN schedule differs from prior external pin")
    schedule = _signed_body(signed_schedule.receipt_bytes, signed_schedule.signature_hex,
                            key=schedule_key, domain=_PLAN_DOMAIN, name="schedule")
    capture = _signed_body(signed_capture.receipt_bytes, signed_capture.signature_hex,
                           key=recorder_key, domain=_CAPTURE_DOMAIN, name="capture")
    scheduled_inventory = _inventory_from_complete_stream(context, scheduled)
    observed_inventory = _inventory_from_complete_stream(context, observed)
    stream = _stream_sha256(scheduled)
    inventory_sha = training_inventory_sha256(scheduled_inventory)
    if schedule != {
        "schema": SCHEMA, "type": "schedule",
        "context": asdict(context), "count": len(scheduled),
        "ordered_stream_sha256": stream,
        "ordered_family_roster_sha256": _family_roster_sha256(scheduled),
        "train_inventory_sha256": inventory_sha,
        "host_custody_qualified": False,
    }:
        raise TrainingExposureAttestationError("TRAIN precommit metadata mismatch")
    if (capture != {
        "schema": SCHEMA, "type": "capture",
        "context": asdict(context), "count": len(observed),
        "schedule_sha256": pinned,
        "ordered_stream_sha256": stream,
        "ordered_family_roster_sha256": _family_roster_sha256(observed),
        "chain_head_sha256": _chain_sha256(observed),
        "train_inventory_sha256": inventory_sha,
        "checkpoint_sha256": checkpoint,
        "physical_execution_authenticated": False,
    } or _stream_sha256(observed) != stream
            or observed_inventory != scheduled_inventory):
        raise TrainingExposureAttestationError(
            "TRAIN recorder evidence incomplete, mismatched or stale")
    return VerifiedTrainInventory(
        inventory=observed_inventory, inventory_sha256=inventory_sha,
        schedule_sha256=pinned,
        capture_sha256=sha256(signed_capture.receipt_bytes).hexdigest(),
        checkpoint_sha256=checkpoint,
    )
