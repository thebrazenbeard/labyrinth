"""Opt-in, evaluator-signed development FAIL -> HF successor binding.

This is an operator-owned entry point, not a model-controlled graduation gate.
The independent *development* evaluator must keep the HMAC key outside the
trainee/taskmaster and sign only outcomes it actually measured. A valid MAC
authenticates the receipt's origin, not neural learning, case independence,
or the truthfulness of that evaluator. The caller must durably reject consumed
event IDs and manage key custody before invoking this function; no local
source-only helper can establish those external runtime properties.

The ordinary HF pilot remains an experimental API: its successor route is
explicitly *unqualified research only*. Only callers using
``run_exam_bound_successor`` get a signed development-transition bound. This
in-process interface guard is not a defense against arbitrary malicious code
running inside the Python trainer, and neither path is a private graduation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import re
from pathlib import Path
from typing import Any, Callable

from .hf_pilot import PilotConfig, _authorized_development_resume, run_pilot
from .hf_resume_bridge import ResumeContext, preflight_resume
from .journey import JourneyState, TRAINING, _validate_state
from .regenerative_maze import generate_maze_plan
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.hf-development-transition.v1"
_DOMAIN = b"labyrinth.hf-development-transition.v1\x00"
_ROSTER_DOMAIN = b"labyrinth.development-public-roster.v1\x00"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_FIELDS = frozenset({
    "schema", "event_id", "exam_id", "exam_kind", "exam_status",
    "objective_id", "scope_fingerprint", "from_cycle", "to_cycle",
    "predecessor_report_sha256", "predecessor_lora_value_sha256",
    "resume_authority_ref", "weak_skills", "exam_case_commitment_sha256",
})
MAX_RECEIPT_BYTES = 65_536
MAX_DEVELOPMENT_CASES = 512
_BOUND_RECEIPT_DOMAIN = b"labyrinth.hf-bound-development-reservation.v1\0"
_MAX_FENCE = 2**63 - 1


@dataclass(frozen=True)
class DevelopmentCaseRef:
    """Public development case commitment fields; contains no prompt or gold."""

    case_id: str
    skill: str
    prompt_sha256: str
    structure_key: str


def development_case_commitment(cases: tuple[DevelopmentCaseRef, ...]) -> str:
    """Canonical public roster digest for the *development* signed receipt.

    The independent evaluator must commit this exact digest before scoring;
    this function does not authenticate custody or reveal private gold.
    This is a distinct domain from independent *final* heldout manifests.
    """
    if type(cases) is not tuple or not 1 <= len(cases) <= MAX_DEVELOPMENT_CASES:
        raise ValueError("invalid development public roster")
    case_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for case in cases:
        if type(case) is not DevelopmentCaseRef:
            raise ValueError("invalid development public case record")
        for label, value in (("case ID", case.case_id), ("skill", case.skill),
                             ("structure key", case.structure_key)):
            if type(value) is not str or not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"invalid development {label}")
        if type(case.prompt_sha256) is not str or not _DIGEST.fullmatch(case.prompt_sha256):
            raise ValueError("invalid development prompt digest")
        if case.case_id in case_ids or case.prompt_sha256 in prompt_hashes:
            raise ValueError("duplicate development case or prompt")
        case_ids.add(case.case_id)
        prompt_hashes.add(case.prompt_sha256)
    public = [(case.case_id, "development", case.skill, case.prompt_sha256,
               case.structure_key) for case in sorted(cases, key=lambda item: item.case_id)]
    payload = json.dumps(public, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(_ROSTER_DOMAIN + payload).hexdigest()


@dataclass(frozen=True)
class VerifiedDevelopmentTransition:
    """Authenticated *development* remediation scope, not training proof."""

    event_id: str
    exam_id: str
    weak_skills: tuple[str, ...]
    exam_case_commitment_sha256: str


@dataclass(frozen=True)
class BoundReservationRequest:
    """Typed source challenge bound to THIS successor; not host attestation."""

    event_id: str
    exam_id: str
    run_id: str
    candidate_artifact_sha256: str
    scope_fingerprint: str
    cycle_index: int
    predecessor_report_sha256: str
    predecessor_lora_value_sha256: str
    authority_ref: str
    predecessor_path_sha256: str
    output_path_sha256: str
    epoch: int
    fence: int

    def __post_init__(self) -> None:
        for label in ("event_id", "exam_id", "run_id", "authority_ref"):
            value = getattr(self, label)
            if type(value) is not str or not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"invalid bound reservation {label}")
        for label in ("candidate_artifact_sha256", "scope_fingerprint",
                      "predecessor_report_sha256", "predecessor_lora_value_sha256",
                      "predecessor_path_sha256", "output_path_sha256"):
            value = getattr(self, label)
            if type(value) is not str or not _DIGEST.fullmatch(value):
                raise ValueError(f"invalid bound reservation {label}")
        if (type(self.cycle_index) is not int or not 1 <= self.cycle_index <= 7
                or type(self.epoch) is not int or not 1 <= self.epoch < _MAX_FENCE
                or type(self.fence) is not int or not 0 <= self.fence < _MAX_FENCE):
            raise ValueError("invalid bound reservation cycle or fence")


@dataclass(frozen=True)
class BoundReservationReceipt:
    """Receipt cryptographically bound to the exact requested successor."""

    request: BoundReservationRequest
    sink_id: str
    transaction_id: str
    signature_hex: str
    graduation_authorized: bool = False
    host_custody_verified: bool = False


def bound_reservation_receipt_payload(receipt: BoundReservationReceipt) -> bytes:
    """Canonical HMAC payload; only a separately held sink key lends authority."""
    if type(receipt) is not BoundReservationReceipt or type(receipt.request) is not BoundReservationRequest:
        raise ValueError("typed bound reservation receipt required")
    receipt.request.__post_init__()
    for label in ("sink_id", "transaction_id"):
        value = getattr(receipt, label)
        if type(value) is not str or not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"invalid bound receipt {label}")
    if (receipt.graduation_authorized is not False
            or receipt.host_custody_verified is not False):
        raise ValueError("bound receipt cannot claim graduation or independently hosted custody")
    body = {
        "request": asdict(receipt.request),
        "sink_id": receipt.sink_id,
        "transaction_id": receipt.transaction_id,
        "graduation_authorized": False,
        "host_custody_verified": False,
    }
    return _BOUND_RECEIPT_DOMAIN + (json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("ascii")


@dataclass(frozen=True)
class BoundReservationAuthority:
    """Opt-in typed callback with independently pinned receipt verifier key.

    The host must pin run/candidate/fence, retain the verifier key separately,
    and enforce durable one-shot state. In-process values are not such proof.
    The legacy event-ID-only callback remains accessible/unqualified.
    """

    sink_id: str
    run_id: str
    candidate_artifact_sha256: str
    epoch: int
    fence: int
    receipt_verification_key: bytes = field(repr=False, compare=False)
    reserve: Callable[[BoundReservationRequest], BoundReservationReceipt] = field(
        repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        for label in ("sink_id", "run_id"):
            value = getattr(self, label)
            if type(value) is not str or not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"invalid bound reservation {label}")
        if (type(self.candidate_artifact_sha256) is not str
                or not _DIGEST.fullmatch(self.candidate_artifact_sha256)):
            raise ValueError("invalid bound reservation candidate artifact")
        if (type(self.epoch) is not int or not 1 <= self.epoch < _MAX_FENCE
                or type(self.fence) is not int or not 0 <= self.fence < _MAX_FENCE
                or type(self.receipt_verification_key) is not bytes
                or len(self.receipt_verification_key) < 32 or not callable(self.reserve)):
            raise ValueError("invalid bound reservation authority and key")


def _path_commitment(path: str | Path) -> str:
    if not isinstance(path, (str, Path)):
        raise ValueError("bound reservation requires a valid local path")
    resolved = str(Path(path).expanduser().resolve(strict=False))
    return hashlib.sha256(b"labyrinth.hf-reservation-path.v1\0" +
                          resolved.encode("utf-8")).hexdigest()


def _reserve_bound_successor(
    authority: BoundReservationAuthority, verified: VerifiedDevelopmentTransition,
    scope: TrainingScopeSpec, context: ResumeContext, config: PilotConfig,
    predecessor_run_dir: str | Path, independent_development_key: bytes,
) -> None:
    if type(authority) is not BoundReservationAuthority:
        raise ValueError("typed bound reservation authority required")
    authority.__post_init__()
    if hmac.compare_digest(authority.receipt_verification_key,
                           independent_development_key):
        raise ValueError("bound reservation key must differ from development evaluator key")
    request = BoundReservationRequest(
        event_id=verified.event_id, exam_id=verified.exam_id,
        run_id=authority.run_id,
        candidate_artifact_sha256=authority.candidate_artifact_sha256,
        scope_fingerprint=scope.fingerprint, cycle_index=context.cycle_index,
        predecessor_report_sha256=context.predecessor_report_sha256,
        predecessor_lora_value_sha256=context.predecessor_lora_value_sha256,
        authority_ref=context.authority_ref,
        predecessor_path_sha256=_path_commitment(predecessor_run_dir),
        output_path_sha256=_path_commitment(config.output_dir),
        epoch=authority.epoch, fence=authority.fence,
    )
    receipt = authority.reserve(request)
    if (type(receipt) is not BoundReservationReceipt
            or receipt.request != request or receipt.sink_id != authority.sink_id
            or type(receipt.signature_hex) is not str
            or not _DIGEST.fullmatch(receipt.signature_hex)):
        raise ValueError("bound reservation receipt mismatches this candidate")
    expected = hmac.new(
        authority.receipt_verification_key,
        bound_reservation_receipt_payload(receipt), hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(receipt.signature_hex, expected):
        raise ValueError("bound reservation receipt signature invalid")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate signed receipt field")
        result[key] = value
    return result


def _nonfinite(_value: str) -> None:
    raise ValueError("non-finite receipt number")


def verify_development_transition(
    scope: TrainingScopeSpec,
    context: ResumeContext,
    receipt_bytes: bytes,
    signature_hex: str,
    *,
    independent_development_key: bytes,
) -> VerifiedDevelopmentTransition:
    """Verify the operator-pinned predecessor and signed failed exam.

    No caller-provided weakness/cycle overrides are accepted. Only exact
    signed bytes are parsed; duplicate keys, status substitutions, and a
    private/graduation exam are rejected before any model or files are used.
    """
    if type(scope) is not TrainingScopeSpec or type(context) is not ResumeContext:
        raise ValueError("expected immutable training scope and resume context")
    TrainingScopeSpec.__post_init__(scope)
    ResumeContext.__post_init__(context)
    if context.allow_legacy_unscoped:
        raise ValueError("signed successor requires scoped predecessor, not legacy bootstrap")
    if scope.fingerprint != context.scope_fingerprint:
        raise ValueError("signed successor scope differs from resume scope")
    if (type(independent_development_key) is not bytes
            or len(independent_development_key) < 32):
        raise ValueError("trusted development evaluator requires a 32+ byte secret")
    if (type(receipt_bytes) is not bytes or not 0 < len(receipt_bytes) <= MAX_RECEIPT_BYTES
            or type(signature_hex) is not str or not _DIGEST.fullmatch(signature_hex)):
        raise ValueError("invalid signed development receipt or signature")
    expected = hmac.new(independent_development_key, _DOMAIN + receipt_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_hex):
        raise ValueError("development receipt signature not authenticated")
    try:
        receipt = json.loads(
            receipt_bytes.decode("utf-8"), object_pairs_hook=_unique_pairs,
            parse_constant=_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed development receipt") from exc
    if type(receipt) is not dict or receipt.keys() != _FIELDS:
        raise ValueError("incorrect development receipt fields")
    if (receipt["schema"] != SCHEMA or receipt["exam_kind"] != "development"
            or receipt["exam_status"] != "FAIL"):
        raise ValueError("only authenticated development FAIL may regenerate training")
    for field in ("event_id", "exam_id", "objective_id", "resume_authority_ref"):
        if type(receipt[field]) is not str or not _IDENTIFIER.fullmatch(receipt[field]):
            raise ValueError(f"invalid {field} in development receipt")
    for field in ("scope_fingerprint", "predecessor_report_sha256",
                  "predecessor_lora_value_sha256", "exam_case_commitment_sha256"):
        if type(receipt[field]) is not str or not _DIGEST.fullmatch(receipt[field]):
            raise ValueError(f"invalid {field} in development receipt")
    if (receipt["objective_id"] != scope.objective_id
            or receipt["scope_fingerprint"] != scope.fingerprint
            or type(receipt["from_cycle"]) is not int
            or type(receipt["to_cycle"]) is not int
            or receipt["from_cycle"] != context.cycle_index - 1
            or receipt["to_cycle"] != context.cycle_index
            or receipt["predecessor_report_sha256"] != context.predecessor_report_sha256
            or receipt["predecessor_lora_value_sha256"] != context.predecessor_lora_value_sha256
            or receipt["resume_authority_ref"] != context.authority_ref):
        raise ValueError("development receipt does not bind this predecessor and successor")
    weak = receipt["weak_skills"]
    if (type(weak) is not list or not 1 <= len(weak) <= len(scope.skills)
            or any(type(skill) is not str or skill not in scope.skills for skill in weak)
            or len(set(weak)) != len(weak)):
        raise ValueError("development weaknesses must be nonempty declared skills")
    return VerifiedDevelopmentTransition(
        event_id=receipt["event_id"], exam_id=receipt["exam_id"],
        weak_skills=tuple(weak),
        exam_case_commitment_sha256=receipt["exam_case_commitment_sha256"],
    )


def run_journey_bound_successor(
    config: PilotConfig,
    scope: TrainingScopeSpec,
    predecessor_run_dir: str | Path,
    resume_context: ResumeContext,
    receipt_bytes: bytes,
    signature_hex: str,
    *,
    journey: JourneyState,
    evaluator_precommitment_sha256: str,
    evaluator_public_roster: tuple[DevelopmentCaseRef, ...],
    independent_development_key: bytes,
    reserve_development_event: Callable[[str], bool] | None = None,
    bound_reservation_authority: BoundReservationAuthority | None = None,
) -> dict[str, Any]:
    """Strict, opt-in Journey + signed development successor entry point.

    The caller must obtain ``journey`` from its separately retained state and
    pin the evaluator's public case roster/commitment *before* the exam; the signature alone
    does not establish that temporal or independent custody. Journey checkpoint
    identities must use the exact LoRA tensor-value SHA-256 convention used by
    ``ResumeContext``. Neither local state nor a matching digest is an external
    attestation of training, evaluator independence, or final graduation.

    All checks precede predecessor preflight and event reservation, so invalid
    lineage cannot consume a signed development event or start an optimizer.
    """
    verified = verify_development_transition(
        scope, resume_context, receipt_bytes, signature_hex,
        independent_development_key=independent_development_key,
    )
    # The caller must present the complete evaluator-public case roster, not
    # merely repeat a signed 64-character digest. This freezes case IDs, prompt
    # hashes, skills and structures without transporting private answer keys.
    observed_roster_sha256 = development_case_commitment(evaluator_public_roster)
    if (type(evaluator_precommitment_sha256) is not str
            or not _DIGEST.fullmatch(evaluator_precommitment_sha256)
            or not hmac.compare_digest(evaluator_precommitment_sha256,
                                       verified.exam_case_commitment_sha256)
            or not hmac.compare_digest(observed_roster_sha256,
                                       verified.exam_case_commitment_sha256)):
        raise ValueError("signed exam does not match independently precommitted development cases")
    _validate_state(journey)
    if (journey.scope != scope or journey.scope.fingerprint != resume_context.scope_fingerprint
            or journey.phase != TRAINING
            or journey.cycle_index != resume_context.cycle_index
            or not journey.attempts or not journey.sealed_exam_ids
            or not journey.exam_event_statuses):
        raise ValueError("regenerated Journey does not match declared successor")
    last = journey.attempts[-1]
    if (tuple(sorted(zip(last.exam_case_ids, last.exam_prompt_sha256)))
            != tuple(sorted((case.case_id, case.prompt_sha256)
                            for case in evaluator_public_roster))
            or any(case.skill not in scope.skills for case in evaluator_public_roster)):
        raise ValueError("signed development roster differs from Journey examination cases")
    if (last.cycle_index != resume_context.cycle_index - 1
            or last.exam_status != "FAIL"
            or last.exam_id != verified.exam_id
            or last.weak_skills != verified.weak_skills
            or journey.sealed_exam_ids[-1] != verified.exam_id
            or journey.exam_event_statuses[-1] != "FAIL"
            or not last.exam_case_ids
            or len(last.exam_case_ids) != len(last.exam_prompt_sha256)
            or not set(last.exam_case_ids).issubset(journey.sealed_exam_case_ids)
            or not set(last.exam_prompt_sha256).issubset(journey.sealed_exam_prompt_sha256)
            or not set(last.training_case_ids).issubset(journey.used_training_case_ids)
            or not set(last.training_prompt_sha256).issubset(journey.used_training_prompt_sha256)):
        raise ValueError("signed FAIL does not match the immediately preceding Journey examination")
    if (journey.attempts[0].checkpoint_before_sha256 != journey.baseline_checkpoint_sha256
            or any(left.checkpoint_after_sha256 != right.checkpoint_before_sha256
                   for left, right in zip(journey.attempts, journey.attempts[1:]))
            or any(attempt.checkpoint_before_sha256 == attempt.checkpoint_after_sha256
                   for attempt in journey.attempts)):
        raise ValueError("Journey learner checkpoint chain is discontinuous")
    if (journey.checkpoint_sha256 != resume_context.predecessor_lora_value_sha256
            or last.checkpoint_after_sha256 != journey.checkpoint_sha256):
        raise ValueError("Journey learner checkpoint differs from predecessor LoRA identity")
    if journey.maze != generate_maze_plan(
        scope, resume_context.cycle_index, weak_skills=verified.weak_skills,
    ):
        raise ValueError("regenerated Journey maze does not derive from signed weaknesses")

    return run_exam_bound_successor(
        config, scope, predecessor_run_dir, resume_context, receipt_bytes, signature_hex,
        independent_development_key=independent_development_key,
        reserve_development_event=reserve_development_event,
        bound_reservation_authority=bound_reservation_authority,
    )


def run_exam_bound_successor(
    config: PilotConfig,
    scope: TrainingScopeSpec,
    predecessor_run_dir: str | Path,
    resume_context: ResumeContext,
    receipt_bytes: bytes,
    signature_hex: str,
    *,
    independent_development_key: bytes,
    reserve_development_event: Callable[[str], bool] | None = None,
    bound_reservation_authority: BoundReservationAuthority | None = None,
) -> dict[str, Any]:
    """Run an opt-in neural successor with only signed remediation hints.

    The legacy host-supplied event-ID-only callback does NOT bind the neural
    invocation's run/candidate/scope/cycle, and remains UNQUALIFIED. A callback
    prepared for another run using the same event string may accept this run.
    The optional bound authority path challenges the host with an exact typed
    invocation and rejects a mismatched or unsigned receipt before training.
    Even that path cannot establish independently hosted custody or one-shot
    persistence without external deployment and an immutable authority.
    Read-only predecessor/configuration preflight runs *before* the reservation,
    so a known invalid path or adapter cannot burn an authentic exam event.
    The HF pilot repeats its checks after reservation to detect TOCTOU changes.
    Failures once reserved may have partial effects and must remain consumed
    pending host-owned, durable recovery/reconciliation. Never auto-release a
    consumed event merely because the local process raised an exception.
    Actual neural loading and training remain HF pilot duties.
    """
    verified = verify_development_transition(
        scope, resume_context, receipt_bytes, signature_hex,
        independent_development_key=independent_development_key,
    )
    if (reserve_development_event is None) == (bound_reservation_authority is None):
        raise ValueError("legacy and bound reservation paths are mutually exclusive")
    if (bound_reservation_authority is not None
            and type(bound_reservation_authority) is not BoundReservationAuthority):
        raise ValueError("typed bound reservation authority required")
    if (reserve_development_event is not None
            and not callable(reserve_development_event)):
        raise ValueError("legacy reservation callback must be callable")
    # No model training or output creation happens during this read-only
    # preflight. It rejects malformed/missing local model and predecessor
    # files while the signed development event is still available to retry.
    preflight_resume(config, predecessor_run_dir, resume_context)
    if bound_reservation_authority is not None:
        _reserve_bound_successor(
            bound_reservation_authority, verified, scope, resume_context,
            config, predecessor_run_dir, independent_development_key,
        )
    elif reserve_development_event(verified.event_id) is not True:
        raise ValueError("development transition event is not newly reserved")
    # All successor scope, cycle and weakness arguments derive from signed
    # evidence, never from a separate operator-supplied weakness override.
    authorization = (
        config, scope.fingerprint, resume_context.cycle_index,
        str(Path(predecessor_run_dir).expanduser().resolve(strict=False)),
        resume_context, verified.weak_skills,
    )
    token = _authorized_development_resume.set(authorization)
    try:
        return run_pilot(
            config, scope=scope, cycle_index=resume_context.cycle_index,
            predecessor_run_dir=predecessor_run_dir, resume_context=resume_context,
            weak_skills=verified.weak_skills,
        )
    finally:
        _authorized_development_resume.reset(token)
