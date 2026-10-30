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

from dataclasses import dataclass
import hashlib
import hmac
import json
import re
from pathlib import Path
from typing import Any, Callable

from .hf_pilot import PilotConfig, _authorized_development_resume, run_pilot
from .hf_resume_bridge import ResumeContext, preflight_resume
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.hf-development-transition.v1"
_DOMAIN = b"labyrinth.hf-development-transition.v1\x00"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_FIELDS = frozenset({
    "schema", "event_id", "exam_id", "exam_kind", "exam_status",
    "objective_id", "scope_fingerprint", "from_cycle", "to_cycle",
    "predecessor_report_sha256", "predecessor_lora_value_sha256",
    "resume_authority_ref", "weak_skills", "exam_case_commitment_sha256",
})
MAX_RECEIPT_BYTES = 65_536


@dataclass(frozen=True)
class VerifiedDevelopmentTransition:
    """Authenticated *development* remediation scope, not training proof."""

    event_id: str
    exam_id: str
    weak_skills: tuple[str, ...]


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
    reserve_development_event: Callable[[str], bool],
) -> dict[str, Any]:
    """Run an opt-in neural successor with only signed remediation hints.

    The host-supplied reservation callback must atomically persist consumption
    of the event ID before training, returning exactly True only on first use.
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
    if not callable(reserve_development_event):
        raise ValueError("trusted durable event reservation callback is required")
    # No model training or output creation happens during this read-only
    # preflight. It rejects malformed/missing local model and predecessor
    # files while the signed development event is still available to retry.
    preflight_resume(config, predecessor_run_dir, resume_context)
    if reserve_development_event(verified.event_id) is not True:
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
