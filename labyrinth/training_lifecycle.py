"""Fail-closed routing for Minotaur trainee training and external re-exams.

This module records public, objective feedback and the specific remediation it
requires.  It has no scoring threshold that can authorize an exit.  The only
transition to ``CUSTODY_AUTHORIZED`` is a ``GRADUATED`` result returned by the
existing atomically persisted graduation custody gate.

The resulting lifecycle state is operational routing data, not a portable
release credential: a release operator must retain and verify the gate's
independent custody database.  In particular this module never accepts a
boolean, an advisory/neural score, or a caller-created ``GateDecision``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

from .graduation_gate import GateDecision, GraduationPolicy
from .graduation_state import decide_campaign


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_PHASES = frozenset({"CHALLENGES", "TARGETED_RETRAINING", "EXTERNAL_REEXAM",
                     "REEXAM_EXHAUSTED", "CUSTODY_AUTHORIZED"})


@dataclass(frozen=True)
class ObjectiveFeedback:
    """A bounded checker result used only to choose training remediation."""

    challenge_id: str
    score: float
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.challenge_id, str) or not _ID.fullmatch(self.challenge_id):
            raise ValueError("invalid challenge ID")
        if (type(self.score) not in (int, float) or not math.isfinite(self.score)
                or not 0 <= self.score <= 1):
            raise ValueError("objective score must be finite and in [0, 1]")
        if (not isinstance(self.targets, tuple) or any(type(item) is not str
                or not _ID.fullmatch(item) for item in self.targets)
                or len(set(self.targets)) != len(self.targets)):
            raise ValueError("targets must be unique bounded IDs")


@dataclass(frozen=True)
class LifecycleState:
    """Replayable training routing state; never use it as an exit credential."""

    phase: str = "CHALLENGES"
    public_feedback: tuple[ObjectiveFeedback, ...] = ()
    pending_targets: tuple[str, ...] = ()
    consumed_receipt_sha256s: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError("invalid lifecycle phase")
        if (not isinstance(self.public_feedback, tuple)
                or any(not isinstance(item, ObjectiveFeedback) for item in self.public_feedback)
                or len({item.challenge_id for item in self.public_feedback}) != len(self.public_feedback)):
            raise ValueError("feedback must contain unique objective challenges")
        if (not isinstance(self.pending_targets, tuple) or any(type(item) is not str
                or not _ID.fullmatch(item) for item in self.pending_targets)
                or len(set(self.pending_targets)) != len(self.pending_targets)):
            raise ValueError("invalid pending remediation targets")
        if (type(self.consumed_receipt_sha256s) is not frozenset
                or any(type(item) is not str or not re.fullmatch(r"[0-9a-f]{64}", item)
                       for item in self.consumed_receipt_sha256s)):
            raise ValueError("invalid consumed exam evidence")
        if self.phase == "TARGETED_RETRAINING" and not self.pending_targets:
            raise ValueError("targeted retraining requires a target")
        if self.phase in {"CHALLENGES", "EXTERNAL_REEXAM", "REEXAM_EXHAUSTED",
                          "CUSTODY_AUTHORIZED"} and self.pending_targets:
            raise ValueError("only retraining may retain pending targets")


@dataclass(frozen=True)
class ReexamResult:
    """Gate output plus the next routing state; neither is a self-signed release."""

    state: LifecycleState
    decision: GateDecision


def record_public_bounded_check(state: LifecycleState, feedback: ObjectiveFeedback) -> LifecycleState:
    """Record public checker feedback and route only to remediation or re-exam.

    Even a score of 1.0 only permits requesting an external private re-exam;
    it cannot produce ``CUSTODY_AUTHORIZED``.
    """
    if not isinstance(state, LifecycleState) or not isinstance(feedback, ObjectiveFeedback):
        raise TypeError("expected lifecycle state and objective feedback")
    if any(item.challenge_id == feedback.challenge_id for item in state.public_feedback):
        raise ValueError("replayed objective challenge feedback")
    if state.phase != "CHALLENGES":
        raise ValueError("objective checks are accepted only during challenges")
    history = state.public_feedback + (feedback,)
    if feedback.targets:
        return LifecycleState("TARGETED_RETRAINING", history, feedback.targets,
                              state.consumed_receipt_sha256s)
    return LifecycleState("EXTERNAL_REEXAM", history, (), state.consumed_receipt_sha256s)


def complete_targeted_retraining(state: LifecycleState, target: str) -> LifecycleState:
    """Record a remediation pass; no claimed score can skip private re-exam."""
    if not isinstance(state, LifecycleState):
        raise TypeError("expected lifecycle state")
    if state.phase != "TARGETED_RETRAINING":
        raise ValueError("retraining is not currently requested")
    if type(target) is not str or target not in state.pending_targets:
        raise ValueError("unknown remediation target")
    pending = tuple(item for item in state.pending_targets if item != target)
    phase = "EXTERNAL_REEXAM" if not pending else "TARGETED_RETRAINING"
    return LifecycleState(phase, state.public_feedback, pending, state.consumed_receipt_sha256s)


def submit_external_reexam(
    state: LifecycleState,
    custody_db: Path,
    policy: GraduationPolicy,
    receipt_bytes: bytes | None,
    signature_hex: str | None,
    *,
    independent_evaluator_key: bytes,
    state_key: bytes,
) -> ReexamResult:
    """Submit evidence to the sole exit authority: external gate plus custody.

    No key is generated here.  Invalid/missing/forged evidence leaves this
    routing state unchanged.  A valid failed exam routes back to targeted
    retraining; exhausted attempts remain explicitly blocked.
    """
    if not isinstance(state, LifecycleState):
        raise TypeError("expected lifecycle state")
    if state.phase != "EXTERNAL_REEXAM":
        raise ValueError("external re-exam is not currently requested")
    digest = hashlib.sha256(receipt_bytes).hexdigest() if type(receipt_bytes) is bytes else None
    if digest is not None and digest in state.consumed_receipt_sha256s:
        # Let the custody gate produce the canonical fail-closed decision, but
        # do not permit this local routing layer to revive a consumed attempt.
        decision = decide_campaign(custody_db, policy, receipt_bytes, signature_hex,
                                   independent_evaluator_key=independent_evaluator_key,
                                   state_key=state_key)
        return ReexamResult(state, decision)
    decision = decide_campaign(custody_db, policy, receipt_bytes, signature_hex,
                               independent_evaluator_key=independent_evaluator_key,
                               state_key=state_key)
    if decision.disposition == "BLOCKED":
        return ReexamResult(state, decision)
    consumed = state.consumed_receipt_sha256s | ({digest} if digest is not None else set())
    if decision.disposition == "GRADUATED":
        next_state = LifecycleState("CUSTODY_AUTHORIZED", state.public_feedback, (), consumed)
    elif decision.disposition == "EXHAUSTED":
        next_state = LifecycleState("REEXAM_EXHAUSTED", state.public_feedback, (), consumed)
    else:
        next_state = LifecycleState("TARGETED_RETRAINING", state.public_feedback,
                                    ("external-reexam-failed",), consumed)
    return ReexamResult(next_state, decision)
