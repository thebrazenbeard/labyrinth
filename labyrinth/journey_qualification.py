"""Narrow handoff from a developmental Journey to private graduation custody.

This module has two deliberately separate entry points:

* :func:`prepare_public_qualification_request` may run on the trainer. It
  produces **only** model/scope identity metadata, never hidden cases,
  development hints, credentials, or a claim of graduation.
* :func:`decide_in_evaluator_custody` must run in the independently controlled
  evaluator process with its *precommitted* GraduationPolicy, durable GateState,
  and evaluator-only secret. It delegates every actual admission decision to
  the existing signed :func:`graduation_gate.decide_graduation`.

No function rewrites JourneyState, creates a signing key, signs a receipt,
loads a neural checkpoint, or proves that evaluator custody is independent.
The caller must separately verify actual checkpoint/source identity, the
freshness of the private exam, and trusted GateState durability. A synthetic
signed test fixture is not a real independent examiner.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .graduation_gate import GateDecision, GateState, GraduationPolicy, decide_graduation
from .journey import (
    DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION,
    JourneyState,
    _validate_state,
)


SCHEMA = "labyrinth.journey-qualification-request.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)


@dataclass(frozen=True)
class PublicQualificationRequest:
    """Identity-only handoff; no evaluator authority or private gold."""

    objective_id: str
    scope_fingerprint: str
    candidate_checkpoint_sha256: str
    entrant_checkpoint_sha256: str
    development_cycle_index: int
    development_exam_id: str

    def __post_init__(self) -> None:
        if (type(self.objective_id) is not str
                or _IDENTIFIER.fullmatch(self.objective_id) is None):
            raise ValueError("invalid objective identity")
        if (type(self.development_exam_id) is not str
                or _IDENTIFIER.fullmatch(self.development_exam_id) is None):
            raise ValueError("invalid development examination identity")
        for field in (
            "scope_fingerprint", "candidate_checkpoint_sha256", "entrant_checkpoint_sha256",
        ):
            value = getattr(self, field)
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError(f"invalid {field}")
        if type(self.development_cycle_index) is not int or not 0 <= self.development_cycle_index < 8:
            raise ValueError("development cycle index must be in [0,8)")

    def to_public_dict(self) -> dict[str, str | int]:
        """No evaluation items, scores, weak-skill hints, or secrets."""
        self.__post_init__()
        return {
            "schema": SCHEMA,
            "objective_id": self.objective_id,
            "scope_fingerprint": self.scope_fingerprint,
            "candidate_checkpoint_sha256": self.candidate_checkpoint_sha256,
            "entrant_checkpoint_sha256": self.entrant_checkpoint_sha256,
            "development_cycle_index": self.development_cycle_index,
            "development_exam_id": self.development_exam_id,
            "qualification": "NOT_ESTABLISHED",
        }


def prepare_public_qualification_request(journey: JourneyState) -> PublicQualificationRequest:
    """Start an *external* qualification campaign only after development PASS.

    This is a request for examination, not evidence of qualification. An
    untrusted trainer can fabricate such a request; evaluator custody must
    independently bind the actual candidate checkpoint and training history.
    """
    _validate_state(journey)
    if (journey.phase != DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION
            or not journey.attempts or journey.attempts[-1].exam_status != "PASS"):
        raise ValueError("private qualification requires developmental pass")
    result = PublicQualificationRequest(
        objective_id=journey.scope.objective_id,
        scope_fingerprint=journey.scope.fingerprint,
        candidate_checkpoint_sha256=journey.checkpoint_sha256,
        entrant_checkpoint_sha256=journey.baseline_checkpoint_sha256,
        development_cycle_index=journey.cycle_index,
        development_exam_id=journey.attempts[-1].exam_id,
    )
    result.__post_init__()
    return result


def decide_in_evaluator_custody(
    request: PublicQualificationRequest,
    policy: GraduationPolicy,
    trusted_gate_state: GateState,
    signed_receipt_bytes: bytes | None,
    receipt_signature_hex: str | None,
    *,
    expected_scope_fingerprint: str,
    expected_development_exam_id: str,
    independent_evaluator_key: bytes | None,
) -> GateDecision:
    """Bind public developmental identity to the *existing* private gate.

    Call from evaluator-controlled infrastructure ONLY. The HMAC key is used
    entirely inside this call and never copied into JourneyState, its public
    request, or a development training receipt. The underlying gate requires
    true paired private evidence, fresh-case separation, pinned sources and
    conditions. The `expected_scope_fingerprint` and
    `expected_development_exam_id` must come from the evaluator's own frozen
    precommit, **not** be uncritically copied from the submitted request.
    An unsigned local Minotaur receipt always yields BLOCKED.

    `GraduationPolicy.base_sha256` identifies the pretrained *base* model,
    whereas `entrant_checkpoint_sha256` is the learner's initial checkpoint.
    They need not be the same; the evaluator must independently establish
    their provenance. A public hash alone proves no physical checkpoint.
    """
    if type(request) is not PublicQualificationRequest:
        raise TypeError("expected PublicQualificationRequest")
    request.__post_init__()
    if not isinstance(policy, GraduationPolicy) or not isinstance(trusted_gate_state, GateState):
        raise TypeError("expected frozen GraduationPolicy and evaluator-held GateState")
    GraduationPolicy.__post_init__(policy)
    if (type(expected_scope_fingerprint) is not str
            or _SHA256.fullmatch(expected_scope_fingerprint) is None):
        return GateDecision("BLOCKED", ("missing trusted predeclared scope binding",), trusted_gate_state)
    if (type(expected_development_exam_id) is not str
            or _IDENTIFIER.fullmatch(expected_development_exam_id) is None):
        return GateDecision("BLOCKED", ("missing trusted developmental exam binding",), trusted_gate_state)
    if (request.scope_fingerprint != expected_scope_fingerprint
            or request.development_exam_id != expected_development_exam_id
            or request.objective_id != policy.objective_id
            or request.candidate_checkpoint_sha256 != policy.trainee_sha256):
        return GateDecision(
            "BLOCKED", ("qualification request does not match pinned scope, candidate and objective",),
            trusted_gate_state,
        )
    # Do not inspect or leak the signed receipt. This gate already checks
    # all strict private-trial fields, HMAC, nonce, replay and status rules.
    return decide_graduation(
        policy, trusted_gate_state, signed_receipt_bytes, receipt_signature_hex,
        independent_evaluator_key=independent_evaluator_key,
    )
