"""R25 source-only handoff from admitted training evidence to Minotaur's gate.

This contract never graduates or deploys a model on its own. It checks that
the declared learning intent, admitted curriculum and reported adapter lineage
refer to one exact subject, then delegates *signed* exam evidence to the
existing Minotaur graduation gate. Training receipts and content hashes are
self-declarations, not host, rights, or independent evaluator attestations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .graduation_gate import GateState, GraduationPolicy, decide_graduation


_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
# Sanity floor for completed optimizer steps, not a host-timing attestation.
# A claimed 1+ neural optimizer steps in zero/subnormal total time is invalid.
_MIN_DECLARED_SECONDS_PER_STEP = 0.001


@dataclass(frozen=True)
class LearningIntent:
    objective_id: str
    objective_kind: str
    intent_sha256: str


@dataclass(frozen=True)
class CurriculumAdmission:
    objective_id: str
    intent_sha256: str
    manifest_sha256: str
    admitted_corpus_sha256: str
    expected_training_source_sha256: str
    expected_tokenizer_sha256: str
    expected_initial_adapter_sha256: str
    evaluator_sha256: str
    conditions_sha256: str
    split: str
    admitted_case_count: int
    rights_approved: bool
    eval_overlap_detected: bool


@dataclass(frozen=True)
class AdapterTrainingReceipt:
    objective_id: str
    manifest_sha256: str
    admitted_corpus_sha256: str
    base_sha256: str
    initial_adapter_sha256: str
    trained_adapter_sha256: str
    tokenizer_sha256: str
    source_sha256: str
    completed_steps: int
    optimizer_budget_seconds: float
    optimizer_elapsed_seconds: float
    time_integrity_status: str


@dataclass(frozen=True)
class BridgeDecision:
    status: str
    reasons: tuple[str, ...]
    corridor_action: str
    minotaur_gate_disposition: str
    gate_state: GateState
    independently_verified_host_receipts: bool = False
    qualified_neural_improvement: bool = False
    promotion_allowed: bool = False


def _hash(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _seconds(value: object) -> bool:
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _blocked(reason: str, state: GateState) -> BridgeDecision:
    return BridgeDecision("BLOCKED", (reason,), "REMAIN_IN_LABYRINTH", "NOT_CALLED", state)


def assess_graduation_handoff(
    intent: LearningIntent,
    curriculum: CurriculumAdmission,
    training: AdapterTrainingReceipt | None,
    policy: GraduationPolicy,
    state: GateState,
    *,
    exam_receipt_bytes: bytes | None = None,
    exam_signature_hex: str | None = None,
    independent_evaluator_key: bytes | None = None,
) -> BridgeDecision:
    """Assess a source-level candidate, then defer signed exam to Minotaur.

    An unsigned score, trainer's PASS assertion or public/curriculum-only
    observation cannot leave the training corridor. Even an existing gate
    disposition of GRADUATED is *not* deployment/promotion authorization:
    a separately custodied real evaluator and host evidence are necessary.
    """
    if not isinstance(state, GateState):
        raise TypeError("GateState must come from a trusted state custodian")
    if (not isinstance(intent, LearningIntent)
            or not isinstance(curriculum, CurriculumAdmission)
            or not isinstance(policy, GraduationPolicy)):
        return _blocked("MALFORMED_HANDOFF", state)
    if (not _id(intent.objective_id)
            or intent.objective_kind not in ("skill_mastery", "training_effect")
            or not _hash(intent.intent_sha256)):
        return _blocked("INVALID_LEARNING_INTENT", state)
    if (not _id(curriculum.objective_id)
            or not all(_hash(v) for v in (
                curriculum.intent_sha256, curriculum.manifest_sha256,
                curriculum.admitted_corpus_sha256,
                curriculum.expected_training_source_sha256,
                curriculum.expected_tokenizer_sha256,
                curriculum.expected_initial_adapter_sha256,
                curriculum.evaluator_sha256, curriculum.conditions_sha256,
            ))
            or curriculum.split != "train"
            or type(curriculum.admitted_case_count) is not int
            or not 8 <= curriculum.admitted_case_count <= 1_000_000
            or type(curriculum.rights_approved) is not bool
            or type(curriculum.eval_overlap_detected) is not bool):
        return _blocked("INVALID_CURRICULUM_ADMISSION", state)
    if not curriculum.rights_approved or curriculum.eval_overlap_detected:
        return _blocked("RIGHTS_OR_EVALUATION_LEAKAGE_HOLD", state)
    if (intent.objective_id != curriculum.objective_id
            or intent.intent_sha256 != curriculum.intent_sha256
            or intent.objective_id != policy.objective_id
            or intent.objective_kind != policy.objective_kind
            or curriculum.evaluator_sha256 != policy.expected_evaluator_sha256
            or curriculum.conditions_sha256 != policy.expected_conditions_sha256):
        return _blocked("INTENT_OR_EXAM_POLICY_BINDING_MISMATCH", state)
    if training is None:
        return _blocked("NO_NEURAL_ADAPTER_RECEIPT", state)
    if not isinstance(training, AdapterTrainingReceipt):
        return _blocked("MALFORMED_TRAINING_RECEIPT", state)
    if (not _id(training.objective_id)
            or not all(_hash(v) for v in (
                training.manifest_sha256, training.admitted_corpus_sha256,
                training.base_sha256,
                training.initial_adapter_sha256, training.trained_adapter_sha256,
                training.tokenizer_sha256, training.source_sha256,
            ))
            or type(training.completed_steps) is not int
            or not 1 <= training.completed_steps <= 100_000
            or not _seconds(training.optimizer_budget_seconds)
            or not _seconds(training.optimizer_elapsed_seconds)
            or training.optimizer_budget_seconds == 0
            or training.optimizer_elapsed_seconds < (
                training.completed_steps * _MIN_DECLARED_SECONDS_PER_STEP
            )):
        return _blocked("INVALID_TRAINING_RECEIPT", state)
    if (training.time_integrity_status != "VALID_TIME_BUDGET"
            or training.optimizer_elapsed_seconds > training.optimizer_budget_seconds):
        return _blocked("INVALID_OR_OVERRUN_TRAINING_BUDGET", state)
    if (training.objective_id != intent.objective_id
            or training.manifest_sha256 != curriculum.manifest_sha256
            or training.admitted_corpus_sha256 != curriculum.admitted_corpus_sha256
            or training.source_sha256 != curriculum.expected_training_source_sha256
            or training.tokenizer_sha256 != curriculum.expected_tokenizer_sha256
            or training.initial_adapter_sha256 != curriculum.expected_initial_adapter_sha256
            or training.base_sha256 != policy.base_sha256
            or training.trained_adapter_sha256 != policy.trainee_sha256
            or training.trained_adapter_sha256 in (
                training.base_sha256, training.initial_adapter_sha256,
            )):
        return _blocked("ADAPTER_OR_CURRICULUM_LINEAGE_MISMATCH", state)

    if exam_receipt_bytes is None or exam_signature_hex is None or independent_evaluator_key is None:
        return BridgeDecision("EXAM_REQUIRED", ("independent signed examination missing",),
                              "REQUEST_INDEPENDENT_MINOTAUR_EXAM", "NOT_CALLED", state)

    result = decide_graduation(policy, state, exam_receipt_bytes, exam_signature_hex,
                               independent_evaluator_key=independent_evaluator_key)
    if result.disposition == "GRADUATED":
        if state.graduated is True:
            # Minotaur's existing gate deliberately returns an earlier
            # custodian-owned graduation before checking a new signature.
            # Preserve that historical state without claiming this receipt
            # was authenticated in the present call.
            return BridgeDecision("PRIOR_TRUSTED_GATE_GRADUATION_REPORTED",
                                  ("prior graduation, no new exam verified",),
                                  "REQUIRES_INDEPENDENT_CUSTODY_REVIEW",
                                  result.disposition, result.state)
        # A signed *source-level* decision is not evidence of actual examiner
        # independence or host-executed neural improvement.
        return BridgeDecision("SIGNED_GATE_REPORTED_GRADUATED", (),
                              "REQUIRES_INDEPENDENT_CUSTODY_REVIEW",
                              result.disposition, result.state)
    return BridgeDecision("GATE_" + result.disposition, result.reasons,
                          "REMAIN_IN_LABYRINTH", result.disposition, result.state)
