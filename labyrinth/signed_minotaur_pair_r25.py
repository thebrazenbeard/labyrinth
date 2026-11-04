"""Opt-in, signed-pair development scores -> regenerative Minotaur adapter.

The existing ``face_minotaur`` accepts caller-supplied per-case diagnostics.
This bridge first authenticates two independently scored development receipts
for the *same* precommitted case roster: the changed candidate checkpoint and
the initial baseline. The signer reports binary-correctness aggregates only.
This bridge binds every public case identity/split/prompt to that roster and
checks each diagnostic group's total and number correct against BOTH signed
aggregates before calling the existing Journey transition.

An HMAC verifies an evaluator claim, not the actual inference origin. The
evaluator key, objective/precommit, durable one-shot ledger and model artifacts
must be controlled outside the trainer; passing them from a trainer process is
NOT independent custody. The assessment does not sign case-level outcomes,
so different per-case distributions with unchanged group totals cannot be
disambiguated and must not be used as per-case remediation evidence. This is
DEVELOPMENT ONLY: no private exam, graduation, host-resume, or model-learning
authority is granted.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import hmac
import re

from .blind_evaluation_custody import (
    BlindCustodyError, SignedAssessment, verify_assessment,
)
from .blind_transition_bridge import PublicAssessmentCase, _bridge_roster
from .journey import AWAITING_EXAM, JourneyState, _validate_state, face_minotaur
from .minotaur_exam import MinotaurExamReceipt
from .objective_contract import ObjectiveContract


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


@dataclass(frozen=True)
class SignedMinotaurPairPrecommit:
    """Expected identities frozen by an independent evaluator *before* scoring.

    This Python record is NOT proof of the independent custodian. It is the
    expected contract read back from the externally owned precommit record.
    """

    objective_id: str
    scope_fingerprint: str
    objective_contract_sha256: str
    exam_id: str
    cycle_index: int
    candidate_run_id: str
    baseline_run_id: str
    candidate_checkpoint_sha256: str
    baseline_checkpoint_sha256: str
    public_exam_sha256: str

    def __post_init__(self) -> None:
        for field in ("objective_id", "exam_id", "candidate_run_id", "baseline_run_id"):
            value = getattr(self, field)
            if type(value) is not str or not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"invalid independently precommitted {field}")
        if self.candidate_run_id == self.baseline_run_id:
            raise ValueError("candidate and baseline must have different signed run identities")
        for field in (
            "scope_fingerprint", "objective_contract_sha256",
            "candidate_checkpoint_sha256", "baseline_checkpoint_sha256",
            "public_exam_sha256",
        ):
            value = getattr(self, field)
            if type(value) is not str or not _SHA256.fullmatch(value):
                raise ValueError(f"invalid independently precommitted {field}")
        if self.candidate_checkpoint_sha256 == self.baseline_checkpoint_sha256:
            raise ValueError("candidate and baseline checkpoints must be different")
        if type(self.cycle_index) is not int or not 0 <= self.cycle_index < 8:
            raise ValueError("invalid independently precommitted development cycle")


@dataclass(frozen=True)
class SignedMinotaurPairTransition:
    """Result of authenticated development-only group reconciliation."""

    next_journey: JourneyState
    candidate_receipt_sha256: str
    baseline_receipt_sha256: str
    qualification: str = "NOT_ESTABLISHED"
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    host_custody_verified: bool = False

    def __post_init__(self) -> None:
        if (self.qualification != "NOT_ESTABLISHED"
                or self.graduation_authorized is not False
                or self.model_execution_authenticated is not False
                or self.host_custody_verified is not False):
            raise ValueError("a development pair cannot claim independent qualification")
        _validate_state(self.next_journey)
        for value in (self.candidate_receipt_sha256, self.baseline_receipt_sha256):
            if type(value) is not str or not _SHA256.fullmatch(value):
                raise ValueError("invalid signed development receipt digest")


def _checked_signed_assessment(
    signed: SignedAssessment,
    *,
    evaluator_key: bytes,
    run_id: str,
    checkpoint: str,
    expected_objective_id: str,
    expected_public_exam_sha256: str,
) -> dict:
    if (type(signed) is not SignedAssessment
            or signed.graduation_authorized is not False):
        raise ValueError("only exact, non-graduating development assessments are accepted")
    try:
        scored = verify_assessment(
            signed.receipt_bytes, signed.signature_hex,
            evaluator_key=evaluator_key, run_id=run_id,
            checkpoint_sha256=checkpoint,
        )
    except (BlindCustodyError, ValueError) as exc:
        raise ValueError("signed development assessment failed verification") from exc
    if (scored["objective_id"] != expected_objective_id
            or not hmac.compare_digest(scored["public_exam_sha256"],
                                       expected_public_exam_sha256)):
        raise ValueError("scored assessment diverges from precommitted objective or roster")
    return scored


def _aggregate_cases(exam: MinotaurExamReceipt) -> dict[tuple[str, str], tuple[int, int, int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for case in exam.cases:
        # BlindCustody's independently signed assessor scores a binary exact
        # answer. Numeric partial-credit diagnostics are not compatible with
        # that assessor and cannot be silently rounded into signed evidence.
        if (type(case.score) not in (int, float) or case.score not in (0, 1)
                or type(case.baseline_score) not in (int, float)
                or case.baseline_score not in (0, 1)):
            raise ValueError("signed-pair development checks require binary per-case diagnostics")
        row = grouped[(case.split, case.skill)]
        row[0] += 1
        row[1] += int(case.score)
        row[2] += int(case.baseline_score)
    return {key: tuple(row) for key, row in grouped.items()}


def apply_signed_minotaur_pair(
    *,
    contract: ObjectiveContract,
    journey: JourneyState,
    exam: MinotaurExamReceipt,
    public_cases: tuple[PublicAssessmentCase, ...],
    candidate_assessment: SignedAssessment,
    baseline_assessment: SignedAssessment,
    evaluator_key: bytes,
    trusted_precommit: SignedMinotaurPairPrecommit,
) -> SignedMinotaurPairTransition:
    """Authenticate paired development aggregates, then face the Minotaur.

    Requires an evaluator independently precommitted to *both* run IDs,
    checkpoint hashes, objective contract, current cycle and public roster.
    There is no implicit transition for unsigned cases, partial credit, or
    score groups missing the declared skill/split combinations.
    """
    if (type(contract) is not ObjectiveContract or type(journey) is not JourneyState
            or type(exam) is not MinotaurExamReceipt
            or type(trusted_precommit) is not SignedMinotaurPairPrecommit
            or type(candidate_assessment) is not SignedAssessment
            or type(baseline_assessment) is not SignedAssessment
            or type(public_cases) is not tuple
            or any(type(case) is not PublicAssessmentCase for case in public_cases)):
        raise ValueError("exact development contract, journey, signed pair, roster and trusted precommit required")
    ObjectiveContract.__post_init__(contract)
    _validate_state(journey)
    MinotaurExamReceipt.__post_init__(exam)
    SignedMinotaurPairPrecommit.__post_init__(trusted_precommit)
    # Frozen dataclasses remain mutable via object.__setattr__. Capture a
    # deep, locally owned snapshot *before* invoking any signed-evaluator or
    # roster callback, and use ONLY the snapshot for all authority decisions.
    # Recheck the original inputs before returning; this detects a signed
    # verifier that mutates a precommit, objective policy, receipt or Journey.
    sources = (contract, journey, exam, public_cases, candidate_assessment,
               baseline_assessment, trusted_precommit)
    contract, journey, exam, public_cases, candidate_assessment, baseline_assessment, trusted_precommit = (
        deepcopy(sources)
    )

    def reject_changed_sources() -> None:
        if any(source != pinned for source, pinned in zip(
            sources, (contract, journey, exam, public_cases,
                      candidate_assessment, baseline_assessment, trusted_precommit),
        )):
            raise ValueError("mutated Minotaur authority, policy, or evidence input during signed verification")

    # The deep clone itself must never conceal an intermediate mutation.
    reject_changed_sources()
    ObjectiveContract.__post_init__(contract)
    _validate_state(journey)
    MinotaurExamReceipt.__post_init__(exam)
    SignedMinotaurPairPrecommit.__post_init__(trusted_precommit)
    if (journey.phase != AWAITING_EXAM
            or contract.scope != journey.scope
            or trusted_precommit.objective_id != journey.scope.objective_id
            or trusted_precommit.scope_fingerprint != journey.scope.fingerprint
            or trusted_precommit.objective_contract_sha256 != contract.fingerprint
            or trusted_precommit.exam_id != exam.exam_id
            or trusted_precommit.cycle_index != journey.cycle_index
            or trusted_precommit.candidate_checkpoint_sha256 != journey.checkpoint_sha256
            or trusted_precommit.baseline_checkpoint_sha256 != journey.baseline_checkpoint_sha256
            or exam.exam_kind != "development"):
        raise ValueError("Minotaur pair diverges from operator-precommitted learning objective, cycle or checkpoint")

    # Verify the source of both aggregates before inspecting any submitted
    # case-level score. Both must bind to an independently committed same exam.
    candidate = _checked_signed_assessment(
        candidate_assessment, evaluator_key=evaluator_key,
        run_id=trusted_precommit.candidate_run_id,
        checkpoint=trusted_precommit.candidate_checkpoint_sha256,
        expected_objective_id=trusted_precommit.objective_id,
        expected_public_exam_sha256=trusted_precommit.public_exam_sha256,
    )
    reject_changed_sources()
    baseline = _checked_signed_assessment(
        baseline_assessment, evaluator_key=evaluator_key,
        run_id=trusted_precommit.baseline_run_id,
        checkpoint=trusted_precommit.baseline_checkpoint_sha256,
        expected_objective_id=trusted_precommit.objective_id,
        expected_public_exam_sha256=trusted_precommit.public_exam_sha256,
    )
    reject_changed_sources()

    _bridge_roster(
        public_cases, expected_digest=trusted_precommit.public_exam_sha256,
        scope=journey.scope,
    )
    case_map = {case.case_id: case for case in exam.cases}
    if len(case_map) != len(public_cases) or set(case_map) != {
        case.case_id for case in public_cases
    }:
        raise ValueError("Minotaur case IDs do not cover the independent public exam roster")
    for public in public_cases:
        private = case_map[public.case_id]
        if (private.skill != public.skill or private.split != public.split
                or not hmac.compare_digest(private.prompt_sha256,
                                           public.prompt_sha256)):
            raise ValueError("Minotaur case identity, split or prompt differs from committed roster")

    grouped = _aggregate_cases(exam)
    candidate_groups = {(item["split"], item["skill"]): item
                        for item in candidate["groups"]}
    baseline_groups = {(item["split"], item["skill"]): item
                       for item in baseline["groups"]}
    if set(grouped) != set(candidate_groups) or set(grouped) != set(baseline_groups):
        raise ValueError("development aggregate groups do not exactly cover the Minotaur exam")
    for key, (total, correct, baseline_correct) in grouped.items():
        if (candidate_groups[key]["total"] != total
                or candidate_groups[key]["correct"] != correct
                or baseline_groups[key]["total"] != total
                or baseline_groups[key]["correct"] != baseline_correct):
            raise ValueError("per-case Minotaur diagnostic contradicts independently signed group counts")

    # The declared rubric values are intent, not an executable evaluator.
    # Check the observable numeric floors in addition to existing Minotaur
    # fixed thresholds, never lower the latter based on a caller declaration.
    for target in contract.targets:
        held = grouped[("heldout", target.skill)]
        transfer = grouped[("transfer", target.skill)]
        count = held[0] + transfer[0]
        correct = held[1] + transfer[1]
        baseline_correct = held[2] + transfer[2]
        if (correct * 100 < target.minimum_score_percent * count
                or (correct - baseline_correct) * 100 < target.minimum_gain_percent * count):
            raise ValueError("candidate does not satisfy declared observable development target")

    reject_changed_sources()
    next_journey = face_minotaur(journey, exam)
    reject_changed_sources()
    result = SignedMinotaurPairTransition(
        next_journey=next_journey,
        candidate_receipt_sha256=sha256(candidate_assessment.receipt_bytes).hexdigest(),
        baseline_receipt_sha256=sha256(baseline_assessment.receipt_bytes).hexdigest(),
    )
    result.__post_init__()
    return result
