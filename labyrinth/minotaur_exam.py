"""The Minotaur's *internal* development exam, never graduation authority.

Only externally scored, held-out and transfer outcomes belong here.  Neither
the trainee nor a task-generating Minotaur model can certify the facts in a
receipt.  This source-level gate is deterministic, but unauthenticated:
production graduation still requires ``labyrinth.graduation_gate`` and an
independently operated, signed evaluator.  An internal PASS is not that proof.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import re

from .training_scope import TrainingScopeSpec


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_REQUIRED_SLICES = ("heldout", "transfer")

# An internal development diagnostic.  Never weaken the stronger, separately
# signed private-exam requirements in graduation_gate.py.
MIN_EXAMPLES_PER_SKILL_SLICE = 4
MIN_ACCURACY = 0.75
MIN_GAIN_OVER_BASELINE = 0.10
MAX_EXAM_CASES = 2048


def _identifier(value: object, label: str) -> None:
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError(f"invalid {label}")


def _score(value: object, label: str) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 1):
        raise ValueError(f"{label} must be finite and between 0 and 1")


@dataclass(frozen=True)
class ExamCaseResult:
    """One externally scored case; no prompt, gold answer or feedback stored."""

    case_id: str
    skill: str
    split: str
    score: float
    baseline_score: float
    prompt_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case_id")
        _identifier(self.skill, "skill")
        if self.split not in _REQUIRED_SLICES:
            raise ValueError("Minotaur exam accepts heldout and transfer only")
        _score(self.score, "score")
        _score(self.baseline_score, "baseline_score")
        if type(self.prompt_sha256) is not str or not _SHA256.fullmatch(self.prompt_sha256):
            raise ValueError("invalid prompt_sha256")


@dataclass(frozen=True)
class MinotaurExamReceipt:
    """An in-process score summary from an operator-owned evaluator.

    ``evaluator_id`` is metadata, NOT cryptographic authentication.  The
    caller must establish actual evaluator identity and challenge custody.
    """

    exam_id: str
    objective_id: str
    cycle_index: int
    checkpoint_sha256: str
    baseline_sha256: str
    evaluator_id: str
    cases: tuple[ExamCaseResult, ...]
    exam_kind: str = "development"

    def __post_init__(self) -> None:
        for label in ("exam_id", "objective_id", "evaluator_id"):
            _identifier(getattr(self, label), label)
        if self.exam_kind != "development":
            raise ValueError("private final exams must use the separate signed graduation gate")
        if type(self.cycle_index) is not int or self.cycle_index < 0:
            raise ValueError("cycle_index must be nonnegative integer")
        for label in ("checkpoint_sha256", "baseline_sha256"):
            value = getattr(self, label)
            if type(value) is not str or not _SHA256.fullmatch(value):
                raise ValueError(f"invalid {label}")
        if type(self.cases) is not tuple or not 1 <= len(self.cases) <= MAX_EXAM_CASES:
            raise ValueError("exam cases must be a nonempty bounded tuple")
        seen: set[str] = set()
        seen_prompts: set[str] = set()
        for case in self.cases:
            if type(case) is not ExamCaseResult:
                raise ValueError("invalid case record")
            ExamCaseResult.__post_init__(case)
            if case.case_id in seen:
                raise ValueError("duplicate exam case_id")
            if case.prompt_sha256 in seen_prompts:
                raise ValueError("duplicate exam prompt commitment")
            seen.add(case.case_id)
            seen_prompts.add(case.prompt_sha256)


@dataclass(frozen=True)
class ExamDecision:
    """Provisional gate decision, with no capability to sign or graduate."""

    status: str  # PASS, FAIL, INCONCLUSIVE
    weak_skills: tuple[str, ...]
    reason: str
    exam_id: str
    # Keep exact skill averages for inspectable, non-oracular decisions.
    mean_accuracy_by_skill: tuple[tuple[str, float], ...] = ()
    mean_gain_by_skill: tuple[tuple[str, float], ...] = ()


def evaluate_minotaur(scope: TrainingScopeSpec, receipt: MinotaurExamReceipt) -> ExamDecision:
    """Evaluate coverage and paired gains; do not use this for formal exit.

    A PASS means *provisionally sufficient*, pending independently signed
    graduation.  This function neither observes a model nor trusts a claimed
    pass verdict supplied by a trainee.  Exam cases must never enter training.
    """
    if type(scope) is not TrainingScopeSpec or type(receipt) is not MinotaurExamReceipt:
        raise ValueError("expected TrainingScopeSpec and MinotaurExamReceipt")
    # Frozen dataclasses can be mutated through object.__setattr__.
    TrainingScopeSpec.__post_init__(scope)
    MinotaurExamReceipt.__post_init__(receipt)
    if receipt.objective_id != scope.objective_id:
        raise ValueError("receipt objective does not match declared learning scope")
    if receipt.cycle_index >= scope.max_cycles:
        raise ValueError("exam cycle is outside declared scope budget")
    if any(case.skill not in scope.skills for case in receipt.cases):
        raise ValueError("exam contains an out-of-scope skill")

    outcomes: dict[tuple[str, str], list[ExamCaseResult]] = defaultdict(list)
    for case in receipt.cases:
        outcomes[(case.skill, case.split)].append(case)

    missing = tuple(
        skill for skill in scope.skills
        if any(len(outcomes[(skill, section)]) < MIN_EXAMPLES_PER_SKILL_SLICE
               for section in _REQUIRED_SLICES)
    )
    if missing:
        return ExamDecision(
            "INCONCLUSIVE", missing, "insufficient heldout/transfer coverage", receipt.exam_id
        )

    accuracy: list[tuple[str, float]] = []
    gains: list[tuple[str, float]] = []
    weak: list[str] = []
    for skill in scope.skills:
        results = outcomes[(skill, "heldout")] + outcomes[(skill, "transfer")]
        average = sum(case.score for case in results) / len(results)
        baseline = sum(case.baseline_score for case in results) / len(results)
        heldout = outcomes[(skill, "heldout")]
        heldout_average = sum(case.score for case in heldout) / len(heldout)
        heldout_baseline = sum(case.baseline_score for case in heldout) / len(heldout)
        transfer = outcomes[(skill, "transfer")]
        transfer_average = sum(case.score for case in transfer) / len(transfer)
        transfer_baseline = sum(case.baseline_score for case in transfer) / len(transfer)
        gain = average - baseline
        accuracy.append((skill, average))
        gains.append((skill, gain))
        # Each independently covered development split must demonstrate mastery;
        # a strong transfer average must not conceal weak held-out performance.
        if (heldout_average < MIN_ACCURACY or transfer_average < MIN_ACCURACY
                or heldout_average < heldout_baseline
                or transfer_average < transfer_baseline
                or gain < MIN_GAIN_OVER_BASELINE):
            weak.append(skill)

    if receipt.checkpoint_sha256 == receipt.baseline_sha256:
        return ExamDecision(
            "FAIL", tuple(scope.skills), "learner checkpoint did not change", receipt.exam_id,
            tuple(accuracy), tuple(gains),
        )
    if weak:
        return ExamDecision(
            "FAIL", tuple(weak), "declared skill mastery or gain was not demonstrated",
            receipt.exam_id, tuple(accuracy), tuple(gains),
        )
    return ExamDecision(
        "PASS", (), "internal development threshold met; independent attestation required",
        receipt.exam_id, tuple(accuracy), tuple(gains),
    )
