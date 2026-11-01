"""Operator-approved, counterfactual Minotaur TRAIN challenge selector.

This SOURCE-ONLY curriculum component addresses a specific public development
failure: a frozen-feature neural head scored 12/16 on easy held-out templates,
then 4/16 on hostile numerical counterfactuals. A lexical cue shared by all
four answers cannot guide a balanced same-wording quartet. Every admitted
family therefore has four *identical lexical scaffolds*, distinct audited
measurements and one example of each taskmaster tactic. The operator, not the
model, approves the entire TRAIN roster and supplies a separately pinned hash.

After recording observed TRAIN predictions, the selector completes any
partially visited quartet, otherwise seeks new lexical/numerical surfaces and
nearby counterfactuals of *previously misclassified TRAIN evidence*. It does
not generate cases, infer an exam key, score an independent evaluation, train
weights, administer a final, or authorize graduation. Training predictions,
the approval process, and external pin custody are operator-trusted inputs;
these in-process checks cannot authenticate an independent trainer or model.

This four-tactic public taskmaster curriculum is distinct from the separate
three-tactic MinotaurTrainingPolicy's arithmetic room API. It does not replace
that policy, the atomic registry receipts, or the ToyLinearAdapter bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


SCHEMA = "labyrinth.minotaur-counterfactual-train-corridors.r92.v1"
TACTICS = ("practice-easy", "contrast-medium", "transfer-hard", "review-easy")
METRICS = ("prerequisite", "paired_method", "retention_now", "retention_before")
MAX_APPROVED_TRAIN_CASES = 256
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COUNT = re.compile(r"(?<![A-Za-z0-9])(?:10|[0-9])/10(?![0-9])", re.ASCII)
_TOKEN = re.compile(r"[a-z]+", re.ASCII)
# Audited visible spellings only. Aliases are explicit so a numeric sequence
# cannot be assigned arbitrary hidden metric semantics by ``metric_order``.
_VISIBLE_METRIC_ALIASES = {
    "basic": "prerequisite",
    "paired": "paired_method",
    "current": "retention_now",
    "historical": "retention_before",
    "prerequisite": "prerequisite",
    "paired_method": "paired_method",
    "retention_now": "retention_now",
    "retention_before": "retention_before",
}
_VISIBLE_PAIR = re.compile(
    r"(?<![A-Za-z0-9_])(?P<name>"
    + "|".join(re.escape(name) for name in _VISIBLE_METRIC_ALIASES)
    + r") +(?P<count>(?:10|[0-9])/10)(?![0-9])",
    re.ASCII,
)


def _id(value: object, name: str) -> None:
    if type(value) is not str or _IDENT.fullmatch(value) is None:
        raise ValueError(f"invalid TRAIN {name}")


def _hash(value: object, name: str) -> None:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError(f"invalid TRAIN {name}")


def _norm(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _mask(prompt: str) -> str:
    # The *exact* visible scaffold is controlled: capitalization, whitespace,
    # and punctuation can encode tactic labels just as additional words can.
    return _COUNT.sub("?/10", prompt)


def _rubric(basic: int, paired: int, now: int, before: int) -> str:
    """Explicit observable evidence rule; retention regression takes priority."""
    if before >= 8 and now < 6:
        return TACTICS[3]
    if basic < 6:
        return TACTICS[0]
    if paired < 7:
        return TACTICS[1]
    return TACTICS[2]


@dataclass(frozen=True)
class ApprovedTrainCounterfactual:
    """One *operator*-approved TRAIN case, including private routing rubric.

    The host must not pass these approvals to the learner. ``prompt_sha256``
    is an explicit approval commitment, not inferred evaluation provenance.
    Four explicitly named denominator-ten measurements in ``prompt`` must
    match their typed values and the actual visible ``metric_order`` exactly.
    """

    case_id: str
    family: str
    prompt: str
    prompt_sha256: str
    metric_order: tuple[str, str, str, str]
    prerequisite: int
    paired_method: int
    retention_now: int
    retention_before: int
    expected_tactic: str
    split: str = "train"

    def __post_init__(self) -> None:
        _id(self.case_id, "case_id")
        _id(self.family, "family")
        if type(self.split) is not str or self.split != "train":
            raise ValueError("only operator-approved TRAIN cases are eligible")
        if (type(self.prompt) is not str or not 40 <= len(self.prompt) <= 800
                or any(not 32 <= ord(ch) <= 126 for ch in self.prompt)):
            raise ValueError("TRAIN prompt must be bounded printable ASCII")
        _hash(self.prompt_sha256, "prompt digest")
        if hashlib.sha256(self.prompt.encode("ascii")).hexdigest() != self.prompt_sha256:
            raise ValueError("TRAIN prompt does not match approved commitment")
        if (type(self.metric_order) is not tuple or len(self.metric_order) != 4
                or any(type(metric) is not str for metric in self.metric_order)
                or set(self.metric_order) != set(METRICS)):
            raise ValueError("TRAIN measure order must name all four distinct metrics")
        values = {name: getattr(self, name) for name in METRICS}
        if any(type(value) is not int or not 0 <= value <= 10 for value in values.values()):
            raise ValueError("TRAIN measures must be integer counts 0..10")
        counts = list(_COUNT.finditer(self.prompt))
        visible = list(_VISIBLE_PAIR.finditer(self.prompt))
        # No extra/unbound count may hide elsewhere in the text; every count
        # must be coupled to one explicit approved name, exactly once.
        if (len(counts) != 4 or len(visible) != 4
                or [match.span() for match in counts]
                != [match.span("count") for match in visible]):
            raise ValueError("TRAIN measures must be four explicitly named prompt counts")
        visible_order = tuple(_VISIBLE_METRIC_ALIASES[match.group("name")]
                              for match in visible)
        if len(set(visible_order)) != 4 or visible_order != self.metric_order:
            raise ValueError("TRAIN measure order disagrees with visible prompt names")
        if any(int(match.group("count").split("/")[0]) != values[name]
               for match, name in zip(visible, visible_order)):
            raise ValueError("TRAIN measures differ from approved prompt")
        if (type(self.expected_tactic) is not str or self.expected_tactic not in TACTICS
                or self.expected_tactic != _rubric(*(values[name] for name in METRICS))):
            raise ValueError("TRAIN label contradicts the audited numeric rubric")


@dataclass(frozen=True)
class ObservedTrainPrediction:
    """A trusted operator reports one model prediction on approved TRAIN only."""

    case_id: str
    prompt_sha256: str
    predicted_tactic: str
    split: str = "train"

    def __post_init__(self) -> None:
        _id(self.case_id, "feedback case_id")
        _hash(self.prompt_sha256, "feedback prompt")
        if type(self.predicted_tactic) is not str or self.predicted_tactic not in TACTICS:
            raise ValueError("unknown TRAIN prediction tactic")
        if type(self.split) is not str or self.split != "train":
            raise ValueError("only observed TRAIN feedback may steer selection")


@dataclass(frozen=True)
class CounterfactualTrainPromptView:
    """The only learner-facing portion; no tactic label or evidence key."""

    prompt: str
    prompt_sha256: str
    split: str = "train"


@dataclass(frozen=True)
class SelectedCounterfactualTrainCorridor:
    """Trainer-side plan; only ``views`` are suitable for learner inference."""

    family: str
    views: tuple[CounterfactualTrainPromptView, ...]
    priority: str
    lexical_novelty: float
    numerical_novelty: float
    failure_proximity: float
    approved_roster_sha256: str

    @property
    def independent_exam(self) -> str:
        return "NOT_RUN"

    @property
    def graduation_authorized(self) -> bool:
        return False


def _validated_roster(approved_train_roster: tuple[ApprovedTrainCounterfactual, ...]):
    if (type(approved_train_roster) is not tuple
            or not 4 <= len(approved_train_roster) <= MAX_APPROVED_TRAIN_CASES
            or len(approved_train_roster) % 4):
        raise ValueError("a bounded, complete operator-approved TRAIN quartet roster is required")
    by_family: dict[str, list[ApprovedTrainCounterfactual]] = {}
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_prompts: set[str] = set()
    wire: list[dict[str, Any]] = []
    for case in approved_train_roster:
        if type(case) is not ApprovedTrainCounterfactual:
            raise ValueError("exact approved TRAIN case type required")
        # A frozen dataclass can still be changed with object.__setattr__.
        ApprovedTrainCounterfactual.__post_init__(case)
        normalized = _norm(case.prompt)
        if (case.case_id in seen_ids or case.prompt_sha256 in seen_hashes
                or normalized in seen_prompts):
            raise ValueError("duplicate TRAIN case identity or prompt")
        seen_ids.add(case.case_id)
        seen_hashes.add(case.prompt_sha256)
        seen_prompts.add(normalized)
        by_family.setdefault(case.family, []).append(case)
        wire.append({
            "case_id": case.case_id, "family": case.family, "prompt": case.prompt,
            "prompt_sha256": case.prompt_sha256, "metric_order": list(case.metric_order),
            **{name: getattr(case, name) for name in METRICS},
            "expected_tactic": case.expected_tactic, "split": case.split,
        })
    for family, cases in by_family.items():
        if (len(cases) != 4 or {case.expected_tactic for case in cases} != set(TACTICS)
                or len({_mask(case.prompt) for case in cases}) != 1
                or len({tuple(getattr(case, name) for name in METRICS)
                        for case in cases}) != 4):
            raise ValueError(f"TRAIN family {family} lacks a balanced identical lexical scaffold")
    canonical = json.dumps({"schema": SCHEMA, "approved_train_roster": wire},
                           sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("ascii")
    return by_family, hashlib.sha256(canonical).hexdigest()


def fingerprint_train_roster(
    approved_train_roster: tuple[ApprovedTrainCounterfactual, ...],
) -> str:
    """Compute a public source digest; the *operator* must pin it externally.

    This calculation alone does not confer approval or authenticate custody.
    The same bytes are verified against the external pin during selection.
    """
    return _validated_roster(approved_train_roster)[1]


def _distance(a: ApprovedTrainCounterfactual, b: ApprovedTrainCounterfactual) -> int:
    return sum(abs(getattr(a, name) - getattr(b, name)) for name in METRICS)


def _lexical_novelty(
    candidate: list[ApprovedTrainCounterfactual],
    observed: list[ApprovedTrainCounterfactual],
) -> float:
    if not observed:
        return 0.0
    current = set(_TOKEN.findall(_mask(candidate[0].prompt).lower()))
    return 1.0 - max(len(current & other) / len(current | other)
                     for other in (set(_TOKEN.findall(_mask(case.prompt).lower()))
                                   for case in observed))


def select_counterfactual_train_corridor(
    *,
    approved_train_roster: tuple[ApprovedTrainCounterfactual, ...],
    expected_roster_sha256: str,
    observed_train_feedback: tuple[ObservedTrainPrediction, ...] = (),
) -> SelectedCounterfactualTrainCorridor:
    """Select a novel, balanced TRAIN quartet; never mint a new case.

    A quartet is a controlled intervention: identical words and all four
    class labels. Completing a partially visited quartet takes precedence.
    Otherwise failures are matched by *same gold TRAIN tactic and numeric
    evidence proximity* even across differing lexical shells, while lexical
    and numeric novelty discourage repeated easy templates. Scores are
    scheduling heuristics, not claims about model quality or difficulty.
    """
    by_family, actual_pin = _validated_roster(approved_train_roster)
    _hash(expected_roster_sha256, "external roster pin")
    if actual_pin != expected_roster_sha256:
        raise ValueError("operator-approved TRAIN roster changed against external pin")
    if (type(observed_train_feedback) is not tuple
            or len(observed_train_feedback) > len(approved_train_roster)):
        raise ValueError("invalid bounded observed TRAIN feedback stream")
    approved = {case.case_id: case for case in approved_train_roster}
    seen: set[str] = set()
    observed: list[ApprovedTrainCounterfactual] = []
    failures: list[ApprovedTrainCounterfactual] = []
    for event in observed_train_feedback:
        if type(event) is not ObservedTrainPrediction:
            raise ValueError("exact observed TRAIN prediction type required")
        ObservedTrainPrediction.__post_init__(event)
        case = approved.get(event.case_id)
        if case is None or case.prompt_sha256 != event.prompt_sha256:
            raise ValueError("TRAIN feedback is not on the operator-approved roster")
        if case.case_id in seen:
            raise ValueError("duplicate observed TRAIN case")
        seen.add(case.case_id)
        observed.append(case)
        if event.predicted_tactic != case.expected_tactic:
            failures.append(case)

    partial = [family for family, cases in by_family.items()
               if 0 < sum(case.case_id in seen for case in cases) < 4]
    if len(partial) > 1:
        raise ValueError("multiple partial TRAIN counterfactual quartets")
    if partial:
        family = partial[0]
        candidate = [case for case in by_family[family] if case.case_id not in seen]
        priority = "complete_partial_counterfactual"
        lexical_novelty = numeric_novelty = failure_proximity = 0.0
    else:
        available = {family: cases for family, cases in by_family.items()
                     if all(case.case_id not in seen for case in cases)}
        if not available:
            raise ValueError("operator-approved counterfactual TRAIN roster exhausted")

        ranked = []
        for family, cases in available.items():
            lexical = _lexical_novelty(cases, observed)
            numeric = (sum(min(_distance(case, old) for old in observed) / 40.0
                           for case in cases) / 4.0) if observed else 0.0
            # Each label gets the same mass; failed evidence cannot be
            # drowned by more common/easier classes. A threshold of 12/40
            # keeps far-off but same-label failures from steering unrelated
            # future TRAIN cases indefinitely.
            proximity = sum(max(0.0, max(
                ((12 - _distance(case, error)) / 12.0
                 for error in failures if error.expected_tactic == case.expected_tactic),
                default=0.0,
            )) for case in cases) / 4.0
            score = 8.0 * proximity + lexical + 0.3 * numeric
            ranked.append((-score, -proximity, -lexical, -numeric, family,
                           cases, lexical, numeric, proximity))
        ranked.sort(key=lambda row: row[:5])
        (_, _, _, _, family, candidate, lexical_novelty,
         numeric_novelty, failure_proximity) = ranked[0]
        priority = ("repair_train_failures" if failure_proximity > 0 else
                    "novel_train_surface" if observed else "balanced_bootstrap")

    # Deterministic *label-independent* order. This returns only prompt text
    # and commitments to inference; labels and metric-order stay host-private.
    ordered = sorted(candidate, key=lambda case: hashlib.sha256(
        b"train-view-order\0" + bytes.fromhex(actual_pin)
        + bytes.fromhex(case.prompt_sha256)).digest())
    return SelectedCounterfactualTrainCorridor(
        family=family,
        views=tuple(CounterfactualTrainPromptView(case.prompt, case.prompt_sha256)
                    for case in ordered),
        priority=priority,
        lexical_novelty=lexical_novelty,
        numerical_novelty=numeric_novelty,
        failure_proximity=failure_proximity,
        approved_roster_sha256=actual_pin,
    )
