"""Two-stage source-only bridge from adaptive Minotaur TRAIN to HF SGD.

Feedback from a prompt-only learner affects the next TRAIN chamber during
staging. Only after all TRAIN cases are collected is their optimizer visit
order pinned for an HF run. No neural weights are updated by staging, and the
roster is neither independent exposure custody nor graduation authority.
An explicit cycle-zero ``run_pilot(minotaur_roster=...)`` can consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
import re
from typing import Callable

from .hf_pilot import PilotConfig, _training_visit_exposures
from .journey_challenges import TraineeRoomView
from .journey_policy_runner import TrainingCorridorRunner, _ANSWER
from .maze import Challenge
from .training_scope import TrainingScopeSpec


STAGED_REPORT_SCHEMA = "labyrinth.minotaur-hf-staged-train.v1"
_INSTRUCTION = "\nRespond with exactly one line: ANSWER: <integer>."
_PRACTICE = re.compile(r"Calculate (\d+) ([+-]) (\d+)\.\Z", re.ASCII)
_CONTRAST = re.compile(
    r"Expression A is (\d+) \+ (\d+)\. Expression B is \1 - \2\. "
    r"Calculate A minus B\.\Z", re.ASCII,
)
_TRANSFER_PLUS = re.compile(
    r"A workshop starts with (\d+) components and receives (\d+) more\. "
    r"How many components does it now have\?\Z", re.ASCII,
)
_TRANSFER_MINUS = re.compile(
    r"A workshop starts with (\d+) components and uses (\d+)\. "
    r"How many components remain, allowing a negative balance\?\Z", re.ASCII,
)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


def _source_arithmetic(case: Challenge) -> tuple[str, tuple[str, int, int]]:
    """Recompute TRAIN gold and conservative semantic key from approved grammar.

    Contrast reduces to b+b and must count as overlap with heldout addition.
    """
    if (type(case) is not Challenge or case.split != "train"
            or case.kind not in ("arithmetic", "arithmetic_contrast")
            or case.tier != 1 or type(case.prompt) is not str
            or not 0 < len(case.prompt.encode("utf-8")) <= 512
            or not case.prompt.endswith(_INSTRUCTION)):
        raise ValueError("unsupported staged TRAIN arithmetic challenge")
    prompt = case.prompt[:-len(_INSTRUCTION)]
    practice = _PRACTICE.fullmatch(prompt)
    contrast = _CONTRAST.fullmatch(prompt)
    plus = _TRANSFER_PLUS.fullmatch(prompt)
    minus = _TRANSFER_MINUS.fullmatch(prompt)
    numerals: tuple[str, str]
    if practice is not None and case.kind == "arithmetic":
        numerals = practice[1], practice[3]
        a, op, b = int(practice[1]), practice[2], int(practice[3])
        answer = a + b if op == "+" else a - b
    elif contrast is not None and case.kind == "arithmetic_contrast":
        numerals = contrast[1], contrast[2]
        a, b, op, answer = int(contrast[1]), int(contrast[2]), "+", 2 * int(contrast[2])
        if not 1 <= a <= 99999:
            raise ValueError("staged TRAIN contrast operand exceeds generator bounds")
        a = b
    elif plus is not None and case.kind == "arithmetic":
        numerals = plus[1], plus[2]
        a, b, op = int(plus[1]), int(plus[2]), "+"
        answer = a + b
    elif minus is not None and case.kind == "arithmetic":
        numerals = minus[1], minus[2]
        a, b, op = int(minus[1]), int(minus[2]), "-"
        answer = a - b
    else:
        raise ValueError("unrecognized staged TRAIN tactic grammar")
    if (not 1 <= a <= 99999 or not 1 <= b <= 99999
            or any(str(int(raw)) != raw for raw in numerals)):
        raise ValueError("staged TRAIN arithmetic operands exceed generator bounds")
    if op == "+":
        a, b = sorted((a, b))
    return str(answer), (op, a, b)


def _source_arithmetic_atoms(case: Challenge) -> frozenset[tuple[str, int, int]]:
    """All arithmetic computations exposed inside one validated TRAIN case.

    A contrast prompt explicitly exposes ``a+b`` and ``a-b`` before asking
    for their difference ``b+b``. Source novelty must account for all three,
    or a later TRAIN challenge can repeat a constituent computation while
    evading the older single-key overlap check.
    """
    _answer, reduced_key = _source_arithmetic(case)
    if case.kind != "arithmetic_contrast":
        return frozenset((reduced_key,))
    matched = _CONTRAST.fullmatch(case.prompt[:-len(_INSTRUCTION)])
    if matched is None:  # Already rejected by the canonical source oracle.
        raise ValueError("unrecognized staged TRAIN contrast grammar")
    a, b = int(matched[1]), int(matched[2])
    return frozenset({
        ("+", min(a, b), max(a, b)),
        ("-", a, b),
        ("+", b, b),
    })


def snapshot_staged_roster(
    roster: "MinotaurHfTrainRoster", config: PilotConfig,
    scope: TrainingScopeSpec,
) -> tuple[list[Challenge], dict[str, object]]:
    """Detach verified private TRAIN gold; persist only answerless public source."""
    if (type(roster) is not MinotaurHfTrainRoster
            or type(scope) is not TrainingScopeSpec or type(config) is not PilotConfig):
        raise ValueError("exact staged roster, scope and HF config required")
    TrainingScopeSpec.__post_init__(scope)
    if (roster.scope_sha256 != scope.fingerprint
            or type(roster.cycle_index) is not int or roster.cycle_index != 0
            or scope.skills != ("arithmetic",) or config.tiers != (1,)
            or config.train_count != len(roster.train_cases)
            or not 2 <= len(roster.train_cases) <= 32
            or type(roster.train_cases) is not tuple):
        raise ValueError("staged TRAIN roster differs from bounded cycle-zero objective")
    copied: list[Challenge] = []
    public: list[dict[str, str]] = []
    keys: set[tuple[str, int, int]] = set()
    ids: set[str] = set()
    prompts: set[str] = set()
    for original in roster.train_cases:
        if (type(original) is not Challenge or type(original.id) is not str
                or not original.id.startswith("train-taskmaster-")
                or len(original.id) > 200 or type(original.answer) is not str
                or len(original.answer) > 64 or type(original.metadata) is not tuple):
            raise ValueError("untrusted or malformed staged TRAIN case")
        case = _primitive_train_snapshot(original)
        computed, _semantic = _source_arithmetic(case)
        atoms = _source_arithmetic_atoms(case)
        if (case.answer != computed or case.id in ids or case.prompt in prompts
                or atoms.intersection(keys)):
            raise ValueError("staged TRAIN answer/identity/semantic overlap invalid")
        ids.add(case.id)
        prompts.add(case.prompt)
        keys.update(atoms)
        copied.append(case)
        public.append({
            "id": case.id, "kind": case.kind, "prompt": case.prompt,
            "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
        })
    exposures = _training_visit_exposures(copied, seed=config.seed, steps=config.steps)
    if roster.ordered_optimizer_exposures != exposures:
        raise ValueError("staged TRAIN optimizer exposure order was substituted")
    record: dict[str, object] = {
        "schema": STAGED_REPORT_SCHEMA, "scope": scope.to_dict(),
        "scope_sha256": scope.fingerprint, "cycle_index": 0,
        "source_authority": "SELF_REPORTED_TRAIN_ONLY_NOT_HOST_AUTHENTICATED",
        "train": public, "optimizer_exposures": [list(pair) for pair in exposures],
    }
    return copied, record


def replay_staged_roster(
    record: object, config: PilotConfig,
) -> tuple[TrainingScopeSpec, list[Challenge], tuple[tuple[str, str], ...]]:
    """Strict rederivation of source TRAIN labels without storing answer strings."""
    if (type(record) is not dict or set(record) != {
            "schema", "scope", "scope_sha256", "cycle_index",
            "source_authority", "train", "optimizer_exposures",
    } or record["schema"] != STAGED_REPORT_SCHEMA
            or record["source_authority"] != "SELF_REPORTED_TRAIN_ONLY_NOT_HOST_AUTHENTICATED"
            or type(record["train"]) is not list):
        raise ValueError("invalid source-only staged TRAIN report contract")
    scope = TrainingScopeSpec.from_dict(record["scope"])
    if (record["scope_sha256"] != scope.fingerprint
            or type(record["cycle_index"]) is not int or record["cycle_index"] != 0
            or scope.skills != ("arithmetic",) or config.tiers != (1,)
            or len(record["train"]) != config.train_count
            or not 2 <= len(record["train"]) <= 32):
        raise ValueError("staged TRAIN report does not match scope or budget")
    cases: list[Challenge] = []
    ids: set[str] = set()
    prompts: set[str] = set()
    keys: set[tuple[str, int, int]] = set()
    for item in record["train"]:
        if (type(item) is not dict
                or set(item) != {"id", "kind", "prompt", "prompt_sha256"}
                or type(item["id"]) is not str or not item["id"].startswith("train-taskmaster-")
                or len(item["id"]) > 200
                or type(item["kind"]) is not str or type(item["prompt"]) is not str
                or len(item["prompt"]) > 4096
                or type(item["prompt_sha256"]) is not str
                or not _SHA.fullmatch(item["prompt_sha256"])):
            raise ValueError("invalid staged TRAIN public commitment")
        case = Challenge(item["id"], item["kind"], 1, "train", item["prompt"], "")
        computed, _semantic = _source_arithmetic(case)
        atoms = _source_arithmetic_atoms(case)
        case = Challenge(case.id, case.kind, case.tier, case.split, case.prompt, computed)
        if (hashlib.sha256(case.prompt.encode("utf-8")).hexdigest() != item["prompt_sha256"]
                or case.id in ids or case.prompt in prompts or atoms.intersection(keys)):
            raise ValueError("staged TRAIN source case identity/semantic overlap mismatch")
        ids.add(case.id)
        prompts.add(case.prompt)
        keys.update(atoms)
        cases.append(case)
    expected = _training_visit_exposures(cases, seed=config.seed, steps=config.steps)
    if record["optimizer_exposures"] != [list(pair) for pair in expected]:
        raise ValueError("staged TRAIN visit order was altered")
    return scope, cases, expected


@dataclass(frozen=True)
class MinotaurHfTrainRoster:
    """Trainer-private staged TRAIN catalog and exact optimizer visit roster."""

    scope_sha256: str
    cycle_index: int
    train_cases: tuple[Challenge, ...] = field(repr=False)
    ordered_optimizer_exposures: tuple[tuple[str, str], ...]
    # A cycle-zero source replay can independently derive each adaptive room
    # from prior TRAIN feedback. These detached primitive values carry no
    # answer key or privileged host approval; older source-only callers can
    # still construct a roster without them, but adaptive HF admission cannot.
    initial_checkpoint_sha256: str = ""
    adaptive_feedback_rows: tuple[tuple[str, str, str, int, str, str], ...] = ()

    @property
    def physically_authenticated(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


def _primitive_train_snapshot(case: Challenge) -> Challenge:
    """Detach the complete trusted case as immutable primitive supervision.

    A frozen dataclass alone is not custody: same-process code can rewrite its
    attributes with ``object.__setattr__``. In particular, copying only the
    case ID and public prompt hash would miss changed historical gold.
    """
    if (type(case) is not Challenge
            or any(type(getattr(case, name)) is not str
                   for name in ("id", "kind", "split", "prompt", "answer"))
            or type(case.tier) is not int
            or type(case.metadata) is not tuple
            or any(type(pair) is not tuple or len(pair) != 2
                   or any(type(part) is not str for part in pair)
                   for pair in case.metadata)):
        raise RuntimeError("mutated TRAIN material has nonprimitive fields")
    return Challenge(
        id=case.id, kind=case.kind, tier=case.tier, split=case.split,
        prompt=case.prompt, answer=case.answer,
        metadata=tuple((key, value) for key, value in case.metadata),
    )


def _assert_historical_cases_unchanged(
    live_cases: list[Challenge], snapshots: list[Challenge],
) -> None:
    for live, snapshot in zip(live_cases, snapshots, strict=True):
        try:
            unchanged = _primitive_train_snapshot(live) == snapshot
        except RuntimeError as exc:
            raise RuntimeError("historical TRAIN material changed after staging") from exc
        if not unchanged:
            raise RuntimeError("historical TRAIN material changed after staging")


def _bounded_int(value: object, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def collect_minotaur_hf_roster(
    runner: TrainingCorridorRunner,
    config: PilotConfig,
    learner: Callable[[TraineeRoomView], str],
) -> MinotaurHfTrainRoster:
    """Stage exact approved TRAIN cases, then pin their HF visit order.

    The caller owns a fresh single-worker runner and a trusted callback. If a
    callback or admission fails after materializing a chamber, discard the
    partial stage; never replay or treat its incomplete roster as training.
    The private cases can feed ``_train_steps`` with the returned ordered
    visits, but ordinary ``run_pilot`` does not yet consume this catalog.
    """
    if type(runner) is not TrainingCorridorRunner or type(config) is not PilotConfig:
        raise ValueError("exact runner and HF PilotConfig required")
    if not callable(learner):
        raise ValueError("trusted operator learner callback required")
    if (runner.observations or runner.state.pending is not None
            or runner._in_flight is not None or runner.is_complete):
        raise ValueError("fresh TRAIN corridor required for HF roster staging")
    scope = runner.state.scope
    # Match HF's scalar resource contract without inspecting model/output
    # paths or importing a model; staging is a source-only operation.
    if (scope.skills != ("arithmetic",) or type(config.tiers) is not tuple
            or config.tiers != (1,)
            or not _bounded_int(config.seed, 0, 2**31 - 1)
            or not _bounded_int(config.steps, 1, 128)
            or not _bounded_int(config.train_count, 2, 32)
            or config.train_count != runner.state.remaining_count
            or not _bounded_int(config.eval_count, 2, 32)
            or not _bounded_int(config.max_tokens, 32, 1024)
            or not _bounded_int(config.max_new_tokens, 1, 32)
            or not _bounded_int(config.lora_rank, 1, 16)
            or not _bounded_int(config.lora_alpha, 1, 64)
            or type(config.learning_rate) is not float
            or not math.isfinite(config.learning_rate)
            or not 1e-7 <= config.learning_rate <= 0.001
            or type(config.device) is not str
            or config.device not in ("cpu", "cuda")
            or type(config.answer_only) is not bool):
        raise ValueError("HF TRAIN roster requires matching bounded arithmetic scope and budget")

    cases: list[Challenge] = []
    snapshots: list[Challenge] = []
    for _ in range(config.train_count):
        view = runner.next_training_prompt()
        case = runner.operator_pending_train_case(view)
        # Python frozen dataclasses can still be rewritten using
        # object.__setattr__. Keep detached pre-callback values and refuse a
        # mutated public view or trainer-private case before scoring feedback.
        original_view = replace(view)
        original_case = _primitive_train_snapshot(case)
        routed_before = runner._in_flight
        pending_before = runner.state.pending
        if routed_before is None or pending_before is None:
            raise RuntimeError("pending TRAIN scoring material is missing")
        routed_snapshot = replace(
            routed_before, suggestion=replace(routed_before.suggestion),
            challenge=original_case,
        )
        pending_snapshot = replace(pending_before)
        state_before = runner.state
        response = learner(view)  # The learner sees no Challenge or answer.
        if view != original_view or _primitive_train_snapshot(case) != original_case:
            raise RuntimeError("learner callback mutated TRAIN material")
        _assert_historical_cases_unchanged(cases, snapshots)
        if (runner._in_flight is not routed_before
                or routed_before.challenge is not case
                or routed_before != routed_snapshot
                or runner.state is not state_before
                or runner.state.pending is not pending_before
                or pending_before != pending_snapshot):
            raise RuntimeError("pending TRAIN scoring material changed during callback")
        outcome = runner.submit_training_response(response)
        match = _ANSWER.fullmatch(response.strip())
        expected_correct = (match is not None
                            and str(int(match.group(1))) == original_case.answer)
        prompt_sha256 = hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
        if (type(case) is not Challenge or case.split != "train"
                or outcome.case_id != case.id
                or outcome.prompt_sha256 != prompt_sha256
                or view.challenge_id != case.id
                or view.prompt_sha256 != prompt_sha256):
            raise RuntimeError("adaptive TRAIN case differs from scored admission")
        if outcome.correct != expected_correct:
            raise RuntimeError("scored TRAIN feedback differs from pinned supervision")
        cases.append(case)
        snapshots.append(original_case)

    _assert_historical_cases_unchanged(cases, snapshots)
    if (not runner.is_complete or len(runner.observations) != len(cases)
            or len({case.id for case in cases}) != len(cases)
            or len({case.prompt for case in cases}) != len(cases)):
        raise ValueError("staged TRAIN catalog incomplete or duplicated")
    registered = {entry.case_id: entry for entry in runner.training_registry}
    for case in snapshots:
        approval = registered.get(case.id)
        if (approval is None or approval.split != "train"
                or approval.prompt_sha256
                != hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()):
            raise ValueError("TRAIN catalog is not approved by the current registry")

    # The adaptive chamber's room_id and case_seed are derived from the
    # *previous* feedback, not from the static initial maze room IDs. Retain
    # this minimal public route history so the report auditor can replay the
    # assignment without trusting arbitrary caller-supplied case seeds.
    history = runner.state.history
    observed = runner.observations
    if len(history) != len(snapshots) or len(observed) != len(snapshots):
        raise ValueError("adaptive TRAIN source history is incomplete")
    feedback_rows: list[tuple[str, str, str, int, str, str]] = []
    for event, case, outcome in zip(history, snapshots, observed, strict=True):
        assignment, feedback = event.assignment, event.feedback
        prompt_sha = hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
        if (feedback.room_id != assignment.room_id
                or feedback.case_id != case.id
                or feedback.prompt_sha256 != prompt_sha
                or feedback.score != int(outcome.correct)
                or outcome.room_id != assignment.room_id
                or outcome.case_id != case.id
                or outcome.prompt_sha256 != prompt_sha):
            raise ValueError("adaptive TRAIN route history differs from admitted cases")
        feedback_rows.append((
            assignment.room_id, feedback.case_id, feedback.prompt_sha256,
            feedback.score, feedback.checkpoint_before_sha256,
            feedback.checkpoint_after_sha256,
        ))

    return MinotaurHfTrainRoster(
        scope_sha256=scope.fingerprint,
        cycle_index=runner.state.plan.cycle_index,
        train_cases=tuple(snapshots),
        ordered_optimizer_exposures=_training_visit_exposures(
            snapshots, seed=config.seed, steps=config.steps,
        ),
        initial_checkpoint_sha256=runner.state.initial_checkpoint_sha256,
        adaptive_feedback_rows=tuple(feedback_rows),
    )
