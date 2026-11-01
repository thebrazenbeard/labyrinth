"""Two-stage source-only bridge from adaptive Minotaur TRAIN to HF/LoRA SGD.

The room selection is feedback-conditioned: a complete HF minibatch schedule
cannot be committed until this source-controlled TRAIN corridor has generated
its cases. This bridge **first** collects TRAIN-only cases while a pre-existing
learner responds to prompts, **then** pins an ordered roster for HF `_train_steps`.
Consequently it is not online neural-driven curriculum adaptation, a host
precommit, a physical optimizer witness, or evidence of transfer/graduation.
An explicit cycle-zero `run_pilot(minotaur_roster=...)` accepts the resulting
source-only roster. Only operator-trusted code receives TRAIN gold.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Callable

from .hf_pilot import PilotConfig, _training_visit_exposures
from .journey_challenges import TraineeRoomView
from .journey_policy_runner import TrainingCorridorRunner
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
    """Derive TRAIN gold and a conservative semantic key from approved grammar.

    This covers all three *actual* taskmaster arithmetic tactics. Contrast
    computes 2*b, equivalent to the heldout sum b+b, and must never disappear
    from train/eval overlap screening just because its kind differs.
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
        a = b  # A minus B is mathematically b+b for this bounded generator.
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
    # The genuine source generator uses at most five ASCII digits per operand.
    # Reject oversized or noncanonical numerals before any arithmetic work.
    if (not 1 <= a <= 99999 or not 1 <= b <= 99999
            or any(str(int(raw)) != raw for raw in numerals)):
        raise ValueError("staged TRAIN arithmetic operands exceed generator bounds")
    if op == "+":
        a, b = sorted((a, b))
    return str(answer), (op, a, b)


def snapshot_staged_roster(
    roster: MinotaurHfTrainRoster, config: PilotConfig,
    scope: TrainingScopeSpec,
) -> tuple[list[Challenge], dict[str, Any]]:
    """Copy operator TRAIN gold into immutable, separately replayable source data.

    The persistent report stores only public TRAIN prompts and prompt hashes;
    only this trusted process receives actual TRAIN answer strings. Report
    metadata is caller supplied, not external source or optimizer custody.
    """
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
        case = Challenge(original.id, original.kind, original.tier,
                         original.split, original.prompt, original.answer)
        computed, semantic = _source_arithmetic(case)
        if (case.answer != computed or case.id in ids or case.prompt in prompts
                or semantic in keys):
            raise ValueError("staged TRAIN answer/identity/semantic overlap invalid")
        ids.add(case.id)
        prompts.add(case.prompt)
        keys.add(semantic)
        copied.append(case)
        public.append({
            "id": case.id, "kind": case.kind, "prompt": case.prompt,
            "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
        })
    exposures = _training_visit_exposures(copied, seed=config.seed, steps=config.steps)
    if roster.ordered_optimizer_exposures != exposures:
        raise ValueError("staged TRAIN optimizer exposure order was substituted")
    record: dict[str, Any] = {
        "schema": STAGED_REPORT_SCHEMA,
        "scope": scope.to_dict(),
        "scope_sha256": scope.fingerprint,
        "cycle_index": 0,
        "source_authority": "SELF_REPORTED_TRAIN_ONLY_NOT_HOST_AUTHENTICATED",
        "train": public,
        "optimizer_exposures": [list(pair) for pair in exposures],
    }
    return copied, record


def replay_staged_roster(
    record: Any, config: PilotConfig,
) -> tuple[TrainingScopeSpec, list[Challenge], tuple[tuple[str, str], ...]]:
    """Strictly rederive TRAIN-only labels, avoiding persistent gold strings."""
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
        computed, semantic = _source_arithmetic(case)
        case = Challenge(case.id, case.kind, case.tier, case.split, case.prompt, computed)
        if (hashlib.sha256(case.prompt.encode("utf-8")).hexdigest() != item["prompt_sha256"]
                or case.id in ids or case.prompt in prompts or semantic in keys):
            raise ValueError("staged TRAIN source case identity/answer mismatch")
        ids.add(case.id)
        prompts.add(case.prompt)
        keys.add(semantic)
        cases.append(case)
    expected = _training_visit_exposures(cases, seed=config.seed, steps=config.steps)
    if record["optimizer_exposures"] != [list(pair) for pair in expected]:
        raise ValueError("staged TRAIN visit order was altered")
    return scope, cases, expected


@dataclass(frozen=True)
class MinotaurHfTrainRoster:
    """Operator-private catalog usable by `_train_steps` after full staging."""

    scope_sha256: str
    cycle_index: int
    train_cases: tuple[Challenge, ...]
    ordered_optimizer_exposures: tuple[tuple[str, str], ...]

    @property
    def physically_authenticated(self) -> bool:
        """Source-only roster can never claim independent physical custody."""
        return False

    @property
    def graduation_authorized(self) -> bool:
        """Materializing TRAIN is not an authorization to graduate."""
        return False


def collect_minotaur_hf_roster(
    runner: TrainingCorridorRunner, config: PilotConfig,
    learner: Callable[[TraineeRoomView], str],
) -> MinotaurHfTrainRoster:
    """Stage exact approved TRAIN material and pin HF optimizer exposure order.

    The caller must own a fresh, single-worker runner and a trusted learner
    callback. This function advances the runner, so an exception after chamber
    creation aborts the stage and the caller must not reuse its partial roster.
    The returned private cases may be fed to `_train_steps` with
    `expected_visit_exposures=ordered_optimizer_exposures`. `run_pilot` accepts
    the catalog only with explicit cycle-zero scope; no host controls result.
    """
    if type(runner) is not TrainingCorridorRunner or type(config) is not PilotConfig:
        raise ValueError("exact runner and HF PilotConfig required")
    if not callable(learner):
        raise ValueError("trusted operator learner callback required")
    if (runner.observations or runner.state.pending is not None
            or runner._in_flight is not None or runner.is_complete):
        raise ValueError("fresh TRAIN corridor required for HF roster staging")
    scope = runner.state.scope
    if (scope.skills != ("arithmetic",) or config.tiers != (1,)
            or type(config.seed) is not int or not 0 <= config.seed < 2**31
            or type(config.steps) is not int or not 1 <= config.steps <= 128
            or type(config.train_count) is not int
            or config.train_count != runner.state.remaining_count
            or not 2 <= config.train_count <= 64):
        raise ValueError("HF TRAIN roster requires matching bounded arithmetic scope and budget")
    cases: list[Challenge] = []
    for _ in range(config.train_count):
        view = runner.next_training_prompt()
        case = runner.operator_pending_train_case(view)
        # Only this prompt-only view goes to the learner, never the Challenge.
        response = learner(view)
        outcome = runner.submit_training_response(response)
        digest = hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
        if (type(case) is not Challenge or case.split != "train"
                or outcome.case_id != case.id or outcome.prompt_sha256 != digest
                or view.challenge_id != case.id or view.prompt_sha256 != digest):
            raise RuntimeError("adaptive TRAIN case differs from scored admission")
        cases.append(case)
    if (not runner.is_complete or len(runner.observations) != len(cases)
            or len({case.id for case in cases}) != len(cases)
            or len({case.prompt for case in cases}) != len(cases)):
        raise ValueError("staged TRAIN catalog incomplete or duplicated")
    registered = {approval.case_id: approval for approval in runner.training_registry}
    for case in cases:
        admission = registered.get(case.id)
        if (admission is None or admission.split != "train"
                or admission.prompt_sha256
                != hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()):
            raise ValueError("TRAIN catalog is not approved by the current registry")
    return MinotaurHfTrainRoster(
        scope_sha256=scope.fingerprint,
        cycle_index=runner.state.plan.cycle_index,
        train_cases=tuple(cases),
        ordered_optimizer_exposures=_training_visit_exposures(
            cases, seed=config.seed, steps=config.steps,
        ),
    )
