"""Thin DEVELOPMENT_ONLY bridge from a model-directed world to measured trials.

This module does not train a model, create evaluation gold, invent candidate
lineage, or decide which learner deserves promotion. It drives one actual
SagaSession across consequential TRAIN and operator Minotaur decisions, then
returns source-verifiable local evidence to a separately authorized population
selector. A Python navigation callback is operator-installed code, not a safe
sandbox for arbitrary generated code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
import re
import subprocess
import sys
from typing import Callable, Mapping

from .model_saga_runtime import (
    CheckpointPaths, CheckpointRef, SagaSession, StepResult, capture_checkpoint,
)
from .saga_ariadne import verify_saga_ariadne

_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)


@dataclass(frozen=True)
class SagaTrialEvidence:
    """Source-only local evidence; never independent evolutionary fitness."""

    objective_id: str
    entry_checkpoint_sha256: str
    exit_checkpoint_sha256: str
    developmental_status: str
    terminal: bool
    train_spark_count: int
    generation_cycles: int
    ariadne_head_sha256: str
    traversed_branches: tuple[str, ...]
    training_tactics: tuple[str, ...]
    classification: str = "SOURCE_MECHANISM_ONLY_UNQUALIFIED"
    # Deliberately exclude gold, private case identifiers and individual
    # prompts. The caller owns physical checkpoint custody and ancestry.

    def __post_init__(self) -> None:
        if type(self.classification) is not str or self.classification not in (
            "SOURCE_MECHANISM_ONLY_UNQUALIFIED",
            "MODEL_ACTION_INTERFACE_UNQUALIFIED_NOT_NEURAL_EVIDENCE",
        ):
            raise ValueError("saga trial cannot upgrade its evidence classification")


@dataclass(frozen=True)
class CandidateSagaPlan:
    """Operator-created sibling with a unique physical clone of the parent.

    ``make_session`` is trusted local Python code, never model-proposed code;
    its resulting entrance is independently rehashed against the common
    physical parent before any candidate TRAIN effect. A production provider
    should be the population ledger, not a model-written selection script.
    """

    candidate_id: str
    make_session: Callable[[CheckpointPaths], SagaSession]
    choose_threshold_action: Callable[[StepResult], str] | None = None
    model_action_policy: object | None = None


@dataclass(frozen=True)
class CandidateSagaEvidence:
    candidate_id: str
    generation_index: int
    parent_candidate_id: str | None
    parent_checkpoint_sha256: str
    successor: CheckpointRef
    trial: SagaTrialEvidence


@dataclass(frozen=True)
class SagaGenerationEvidence:
    generation_index: int
    parent_candidate_id: str | None
    parent_checkpoint_sha256: str
    candidates: tuple[CandidateSagaEvidence, ...]
    selected_candidate_id: str
    selection_classification: str = "OPERATOR_SCRIPTED_MECHANISM_SELECTION_ONLY"

    def __post_init__(self) -> None:
        if (type(self.selection_classification) is not str
                or self.selection_classification != "OPERATOR_SCRIPTED_MECHANISM_SELECTION_ONLY"):
            raise ValueError("source-only selector cannot assert independently examined fitness")


@dataclass(frozen=True)
class FreshProcessCheckpointControl:
    """Physical fresh-process readback, NOT matched behavioral evaluation."""

    weights_sha256: str
    optimizer_sha256: str | None
    separate_pid: int
    classification: str = "HASH_CONTROL_ONLY_NOT_MODEL_GENERALIZATION"

    def __post_init__(self) -> None:
        if (type(self.classification) is not str
                or self.classification != "HASH_CONTROL_ONLY_NOT_MODEL_GENERALIZATION"):
            raise ValueError("file hash readback cannot assert neural generalization")


@dataclass(frozen=True)
class SagaCampaignEvidence:
    w0: CheckpointRef
    generations: tuple[SagaGenerationEvidence, ...]
    w0_control: FreshProcessCheckpointControl
    classification: str = "MECHANISM_ONLY_NO_NEURAL_EVOLUTION_CLAIM"

    def __post_init__(self) -> None:
        if (type(self.classification) is not str
                or self.classification != "MECHANISM_ONLY_NO_NEURAL_EVOLUTION_CLAIM"):
            raise ValueError("saga mechanism cannot assert proven neural evolution")


def _fresh_worker_environment(
    source: Mapping[str, str], *, platform_name: str,
) -> dict[str, str]:
    """Forward only the process bootstrap variables required by the worker.

    Windows Python startup may require SystemRoot/WINDIR and its temporary
    directory; do not blindly inherit API keys or other parent credentials.
    This protects the subprocess boundary while remaining runnable on VN1.
    """
    environment = {
        "PATH": source.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    if platform_name == "nt":
        for key in ("SystemRoot", "WINDIR", "TEMP", "TMP"):
            if source.get(key):
                environment[key] = source[key]
    return environment


def fresh_process_checkpoint_control(paths: CheckpointPaths) -> FreshProcessCheckpointControl:
    """Reopen W0 in a true child process and cross-check parent hash readback.

    This is an intentionally narrow *file* control. It does not evaluate
    semantic ability or infer a model in a separate process, and never sends
    expected answers, scoring keys or private exam material to a child worker.
    """
    parent = capture_checkpoint(paths)
    command = [sys.executable, "-m", "labyrinth.evolutionary_saga_runner_v1",
               "--fresh-checkpoint", parent.weights_path]
    if parent.optimizer_path is not None:
        command.append(parent.optimizer_path)
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=180,
            # This child needs imports and file access, not a copied parent
            # credential environment or any model evaluation answer keys.
            env=_fresh_worker_environment(os.environ, platform_name=os.name),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("fresh-process W0 control failed; no recovery or fitness claim") from exc
    if len(completed.stdout) > 4096:
        raise ValueError("fresh-process W0 control response exceeded strict bound")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("fresh-process W0 control returned malformed JSON") from exc
    if (type(value) is not dict or set(value) != {"weights_sha256", "optimizer_sha256", "pid"}
            or type(value["pid"]) is not int or value["pid"] <= 0
            or value["pid"] == os.getpid()
            or value["weights_sha256"] != parent.weights_sha256
            or value["optimizer_sha256"] != parent.optimizer_sha256):
        raise ValueError("fresh-process W0 physical checkpoint control disagrees")
    # Detect parent modification between the two independent readbacks.
    if capture_checkpoint(paths) != parent:
        raise ValueError("W0 changed during fresh-process control")
    return FreshProcessCheckpointControl(
        parent.weights_sha256, parent.optimizer_sha256, value["pid"],
    )


def _shortest_path_action(session: SagaSession) -> str:
    """Find a bounded source-world route toward the next approved chamber."""
    if session.location == "chamber":
        return "train"
    pending = deque(((session.location, ()),))
    seen = set()
    while pending:
        node, actions = pending.popleft()
        if node == "chamber":
            if not actions:
                raise ValueError("cannot move from a non-chamber without a path")
            return actions[0]
        if node in seen:
            continue
        seen.add(node)
        for direction, successor in session.nodes[node].exits:
            pending.append((successor, actions + (direction,)))
    raise ValueError("model story has no path to its training chamber")


def run_saga_trial(
    session: SagaSession, *,
    choose_threshold_action: Callable[[StepResult], str],
) -> SagaTrialEvidence:
    """Execute a fresh bounded episode and return inspectable source evidence.

    The learner/scene/minotaur models are wired into ``session`` by an
    operator. This runner never receives an examiner answer, private item or
    TRAIN gold. A candidate cannot be called evolutionarily fit solely on a
    development PASS: population selection needs its own measured heldout
    policy and the private graduation gate remains separate.
    """
    if type(session) is not SagaSession or not callable(choose_threshold_action):
        raise ValueError("exact source saga and authorized navigation callback required")
    if (session.action_count or session.training_sparks or session.terminated
            or session.truncated or session.indeterminate):
        raise ValueError("evolution trial requires a fresh, unmodified saga entry")
    root = session.entrance.checkpoint.weights_sha256
    chosen: list[str] = []
    for _ in range(session.action_budget + 1):
        if session.terminated or session.truncated or session.indeterminate:
            break
        if session.phase == "MINOTAUR":
            session.step("face")
        elif session.location == "threshold":
            action = choose_threshold_action(session.observe())
            choices = {movement for movement, _ in session.nodes["threshold"].exits}
            if type(action) is not str or action not in choices:
                raise ValueError("navigation must choose one authorized threshold branch")
            chosen.append(action)
            session.step(action)
        else:
            session.step(_shortest_path_action(session))
    if session.indeterminate:
        raise ValueError("TRAIN left a possibly partial physical update; no success evidence")
    if not session.terminated:
        raise ValueError("candidate saga exhausted action budget without developmental verdict")
    head = verify_saga_ariadne(session.ariadne_events, require_exit=True)
    events = [item["summary"] for item in session.ariadne_events if item["kind"] == "train"]
    return SagaTrialEvidence(
        objective_id=session.scope.objective_id,
        entry_checkpoint_sha256=root,
        exit_checkpoint_sha256=session.checkpoint.weights_sha256,
        developmental_status=session.development_status,
        terminal=True,
        train_spark_count=len(session.training_sparks),
        generation_cycles=session.journey.cycle_index + 1,
        ariadne_head_sha256=head,
        traversed_branches=tuple(chosen),
        training_tactics=tuple(item["train_tactic"] for item in events),
    )


def run_theseus_saga_trial(session: SagaSession, *, model_policy: object) -> SagaTrialEvidence:
    """Use the separately installed model-action interface, with no auto-steps.

    The policy may be a genuine model adapter or scripted fixture; these
    source mechanics cannot attest which. The integrated ``drive_theseus``
    runtime must be present, and invalid model actions stop without fallback.
    """
    if (type(session) is not SagaSession or session.action_count
            or session.training_sparks or session.terminated or session.truncated
            or session.indeterminate):
        raise ValueError("model-directed trial requires a fresh exact SagaSession")
    try:
        from .theseus_agent_loop_v1 import drive_theseus
    except ImportError as exc:
        raise ValueError("Theseus model-action runtime not integrated; cannot claim model control") from exc
    receipt = drive_theseus(session, model_policy, max_decisions=session.action_budget)
    if (receipt.status != "DEVELOPMENT_EPISODE_ENDED" or not session.terminated
            or session.indeterminate or session.truncated):
        raise ValueError("Theseus did not complete a valid DEVELOPMENT_ONLY episode")
    head = verify_saga_ariadne(session.ariadne_events, require_exit=True)
    trains = [item["summary"] for item in session.ariadne_events if item["kind"] == "train"]
    return SagaTrialEvidence(
        objective_id=session.scope.objective_id,
        entry_checkpoint_sha256=session.entrance.checkpoint.weights_sha256,
        exit_checkpoint_sha256=session.checkpoint.weights_sha256,
        developmental_status=session.development_status,
        terminal=True,
        train_spark_count=len(session.training_sparks),
        generation_cycles=session.journey.cycle_index + 1,
        ariadne_head_sha256=head,
        traversed_branches=tuple(event.action for event in receipt.decisions),
        training_tactics=tuple(item["train_tactic"] for item in trains),
        classification="MODEL_ACTION_INTERFACE_UNQUALIFIED_NOT_NEURAL_EVIDENCE",
    )


def run_mechanism_campaign(
    w0: CheckpointPaths,
    generations: tuple[tuple[CandidateSagaPlan, ...], ...],
    *, select_winner: Callable[[int, tuple[CandidateSagaEvidence, ...]], str],
) -> SagaCampaignEvidence:
    """Execute a two-or-more-generation *mechanical* lineage walkthrough.

    The selector is an operator-installed adapter and its choice is never
    elevated to independent generalization, neural evolution or graduation.
    A real population authority must additionally bind its immutable manifest,
    measured fitness and fresh-process behavioral controls; here we verify the
    **physical checkpoint ancestry only**. No optimizer-less mock becomes a
    proven full-model lineage merely by surviving this source exercise.
    """
    if (type(generations) is not tuple or len(generations) < 2
            or not callable(select_winner)):
        raise ValueError("campaign needs at least two generations and an owner selector")
    root = capture_checkpoint(w0)
    control = fresh_process_checkpoint_control(w0)
    parent = root
    parent_id: str | None = None
    generations_out: list[SagaGenerationEvidence] = []
    used_ids: set[str] = set()
    entrant_paths: set[str] = {root.weights_path}
    selected_lineage: list[CheckpointRef] = [root]
    for generation, siblings in enumerate(generations):
        if type(siblings) is not tuple or len(siblings) < 2:
            raise ValueError("each generation requires at least two sibling pathways")
        parent_paths = CheckpointPaths(parent.weights_path, parent.optimizer_path)
        reports: list[CandidateSagaEvidence] = []
        for plan in siblings:
            if (type(plan) is not CandidateSagaPlan
                    or type(plan.candidate_id) is not str
                    or _ID.fullmatch(plan.candidate_id) is None
                    or plan.candidate_id in used_ids
                    or not callable(plan.make_session)
                    or (callable(plan.choose_threshold_action)
                        == (plan.model_action_policy is not None))):
                raise ValueError("candidate identity and launch callbacks must be unique and approved")
            used_ids.add(plan.candidate_id)
            if capture_checkpoint(parent_paths) != parent:
                raise ValueError("selected parent checkpoint changed before child entry")
            session = plan.make_session(parent_paths)
            if type(session) is not SagaSession:
                raise ValueError("candidate factory did not produce an actual SagaSession")
            entrance = session.entrance.checkpoint
            if (entrance.weights_sha256 != parent.weights_sha256
                    or entrance.optimizer_sha256 != parent.optimizer_sha256
                    or entrance.weights_path in entrant_paths):
                raise ValueError("candidate must enter with a unique physical clone of selected parent")
            entrant_paths.add(entrance.weights_path)
            if plan.model_action_policy is None:
                trial = run_saga_trial(
                    session, choose_threshold_action=plan.choose_threshold_action,
                )
            else:
                trial = run_theseus_saga_trial(session, model_policy=plan.model_action_policy)
            actual = capture_checkpoint(CheckpointPaths(
                session.checkpoint.weights_path, session.checkpoint.optimizer_path,
            ))
            if actual != session.checkpoint or trial.exit_checkpoint_sha256 != actual.weights_sha256:
                raise ValueError("candidate successor failed physical readback")
            if capture_checkpoint(parent_paths) != parent:
                raise ValueError("candidate mutated the retained parent checkpoint")
            reports.append(CandidateSagaEvidence(
                plan.candidate_id, generation, parent_id,
                parent.weights_sha256, actual, trial,
            ))
        proposals = tuple(reports)
        chosen = select_winner(generation, proposals)
        if type(chosen) is not str or chosen not in {p.candidate_id for p in proposals}:
            raise ValueError("selector must identify an actual candidate from this generation")
        selected = next(candidate for candidate in proposals if candidate.candidate_id == chosen)
        if selected.trial.developmental_status != "PASS":
            raise ValueError("cannot advance a development-failed sibling as champion")
        generations_out.append(SagaGenerationEvidence(
            generation, parent_id, parent.weights_sha256, proposals, chosen,
        ))
        parent, parent_id = selected.successor, chosen
        selected_lineage.append(parent)
    for ref in selected_lineage:
        if capture_checkpoint(CheckpointPaths(ref.weights_path, ref.optimizer_path)) != ref:
            raise ValueError("retained selected lineage artifact changed during campaign")
    if capture_checkpoint(w0) != root:
        raise ValueError("original W0 changed during campaign")
    return SagaCampaignEvidence(root, tuple(generations_out), control)


def _fresh_checkpoint_child(argv: list[str]) -> None:
    """No child inference or scoring: read-only physical checkpoint hashes."""
    if not 2 <= len(argv) <= 3 or argv[0] != "--fresh-checkpoint":
        raise SystemExit("explicit --fresh-checkpoint <weights> [optimizer] required")
    ref = capture_checkpoint(CheckpointPaths(argv[1], argv[2] if len(argv) == 3 else None))
    print(json.dumps({
        "weights_sha256": ref.weights_sha256,
        "optimizer_sha256": ref.optimizer_sha256,
        "pid": os.getpid(),
    }, separators=(",", ":")))


if __name__ == "__main__":
    _fresh_checkpoint_child(sys.argv[1:])


__all__ = [
    "SagaTrialEvidence", "CandidateSagaPlan", "CandidateSagaEvidence",
    "SagaGenerationEvidence", "FreshProcessCheckpointControl", "SagaCampaignEvidence",
    "run_saga_trial", "run_mechanism_campaign", "fresh_process_checkpoint_control",
    "run_theseus_saga_trial",
]
