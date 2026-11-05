"""Original, model-directed Labyrinth episode over existing TRAIN/exam machinery.

The story is a *presentation and control* layer. Models may propose room prose,
branching paths and Minotaur development pressure, but cannot supply training
gold, grade their own exam, or authorize graduation. The trusted operator owns
the learner's actual checkpoint files, supervised update and development exam.

This is a DEVELOPMENT_ONLY in-process experiment. File hashing proves only that
bytes were present and changed, not neural learning, immutability or custody.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import stat
from typing import Callable, Protocol

from .journey import (
    AWAITING_EXAM, EXHAUSTED, TRAINING,
    DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION,
    begin_journey, complete_maze, face_minotaur,
)
from .adaptive_corridors import offer_next_chamber
from .journey_policy_runner import PedagogicalRoute, TrainingCorridorRunner
from .maze import Challenge
from .minotaur_exam import ExamCaseResult, MinotaurExamReceipt, evaluate_minotaur
from .saga_ariadne import append_saga_event, verify_saga_ariadne
from .training_scope import TrainingScopeSpec


_ID = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z", re.ASCII)
_ACTIONS = frozenset(("left", "right", "forward", "back"))
_ANSWER = re.compile(r"ANSWER: ([+-]?\d+)\Z", re.ASCII)
_MAX_ARTIFACT_BYTES = 16 * 1024 ** 3
_PUBLIC_OUTCOME_WINDOW = 16


@dataclass(frozen=True)
class CheckpointPaths:
    """Operator-owned, locally readable weights and optional optimizer files.

    A missing optimizer is allowed for DEVELOPMENT_ONLY but makes exact
    optimizer/scheduler/RNG recovery unestablished. Paths are not remote URIs.
    """

    weights: str
    optimizer: str | None = None


@dataclass(frozen=True)
class CheckpointRef:
    weights_path: str
    weights_sha256: str
    optimizer_path: str | None
    optimizer_sha256: str | None
    continuity: str


def _hash_regular_file(path: str) -> tuple[str, str]:
    """Hash actual on-disk bytes using a no-follow file descriptor."""
    if type(path) is not str or not 1 <= len(path) <= 4096:
        raise ValueError("checkpoint needs a bounded local path")
    file_path = Path(path)
    if not file_path.is_absolute():
        raise ValueError("checkpoint path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(file_path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_ARTIFACT_BYTES:
            raise ValueError("checkpoint must be a nonempty bounded regular file")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(fd), "rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ValueError("checkpoint changed while hashing")
        return str(file_path), digest.hexdigest()
    finally:
        os.close(fd)


def capture_checkpoint(paths: CheckpointPaths) -> CheckpointRef:
    """Read actual files before claiming an entrance or retained state."""
    if type(paths) is not CheckpointPaths:
        raise ValueError("exact checkpoint file paths required")
    weights_path, weights_sha = _hash_regular_file(paths.weights)
    optimizer_path = optimizer_sha = None
    if paths.optimizer is not None:
        optimizer_path, optimizer_sha = _hash_regular_file(paths.optimizer)
        if optimizer_path == weights_path:
            raise ValueError("optimizer and model must be distinct artifacts")
    return CheckpointRef(
        weights_path, weights_sha, optimizer_path, optimizer_sha,
        "FULL_ARTIFACTS_PRESENT_UNQUALIFIED" if optimizer_path else "WEIGHTS_ONLY_CONTINUITY",
    )


@dataclass(frozen=True)
class BaselineMeasurement:
    """Actual pre-TRAIN arithmetic probe; never a heldout/private evaluation."""

    case_count: int
    correct_count: int
    prompt_sha256: tuple[str, ...]
    source: str = "operator_train_safe_probe"


@dataclass(frozen=True)
class EntranceState:
    objective_id: str
    scope_fingerprint: str
    checkpoint: CheckpointRef
    baseline: BaselineMeasurement


@dataclass(frozen=True)
class TrainingSpark:
    """One physically rehashed TRAIN update quantum, distinct from story prose.

    A source-provided weight change is measured, not automatically a neural
    learning or quality gain. No synthetic loss metric is invented.
    """

    cycle_index: int
    room_index: int
    skill: str
    before_sha256: str
    after: CheckpointRef
    train_correct: bool
    loss: float | None = None


@dataclass(frozen=True)
class SceneRequest:
    """Safe model input: no training prompt, answer, private exam or scorer."""

    objective_id: str
    domain: str
    skill: str
    cycle_index: int
    room_index: int
    specificity: int
    weak_skills: tuple[str, ...]
    prior_training_correct: tuple[bool, ...]
    public_seed: int
    prior_training_tactics: tuple[str, ...] = ()
    # Cumulative bounded public experience, including failed past cycles.
    # The source keeps answers, prompts and case identifiers private.
    prior_training_outcomes: tuple[PedagogicalOutcome, ...] = ()


@dataclass(frozen=True)
class SceneNode:
    node_id: str
    prose: str
    exits: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PedagogicalEncounter:
    """Model-proposed learning consequence of entering one initial branch.

    The values select only an already approved source-owned TRAIN grammar;
    no gold, training prompt, grade or private exam material can be supplied.
    """

    branch_node_id: str
    tactic: str
    difficulty_delta: int


@dataclass(frozen=True)
class WorldProposal:
    """A model-proposed map and optional bounded path-to-TRAIN consequences."""

    nodes: tuple[SceneNode, ...]
    encounters: tuple[PedagogicalEncounter, ...] = ()


def _check_world(world: WorldProposal) -> str:
    if (type(world) is not WorldProposal or type(world.nodes) is not tuple
            or not 4 <= len(world.nodes) <= 8
            or any(type(node) is not SceneNode
                   or type(node.node_id) is not str or not _ID.fullmatch(node.node_id)
                   for node in world.nodes)):
        raise ValueError("model world must contain 4-8 bounded scenes")
    nodes = {node.node_id: node for node in world.nodes}
    if len(nodes) != len(world.nodes) or "threshold" not in nodes or "chamber" not in nodes:
        raise ValueError("world needs unique scenes, entrance and training chamber")
    for node in world.nodes:
        if (type(node.node_id) is not str or not _ID.fullmatch(node.node_id)
                or type(node.prose) is not str or not 1 <= len(node.prose) <= 1000
                or "ANSWER:" in node.prose or type(node.exits) is not tuple
                or len(node.exits) > 4):
            raise ValueError("invalid model-proposed story scene")
        if (any(type(item) is not tuple or len(item) != 2
                or type(item[0]) is not str or type(item[1]) is not str
                or item[0] not in _ACTIONS or item[1] not in nodes for item in node.exits)
                or len({action for action, _ in node.exits}) != len(node.exits)):
            raise ValueError("invalid story exits")
    if (len(nodes["threshold"].exits) < 2
            or len({target for _, target in nodes["threshold"].exits}) < 2
            or nodes["chamber"].exits):
        raise ValueError("world must branch from entrance and terminate at training")
    visited, queue = set(), ["threshold"]
    while queue:
        node_id = queue.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        queue.extend(target for _, target in nodes[node_id].exits)
    if visited != set(nodes):
        raise ValueError("every story scene must be reachable from the entrance")
    if "chamber" not in visited:
        raise ValueError("model world has no reachable training chamber")
    can_reach_chamber = {"chamber"}
    while True:
        earlier_size = len(can_reach_chamber)
        for node in world.nodes:
            if any(target in can_reach_chamber for _, target in node.exits):
                can_reach_chamber.add(node.node_id)
        if len(can_reach_chamber) == earlier_size:
            break
    if can_reach_chamber != set(nodes):
        raise ValueError("every narrative branch must have a route to training")
    if type(world.encounters) is not tuple:
        raise ValueError("pedagogical encounters must be an immutable tuple")
    direct_branches = {target for _, target in nodes["threshold"].exits}
    if world.encounters:
        if (len(world.encounters) != len(direct_branches)
                or any(type(encounter) is not PedagogicalEncounter
                       or type(encounter.branch_node_id) is not str
                       or encounter.branch_node_id not in direct_branches
                       for encounter in world.encounters)):
            raise ValueError("pedagogical encounters must cover each direct branch once")
        if len({encounter.branch_node_id for encounter in world.encounters}) != len(direct_branches):
            raise ValueError("duplicated pedagogical branch consequence")
        for encounter in world.encounters:
            PedagogicalRoute(encounter.tactic, encounter.difficulty_delta)
        if len({(e.tactic, e.difficulty_delta) for e in world.encounters}) < 2:
            raise ValueError("distinct choices must change the TRAIN encounter")
        # A raw shift differs from another shift yet can disappear at the
        # difficulty clamps (e.g. level 5 with 0 vs +1). Reject even these
        # latent equivalences, because the model world must *always* provide
        # meaningfully different source-controlled TRAIN exposure for distinct
        # entrance choices, regardless of live policy adjustment.
        for allowed_baseline in range(1, 6):
            normalized = {
                (e.tactic, max(1, min(5, allowed_baseline + e.difficulty_delta)))
                for e in world.encounters
            }
            if len(normalized) != len(world.encounters):
                raise ValueError("model world branches collapse to the same TRAIN encounter")
    # Bind graph structure and bounded pedagogical consequences to the world
    # fingerprint, never the cosmetic prose. Legacy non-pedagogical proposals
    # retain their preexisting topology hash.
    structure = sorted((node.node_id, sorted(node.exits)) for node in world.nodes)
    if world.encounters:
        encoded_subject = {
            "graph": structure,
            "train_encounters": sorted(
                (e.branch_node_id, e.tactic, e.difficulty_delta)
                for e in world.encounters
            ),
        }
    else:
        encoded_subject = structure
    encoded = json.dumps(encoded_subject, separators=(",", ":"),
                         sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExamPressure:
    """Model suggestion, not authority to construct gold or assign grades."""

    focus_skill: str
    difficulty: int
    tactic: str


@dataclass(frozen=True)
class ExamCoveragePolicy:
    """Operator-owned exam intensity declared independent of skill narrowness.

    For arithmetic DEVELOPMENT_ONLY the source scorer always requires both
    heldout and narrative transfer splits (four fresh cases per split). A
    numeric lower bound must be explicitly chosen for the actual objective;
    the untrusted Minotaur cannot relax it in response to observed weaknesses.
    """

    minimum_difficulty: int

    def __post_init__(self) -> None:
        if type(self.minimum_difficulty) is not int or not 1 <= self.minimum_difficulty <= 5:
            raise ValueError("operator arithmetic exam difficulty must be in [1, 5]")


@dataclass(frozen=True)
class PedagogicalOutcome:
    """Public TRAIN aggregate that can guide the next development pressure."""

    tactic: str
    difficulty: int
    correct: bool


@dataclass(frozen=True)
class ExamPressureContext:
    objective_id: str
    cycle_index: int
    skills: tuple[str, ...]
    weak_skills: tuple[str, ...]
    baseline_correct: int
    baseline_count: int
    prior_training_outcomes: tuple[PedagogicalOutcome, ...] = ()
    minimum_difficulty: int = 1


@dataclass(frozen=True)
class ExaminerRequest:
    """Trusted developmental evaluator input; exam seed stays away from models."""

    objective_id: str
    scope_fingerprint: str
    cycle_index: int
    checkpoint_sha256: str
    baseline_sha256: str
    pressure: ExamPressure
    fresh_seed: int
    minimum_difficulty: int = 1


@dataclass(frozen=True)
class StepResult:
    observation: str
    event: str
    available_actions: tuple[str, ...]
    terminated: bool
    truncated: bool
    checkpoint_sha256: str
    development_status: str
    train_correct: bool | None = None


class SagaLearner(Protocol):
    def predict(self, prompt: str) -> str: ...
    def train_supervised(self, case: Challenge) -> CheckpointPaths: ...


class SceneDirector(Protocol):
    def propose_world(self, context: SceneRequest) -> WorldProposal: ...


class MinotaurDirector(Protocol):
    def propose_pressure(self, context: ExamPressureContext) -> ExamPressure: ...


class DevelopmentExaminer(Protocol):
    def assess(self, request: ExaminerRequest, predict: Callable[[str], str]) -> MinotaurExamReceipt: ...


class OperatorArithmeticDevelopmentExaminer:
    """Local heldout/transfer scorer with model-selected *pressure*, not gold.

    This is an operator-owned DEVELOPMENT examiner only. The caller must
    provide a separately frozen W0 predictor; its independence is not proven
    here. Test cases and gold never enter the scene model or TRAIN hook.
    """

    def __init__(self, baseline_predict: Callable[[str], str]) -> None:
        if not callable(baseline_predict):
            raise ValueError("frozen baseline predictor callback required")
        self._baseline_predict = baseline_predict

    @staticmethod
    def _score(predictor: Callable[[str], str], prompt: str, expected: int) -> float:
        response = predictor(prompt)
        if type(response) is not str or len(response) > 4096:
            raise ValueError("exam inference must return bounded text")
        matched = _ANSWER.fullmatch(response.strip())
        return float(matched is not None and int(matched.group(1)) == expected)

    def assess(self, request: ExaminerRequest, predict: Callable[[str], str]) -> MinotaurExamReceipt:
        if type(request) is not ExaminerRequest or not callable(predict):
            raise ValueError("exact developmental examiner request and prediction callback required")
        pressure = request.pressure
        if (type(request.minimum_difficulty) is not int
                or not 1 <= request.minimum_difficulty <= 5):
            raise ValueError("operator minimum exam difficulty must be in [1, 5]")
        if (type(pressure) is not ExamPressure or pressure.focus_skill != "arithmetic"
                or type(pressure.difficulty) is not int or not 1 <= pressure.difficulty <= 5
                or pressure.difficulty < request.minimum_difficulty
                or pressure.tactic not in ("holdout", "transfer", "mixed")):
            raise ValueError("unsupported arithmetic development pressure")
        # Domain separate from TRAIN routing and from a second development
        # attempt. The public model-selected pressure changes case generation,
        # not answer authority; the scorer is deterministic and source-owned.
        material = (f"labyrinth-operator-math-exam-v1\x1f{request.fresh_seed}\x1f"
                    f"{pressure.focus_skill}\x1f{pressure.difficulty}\x1f{pressure.tactic}")
        seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        low = 10 ** (pressure.difficulty - 1)
        high = min(99999, low * 10 - 1)
        cases: list[ExamCaseResult] = []
        for split in ("heldout", "transfer"):
            for index in range(4):
                a, b = rng.randint(low, high), rng.randint(low, high)
                # A model can intensify difficulty and select the transfer
                # tactic but cannot insert gold. Each split remains explicit.
                addition = rng.choice((True, False))
                expected = a + b if addition else a - b
                if split == "transfer":
                    prompt = (f"A vessel carries {a} units, then "
                              + (f"receives {b} units. " if addition else f"spends {b} units. ")
                              + "What is its final balance?\nRespond with exactly one line: ANSWER: <integer>.")
                else:
                    prompt = (f"Compute {a} {'+' if addition else '-'} {b}.\n"
                              "Respond with exactly one line: ANSWER: <integer>.")
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                cases.append(ExamCaseResult(
                    case_id=f"exam-dev-{request.cycle_index}-{split}-{index}-{prompt_hash[:12]}",
                    skill="arithmetic", split=split,
                    score=self._score(predict, prompt, expected),
                    baseline_score=self._score(self._baseline_predict, prompt, expected),
                    prompt_sha256=prompt_hash,
                ))
        return MinotaurExamReceipt(
            exam_id=f"dev-{request.cycle_index}-{request.fresh_seed:x}",
            objective_id=request.objective_id, cycle_index=request.cycle_index,
            checkpoint_sha256=request.checkpoint_sha256,
            baseline_sha256=request.baseline_sha256,
            evaluator_id="operator-local-arithmetic-dev-v1", cases=tuple(cases),
        )


class SagaSession:
    """Bounded original text journey whose TRAIN/Minotaur states are real source contracts."""

    def __init__(
        self, scope: TrainingScopeSpec, *, learner: SagaLearner,
        checkpoint: CheckpointPaths, baseline_train_cases: tuple[Challenge, ...],
        scene_director: SceneDirector, minotaur_director: MinotaurDirector,
        examiner: DevelopmentExaminer, exam_coverage_policy: ExamCoveragePolicy,
        action_budget: int = 64,
        require_pedagogical_encounters: bool = True,
    ) -> None:
        if type(scope) is not TrainingScopeSpec or scope.skills != ("arithmetic",):
            raise ValueError("first live saga episode supports arithmetic TRAIN only")
        TrainingScopeSpec.__post_init__(scope)
        if type(exam_coverage_policy) is not ExamCoveragePolicy:
            raise ValueError("explicit operator-owned exam coverage policy is required")
        ExamCoveragePolicy.__post_init__(exam_coverage_policy)
        if type(action_budget) is not int or not 6 <= action_budget <= 512:
            raise ValueError("action budget must be in [6, 512]")
        if type(require_pedagogical_encounters) is not bool:
            raise ValueError("pedagogical encounters strictness must be a boolean")
        if (type(baseline_train_cases) is not tuple
                or not 2 <= len(baseline_train_cases) <= 16
                or any(type(case) is not Challenge or case.split != "train"
                       or case.kind != "arithmetic" for case in baseline_train_cases)):
            raise ValueError("baseline needs 2-16 source-owned TRAIN arithmetic probes")
        if (len({case.id for case in baseline_train_cases}) != len(baseline_train_cases)
                or len({case.prompt for case in baseline_train_cases}) != len(baseline_train_cases)):
            raise ValueError("baseline TRAIN probes must be distinct")
        if not callable(getattr(learner, "predict", None)) or not callable(getattr(learner, "train_supervised", None)):
            raise ValueError("operator-owned learner must implement predict and train_supervised")
        for obj, method in ((scene_director, "propose_world"),
                            (minotaur_director, "propose_pressure"),
                            (examiner, "assess")):
            if not callable(getattr(obj, method, None)):
                raise ValueError(f"missing operator-configured {method} collaborator")
        self.scope = scope
        self.learner = learner
        self.scene_director = scene_director
        self.minotaur_director = minotaur_director
        self.examiner = examiner
        self.exam_coverage_policy = exam_coverage_policy
        self.action_budget = action_budget
        self.require_pedagogical_encounters = require_pedagogical_encounters
        self.action_count = 0
        self.terminated = False
        self.truncated = False
        # Once a trainer may have written outside this process, an exception
        # makes the state ambiguous. Never retry an unreceipted update.
        self.indeterminate = False
        self.development_status = "NOT_EVALUATED"
        self.checkpoint = capture_checkpoint(checkpoint)
        self._used_weights_paths = {self.checkpoint.weights_path}
        self._used_optimizer_paths = ({self.checkpoint.optimizer_path}
                                      if self.checkpoint.optimizer_path else set())
        baseline_correct = 0
        prompt_hashes: list[str] = []
        for case in baseline_train_cases:
            response = learner.predict(case.prompt)
            if type(response) is not str or len(response) > 4096:
                raise ValueError("learner returned invalid baseline response")
            match = _ANSWER.fullmatch(response.strip())
            baseline_correct += int(match is not None and str(int(match.group(1))) == case.answer)
            prompt_hashes.append(hashlib.sha256(case.prompt.encode("utf-8")).hexdigest())
        self.entrance = EntranceState(
            scope.objective_id, scope.fingerprint, self.checkpoint,
            BaselineMeasurement(len(baseline_train_cases), baseline_correct, tuple(prompt_hashes)),
        )
        self.checkpoint_refs = [self.checkpoint]
        self.training_sparks: list[TrainingSpark] = []
        self.ariadne_events: list[dict] = []
        self._record("entrance", 0, self.checkpoint.weights_sha256,
                     self.checkpoint.weights_sha256, {
                         "checkpoint_path": self.checkpoint.weights_path,
                         "optimizer_status": self.checkpoint.continuity,
                         "optimizer_sha256": self.checkpoint.optimizer_sha256,
                         "baseline_count": self.entrance.baseline.case_count,
                         "baseline_correct": self.entrance.baseline.correct_count,
                     })
        self._baseline_case_ids = frozenset(case.id for case in baseline_train_cases)
        self._baseline_prompt_hashes = frozenset(prompt_hashes)
        self.journey = begin_journey(scope, checkpoint_sha256=self.checkpoint.weights_sha256)
        self.runner = TrainingCorridorRunner.begin(
            scope, self.journey.maze, checkpoint_sha256=self.checkpoint.weights_sha256,
        )
        self._public_training_history: list[PedagogicalOutcome] = []
        self._cycle_first_topology: dict[int, str] = {}
        self._exam_attempt = 0
        self._prepare_world()

    def _record(self, kind: str, cycle: int, before: str, after: str, summary: dict) -> None:
        append_saga_event(
            self.ariadne_events, kind=kind, objective_id=self.scope.objective_id,
            scope_fingerprint=self.scope.fingerprint, cycle_index=cycle,
            before=before, after=after, summary=summary,
        )
        verify_saga_ariadne(self.ariadne_events)

    def _prepare_world(self) -> None:
        # Preview the authorized next assignment without generating or
        # exposing its TRAIN task. The path actually walked will now select
        # the approved tactic/difficulty before source-owned case creation.
        preview = offer_next_chamber(self.runner.state).pending
        if preview is None:
            raise ValueError("a living world needs an authorized next chamber")
        self.view = None
        recent_outcomes = tuple(self._public_training_history[-_PUBLIC_OUTCOME_WINDOW:])
        context = SceneRequest(
            objective_id=self.scope.objective_id, domain=self.scope.domain,
            skill=preview.skill, cycle_index=self.journey.cycle_index,
            room_index=len(self.runner.observations), specificity=self.scope.specificity,
            weak_skills=(self.journey.attempts[-1].weak_skills if self.journey.attempts else ()),
            prior_training_correct=tuple(item.correct for item in recent_outcomes),
            public_seed=int.from_bytes(hashlib.sha256(
                ("labyrinth-original-world-v1\x1f" + self.scope.fingerprint + "\x1f"
                 + str(self.journey.cycle_index) + "\x1f" + preview.room_id).encode()
            ).digest()[:8], "big"),
            prior_training_tactics=tuple(item.tactic for item in recent_outcomes),
            prior_training_outcomes=recent_outcomes,
        )
        world = self.scene_director.propose_world(context)
        if (type(world) is WorldProposal and self.require_pedagogical_encounters
                and not world.encounters):
            raise ValueError("living labyrinth requires consequential pedagogical encounters")
        topology = _check_world(world)
        cycle = self.journey.cycle_index
        if cycle not in self._cycle_first_topology:
            if cycle and topology == self._cycle_first_topology.get(cycle - 1):
                raise ValueError("regenerated world topology must differ from previous cycle")
            self._cycle_first_topology[cycle] = topology
        self.world = world
        self.nodes = {node.node_id: node for node in world.nodes}
        self._encounter_by_branch = {
            encounter.branch_node_id: PedagogicalRoute(
                encounter.tactic, encounter.difficulty_delta,
            ) for encounter in world.encounters
        }
        self._selected_route = None
        self._selected_branch = None
        self.location = "threshold"
        self.phase = "EXPLORE"
        self._record("scene", cycle, self.checkpoint.weights_sha256,
                     self.checkpoint.weights_sha256, {
                         "skill": preview.skill,
                         "room_index": len(self.runner.observations),
                         "topology_sha256": topology,
                     })

    def _current_actions(self) -> tuple[str, ...]:
        if self.terminated or self.truncated or self.indeterminate:
            return ()
        if self.phase == "MINOTAUR":
            return ("face", "look")
        node = self.nodes[self.location]
        return ("look",) + tuple(action for action, _ in node.exits) + (("train",) if self.location == "chamber" else ())

    def observe(self) -> StepResult:
        if self.indeterminate:
            msg = "Training state is indeterminate after a possible partial update; manual checkpoint recovery is required."
        elif self.terminated:
            msg = "The development journey has ended; no formal graduation has been established."
        elif self.truncated:
            msg = "The authorized action budget ended; no further training or examination is allowed."
        elif self.phase == "MINOTAUR":
            msg = "The Minotaur awaits a measured development confrontation; no verdict has been assumed."
        else:
            msg = self.nodes[self.location].prose
        return StepResult(
            observation=msg, event="OBSERVE", available_actions=self._current_actions(),
            terminated=self.terminated, truncated=self.truncated,
            checkpoint_sha256=self.checkpoint.weights_sha256,
            development_status=self.development_status,
        )

    def _train(self) -> StepResult:
        # Only the prompt-only view is provided to prediction. The trusted
        # operator-side update gets TRAIN supervision *after* prediction.
        if type(self._selected_branch) is not str or self._selected_branch not in self.nodes:
            raise ValueError("TRAIN lacks a recorded chosen world branch")
        view = self.view
        response = self.learner.predict(view.prompt)
        if type(response) is not str or len(response) > 4096:
            raise ValueError("learner returned invalid TRAIN response")
        case = self.runner.operator_pending_train_case(view)
        previous = self.checkpoint
        self._verify_historical_artifacts()
        old_development_status = self.development_status
        self.indeterminate = True
        self.development_status = "INDETERMINATE"
        paths = self.learner.train_supervised(case)
        # Check every retained checkpoint, not only immediate predecessor:
        # otherwise a later spark could silently rewrite W0 or an older Wt.
        self._verify_historical_artifacts(after_training=True)
        successor = capture_checkpoint(paths)
        if (successor.weights_path in self._used_weights_paths
                or successor.weights_sha256 == previous.weights_sha256):
            raise ValueError("TRAIN hook did not produce a new write-once checkpoint state")
        if successor.optimizer_path is not None and successor.optimizer_path in self._used_optimizer_paths:
            raise ValueError("optimizer artifact path was reused")
        if (previous.optimizer_path is not None and successor.optimizer_path is None):
            raise ValueError("cannot silently discard optimizer state during TRAIN")
        # This is local byte readback, not a hardware witness or proof of
        # effectiveness. A crashed hook can still leave an orphan artifact.
        outcome = self.runner.submit_training_response(
            response, checkpoint_after_sha256=successor.weights_sha256,
        )
        self._used_weights_paths.add(successor.weights_path)
        if successor.optimizer_path is not None:
            self._used_optimizer_paths.add(successor.optimizer_path)
        self.checkpoint = successor
        self.checkpoint_refs.append(successor)
        self._public_training_history.append(PedagogicalOutcome(
            outcome.tactic, outcome.difficulty, outcome.correct,
        ))
        self.training_sparks.append(TrainingSpark(
            cycle_index=self.journey.cycle_index,
            room_index=len(self.runner.observations) - 1,
            skill=view.skill, before_sha256=previous.weights_sha256,
            after=successor, train_correct=outcome.correct,
        ))
        self._record("train", self.journey.cycle_index,
                     previous.weights_sha256, successor.weights_sha256, {
                         "skill": view.skill,
                         "room_index": len(self.runner.observations) - 1,
                         "train_correct": outcome.correct,
                         "policy_before_sha256": outcome.policy_before_sha256,
                         "policy_after_sha256": outcome.policy_after_sha256,
                         "checkpoint_path": successor.weights_path,
                         "branch_node_id": self._selected_branch,
                         "train_tactic": outcome.tactic,
                         "train_difficulty": outcome.difficulty,
                         "train_prompt_sha256": view.prompt_sha256,
                     })
        if self.runner.is_complete:
            receipt = self.runner.self_reported_completion_receipt()
            self.journey = complete_maze(self.journey, receipt)
            self.phase = "MINOTAUR"
            self.view = None
            next_text = "The corridor is complete. The Minotaur waits with separately held development trials."
        else:
            self._prepare_world()
            next_text = self.nodes[self.location].prose
        # Only after the training hook, all old/new checkpoint readbacks,
        # runner update and Ariadne record have completed can another action
        # be allowed. Any exception above leaves this session quarantined.
        self.indeterminate = False
        self.development_status = old_development_status
        return self._result("TRAIN_COMPLETED", next_text, outcome.correct)

    def _verify_historical_artifacts(self, *, after_training: bool = False) -> None:
        """Detect in-place edits to every prior weight/optimizer file.

        This is intentionally expensive for large checkpoints. A future
        host-owned immutable CAS/witness should replace O(history) rehashing.
        """
        for prior in self.checkpoint_refs:
            readback = capture_checkpoint(CheckpointPaths(prior.weights_path, prior.optimizer_path))
            if readback != prior:
                reason = ("TRAIN hook overwrote a retained checkpoint"
                          if after_training else "retained checkpoint artifact changed unexpectedly")
                raise ValueError(reason)

    def _face_minotaur(self) -> StepResult:
        recent_outcomes = tuple(self._public_training_history[-_PUBLIC_OUTCOME_WINDOW:])
        context = ExamPressureContext(
            objective_id=self.scope.objective_id, cycle_index=self.journey.cycle_index,
            skills=self.scope.skills,
            weak_skills=self.journey.attempts[-1].weak_skills if self.journey.attempts else (),
            baseline_correct=self.entrance.baseline.correct_count,
            baseline_count=self.entrance.baseline.case_count,
            prior_training_outcomes=recent_outcomes,
            minimum_difficulty=self.exam_coverage_policy.minimum_difficulty,
        )
        proposed_pressure = self.minotaur_director.propose_pressure(context)
        if (type(proposed_pressure) is not ExamPressure
                or proposed_pressure.focus_skill not in self.scope.skills
                or type(proposed_pressure.difficulty) is not int
                or not 1 <= proposed_pressure.difficulty <= 5
                or proposed_pressure.tactic not in ("holdout", "transfer", "mixed")):
            raise ValueError("invalid Minotaur pressure proposal")
        # Difficulty coverage is operator-owned by the explicit exam rubric,
        # not training specificity (which measures scope narrowness). The
        # Minotaur may intensify but never relax this owner-set requirement.
        pressure = ExamPressure(
            proposed_pressure.focus_skill,
            max(self.exam_coverage_policy.minimum_difficulty, proposed_pressure.difficulty),
            proposed_pressure.tactic,
        )
        exam_seed = int.from_bytes(hashlib.sha256(
            ("labyrinth-operator-development-exam-v1\x1f" + self.scope.fingerprint
             + "\x1f" + str(self.journey.cycle_index) + "\x1f"
             + str(self._exam_attempt)).encode()
        ).digest()[:8], "big")
        request = ExaminerRequest(
            self.scope.objective_id, self.scope.fingerprint,
            self.journey.cycle_index, self.checkpoint.weights_sha256,
            self.entrance.checkpoint.weights_sha256, pressure, exam_seed,
            minimum_difficulty=self.exam_coverage_policy.minimum_difficulty,
        )
        self._exam_attempt += 1
        receipt = self.examiner.assess(request, self.learner.predict)
        if type(receipt) is not MinotaurExamReceipt:
            raise ValueError("trusted examiner must return exact development evidence")
        if (set(case.case_id for case in receipt.cases) & self._baseline_case_ids
                or set(case.prompt_sha256 for case in receipt.cases) & self._baseline_prompt_hashes):
            raise ValueError("development exam reused the entrance TRAIN baseline")
        previous_cycle = self.journey.cycle_index
        decision = evaluate_minotaur(self.scope, receipt)
        updated = face_minotaur(self.journey, receipt)
        self.journey = updated
        self.development_status = decision.status
        self._record("verdict", previous_cycle,
                     self.checkpoint.weights_sha256, self.checkpoint.weights_sha256, {
                         "status": decision.status,
                         "weak_skills": list(decision.weak_skills),
                         "exam_count": len(receipt.cases),
                         "candidate_score": sum(case.score for case in receipt.cases),
                         "baseline_score": sum(case.baseline_score for case in receipt.cases),
                         "focus_skill": pressure.focus_skill,
                         "pressure_difficulty": pressure.difficulty,
                         "pressure_tactic": pressure.tactic,
                     })
        if updated.phase == TRAINING:
            self.runner = self.runner.regenerate(updated)
            self._record("regeneration", updated.cycle_index,
                         self.checkpoint.weights_sha256, self.checkpoint.weights_sha256, {
                             "weak_skills": list(decision.weak_skills),
                             "new_maze_plan_sha256": hashlib.sha256(
                                 repr(updated.maze).encode("utf-8")
                             ).hexdigest(),
                         })
            self._prepare_world()
            story = "The Minotaur found weakness; the labyrinth regenerated without resetting the checkpoint. " + self.nodes[self.location].prose
        elif updated.phase == AWAITING_EXAM:
            story = "The Minotaur's evidence was inconclusive; another fresh development trial is required."
        elif updated.phase == DEVELOPMENT_PASSED_PENDING_PRIVATE_QUALIFICATION:
            self.terminated = True
            story = "Development criteria passed; private graduation remains unestablished."
        elif updated.phase == EXHAUSTED:
            self.terminated = True
            story = "The journey budget ended without formal graduation."
        else:
            raise ValueError("unexpected Minotaur development phase")
        if self.terminated:
            self._record("exit", updated.cycle_index,
                         self.checkpoint.weights_sha256, self.checkpoint.weights_sha256,
                         {"phase": updated.phase})
        return self._result("MINOTAUR_" + decision.status, story)

    def _result(self, event: str, text: str, correct: bool | None = None) -> StepResult:
        return StepResult(
            observation=text, event=event, available_actions=self._current_actions(),
            terminated=self.terminated, truncated=self.truncated,
            checkpoint_sha256=self.checkpoint.weights_sha256,
            development_status=self.development_status, train_correct=correct,
        )

    def step(self, action: str) -> StepResult:
        if self.indeterminate:
            raise ValueError("indeterminate training state requires manual checkpoint recovery")
        if self.terminated or self.truncated:
            raise ValueError("development episode has ended")
        if type(action) is not str or not 1 <= len(action) <= 16 or action not in self._current_actions():
            raise ValueError("invalid action for the current scene")
        self.action_count += 1
        if self.action_count > self.action_budget:
            self.truncated = True
            self._record("exit", self.journey.cycle_index,
                         self.checkpoint.weights_sha256, self.checkpoint.weights_sha256,
                         {"phase": "ACTION_BUDGET_EXHAUSTED"})
            return self._result("ACTION_BUDGET_EXHAUSTED", "The authorized action budget ended.")
        if action == "look":
            return self.observe()
        if self.phase == "MINOTAUR":
            return self._face_minotaur()
        if action == "train":
            return self._train()
        next_node = dict(self.nodes[self.location].exits)[action]
        if self.location == "threshold":
            self._selected_route = self._encounter_by_branch.get(next_node)
            self._selected_branch = next_node
        if next_node == "chamber" and self.view is None:
            old_development_status = self.development_status
            # Source-side task admission/registry updates are effects too:
            # a partial failure must not be retried as a fresh prompt.
            self.indeterminate = True
            self.development_status = "INDETERMINATE"
            self.view = self.runner.next_training_prompt(route=self._selected_route)
            self.indeterminate = False
            self.development_status = old_development_status
        self.location = next_node
        return self._result("MOVE", self.nodes[next_node].prose)

    def save_receipt_new(self, path: str | Path) -> str:
        """Write once and read back local evidence; never an independent witness.

        Includes absolute operator-local artifact paths for actual recovery;
        keep these reports private if filesystem locations are sensitive.
        """
        from .runner import load_json, save_json_new

        if self.indeterminate:
            raise ValueError("cannot export a successful Ariadne receipt for indeterminate training")

        head = verify_saga_ariadne(
            self.ariadne_events, require_exit=self.terminated or self.truncated,
        )
        self._verify_historical_artifacts()
        manifest = [
            {
                "weights_path": ref.weights_path,
                "weights_sha256": ref.weights_sha256,
                "optimizer_path": ref.optimizer_path,
                "optimizer_sha256": ref.optimizer_sha256,
                "continuity": ref.continuity,
            }
            for ref in self.checkpoint_refs
        ]
        value = {
            "schema": "labyrinth.model-saga-receipt.v1",
            "objective_id": self.scope.objective_id,
            "scope_fingerprint": self.scope.fingerprint,
            "ariadne_events": self.ariadne_events,
            "ariadne_head_sha256": head,
            "checkpoint_manifest": manifest,
            "development_status": self.development_status,
            "session_status": ("TERMINATED" if self.terminated else
                               "TRUNCATED" if self.truncated else "IN_PROGRESS"),
            "qualification": "NOT_ESTABLISHED",
        }
        save_json_new(path, value)
        actual = load_json(path, strict=True)
        if actual != value or verify_saga_ariadne(actual["ariadne_events"]) != head:
            raise ValueError("local Ariadne receipt readback disagrees")
        return head


__all__ = [
    "BaselineMeasurement", "CheckpointPaths", "CheckpointRef", "EntranceState", "TrainingSpark",
    "SceneRequest", "SceneNode", "PedagogicalEncounter", "WorldProposal", "ExamPressure", "ExamCoveragePolicy", "ExamPressureContext", "PedagogicalOutcome",
    "ExaminerRequest", "StepResult", "SagaSession", "capture_checkpoint",
    "OperatorArithmeticDevelopmentExaminer",
]
