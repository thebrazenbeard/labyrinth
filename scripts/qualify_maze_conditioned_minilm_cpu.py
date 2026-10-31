"""Opt-in, offline MiniLM LoRA probe driven by *actual* regenerated Maze rooms.

The approved demonstration objective is binary sentiment polarity. Every
training example is created from a TrainingScopeSpec -> MazePlan -> MazeRoom;
there are no static TRAIN cases. The learner's adapter and head survive a new
maze and are reloaded into a *new model object* before cycle two.

All tests here are public development tests. This harness cannot authorize
graduation, provide a protected final exam, or demonstrate broad generalization.
Dry-run is the default and never loads torch, downloads weights or creates a run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable

from labyrinth.regenerative_maze import MazePlan, generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec, parse_scope_json
from scripts.qualify_minilm_cpu import (
    ARMS, HELDOUT as PUBLIC_DEVELOPMENT, MODEL_ID, TRANSFER as PUBLIC_TRANSFER,
    _sha_file, comparison, plan_output,
)


SCHEMA = "labyrinth.maze-conditioned-minilm-cpu-development.v1"
PREREG_SCHEMA = "labyrinth.maze-conditioned-minilm-five-arm-public-controls.v1"
# The original three-arm physical protocol remains the default. Extra arms
# must be explicitly requested; neither mode qualifies a protected exam.
PREREG_ARMS = ARMS + ("zero_lr", "randomized_labels")
OBJECTIVE = "sentiment.binary_polarity"
SKILLS = ("positive_polarity", "negative_polarity")
SKILL_LABEL = {"positive_polarity": 1, "negative_polarity": 0}
SUBJECTS = (
    "service", "documentation", "repair", "response", "guidance", "answer",
    "solution", "design", "instruction", "report", "delivery", "result",
    "support", "process", "diagnosis", "schedule", "tutorial", "interface",
    "workshop", "explanation", "communication", "review", "implementation",
    "experience",
)
POSITIVES = (
    "excellent", "thoughtful", "reliable", "helpful", "clear", "effective",
    "accurate", "pleasant", "impressive", "valuable", "efficient", "useful",
    "wonderful", "friendly", "reassuring", "satisfying", "successful",
)
NEGATIVES = (
    "terrible", "confusing", "unreliable", "useless", "disappointing",
    "inaccurate", "hostile", "frustrating", "misleading", "wasteful",
    "unpleasant", "broken", "awful", "distressing", "ineffective",
    "unsatisfactory", "harmful",
)
TEMPLATES = (
    "The {subject} was {adjective}.",
    "I found the {subject} {adjective}.",
    "Our team called the {subject} {adjective}.",
    "After trying it, the {subject} seemed {adjective}.",
    "The customer thought the {subject} was {adjective}.",
    "In my opinion, the {subject} felt {adjective}.",
)


@dataclass(frozen=True)
class TrainingExample:
    cycle_index: int
    room_id: str
    case_seed: int
    skill: str
    exposure_id: str
    prompt: str
    label: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "room_id": self.room_id,
            "case_seed": self.case_seed,
            "skill": self.skill,
            "exposure_id": self.exposure_id,
            "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "label": self.label,
            "split": "train",
        }


@dataclass(frozen=True)
class TrainingCycle:
    plan: MazePlan
    examples: tuple[TrainingExample, ...]


@dataclass(frozen=True)
class OptimizerContinuityRequest:
    """Opt in to same-process AdamW continuity, not independent authority.

    The predecessor digest and fence are caller-asserted values. This runner
    cannot establish their ownership or freshness; it will never report them
    as host-issued credentials or claim recovery after a process crash.
    """

    logical_run_id: str
    prior_checkpoint_sha256: str
    fence: int

    def validate(self) -> None:
        if (type(self.logical_run_id) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}",
                                self.logical_run_id, re.ASCII) is None
                or type(self.prior_checkpoint_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}",
                                self.prior_checkpoint_sha256, re.ASCII) is None
                or type(self.fence) is not int
                or not 0 <= self.fence < 2**63 - 1):
            raise ValueError("optimizer continuity requires bounded operator pins")


@dataclass(frozen=True)
class MeasuredNeuralExecutionRequest:
    """Caller-supplied external pins for an opt-in signed TRAIN schedule.

    Supplying these values does not establish that an independent host owns
    them. The runner verifies the exact signed-source schedule before loading
    the neural backend, then uses one schedule for all five matched arms.
    """

    schedule: Any
    independently_pinned_schedule_sha256: str
    run_id: str
    candidate_artifact_sha256: str
    predecessor_checkpoint_sha256: str
    scope_fingerprint: str
    fence: int


def default_scope() -> TrainingScopeSpec:
    return TrainingScopeSpec(
        objective_id=OBJECTIVE, domain="sentiment", skills=SKILLS,
        specificity=2, scale=3, max_cycles=2, seed=20261030,
    )


def validate_scope(scope: TrainingScopeSpec) -> None:
    """Require an exact approved semantic provider and bounded CPU work."""
    if type(scope) is not TrainingScopeSpec:
        raise ValueError("scope must be a TrainingScopeSpec")
    TrainingScopeSpec.__post_init__(scope)
    if (scope.objective_id != OBJECTIVE or scope.domain != "sentiment"
            or scope.skills != SKILLS):
        raise ValueError("no approved generator for this declared objective/skills")
    if scope.max_cycles != 2:
        raise ValueError("continuity probe requires exactly two maze cycles")
    if not 8 <= scope.total_chambers <= 24:
        raise ValueError("CPU probe requires 8..24 training rooms per cycle")


def _normalized(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def _example(scope: TrainingScopeSpec, plan: MazePlan, room: Any) -> TrainingExample:
    if room.skill not in SKILL_LABEL or room.purpose != "train":
        raise ValueError("room has no approved training generator")
    # Seed is from the real Maze plan. No room ID/seed/answer is inserted in
    # model-visible text: unique IDs must not conceal duplicate prompts.
    rng = random.Random(room.case_seed)
    adjective = rng.choice(POSITIVES if SKILL_LABEL[room.skill] else NEGATIVES)
    prompt = rng.choice(TEMPLATES).format(
        subject=rng.choice(SUBJECTS), adjective=adjective,
    )
    exposure_id = hashlib.sha256(
        f"train\x1f{scope.fingerprint}\x1f{plan.cycle_index}\x1f"
        f"{room.room_id}\x1f{room.case_seed}\x1f{prompt}".encode("utf-8")
    ).hexdigest()
    return TrainingExample(
        cycle_index=plan.cycle_index, room_id=room.room_id,
        case_seed=room.case_seed, skill=room.skill, exposure_id=exposure_id,
        prompt=prompt, label=SKILL_LABEL[room.skill],
    )


def validate_cycles(cycles: tuple[TrainingCycle, ...]) -> dict[str, Any]:
    """Reject repeated cases and any public development/train contamination."""
    if type(cycles) is not tuple or len(cycles) != 2:
        raise ValueError("expected exactly two generated training cycles")
    dev = PUBLIC_DEVELOPMENT + PUBLIC_TRANSFER
    if (len({_normalized(prompt) for prompt, _ in dev}) != len(dev)
            or any(label not in (0, 1) for _, label in dev)):
        raise ValueError("invalid public development examples")
    seen_prompts = {_normalized(prompt) for prompt, _ in dev}
    ids: set[str] = set()
    seeds: set[int] = set()
    exposure_ids: set[str] = set()
    counts = []
    for index, cycle in enumerate(cycles):
        plan = cycle.plan
        if plan.cycle_index != index or len(cycle.examples) != len(plan.rooms):
            raise ValueError("training cycle and maze plan mismatch")
        if (tuple(row.room_id for row in cycle.examples)
                != tuple(room.room_id for room in plan.rooms)):
            raise ValueError("example order is not bound to the Maze rooms")
        labels = {0: 0, 1: 0}
        for row, room in zip(cycle.examples, plan.rooms):
            if (row.cycle_index != index or row.case_seed != room.case_seed
                    or row.skill != room.skill or row.label != SKILL_LABEL.get(room.skill)
                    or row != _example_for_validation(plan, room, row)):
                raise ValueError("training example diverged from its authentic room")
            key = _normalized(row.prompt)
            if key in seen_prompts:
                raise ValueError("repeated training case or public development overlap")
            if (row.room_id in ids or row.case_seed in seeds
                    or row.exposure_id in exposure_ids):
                raise ValueError("room/case-seed/exposure identity repeated")
            ids.add(row.room_id)
            seeds.add(row.case_seed)
            exposure_ids.add(row.exposure_id)
            seen_prompts.add(key)
            labels[row.label] += 1
        if any(value < 2 for value in labels.values()):
            raise ValueError("each training cycle must cover both skills")
        counts.append(labels)
    return {
        "cycle_label_counts": counts,
        "unique_room_ids": len(ids), "unique_case_seeds": len(seeds),
        "unique_training_prompts": len(seen_prompts) - len(dev),
        "unique_exposure_ids": len(exposure_ids),
        "public_development_count": len(PUBLIC_DEVELOPMENT),
        "public_transfer_count": len(PUBLIC_TRANSFER),
    }


def _example_for_validation(plan: MazePlan, room: Any, row: TrainingExample) -> TrainingExample:
    # The scope fingerprint and plan are part of exposure identity; the
    # declaration is reconstructed from the same approved scope by the caller.
    # Validate text deterministically from the stored Maze seed, separately
    # from the model training loop.
    rng = random.Random(room.case_seed)
    adjective = rng.choice(POSITIVES if SKILL_LABEL[room.skill] else NEGATIVES)
    prompt = rng.choice(TEMPLATES).format(
        subject=rng.choice(SUBJECTS), adjective=adjective,
    )
    expected_id = hashlib.sha256(
        f"train\x1f{plan.scope_fingerprint}\x1f{plan.cycle_index}\x1f"
        f"{room.room_id}\x1f{room.case_seed}\x1f{prompt}".encode("utf-8")
    ).hexdigest()
    return TrainingExample(plan.cycle_index, room.room_id, room.case_seed,
                           room.skill, expected_id, prompt, SKILL_LABEL[room.skill])


def build_training_cycles(
    scope: TrainingScopeSpec, *, weak_cycle_one: tuple[str, ...] = (),
) -> tuple[TrainingCycle, ...]:
    validate_scope(scope)
    if type(weak_cycle_one) is not tuple:
        raise ValueError("weak_cycle_one must be a tuple of declared skills")
    plans = (
        generate_maze_plan(scope, 0),
        generate_maze_plan(scope, 1, weak_skills=weak_cycle_one),
    )
    cycles = tuple(TrainingCycle(
        plan=plan,
        examples=tuple(_example(scope, plan, room) for room in plan.rooms),
    ) for plan in plans)
    validate_cycles(cycles)
    return cycles


def _rows(cycle: TrainingCycle) -> tuple[tuple[str, int], ...]:
    """Exactly the generated room rows, with labels used only as trainer gold."""
    return tuple((row.prompt, row.label) for row in cycle.examples)


def select_training_batch(cycle: TrainingCycle, step: int) -> tuple[TrainingExample, ...]:
    """Pure selector shared by source tests and the physical optimizer loop."""
    if type(step) is not int or step < 0 or not cycle.examples:
        raise ValueError("step must be a nonnegative integer and cycle nonempty")
    return tuple(cycle.examples[(step * 4 + offset) % len(cycle.examples)]
                 for offset in range(4))


def select_executed_training_batch(
    cycle: TrainingCycle, step: int, arm: str,
    verified_measured_schedule: Any | None = None,
) -> tuple[TrainingExample, ...]:
    """The actual neural loop selector, shared with source custody tests.

    ``verified_measured_schedule`` is only admitted after execute() has
    revalidated its externally pinned digest and full signed-source identity.
    It changes regenerated cycle one's TRAIN ordering, never the control arm
    label assignment or the first-cycle training distribution.
    """
    if verified_measured_schedule is None or cycle.plan.cycle_index == 0:
        return select_training_batch(cycle, step)
    if cycle.plan.cycle_index != 1 or cycle != verified_measured_schedule.cycle:
        raise ValueError("measured neural batch applied to foreign Maze cycle")
    return verified_measured_schedule.batch_for_arm(arm, step)


def plan_signed_measured_sentiment_batches(
    *, declared_scope: TrainingScopeSpec,
    training_cycles: tuple[TrainingCycle, ...],
    steps_per_cycle: int,
    binding: Any,
    training_exposures: tuple[Any, ...],
    expected_training_roster_sha256: str,
    **verification_inputs: Any,
) -> Any:
    """Opt-in TRAIN batch plan from the four-key signed development verifier.

    Unlike a caller-provided `MinotaurDevelopmentFeedback` dataclass, all
    weakness/tactic decisions are recomputed from authenticated assessment,
    candidate lineage, successor receipt, and companion binding bytes. The
    exact first-cycle TRAIN roster must match the actual generated examples.

    This returns a development-only immutable batch schedule, not host custody
    of signing keys, physical model execution, or an authorization to graduate.
    The existing default three-/five-arm physical runner is unchanged; a host
    must independently attest and reserve a real candidate transition before
    allowing this schedule into physical optimizer execution.
    """
    validate_scope(declared_scope)
    validate_cycles(training_cycles)
    if (type(steps_per_cycle) is not int or not 4 <= steps_per_cycle <= 16
            or verification_inputs.get("scope") != declared_scope):
        raise ValueError("signed development scope or batch budget mismatch")
    from labyrinth.measured_weakness_feedback import (
        TrainingExposureRef, training_roster_commitment,
    )

    authentic_first = build_training_cycles(declared_scope)[0]
    if training_cycles[0] != authentic_first:
        raise ValueError("cycle zero differs from its declared TRAIN Maze")
    actual_exposures = tuple(
        TrainingExposureRef(
            row.exposure_id, hashlib.sha256(row.prompt.encode("utf-8")).hexdigest(),
        ) for row in authentic_first.examples
    )
    if (training_exposures != actual_exposures
            or expected_training_roster_sha256
            != training_roster_commitment(actual_exposures)):
        raise ValueError("signed candidate TRAIN roster is not actual first-cycle TRAIN")

    # Import inside this opt-in seam: the sentiment scheduler imports this
    # source harness to reuse its approved generated TRAIN case identities.
    from labyrinth.minotaur_development_feedback import (
        derive_minotaur_development_feedback,
    )
    from labyrinth.measured_sentiment_corridors import (
        plan_measured_sentiment_corridor,
    )

    feedback = derive_minotaur_development_feedback(
        binding=binding,
        training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        **verification_inputs,
    )
    planned = plan_measured_sentiment_corridor(
        declared_scope, feedback, steps_per_cycle=steps_per_cycle,
    )
    if training_cycles[1] != planned.cycle:
        raise ValueError("cycle one was not generated from the signed weak polarity")
    return planned


def configured_arms(preregistered_controls: bool) -> tuple[str, ...]:
    if type(preregistered_controls) is not bool:
        raise ValueError("control mode must be a boolean")
    return PREREG_ARMS if preregistered_controls else ARMS


def preregistered_randomized_labels(
    cycle: TrainingCycle, scope_seed: int,
) -> dict[str, int]:
    """Pinned TRAIN-only label permutation, not an independent holdout.

    Exactly preserves this cycle's binary label counts and Maze visits, while
    excluding an unchanged assignment or deterministic full inversion.
    """
    if (type(cycle) is not TrainingCycle or type(scope_seed) is not int
            or scope_seed < 0 or not cycle.examples
            or any(type(row) is not TrainingExample or row.label not in (0, 1)
                   for row in cycle.examples)):
        raise ValueError("valid generated binary TRAIN cycle and seed required")
    labels = [row.label for row in cycle.examples]
    if min(labels.count(0), labels.count(1)) < 2:
        raise ValueError("randomized control requires both TRAIN classes")
    identity = tuple(labels)
    inverse = tuple(1-label for label in labels)
    source = (
        f"labyrinth.maze-preregistered-labels.v1\0{scope_seed}\0"
        f"{cycle.plan.scope_fingerprint}\0{cycle.plan.cycle_index}"
    ).encode("ascii")
    rng = random.Random(int.from_bytes(hashlib.sha256(source).digest(), "big"))
    for _ in range(128):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        # A one-step run visits only the first four cases. The randomized
        # arm must differ on at least one *actually visited* case even at
        # that minimum budget, not only on unvisited rows farther down.
        if (tuple(shuffled) not in (identity, inverse)
                and any(shuffled[index] != labels[index]
                        for index in range(min(4, len(labels))))):
            return {row.exposure_id: label
                    for row, label in zip(cycle.examples, shuffled)}
    raise ValueError("no nontrivial class-preserving label permutation found")


def optimizer_learning_rate(arm: str, nominal_rate: float) -> float:
    if (type(arm) is not str or arm not in PREREG_ARMS
            or type(nominal_rate) not in (float, int)
            or not math.isfinite(nominal_rate) or not 0 < nominal_rate <= .01):
        raise ValueError("invalid arm or nominal learning rate")
    return 0.0 if arm == "zero_lr" else float(nominal_rate)


def perform_optimizer_step(arm: str, optimizer: Any) -> None:
    """zero_lr intentionally runs optimizer.step(), unlike zero_update."""
    if arm not in PREREG_ARMS:
        raise ValueError("unsupported training control arm")
    if arm != "zero_update":
        optimizer.step()


def training_batch_labels(
    batch: tuple[TrainingExample, ...], arm: str,
    randomized_labels: dict[str, int] | None = None,
) -> tuple[int, ...]:
    """Exactly the labels passed to loss(), with control provenance explicit."""
    if (type(batch) is not tuple or not batch
            or type(arm) is not str or arm not in PREREG_ARMS
            or any(type(row) is not TrainingExample or row.label not in (0, 1)
                   for row in batch)):
        raise ValueError("invalid generated TRAIN batch or arm")
    if arm == "randomized_labels":
        if (type(randomized_labels) is not dict
                or any(row.exposure_id not in randomized_labels
                       or type(randomized_labels[row.exposure_id]) is not int
                       or randomized_labels[row.exposure_id] not in (0, 1)
                       for row in batch)):
            raise ValueError("randomized-label arm requires valid pinned assignments")
        return tuple(randomized_labels[row.exposure_id] for row in batch)
    if randomized_labels is not None:
        raise ValueError("label permutation supplied for a nonrandomized arm")
    return tuple(row.label ^ int(arm == "inverted") for row in batch)


def preregistered_control_comparison(scores: dict[str, float]) -> dict[str, Any]:
    if (type(scores) is not dict or set(scores) != set(PREREG_ARMS)
            or any(type(value) not in (float, int) or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in scores.values())):
        raise ValueError("complete five-arm finite public scores required")
    return {
        "correct_minus_controls": {
            arm: float(scores["correct"] - scores[arm])
            for arm in PREREG_ARMS if arm != "correct"
        },
        "correct_exceeds_every_control": all(
            scores["correct"] > scores[arm]
            for arm in PREREG_ARMS if arm != "correct"
        ),
        "qualification": "PUBLIC_DEVELOPMENT_ONLY",
        "independent_generalization_qualified": False,
    }


def hash_named_weights(named: Iterable[tuple[str, Any]]) -> str:
    """Byte-exact weight identity, including name, shape and dtype.

    This is not an origin attestation. Called on *all* freshly loaded pretrained
    base weights and on frozen PEFT parameters at each cycle boundary.
    """
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(named):
        count += 1
        if type(name) is not str or not name:
            raise ValueError("invalid model tensor name")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if not count:
        raise ValueError("model weight identity requires nonempty tensors")
    return digest.hexdigest()


def require_frozen_identity(reference: str, observed: str) -> None:
    """A regenerated Maze must never silently swap its pretrained backbone."""
    if (type(reference) is not str or len(reference) != 64
            or type(observed) is not str or len(observed) != 64
            or reference != observed):
        raise RuntimeError("frozen pretrained backbone identity changed")


def score_predictions(
    predictions: tuple[int, ...], rows: tuple[tuple[str, int], ...],
) -> float:
    """Public development accuracy calculated without model/backend dependency."""
    if (type(predictions) is not tuple or not rows or len(predictions) != len(rows)
            or any(type(value) is not int or value not in (0, 1)
                   for value in predictions)
            or any(type(label) is not int or label not in (0, 1)
                   for _, label in rows)):
        raise ValueError("invalid bounded public prediction rows")
    return sum(prediction == label for prediction, (_, label)
               in zip(predictions, rows)) / len(rows)


def validate_final_checkpoint_readback(
    saved: dict[str, Any], restored: dict[str, Any],
    in_memory: dict[str, tuple[int, ...]],
    reloaded: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Fail on byte identity or per-case prediction mismatch after *fresh* load.

    Identical aggregate accuracy with swapped individual predictions is NOT a
    pass. This gate checks one saved final checkpoint, not crash recovery or a
    protected examination; the caller must supply the actual fresh readback.
    """
    digests = (
        "trainable_sha256_after", "lora_sha256_after",
        "pretrained_base_sha256", "frozen_peft_sha256_after",
        "adapter_file_sha256", "head_file_sha256",
    )
    for key in digests:
        original, observed = saved.get(key), restored.get(key)
        if (type(original) is not str or len(original) != 64
                or type(observed) is not str or len(observed) != 64
                or original != observed):
            raise RuntimeError(f"final checkpoint readback mismatch: {key}")
    result: dict[str, Any] = {"status": "SELF_REPORTED_FRESH_INSTANCE_PASS"}
    public = {
        "public_development": PUBLIC_DEVELOPMENT,
        "public_transfer": PUBLIC_TRANSFER,
    }
    if set(in_memory) != set(public) or set(reloaded) != set(public):
        raise RuntimeError("final checkpoint readback missing public prediction split")
    for name, rows in public.items():
        before, after = in_memory[name], reloaded[name]
        # Score validation also rejects invalid predictions and row counts.
        before_score = score_predictions(before, rows)
        after_score = score_predictions(after, rows)
        if before != after or before_score != after_score:
            raise RuntimeError(f"final checkpoint readback prediction mismatch: {name}")
        digest = hashlib.sha256(bytes(before)).hexdigest()
        result[name + "_prediction_sha256"] = digest
        result[name + "_accuracy_after_reload"] = after_score
    return result


def execute(
    run_root: Path, *, scope: TrainingScopeSpec,
    cycles: tuple[TrainingCycle, ...], steps_per_cycle: int,
    learning_rate: float, model_id: str = MODEL_ID,
    preregistered_controls: bool = False,
    optimizer_continuity: OptimizerContinuityRequest | None = None,
    measured_batches: MeasuredNeuralExecutionRequest | None = None,
) -> dict[str, Any]:
    validate_scope(scope)
    validate_cycles(cycles)
    if (type(steps_per_cycle) is not int or not 1 <= steps_per_cycle <= 16
            or type(learning_rate) not in (int, float)
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 0.01):
        raise ValueError("unbounded physical experiment configuration")
    authoritative = build_training_cycles(scope)
    # Only the second cycle may be reweighted by an approved weak-skill hint.
    # Rebuild every example from the declaration; never train injected rows.
    weak_skills = tuple(skill for skill in SKILLS
                        if cycles[1].plan.counts_by_skill.get(skill, 0)
                        > authoritative[1].plan.counts_by_skill.get(skill, 0))
    if cycles != build_training_cycles(scope, weak_cycle_one=weak_skills):
        raise ValueError("untrusted training cycles diverged from declared Maze")
    active_arms = configured_arms(preregistered_controls)
    verified_measured_schedule = None
    if measured_batches is not None:
        if (type(measured_batches) is not MeasuredNeuralExecutionRequest
                or not preregistered_controls):
            raise ValueError("measured neural TRAIN batches require typed five-arm opt-in")
        # Import lazily to avoid a module cycle: the source planner imports
        # this probe's immutable TrainingCycle/TrainingExample definitions.
        from labyrinth.measured_neural_batch_binding import (
            verify_precommitted_neural_batch_schedule,
        )
        verified_measured_schedule = verify_precommitted_neural_batch_schedule(
            measured_batches.schedule, training_cycles=cycles,
            expected_schedule_sha256=(
                measured_batches.independently_pinned_schedule_sha256),
            expected_run_id=measured_batches.run_id,
            expected_candidate_artifact_sha256=(
                measured_batches.candidate_artifact_sha256),
            expected_predecessor_checkpoint_sha256=(
                measured_batches.predecessor_checkpoint_sha256),
            expected_scope_fingerprint=measured_batches.scope_fingerprint,
            expected_fence=measured_batches.fence,
        )
        if (verified_measured_schedule.steps_per_cycle != steps_per_cycle
                or weak_skills != (verified_measured_schedule.weak_skill,)):
            raise ValueError("signed weakness and step count must match regenerated Maze")
    if optimizer_continuity is not None:
        if type(optimizer_continuity) is not OptimizerContinuityRequest:
            raise ValueError("invalid optimizer continuity request")
        optimizer_continuity.validate()
        if (measured_batches is not None
                and (optimizer_continuity.logical_run_id != measured_batches.run_id
                     or optimizer_continuity.prior_checkpoint_sha256
                     != measured_batches.predecessor_checkpoint_sha256
                     or optimizer_continuity.fence != measured_batches.fence)):
            raise ValueError("optimizer and measured schedule custody pins diverge")
        # The adapter reads this runner's source bytes to bind all visits and
        # control-label assignments before restoring model or optimizer state.
        from labyrinth.minilm_optimizer_state_resume import (
            MazeOptimizerResumePlan, load_maze_optimizer_snapshot,
            save_maze_optimizer_snapshot,
        )

    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    def adapter_digest(model: Any) -> str:
        digest = hashlib.sha256()
        count = 0
        for name, param in sorted(model.named_parameters()):
            if param.requires_grad:
                count += 1
                digest.update(name.encode("utf-8") + b"\0")
                digest.update(param.detach().cpu().contiguous().numpy().tobytes())
        if not count:
            raise RuntimeError("no trainable LoRA adapter values")
        return digest.hexdigest()

    def trainable_digest(model: Any, head: Any) -> str:
        digest = hashlib.sha256()
        digest.update(adapter_digest(model).encode("ascii") + b"\0")
        for name, param in sorted(head.named_parameters()):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(param.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def create_model(previous: Path | None) -> tuple[Any, Any, str]:
        torch.manual_seed(scope.seed)
        base = AutoModel.from_pretrained(model_id, local_files_only=True)
        # Pin actual pretrained tensor bytes BEFORE PEFT modifies the module.
        # The model ID/cache path alone is insufficient continuity evidence.
        base_sha256 = hash_named_weights(base.state_dict().items())
        if previous is None:
            adapter = get_peft_model(base, LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=4, lora_alpha=8,
                lora_dropout=0.0, bias="none", target_modules=["query", "value"],
            ))
        else:
            adapter = PeftModel.from_pretrained(
                base, previous / "adapter", is_trainable=True,
            )
        head = torch.nn.Linear(base.config.hidden_size, 2)
        if previous is not None:
            head.load_state_dict(load_file(str(previous / "head.safetensors")))
        return adapter, head, base_sha256

    def logits(model: Any, head: Any, prompts: list[str]) -> Any:
        tokens = tokenizer(prompts, padding=True, truncation=True, max_length=64,
                           return_tensors="pt")
        result = model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (result.last_hidden_state * mask).sum(1) / mask.sum(1)
        return head(pooled)

    def predict(model: Any, head: Any, rows: tuple[tuple[str, int], ...]) -> tuple[int, ...]:
        model.eval()
        head.eval()
        with torch.no_grad():
            predicted = logits(model, head, [prompt for prompt, _ in rows]).argmax(-1)
        return tuple(int(value.item()) for value in predicted)

    def score(model: Any, head: Any, rows: tuple[tuple[str, int], ...]) -> float:
        return score_predictions(predict(model, head, rows), rows)

    report: dict[str, Any] = {
        "schema": PREREG_SCHEMA if preregistered_controls else SCHEMA,
        "qualification": "PUBLIC_DEVELOPMENT_ONLY",
        "pretrained_origin": "CACHED_MODEL_ID_NOT_INDEPENDENTLY_ATTESTED",
        "model_id": model_id, "device": "cpu", "cpu_torch_threads": 2,
        "torch_version": str(torch.__version__),
        "harness_sha256": _sha_file(Path(__file__)),
        "scope": scope.to_dict(), "scope_fingerprint": scope.fingerprint,
        "cycle_one_weak_skill_hint": list(weak_skills),
        "weakness_feedback_provenance": "OPERATOR_DECLARED_NOT_MEASURED",
        "steps_per_cycle": steps_per_cycle, "learning_rate": learning_rate,
        "maze_manifest": "maze_manifest.json", "arms": {},
        "zero_update_control": "IDENTICAL_FORWARD_BACKWARD_NO_OPTIMIZER_STEP",
        "reload_method": "NEW_BASE_OBJECT_LOADED_FROM_PRIOR_ADAPTER_AND_HEAD",
        "private_final_exam": "NOT_RUN", "graduation_authorized": False,
        "crash_recovery": "NOT_TESTED",
        "physical_model_byte_custody": "SELF_REPORTED",
    }
    if preregistered_controls:
        report.update({
            "control_preregistration": "OPERATOR_REQUESTED_UNSEALED_SOURCE_ONLY",
            "control_arms": list(active_arms),
            "control_semantics": {
                "correct": "TRUE_TRAIN_LABELS_ADAMW",
                "zero_update": "SAME_FORWARD_BACKWARD_NO_OPTIMIZER_STEP",
                "inverted": "BINARY_LABEL_INVERSION_ADAMW",
                "zero_lr": "SAME_FORWARD_BACKWARD_ADAMW_STEP_AT_LR_ZERO",
                "randomized_labels": "PINNED_CLASS_PRESERVING_TRAIN_LABEL_PERMUTATION_ADAMW",
            },
            "private_exam_independence": "NOT_ESTABLISHED",
        })
    if optimizer_continuity is not None:
        report.update({
            "optimizer_continuity": "SELF_REPORTED_LOCAL_CYCLE_BOUNDARY",
            "optimizer_checkpoint_pin_provenance": "SAME_PROCESS_SAVE_RECEIPT",
            "optimizer_predecessor_pin_provenance": "OPERATOR_DECLARED_UNATTESTED",
            "optimizer_external_monotonic_authority": "NOT_ESTABLISHED",
            "optimizer_crash_resume": "NOT_TESTED",
            "reload_method": "NEW_BASE_OBJECT_FULL_MODEL_OPTIMIZER_AND_RNG_RESTORE",
        })
    if verified_measured_schedule is not None:
        report.update({
            "weakness_feedback_provenance": "SIGNED_DEVELOPMENT_SOURCE_HOST_UNATTESTED",
            "measured_neural_training_schedule": "VERIFIED_PRECOMMITTED_SOURCE_ONLY",
            "measured_neural_training_schedule_sha256": (
                verified_measured_schedule.schedule_sha256),
            "measured_neural_training_source_event_id": (
                verified_measured_schedule.source_event_id),
            "measured_neural_training_source_exam_id": (
                verified_measured_schedule.source_exam_id),
            "measured_neural_training_tactic": verified_measured_schedule.tactic,
            "measured_neural_control_arms": list(PREREG_ARMS),
            "measured_neural_external_pin_attested": False,
            "measured_neural_model_execution_attested": False,
        })
    run_root.mkdir(mode=0o700)
    (run_root / "maze_manifest.json").write_text(json.dumps({
        "schema": PREREG_SCHEMA if preregistered_controls else SCHEMA,
        "scope": scope.to_dict(),
        "cycles": [{
            "cycle_index": cycle.plan.cycle_index,
            "weak_skill_counts": cycle.plan.counts_by_skill,
            "examples": [row.to_dict() for row in cycle.examples],
        } for cycle in cycles],
        "development_prompts_included": False,
        "final_exam_prompts_included": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["maze_manifest_sha256"] = _sha_file(run_root / "maze_manifest.json")

    expected_base_digest = None
    for arm in active_arms:
        print(f"arm={arm} starting", flush=True)
        arm_root = run_root / arm
        arm_root.mkdir()
        previous: Path | None = None
        preceding_state = None
        preceding_lora = None
        preceding_frozen = None
        prior_full_state_sha256: str | None = None
        full_state_plan = None
        cycle_results = []
        for cycle in cycles:
            randomized_labels = (preregistered_randomized_labels(cycle, scope.seed)
                                 if arm == "randomized_labels" else None)
            # In the opt-in continuity branch a fresh pretrained object and
            # freshly initialized head are restored from the *full* snapshot;
            # retaining only LoRA/head would silently cold-start AdamW.
            model, head, pretrained_base = create_model(
                None if optimizer_continuity is not None else previous)
            if expected_base_digest is None:
                expected_base_digest = pretrained_base
            else:
                require_frozen_identity(expected_base_digest, pretrained_base)
            frozen_before = hash_named_weights(
                (name, param) for name, param in model.named_parameters()
                if not param.requires_grad
            )
            if preceding_frozen is not None:
                require_frozen_identity(preceding_frozen, frozen_before)
            params = [p for p in model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError("model has no trainable pretrained adapter")
            optimizer = torch.optim.AdamW(
                params + list(head.parameters()),
                lr=optimizer_learning_rate(arm, learning_rate),
            )
            restored_full_state_sha256 = None
            if optimizer_continuity is not None:
                if full_state_plan is None:
                    full_state_plan = MazeOptimizerResumePlan(
                        scope=scope, cycles=cycles, weak_cycle_one=weak_skills,
                        arm=arm, steps_per_cycle=steps_per_cycle,
                        learning_rate=learning_rate,
                        logical_run_id=optimizer_continuity.logical_run_id,
                        base_checkpoint_sha256=pretrained_base,
                        prior_checkpoint_sha256=(
                            optimizer_continuity.prior_checkpoint_sha256),
                        fence=optimizer_continuity.fence,
                        measured_cycle_one=verified_measured_schedule,
                        measured_schedule_sha256=(
                            measured_batches.independently_pinned_schedule_sha256
                            if measured_batches is not None else None),
                        measured_candidate_artifact_sha256=(
                            measured_batches.candidate_artifact_sha256
                            if measured_batches is not None else None),
                    )
                    full_state_plan.validate()
                else:
                    if previous is None or prior_full_state_sha256 is None:
                        raise RuntimeError("missing prior full optimizer snapshot")
                    # This digest was obtained by the SAME process and is not
                    # an independently controlled external checkpoint pin.
                    receipt = load_maze_optimizer_snapshot(
                        previous / "optimizer-state", encoder=model, head=head,
                        optimizer=optimizer, plan=full_state_plan,
                        completed_steps=steps_per_cycle,
                        independently_pinned_checkpoint_sha256=prior_full_state_sha256,
                    )
                    restored_full_state_sha256 = receipt.sha256
                    if restored_full_state_sha256 != prior_full_state_sha256:
                        raise RuntimeError("optimizer snapshot readback changed")
            before = trainable_digest(model, head)
            lora_before = adapter_digest(model)
            if preceding_state is not None and before != preceding_state:
                raise RuntimeError("adapter/head weights changed across maze regeneration")
            if preceding_lora is not None and lora_before != preceding_lora:
                raise RuntimeError("LoRA values failed checkpoint reload")
            baseline_dev = score(model, head, PUBLIC_DEVELOPMENT)
            examples = cycle.examples
            visits = []
            model.train()
            head.train()
            for step in range(steps_per_cycle):
                batch = select_executed_training_batch(
                    cycle, step, arm, verified_measured_schedule)
                visits.extend(row.exposure_id for row in batch)
                labels = torch.tensor(
                    training_batch_labels(batch, arm, randomized_labels),
                    dtype=torch.long,
                )
                prediction = logits(model, head, [row.prompt for row in batch])
                loss = torch.nn.functional.cross_entropy(prediction, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                perform_optimizer_step(arm, optimizer)
                if arm == "zero_update":
                    optimizer.zero_grad(set_to_none=True)
            after = trainable_digest(model, head)
            lora_after = adapter_digest(model)
            frozen_after = hash_named_weights(
                (name, param) for name, param in model.named_parameters()
                if not param.requires_grad
            )
            require_frozen_identity(frozen_before, frozen_after)
            if arm in ("zero_update", "zero_lr") and (
                    after != before or lora_after != lora_before):
                raise RuntimeError("no-parameter-update control arm mutated")
            if arm not in ("zero_update", "zero_lr") and (
                    after == before or lora_after == lora_before):
                raise RuntimeError("trainable parameters did not change")
            checkpoint = arm_root / f"cycle-{cycle.plan.cycle_index}"
            checkpoint.mkdir()
            model.save_pretrained(str(checkpoint / "adapter"), safe_serialization=True)
            save_file({key: val.detach().cpu().contiguous()
                       for key, val in head.state_dict().items()},
                      str(checkpoint / "head.safetensors"))
            cycle_result = {
                "cycle": cycle.plan.cycle_index,
                "maze_room_ids": [row.room_id for row in examples],
                "maze_case_seeds": [row.case_seed for row in examples],
                "maze_exposure_ids": [row.exposure_id for row in examples],
                "training_visit_exposure_ids": visits,
                "training_visits": len(visits), "unique_training_cases": len(examples),
                "forward_steps": steps_per_cycle, "backward_steps": steps_per_cycle,
                "optimizer_updates": 0 if arm in ("zero_update", "zero_lr")
                else steps_per_cycle,
                "trainable_sha256_before": before, "trainable_sha256_after": after,
                "lora_sha256_before": lora_before, "lora_sha256_after": lora_after,
                "pretrained_base_sha256": pretrained_base,
                "frozen_peft_sha256_before": frozen_before,
                "frozen_peft_sha256_after": frozen_after,
                "adapter_file_sha256": _sha_file(checkpoint / "adapter" / "adapter_model.safetensors"),
                "head_file_sha256": _sha_file(checkpoint / "head.safetensors"),
                "public_development_accuracy_before": baseline_dev,
                "public_development_accuracy": score(model, head, PUBLIC_DEVELOPMENT),
                "public_transfer_accuracy": score(model, head, PUBLIC_TRANSFER),
            }
            if verified_measured_schedule is not None:
                cycle_result.update({
                    "measured_neural_training_applied": (
                        cycle.plan.cycle_index == 1),
                    "measured_neural_training_schedule_sha256": (
                        verified_measured_schedule.schedule_sha256
                        if cycle.plan.cycle_index == 1 else None),
                    "training_visit_exposure_ids_sha256": hashlib.sha256(
                        (json.dumps(visits, separators=(",", ":")) + "\n").encode("ascii")
                    ).hexdigest(),
                })
            if optimizer_continuity is not None:
                completed_steps = (cycle.plan.cycle_index + 1) * steps_per_cycle
                full_receipt = save_maze_optimizer_snapshot(
                    checkpoint / "optimizer-state", encoder=model, head=head,
                    optimizer=optimizer, plan=full_state_plan,
                    completed_steps=completed_steps,
                )
                prior_full_state_sha256 = full_receipt.sha256
                cycle_result.update({
                    "full_optimizer_snapshot_sha256": full_receipt.sha256,
                    "full_optimizer_state_file_sha256": _sha_file(
                        checkpoint / "optimizer-state" / "state.safetensors"),
                    "full_optimizer_completed_steps": completed_steps,
                    "full_optimizer_predecessor_restore_sha256": restored_full_state_sha256,
                    "optimizer_continuity_provenance": "LOCAL_RECEIPT_NO_HOST_FENCE",
                })
            if preregistered_controls:
                cycle_result.update({
                    "optimizer_step_invocations": 0 if arm == "zero_update"
                    else steps_per_cycle,
                    "effective_learning_rate": optimizer_learning_rate(
                        arm, learning_rate),
                    "label_assignment_sha256": hashlib.sha256(
                        json.dumps({
                            row.exposure_id: assigned
                            for row, assigned in zip(
                                cycle.examples,
                                training_batch_labels(cycle.examples, arm,
                                                      randomized_labels),
                            )
                        }, sort_keys=True, separators=(",", ":")).encode("ascii")
                    ).hexdigest(),
                })
            cycle_results.append(cycle_result)
            preceding_state, preceding_lora, preceding_frozen = after, lora_after, frozen_after
            previous = checkpoint
            print(f"arm={arm} cycle={cycle.plan.cycle_index} saved", flush=True)
        # The last checkpoint must be independently consumed by a fresh model
        # instance. An in-memory post-save score is insufficient readback.
        final_predictions = {
            "public_development": predict(model, head, PUBLIC_DEVELOPMENT),
            "public_transfer": predict(model, head, PUBLIC_TRANSFER),
        }
        del model, head, optimizer, params, prediction, loss, labels, batch
        import gc
        gc.collect()
        restored_model, restored_head, restored_base = create_model(previous)
        restored_values = {
            "trainable_sha256_after": trainable_digest(restored_model, restored_head),
            "lora_sha256_after": adapter_digest(restored_model),
            "pretrained_base_sha256": restored_base,
            "frozen_peft_sha256_after": hash_named_weights(
                (name, param) for name, param in restored_model.named_parameters()
                if not param.requires_grad
            ),
            "adapter_file_sha256": _sha_file(previous / "adapter" / "adapter_model.safetensors"),
            "head_file_sha256": _sha_file(previous / "head.safetensors"),
        }
        restored_predictions = {
            "public_development": predict(restored_model, restored_head, PUBLIC_DEVELOPMENT),
            "public_transfer": predict(restored_model, restored_head, PUBLIC_TRANSFER),
        }
        readback = validate_final_checkpoint_readback(
            cycle_results[-1], restored_values, final_predictions, restored_predictions,
        )
        if (readback["public_development_accuracy_after_reload"]
                != cycle_results[-1]["public_development_accuracy"]
                or readback["public_transfer_accuracy_after_reload"]
                != cycle_results[-1]["public_transfer_accuracy"]):
            raise RuntimeError("final checkpoint readback changed public accuracy")
        del restored_model, restored_head
        gc.collect()
        report["arms"][arm] = {
            "initial_trainable_sha256": cycle_results[0]["trainable_sha256_before"],
            "initial_lora_sha256": cycle_results[0]["lora_sha256_before"],
            "final_checkpoint_readback": readback,
            "cycles": cycle_results,
        }
    if len({report["arms"][arm]["initial_trainable_sha256"] for arm in active_arms}) != 1:
        raise RuntimeError("matched controls had different initial trainable parameters")
    if len({report["arms"][arm]["initial_lora_sha256"] for arm in active_arms}) != 1:
        raise RuntimeError("matched controls had different initial LoRA parameters")
    report["pretrained_base_sha256"] = expected_base_digest
    for cycle_index in range(2):
        exposures = {
            tuple(report["arms"][arm]["cycles"][cycle_index]["training_visit_exposure_ids"])
            for arm in active_arms
        }
        if len(exposures) != 1:
            raise RuntimeError("matched controls received nonmatching training visits")
    report["public_development_comparison"] = comparison({
        arm: report["arms"][arm]["cycles"][-1]["public_development_accuracy"]
        for arm in ARMS
    })
    report["public_transfer_comparison"] = comparison({
        arm: report["arms"][arm]["cycles"][-1]["public_transfer_accuracy"]
        for arm in ARMS
    })
    report["public_development_retention"] = {
        arm: (report["arms"][arm]["cycles"][1]["public_development_accuracy"]
              - report["arms"][arm]["cycles"][0]["public_development_accuracy"])
        for arm in active_arms
    }
    if preregistered_controls:
        for split in ("public_development", "public_transfer"):
            report[split + "_five_arm_comparison"] = preregistered_control_comparison({
                arm: report["arms"][arm]["cycles"][-1][split + "_accuracy"]
                for arm in active_arms
            })
    report["learning_qualification"] = "NOT_ESTABLISHED_UNTIL_INDEPENDENT_EVALUATION"
    (run_root / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scope-file", type=Path,
                        help="strict TrainingScopeSpec JSON for the approved polarity objective")
    parser.add_argument("--weak-cycle-one", choices=SKILLS,
                        help="operator-declared weakness; never an independent exam signal")
    parser.add_argument("--steps-per-cycle", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--preregistered-controls", action="store_true",
                        help="opt-in five-arm public probe; unsealed, not private qualification")
    parser.add_argument("--full-optimizer-cycle-continuity", action="store_true",
                        help="opt-in local full-state optimizer+RNG roundtrip; no host-authoritative pin or crash resume")
    parser.add_argument("--optimizer-run-id",
                        help="required caller-declared run ID for full optimizer continuity")
    parser.add_argument("--optimizer-prior-sha256",
                        help="required caller-declared predecessor checkpoint SHA-256; not independently attested")
    parser.add_argument("--optimizer-fence", type=int,
                        help="required caller-declared local fence; not host-owned")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= args.steps_per_cycle <= 16
            or not math.isfinite(args.learning_rate)
            or not 0 < args.learning_rate <= 0.01):
        parser.error("steps must be 1..16 and finite learning rate in (0,.01]")
    try:
        optimizer_request = None
        optimizer_args = (args.optimizer_run_id, args.optimizer_prior_sha256,
                          args.optimizer_fence)
        if args.full_optimizer_cycle_continuity:
            if any(value is None for value in optimizer_args):
                raise ValueError("full optimizer continuity requires run ID, predecessor SHA and fence")
            optimizer_request = OptimizerContinuityRequest(*optimizer_args)
            optimizer_request.validate()
        elif any(value is not None for value in optimizer_args):
            raise ValueError("optimizer pins require --full-optimizer-cycle-continuity")
        scope = default_scope()
        if args.scope_file is not None:
            if args.scope_file.stat().st_size > 16_384:
                raise ValueError("scope file exceeds 16 KiB")
            scope = parse_scope_json(args.scope_file.read_bytes())
        cycles = build_training_cycles(
            scope, weak_cycle_one=((args.weak_cycle_one,)
                                   if args.weak_cycle_one else ()),
        )
        validation = validate_cycles(cycles)
        output = plan_output(args.output_parent, args.run_name)
        if not args.execute:
            print(json.dumps({
                "schema": PREREG_SCHEMA if args.preregistered_controls else SCHEMA,
                "execute": False, "run_root": str(output),
                "scope_fingerprint": scope.fingerprint, "objective_id": scope.objective_id,
                "cycles": [{
                    "cycle_index": c.plan.cycle_index,
                    "room_ids": [r.room_id for r in c.plan.rooms],
                    "case_seeds": [r.case_seed for r in c.plan.rooms],
                    "exposure_ids": [r.exposure_id for r in c.examples],
                    "train_label_counts": c.plan.counts_by_skill,
                } for c in cycles],
                "validation": validation, "model_id": MODEL_ID,
                "weakness_feedback_provenance": "OPERATOR_DECLARED_NOT_MEASURED",
                "public_development_only": True, "graduation_authorized": False,
                "full_optimizer_continuity": (
                    "LOCAL_RECEIPT_NO_HOST_FENCE" if optimizer_request is not None
                    else "DISABLED"),
                **({
                    "arms": list(configured_arms(True)),
                    "control_preregistration": "OPERATOR_REQUESTED_UNSEALED_SOURCE_ONLY",
                } if args.preregistered_controls else {}),
            }, sort_keys=True))
            return 0
        report = execute(
            output, scope=scope, cycles=cycles,
            steps_per_cycle=args.steps_per_cycle,
            learning_rate=args.learning_rate,
            preregistered_controls=args.preregistered_controls,
            optimizer_continuity=optimizer_request,
        )
        print(json.dumps({
            "run_root": str(output),
            "public_development_comparison": report["public_development_comparison"],
            "public_transfer_comparison": report["public_transfer_comparison"],
            "graduation_authorized": False,
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"preflight or execution failure: {exc}\n")


if __name__ == "__main__":
    # `python -m scripts.qualify_maze_conditioned_minilm_cpu` executes these
    # dataclasses under __main__, while the signed batch and optimizer modules
    # import the canonical qualified module. Reimporting would manufacture a
    # second TrainingExample/TrainingCycle class and reject authentic rows.
    # Preserve ONE module identity or fail closed on an existing different
    # definition. This is not a trusted module-substitution authorization.
    import sys
    _canonical_module = "scripts.qualify_maze_conditioned_minilm_cpu"
    _loaded = sys.modules.get(_canonical_module)
    if _loaded is not None and _loaded is not sys.modules["__main__"]:
        raise SystemExit("conflicting pretrained MiniLM runner module identity")
    sys.modules[_canonical_module] = sys.modules["__main__"]
    raise SystemExit(main())
