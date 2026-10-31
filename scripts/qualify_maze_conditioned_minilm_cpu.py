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
        cycle_results = []
        for cycle in cycles:
            randomized_labels = (preregistered_randomized_labels(cycle, scope.seed)
                                 if arm == "randomized_labels" else None)
            model, head, pretrained_base = create_model(previous)
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
            before = trainable_digest(model, head)
            lora_before = adapter_digest(model)
            if preceding_state is not None and before != preceding_state:
                raise RuntimeError("adapter/head weights changed across maze regeneration")
            if preceding_lora is not None and lora_before != preceding_lora:
                raise RuntimeError("LoRA values failed checkpoint reload")
            baseline_dev = score(model, head, PUBLIC_DEVELOPMENT)
            params = [p for p in model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError("model has no trainable pretrained adapter")
            optimizer = torch.optim.AdamW(
                params + list(head.parameters()),
                lr=optimizer_learning_rate(arm, learning_rate),
            )
            examples = cycle.examples
            visits = []
            model.train()
            head.train()
            for step in range(steps_per_cycle):
                batch = select_training_batch(cycle, step)
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
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if (not 1 <= args.steps_per_cycle <= 16
            or not math.isfinite(args.learning_rate)
            or not 0 < args.learning_rate <= 0.01):
        parser.error("steps must be 1..16 and finite learning rate in (0,.01]")
    try:
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
    raise SystemExit(main())
