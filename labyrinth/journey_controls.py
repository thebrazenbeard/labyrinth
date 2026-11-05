"""Matched arithmetic toy controls for the regenerative Labyrinth prototype.

This is not a second Minotaur gate. It holds exposures and development cases
fixed across four toy learners and records the differences caused by correct
targets, zero updates, and deranged train-only labels. It neither uses exam
feedback for supervision nor claims neural-model or general reasoning gain.
"""

from __future__ import annotations

import hashlib
import math

from .evaluation import score_response
from .journey_challenges import prepare_training_rooms
from .journey_toy import _exam_cases
from .regenerative_maze import generate_maze_plan
from .toy_backend import ToyLinearAdapter, _deranged_labels
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.toy-journey-controls.v1"
ARMS = ("baseline", "correct_labels", "zero_update", "corrupted_labels")


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _check_schedule(epochs_by_cycle: tuple[int, ...] | None, cycles: int) -> tuple[int, ...]:
    if epochs_by_cycle is None:
        epochs_by_cycle = (4,) * cycles
    if (type(epochs_by_cycle) is not tuple or len(epochs_by_cycle) != cycles
            or any(type(n) is not int or not 1 <= n <= 128 for n in epochs_by_cycle)
            or sum(epochs_by_cycle) > 256):
        raise ValueError("epochs_by_cycle must be a bounded integer tuple matching every cycle")
    return epochs_by_cycle


def run_matched_toy_controls(
    scope: TrainingScopeSpec, *, learner_seed: int = 0,
    epochs_by_cycle: tuple[int, ...] | None = None,
    learning_rate: float = 0.5, exam_count_per_slice: int = 8,
) -> dict:
    """Run all arms on identical train cases and identical fresh dev cases.

    The number of cycles is fixed *before* any exam, preventing evaluation
    scores from choosing one arm's training budget. All four arms begin from
    the same parameters, and any label derangement is constrained to the
    training split only. Uses private answer keys only inside the local
    development scorer; no answers enter model.generate or future training.

    Same-exposure means case visits (including baseline) and evaluation calls.
    Baseline and zero-update perform no parameter changes by design; this is
    not equalized optimizer arithmetic, forward timing, or neural FLOPs.
    """
    if type(scope) is not TrainingScopeSpec:
        raise ValueError("scope must be a frozen TrainingScopeSpec")
    TrainingScopeSpec.__post_init__(scope)
    if scope.skills != ("arithmetic",) or scope.total_chambers < 2:
        raise ValueError("toy controls require an arithmetic-only scope with >=2 chambers")
    if type(learner_seed) is not int or not 0 <= learner_seed < 2**31:
        raise ValueError("learner_seed must be a nonnegative 31-bit integer")
    epochs_by_cycle = _check_schedule(epochs_by_cycle, scope.max_cycles)
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1):
        raise ValueError("learning_rate must be finite and in (0,1]")
    if type(exam_count_per_slice) is not int or not 4 <= exam_count_per_slice <= 32:
        raise ValueError("exam_count_per_slice must be an integer in [4,32]")

    # Freeze every case schedule before creating any arms. Development gold
    # never informs generated training or optimization schedules.
    schedules = []
    all_train_ids: set[str] = set()
    all_train_prompts: set[str] = set()
    all_eval_ids: set[str] = set()
    all_eval_prompts: set[str] = set()
    for cycle, epochs in enumerate(epochs_by_cycle):
        plan = generate_maze_plan(scope, cycle)
        prepared = prepare_training_rooms(plan)
        train = tuple(room.challenge for room in prepared)
        exam = _exam_cases(scope, cycle, exam_count_per_slice)
        ids = [case.id for case in train]
        prompts = [_hash_prompt(case.prompt) for case in train]
        exam_ids = [case.id for _, case in exam]
        exam_prompts = [_hash_prompt(case.prompt) for _, case in exam]
        if (len(ids) != len(set(ids)) or len(prompts) != len(set(prompts))
                or len(exam_ids) != len(set(exam_ids))
                or len(exam_prompts) != len(set(exam_prompts))
                or set(ids) & (all_train_ids | all_eval_ids)
                or set(prompts) & (all_train_prompts | all_eval_prompts)
                or set(exam_ids) & (all_train_ids | all_eval_ids | set(ids))
                or set(exam_prompts) & (all_train_prompts | all_eval_prompts | set(prompts))):
            raise ValueError("training/evaluation case or prompt reused across cycles")
        # Reject impossible derangements before updating any model parameters.
        _deranged_labels([float(case.answer) for case in train], learner_seed)
        all_train_ids.update(ids)
        all_train_prompts.update(prompts)
        all_eval_ids.update(exam_ids)
        all_eval_prompts.update(exam_prompts)
        schedules.append((plan, train, exam, epochs))

    arms = {arm: ToyLinearAdapter(seed=learner_seed) for arm in ARMS}
    initial_sha256 = arms["baseline"].model_hash
    if any(model.model_hash != initial_sha256 for model in arms.values()):
        raise RuntimeError("control learners did not start from the same initialization")
    cycles: list[dict] = []
    for plan, training_cases, exam_cases, epochs in schedules:
        results: dict[str, dict] = {}
        for arm in ARMS:
            model = arms[arm]
            before = model.model_hash
            receipt = model.train(
                training_cases,
                epochs=epochs,
                learning_rate=learning_rate,
                shuffled_labels=(arm == "corrupted_labels"),
                update_enabled=(arm not in ("baseline", "zero_update")),
            )
            if (receipt["before_model_hash"] != before
                    or receipt["after_model_hash"] != model.model_hash
                    or receipt["case_visits"] != epochs * len(training_cases)):
                raise RuntimeError("toy training receipt does not match actual model state")
            if arm in ("baseline", "zero_update") and model.model_hash != initial_sha256:
                raise RuntimeError("nonupdate control unexpectedly mutated parameters")
            observations = [
                {"case_id": case.id, "section": section,
                 "prompt_sha256": _hash_prompt(case.prompt),
                 "correct": score_response(case, model.generate(case.prompt)).correct}
                for section, case in exam_cases
            ]
            results[arm] = {
                "checkpoint_before_sha256": before,
                "checkpoint_after_sha256": model.model_hash,
                "case_visits": receipt["case_visits"],
                "parameter_updates": receipt["parameter_updates"],
                "changed_training_labels": receipt["changed_labels"],
                "exam_correct": sum(obs["correct"] for obs in observations),
                "exam_total": len(observations),
                "observations": observations,
            }
        if (results["baseline"]["exam_correct"] != results["zero_update"]["exam_correct"]
                or results["baseline"]["checkpoint_after_sha256"]
                != results["zero_update"]["checkpoint_after_sha256"]):
            raise RuntimeError("matched zero-update and baseline disagree")
        if len({results[a]["case_visits"] for a in ARMS}) != 1:
            raise RuntimeError("training case exposure differs among controls")
        cycles.append({
            "cycle_index": plan.cycle_index,
            "training_room_ids": [room.room_id for room in plan.rooms],
            "training_case_ids": [case.id for case in training_cases],
            "training_prompt_sha256": [_hash_prompt(case.prompt) for case in training_cases],
            "exam_case_ids": [case.id for _, case in exam_cases],
            "exam_prompt_sha256": [_hash_prompt(case.prompt) for _, case in exam_cases],
            "epochs": epochs,
            "arms": results,
            "candidate_minus_baseline_correct": (
                results["correct_labels"]["exam_correct"] - results["baseline"]["exam_correct"]
            ),
            "candidate_minus_corrupted_correct": (
                results["correct_labels"]["exam_correct"] - results["corrupted_labels"]["exam_correct"]
            ),
        })
    return {
        "schema": SCHEMA,
        "scope_fingerprint": scope.fingerprint,
        "objective_id": scope.objective_id,
        "initial_checkpoint_sha256": initial_sha256,
        "final_checkpoint_sha256_by_arm": {name: model.model_hash for name, model in arms.items()},
        "cycle_count": scope.max_cycles,
        "arms": list(ARMS),
        "cycles": cycles,
        "formal_graduation": False,
        "qualification": "NOT_ESTABLISHED",
        "limitations": [
            "Four-parameter arithmetic toy; no pretrained neural model or empathy training.",
            "Development tests share a public same-family generator, not independent final custody.",
            "Fixed cycles and matched case visits do not equalize optimizer FLOPs or wall time.",
            "Score differences are developmental observations, not a transfer or graduation claim.",
        ],
    }
