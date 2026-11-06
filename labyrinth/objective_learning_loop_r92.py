"""Executable Declare -> Generate -> Experience -> Learn -> Confront -> Evaluate -> Regenerate.

The *approved*, narrowly defined objective is canonical integer addition and
subtraction. The four-weight ToyLinearAdapter genuinely optimizes that grammar
with normalized SGD. It CANNOT learn the source-priority evidence-audit rubric
implemented separately in objective_challenge_provider_r92. A declaration is
never promoted to a generic task generator or trainable neural architecture.

This is an operator-controlled, public deterministic development experiment:
TRAIN gold stays trainer-side, the toy sees prompt strings only, heldout and
same-family transfer cases never enter the optimizer, Minotaur scores come
from actual inference, and checkpoints are exclusively written and reloaded.
No pretrained/HF/LoRA adaptation, evaluator independence, secret final exam,
externally authenticated custody, or formal graduation is established.

CLI: python -m labyrinth.objective_learning_loop_r92 --checkpoint-dir /tmp/toy-run
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .evaluation import score_response
from .journey import (
    AWAITING_EXAM, TRAINING, TrainingCompletionReceipt, begin_journey,
    complete_maze, face_minotaur,
)
from .journey_challenges import PreparedTrainingRoom, prepare_training_rooms
from .maze import Challenge, generate_maze
from .minotaur_exam import ExamCaseResult, MinotaurExamReceipt, evaluate_minotaur
from .objective_contract import ObjectiveContract, SkillTarget
from .regenerative_maze import generate_maze_plan
from .toy_backend import ToyLinearAdapter
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.objective-learning-loop-r92.v1"
FOCUS = "integer_operator_coefficients"
BEHAVIOR = "Calculate canonical integer addition and subtraction with an exact integer answer."
RUBRIC = "exact_integer_arithmetic_v1"
_PROMPT = re.compile(
    r"Calculate (\d{2,3}) ([+-]) (\d{2,3})\.\n"
    r"Respond with exactly one line: ANSWER: <integer>\.", re.ASCII,
)


def arithmetic_objective(*, seed: int = 7, max_cycles: int = 2) -> ObjectiveContract:
    """Freeze an executable, objectively scorable toy-compatible declaration."""
    scope = TrainingScopeSpec(
        objective_id="operator-coefficient-learning-r92", domain="arithmetic",
        skills=("arithmetic",), specificity=2, scale=4,
        max_cycles=max_cycles, seed=seed,
    )
    return ObjectiveContract(
        scope=scope,
        intention="Learn the numeric coefficients of addition and subtraction from practice.",
        targets=(SkillTarget(
            skill="arithmetic", focus_tags=(FOCUS,),
            observable_behaviors=(BEHAVIOR,), rubric_id=RUBRIC,
            minimum_score_percent=75, minimum_gain_percent=10,
        ),),
        excluded_scopes=("Evidence audit, source priority, and noncanonical language tasks.",),
        forbidden_claims=(
            "Pretrained language model improvement is not established.",
            "Independent examination and formal graduation are not established.",
        ),
    )


def _approved_objective(contract: ObjectiveContract) -> TrainingScopeSpec:
    if type(contract) is not ObjectiveContract:
        raise ValueError("exact ObjectiveContract required")
    ObjectiveContract.__post_init__(contract)
    scope = contract.training_scope
    if (scope.objective_id != "operator-coefficient-learning-r92"
            or scope.domain != "arithmetic" or scope.skills != ("arithmetic",)
            or scope.specificity != 2 or scope.scale != 4
            or scope.total_chambers < 2 or len(contract.targets) != 1):
        raise ValueError("only the approved arithmetic objective is toy-executable")
    target = contract.targets[0]
    if (target.focus_tags != (FOCUS,)
            or target.observable_behaviors != (BEHAVIOR,)
            or target.rubric_id != RUBRIC
            or target.minimum_score_percent != 75
            or target.minimum_gain_percent != 10
            or contract.selected_focus("arithmetic") != (FOCUS,)):
        raise ValueError("only the approved arithmetic objective is toy-executable")
    return scope


def _hash_prompt(challenge: Challenge) -> str:
    return sha256(challenge.prompt.encode("utf-8")).hexdigest()


def _exam_seed(scope: TrainingScopeSpec, cycle_index: int, section: str) -> int:
    raw = f"objective-loop-r92-dev\x1f{scope.fingerprint}\x1f{cycle_index}\x1f{section}"
    return int.from_bytes(sha256(raw.encode("ascii")).digest()[:8], "big") % 2**31


def _development_cases(
    scope: TrainingScopeSpec, cycle_index: int, count: int,
) -> tuple[tuple[str, Challenge], ...]:
    return tuple(
        (section, case)
        for section in ("heldout", "transfer")
        for case in generate_maze(
            seed=_exam_seed(scope, cycle_index, section),
            tiers=(1,), count=count, split="eval",
        )
    )


def _check_train(
    prepared: tuple[PreparedTrainingRoom, ...], state,
) -> None:
    if len(prepared) != len(state.maze.rooms):
        raise ValueError("missing generated objective TRAIN rooms")
    operators: set[str] = set()
    for entry in prepared:
        case = entry.challenge
        match = _PROMPT.fullmatch(case.prompt)
        if (case.split != "train" or case.kind != "arithmetic" or case.tier != 1
                or match is None or dict(case.metadata).get("generator") != "labyrinth-v1"):
            raise ValueError("approved arithmetic provider returned nonexecutable TRAIN material")
        a, op, b = int(match[1]), match[2], int(match[3])
        if case.answer != str(a + b if op == "+" else a - b):
            raise ValueError("arithmetic source gold disagrees with objective rubric")
        operators.add(op)
    if operators != {"+", "-"}:
        raise ValueError("the toy objective must exercise both operator coefficients")
    ids = [entry.challenge.id for entry in prepared]
    prompts = [_hash_prompt(entry.challenge) for entry in prepared]
    forbidden_ids = set(state.used_training_case_ids) | set(state.sealed_exam_case_ids)
    forbidden_prompts = (set(state.used_training_prompt_sha256)
                         | set(state.sealed_exam_prompt_sha256))
    if (len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts)
            or set(ids) & forbidden_ids or set(prompts) & forbidden_prompts):
        raise ValueError("TRAIN batch duplicates previously exposed material")


def _check_exam(exam: tuple[tuple[str, Challenge], ...], prepared, state) -> None:
    ids = [case.id for _, case in exam]
    prompts = [_hash_prompt(case) for _, case in exam]
    training_ids = {entry.challenge.id for entry in prepared}
    training_prompts = {_hash_prompt(entry.challenge) for entry in prepared}
    prior_ids = (set(state.used_training_case_ids) | set(state.sealed_exam_case_ids)
                 | training_ids)
    prior_prompts = (set(state.used_training_prompt_sha256)
                     | set(state.sealed_exam_prompt_sha256) | training_prompts)
    if (len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts)
            or set(ids) & prior_ids or set(prompts) & prior_prompts
            or any(case.split != "eval" or case.kind != "arithmetic"
                   or case.tier != 1 for _, case in exam)):
        raise ValueError("development material overlaps TRAIN or sealed exam material")


def _scores(learner: ToyLinearAdapter, cases: tuple[tuple[str, Challenge], ...]) -> tuple[float, ...]:
    # Only public prompts cross the adapter inference boundary. Gold remains
    # local to the operator-side scorer, and raw replies are not exported.
    return tuple(score_response(case, learner.generate(case.prompt)).points
                 for _, case in cases)


def _store_checkpoint(
    learner: ToyLinearAdapter, directory: Path, index: int,
) -> dict:
    digest = learner.model_hash
    path = directory / f"checkpoint-{index:02d}-{digest}.json"
    saved = learner.save_checkpoint(path)  # O_EXCL; never overwrite prior evidence.
    restored = ToyLinearAdapter.load_checkpoint(path)
    if saved != digest or restored.model_hash != digest:
        raise RuntimeError("persistent toy checkpoint failed byte-level reload")
    return {"index": index, "path": str(path.resolve()), "sha256": digest}


def run_objective_learning_loop(
    contract: ObjectiveContract, *, checkpoint_dir: str | Path,
    learner_seed: int = 0, epochs_by_cycle: tuple[int, ...] | None = None,
    learning_rate: float = 0.5, exam_count_per_slice: int = 8,
) -> dict:
    """Run a real, fixed-budget toy training loop with measured internal exams.

    The operator supplies immutable TRAIN and development generators; no model
    controls its rubric, case identities, optimizer switches, or pass decision.
    The development cases may be pre-scored but *never* added to TRAIN. The
    zero-update sham has matched TRAIN case visits without optimizer writes.
    """
    scope = _approved_objective(contract)
    if type(learner_seed) is not int or not 0 <= learner_seed < 2**31:
        raise ValueError("invalid toy learner seed")
    if epochs_by_cycle is None:
        epochs_by_cycle = (1,) + (80,) * (scope.max_cycles - 1)
    if (type(epochs_by_cycle) is not tuple or len(epochs_by_cycle) != scope.max_cycles
            or any(type(epoch) is not int or not 1 <= epoch <= 1000
                   for epoch in epochs_by_cycle)):
        raise ValueError("one fixed bounded optimizer budget is required per cycle")
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1):
        raise ValueError("learning_rate must be finite and in (0,1]")
    if type(exam_count_per_slice) is not int or not 4 <= exam_count_per_slice <= 32:
        raise ValueError("each development split requires 4..32 cases")
    if not isinstance(checkpoint_dir, (str, Path)) or not str(checkpoint_dir):
        raise ValueError("checkpoint_dir must be a nonempty path")

    # Verify the source-backed objective declaration before any filesystem
    # mutation. The caller owns this directory; all files use exclusive create.
    directory = Path(checkpoint_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    learner = ToyLinearAdapter(seed=learner_seed)
    baseline = ToyLinearAdapter(seed=learner_seed)
    sham = ToyLinearAdapter(seed=learner_seed)
    initial_hash = learner.model_hash
    files = [_store_checkpoint(learner, directory, 0)]
    # Exercise actual checkpoint load before entering the first maze.
    learner = ToyLinearAdapter.load_checkpoint(files[0]["path"])
    state = begin_journey(scope, checkpoint_sha256=initial_hash)
    cycles: list[dict] = []

    for cycle_index, epochs in enumerate(epochs_by_cycle):
        if state.phase != TRAINING:
            break
        before = learner.model_hash
        if state.cycle_index != cycle_index or before != state.checkpoint_sha256:
            raise RuntimeError("retained learner is not the journey's active checkpoint")
        preceding_weak = (list(state.attempts[-1].weak_skills) if state.attempts else [])
        if state.maze != generate_maze_plan(scope, cycle_index, weak_skills=preceding_weak):
            raise RuntimeError("regenerated maze does not follow measured Minotaur weakness")

        # GENERATE source-backed TRAIN material for precisely this generated
        # maze. An ObjectiveContract alone cannot synthesize an executable task.
        prepared = prepare_training_rooms(state.maze)
        _check_train(prepared, state)
        development = _development_cases(scope, cycle_index, exam_count_per_slice)
        _check_exam(development, prepared, state)

        # EXPERIENCE: actual pre-optimizer responses to prompt-only TRAIN views;
        # separate read-only pre-scoring of reserved development cases.
        practice_pre = sum(
            score_response(entry.challenge, learner.generate(entry.trainee_view.prompt)).points
            for entry in prepared
        )
        development_pre = _scores(learner, development)
        frozen_scores = _scores(baseline, development)
        sham_before = sham.model_hash
        if sham_before != initial_hash:
            raise RuntimeError("untrained control diverged from frozen baseline")

        # LEARN: a real normalized-SGD optimizer pass on private TRAIN gold.
        # The schedule was fixed before reading development feedback.
        batch = tuple(entry.challenge for entry in prepared)
        optimizer = learner.train(batch, epochs=epochs, learning_rate=float(learning_rate),
                                  update_enabled=True, shuffled_labels=False)
        after = learner.model_hash
        if (optimizer.get("before_model_hash") != before
                or optimizer.get("after_model_hash") != after
                or type(optimizer.get("parameter_updates")) is not int
                or optimizer["parameter_updates"] <= 0 or after == before):
            raise RuntimeError("no measured toy optimizer update matches model hash readback")
        sham_optimizer = sham.train(
            batch, epochs=epochs, learning_rate=float(learning_rate),
            update_enabled=False, shuffled_labels=False,
        )
        if (sham.model_hash != sham_before
                or sham_optimizer.get("before_model_hash") != sham_before
                or sham_optimizer.get("after_model_hash") != sham_before
                or sham_optimizer.get("parameter_updates") != 0
                or sham_optimizer.get("case_visits") != optimizer.get("case_visits")):
            raise RuntimeError("zero-update control changed or lost matched exposure")
        files.append(_store_checkpoint(learner, directory, cycle_index + 1))
        learner = ToyLinearAdapter.load_checkpoint(files[-1]["path"])
        if learner.model_hash != after:
            raise RuntimeError("checkpoint lost trained parameters at cycle boundary")

        completed = TrainingCompletionReceipt(
            objective_id=scope.objective_id, cycle_index=cycle_index,
            checkpoint_before_sha256=before, checkpoint_after_sha256=after,
            room_case_ids=tuple((entry.room_id, entry.challenge.id,
                                 _hash_prompt(entry.challenge)) for entry in prepared),
        )
        state = complete_maze(state, completed)
        if state.phase != AWAITING_EXAM or state.checkpoint_sha256 != after:
            raise RuntimeError("completed maze and learner optimizer diverged")

        # CONFRONT + EVALUATE: internal source Minotaur sees actual post-update
        # and frozen-baseline scores, not a caller-authored PASS flag. No
        # development gold or individual responses enter the optimizer.
        development_post = _scores(learner, development)
        sham_scores = _scores(sham, development)
        if sham_scores != frozen_scores:
            raise RuntimeError("sham scores diverged despite frozen checkpoint")
        observations = tuple(
            ExamCaseResult(
                case_id=case.id, skill="arithmetic", split=section,
                score=float(post), baseline_score=float(baseline_score),
                prompt_sha256=_hash_prompt(case),
            )
            for (section, case), post, baseline_score in zip(
                development, development_post, frozen_scores
            )
        )
        receipt = MinotaurExamReceipt(
            exam_id=f"objloop-r92-{_exam_seed(scope, cycle_index, 'exam')}",
            objective_id=scope.objective_id, cycle_index=cycle_index,
            checkpoint_sha256=after, baseline_sha256=initial_hash,
            evaluator_id="operator.internal-toy-development-r92", cases=observations,
        )
        decision = evaluate_minotaur(scope, receipt)
        state = face_minotaur(state, receipt)  # Regenerate only on measured FAIL.
        if state.checkpoint_sha256 != learner.model_hash:
            raise RuntimeError("Minotaur handoff erased the trained checkpoint")
        cycles.append({
            "cycle_index": cycle_index,
            "regenerated_from_weak_skills": preceding_weak,
            "training_room_ids": [entry.room_id for entry in prepared],
            "training_case_ids": [entry.challenge.id for entry in prepared],
            "training_prompt_sha256": [_hash_prompt(entry.challenge) for entry in prepared],
            "practice_pre_correct": int(practice_pre),
            "training_case_count": len(prepared),
            "development_case_ids": [case.id for _, case in development],
            "development_prompt_sha256": [_hash_prompt(case) for _, case in development],
            "development_case_count": len(development),
            "development_pre_correct": int(sum(development_pre)),
            "development_post_correct": int(sum(development_post)),
            "frozen_baseline_correct": int(sum(frozen_scores)),
            "sham_correct": int(sum(sham_scores)),
            "checkpoint_before_sha256": before,
            "checkpoint_after_sha256": after,
            "sham_checkpoint_sha256": sham.model_hash,
            "optimizer": {
                "epochs": epochs, "learning_rate": float(learning_rate),
                "case_visits": optimizer["case_visits"],
                "parameter_updates": optimizer["parameter_updates"],
                "before_sha256": before, "after_sha256": after,
            },
            "sham_parameter_updates": sham_optimizer["parameter_updates"],
            "minotaur": {
                "status": decision.status,
                "weak_skills": list(decision.weak_skills),
                "mean_accuracy_by_skill": dict(decision.mean_accuracy_by_skill),
                "mean_gain_by_skill": dict(decision.mean_gain_by_skill),
                "exam_kind": "development",
            },
            "next_phase": state.phase,
        })

    return {
        "schema": SCHEMA,
        "objective_id": scope.objective_id,
        "objective_contract_sha256": contract.fingerprint,
        "scope_fingerprint": scope.fingerprint,
        "backend": "ToyLinearAdapter: four-parameter normalized SGD",
        "baseline_checkpoint_sha256": initial_hash,
        "checkpoint_sha256": learner.model_hash,
        "checkpoint_files": files,
        "phase": state.phase,
        "cycles": cycles,
        "qualification": "NOT_ESTABLISHED",
        "pretrained_improvement_established": False,
        "graduation_authorized": False,
        "limitations": [
            "This is a hand-engineered four-parameter toy, not pretrained/HF/LoRA learning.",
            "Operator-owned deterministic development cases are public, not private or independent.",
            "The transfer split is a new same-grammar sample, not structural transfer.",
            "Checkpoint hashes and local reload are integrity checks, not external custody.",
            "An internal development PASS only awaits separate private qualification.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory for exclusively created, reloadable toy checkpoints")
    args = parser.parse_args(argv)
    report = run_objective_learning_loop(
        arithmetic_objective(), checkpoint_dir=args.checkpoint_dir,
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
