"""Real toy-parameter learning across regenerative Labyrinth journeys.

This source-only arithmetic demonstration uses the four-parameter
``ToyLinearAdapter``. The same learner instance retains changed coefficients
when Minotaur's *internal development* exam sends it through a regenerated
maze. It neither trains a pretrained language model nor establishes formal
Minotaur graduation, cognitive empathy, or structural reasoning transfer.
"""

from __future__ import annotations

import hashlib
import math

from .evaluation import score_response
from .journey import (
    AWAITING_EXAM,
    TRAINING,
    TrainingCompletionReceipt,
    begin_journey,
    complete_maze,
    face_minotaur,
)
from .journey_challenges import prepare_training_rooms
from .maze import Challenge, generate_maze
from .minotaur_exam import ExamCaseResult, MinotaurExamReceipt, evaluate_minotaur
from .toy_backend import ToyLinearAdapter
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.toy-journey.v1"


def _exam_seed(scope: TrainingScopeSpec, cycle_index: int, split: str) -> int:
    # Domain separated from every training room case_seed. The same train/
    # development challenge ID cannot be reused across journey cycles.
    digest = hashlib.sha256(
        f"labyrinth-toy-dev-v1\x1f{scope.fingerprint}\x1f{cycle_index}\x1f{split}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _validate_train_batch(state, prepared) -> None:
    if len(prepared) != len(state.maze.rooms):
        raise ValueError("not all declared maze rooms were materialized")
    ids = {entry.challenge.id for entry in prepared}
    hashes = {_hash_prompt(entry.challenge.prompt) for entry in prepared}
    if (len(ids) != len(prepared) or len(hashes) != len(prepared)
            or ids & (set(state.used_training_case_ids) | set(state.sealed_exam_case_ids))
            or hashes & (set(state.used_training_prompt_sha256) | set(state.sealed_exam_prompt_sha256))):
        # Check before ANY parameter update: a rejected training receipt after
        # learning would have already contaminated the learner.
        raise ValueError("generated training batch overlaps prior training or exam evidence")


def _exam_cases(scope: TrainingScopeSpec, cycle_index: int, count: int) -> tuple[tuple[str, Challenge], ...]:
    cases: list[tuple[str, Challenge]] = []
    for section in ("heldout", "transfer"):
        # Both are separately seeded, untouched eval cases. ToyLinearAdapter's
        # canonical prompt grammar supports *same-family* transfer only; these
        # cannot establish structural transfer to unfamiliar task families.
        generated = generate_maze(
            seed=_exam_seed(scope, cycle_index, section), tiers=(1,), count=count,
            split="eval",
        )
        cases.extend((section, case) for case in generated)
    return tuple(cases)


def run_toy_journey(
    scope: TrainingScopeSpec,
    *,
    learner_seed: int = 0,
    epochs_by_cycle: tuple[int, ...] | None = None,
    learning_rate: float = 0.5,
    exam_count_per_slice: int = 8,
) -> dict:
    """Run bounded arithmetic journeys with actual learner weight updates.

    No hidden downloads, checkpoint writes, or independent evaluator access.
    Every cycle's training task IDs and prompt hashes are preflighted before
    supervision. Scores are computed locally by an operator-owned development
    evaluator, and receive no graduation authority.
    """
    if type(scope) is not TrainingScopeSpec:
        raise ValueError("scope must be a frozen TrainingScopeSpec")
    TrainingScopeSpec.__post_init__(scope)
    if scope.skills != ("arithmetic",):
        raise ValueError("ToyLinearAdapter supports only an arithmetic-only training scope")
    if scope.total_chambers < 2:
        raise ValueError("toy trainer requires at least two arithmetic chambers per cycle")
    if type(learner_seed) is not int or not 0 <= learner_seed < 2**31:
        raise ValueError("learner_seed must be a nonnegative 31-bit integer")
    if epochs_by_cycle is None:
        epochs_by_cycle = (4,) * scope.max_cycles
    if (type(epochs_by_cycle) is not tuple or len(epochs_by_cycle) != scope.max_cycles
            or any(type(epoch) is not int or not 1 <= epoch <= 1000 for epoch in epochs_by_cycle)):
        raise ValueError("epochs_by_cycle must provide one bounded positive epoch count per cycle")
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1):
        raise ValueError("learning_rate must be finite and in (0,1]")
    if type(exam_count_per_slice) is not int or not 4 <= exam_count_per_slice <= 32:
        raise ValueError("exam_count_per_slice must be an integer in [4,32]")

    learner = ToyLinearAdapter(seed=learner_seed)
    frozen_baseline = ToyLinearAdapter(seed=learner_seed)
    baseline_hash = learner.model_hash
    state = begin_journey(scope, checkpoint_sha256=baseline_hash)
    trajectory: list[dict] = []

    for cycle_index, epochs in enumerate(epochs_by_cycle):
        if state.phase != TRAINING:
            break
        if state.cycle_index != cycle_index or learner.model_hash != state.checkpoint_sha256:
            raise RuntimeError("learner state diverged from retained journey checkpoint")

        prepared = prepare_training_rooms(state.maze)
        _validate_train_batch(state, prepared)
        before = learner.model_hash
        trained = learner.train(
            [entry.challenge for entry in prepared],
            epochs=epochs,
            learning_rate=float(learning_rate),
        )
        after = learner.model_hash
        if (trained["before_model_hash"] != before
                or trained["after_model_hash"] != after
                or trained["parameter_updates"] <= 0 or before == after):
            raise RuntimeError("toy train operation did not produce a verified parameter update")
        receipt = TrainingCompletionReceipt(
            objective_id=scope.objective_id,
            cycle_index=cycle_index,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
            room_case_ids=tuple(
                (entry.room_id, entry.challenge.id, _hash_prompt(entry.challenge.prompt))
                for entry in prepared
            ),
        )
        state = complete_maze(state, receipt)
        if state.phase != AWAITING_EXAM or state.checkpoint_sha256 != learner.model_hash:
            raise RuntimeError("journey state and trained learner disagree")

        fresh = _exam_cases(scope, cycle_index, exam_count_per_slice)
        exam_ids = {case.id for _, case in fresh}
        exam_prompts = {_hash_prompt(case.prompt) for _, case in fresh}
        if (
            len(exam_ids) != len(fresh) or len(exam_prompts) != len(fresh)
            or exam_ids & (set(state.used_training_case_ids) | set(state.sealed_exam_case_ids))
            or exam_prompts & (
                set(state.used_training_prompt_sha256) | set(state.sealed_exam_prompt_sha256)
            )
        ):
            raise ValueError("development examination overlaps prior training or examination")

        observations = tuple(
            ExamCaseResult(
                case_id=case.id,
                skill="arithmetic",
                split=section,
                score=score_response(case, learner.generate(case.prompt)).points,
                baseline_score=score_response(case, frozen_baseline.generate(case.prompt)).points,
                prompt_sha256=_hash_prompt(case.prompt),
            )
            for section, case in fresh
        )
        exam = MinotaurExamReceipt(
            exam_id=f"toyexam-{_exam_seed(scope, cycle_index, 'exam')}",
            objective_id=scope.objective_id,
            cycle_index=cycle_index,
            checkpoint_sha256=after,
            baseline_sha256=baseline_hash,
            evaluator_id="internal.toy-development-v1",
            cases=observations,
        )
        decision = evaluate_minotaur(scope, exam)
        state = face_minotaur(state, exam)
        if state.checkpoint_sha256 != learner.model_hash:
            raise RuntimeError("maze regeneration unexpectedly reset learner weights")
        trajectory.append({
            "cycle_index": cycle_index,
            "checkpoint_before_sha256": before,
            "checkpoint_after_sha256": after,
            "parameter_updates": trained["parameter_updates"],
            "epochs": epochs,
            "training_case_count": len(prepared),
            "training_case_ids": [entry.challenge.id for entry in prepared],
            "training_room_ids": [entry.room_id for entry in prepared],
            "exam_case_count": len(observations),
            "exam_case_ids": [entry.case_id for entry in observations],
            "exam_candidate_correct": sum(entry.score for entry in observations),
            "exam_frozen_baseline_correct": sum(entry.baseline_score for entry in observations),
            "exam_status": decision.status,
            "weak_skills": list(decision.weak_skills),
        })

    return {
        "schema": SCHEMA,
        "scope_fingerprint": scope.fingerprint,
        "objective_id": scope.objective_id,
        "baseline_sha256": baseline_hash,
        "final_checkpoint_sha256": learner.model_hash,
        "final_phase": state.phase,
        "cycles": trajectory,
        "attempts": len(state.attempts),
        "formal_graduation": False,
        "qualification": "NOT_ESTABLISHED",
        "limitations": [
            "Four-parameter arithmetic toy learner, not pretrained language-model adaptation.",
            "Internal development scores are not independent or signed private evaluation.",
            "Transfer slice uses a fresh same-family prompt distribution, not structural transfer.",
            "Checkpoint hashes are local integrity receipts, not proof of operator custody.",
        ],
    }
