"""Offline, deterministic cognitive-empathy TRAIN walkthrough (not model training).

Run with ``python -m scripts.experiments.cognitive_train_demo`` from the repo.
Only the approved public TraineeRoomView prompt crosses the mock-learner
boundary.  This intentionally uses fabricated checkpoint hashes and a
fabricated, failing *development* diagnostic to demonstrate regeneration;
neither is evidence of real execution, evaluation, or model improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from labyrinth.cognitive_objective_training_run import (
    CognitiveObjectiveTrainingRun, open_approved_objective_training_run,
)
from labyrinth.journey import TRAINING, begin_journey, complete_maze, face_minotaur
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.objective_contract import ObjectiveContract, SkillTarget
from labyrinth.training_scope import TrainingScopeSpec


POLICIES = ("all-a", "all-b", "all-c", "rotate-abc")
DISCLAIMER = (
    "OFFLINE SOURCE-ONLY DEMO: synthetic A/B/C practice and mock responses. "
    "NO Q6/host authorization, model weights, physical checkpoints, "
    "independent exam, measured learning, mastery, or graduation claim."
)


def _fake_digest(label: str) -> str:
    """A distinguishable *mock* identity, never a file/model digest."""
    return hashlib.sha256(("labyrinth-mock-demo-only:" + label).encode()).hexdigest()


def _demo_contract(seed: int) -> ObjectiveContract:
    scope = TrainingScopeSpec(
        objective_id=f"offline-cognitive-empathy-demo-{seed}",
        domain="empathy", skills=("cognitive_empathy",),
        specificity=5, scale=5, max_cycles=2, seed=seed,
    )
    return ObjectiveContract(
        scope=scope,
        intention="Practice synthetic uncertainty-preserving perspective choices.",
        targets=(SkillTarget(
            skill="cognitive_empathy", focus_tags=("context",),
            observable_behaviors=(
                "Select a perspective without assuming private motives.",
            ),
            rubric_id="perspective_choice_train_v1",
            minimum_score_percent=80, minimum_gain_percent=10,
        ),),
        excluded_scopes=("Emotional or reactive empathy measurement",),
        forbidden_claims=("No human empathy mastery can be inferred.",),
    )


def _mock_response(policy: str, ordinal: int) -> str:
    """An intentionally simple learner, independent of private scoring gold."""
    if policy == "rotate-abc":
        return "ABC"[ordinal % 3]
    return {"all-a": "A", "all-b": "B", "all-c": "C"}[policy]


def _mock_failed_development_diagnostic(waiting) -> MinotaurExamReceipt:
    """Fabricated zero-score DEV event solely to demonstrate cycle reroll.

    No private/held-out evaluation items are loaded or generated here. The
    fabricated identifiers are never suitable for a trusted Minotaur exam.
    """
    cases = tuple(
        ExamCaseResult(
            case_id=f"demo-not-an-exam-{waiting.cycle_index}-{kind}-{index}",
            skill="cognitive_empathy", split=kind,
            score=0.0, baseline_score=0.0,
            prompt_sha256=_fake_digest(
                f"not-an-exam-{waiting.cycle_index}-{kind}-{index}"
            ),
        )
        for kind in ("heldout", "transfer") for index in range(4)
    )
    return MinotaurExamReceipt(
        exam_id=f"demo-forced-failure-{waiting.cycle_index}",
        objective_id=waiting.scope.objective_id,
        cycle_index=waiting.cycle_index,
        checkpoint_sha256=waiting.checkpoint_sha256,
        baseline_sha256=waiting.baseline_checkpoint_sha256,
        evaluator_id="demo-untrusted-fabricated-diagnostic", cases=cases,
    )


def run_demo(*, seed: int = 107, policy: str = "rotate-abc",
             show_prompts: bool = True) -> dict:
    """Return a self-reported source receipt; never assert neural learning."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2^31)")
    if policy not in POLICIES:
        raise ValueError("unsupported deterministic mock policy")
    contract = _demo_contract(seed)
    journey = begin_journey(
        contract.scope, checkpoint_sha256=_fake_digest("synthetic-before"),
    )
    if show_prompts:
        print(DISCLAIMER)
        print(f"Objective: cognitive_empathy | mock policy: {policy} | cycles: 2")

    cycles: list[dict] = []
    exposed_cases: set[str] = set()
    exposed_prompts: set[str] = set()
    for cycle_index in range(2):
        if journey.phase != TRAINING or journey.cycle_index != cycle_index:
            raise RuntimeError("synthetic demo could not enter next TRAIN cycle")
        run = open_approved_objective_training_run(contract, journey)
        if not isinstance(run, CognitiveObjectiveTrainingRun):
            raise RuntimeError("cognitive TRAIN provider routing was not approved")
        total = run.feedback_state.total_rooms
        for room_index in range(total):
            view = run.next_training_prompt()
            if show_prompts:
                print(f"\n[Cycle {cycle_index + 1}/2, room {room_index + 1}/{total}]")
                print(view.prompt)
            # Critical: this mock response NEVER inspects prepared.challenge,
            # answer keys, source gold, or returned score feedback.
            run.submit_training_response(
                _mock_response(policy, cycle_index * total + room_index)
            )
        state = run.feedback_state
        receipt = run.self_reported_completion_receipt(
            checkpoint_after_sha256=_fake_digest(f"synthetic-after-{cycle_index}"),
        )
        case_ids = [entry[1] for entry in receipt.room_case_ids]
        prompt_sha256 = [entry[2] for entry in receipt.room_case_ids]
        if (exposed_cases.intersection(case_ids)
                or exposed_prompts.intersection(prompt_sha256)):
            raise RuntimeError("source demo replayed a previously exposed room")
        exposed_cases.update(case_ids)
        exposed_prompts.update(prompt_sha256)
        cycles.append({
            "cycle_index": cycle_index,
            "rooms_exposed": state.completed_rooms,
            "aggregate_correct_mock_responses": state.correct_rooms,
            "aggregate_incorrect_mock_responses": state.incorrect_rooms,
            "source_only_self_reported_receipt": {
                "mock_checkpoint_before_sha256": receipt.checkpoint_before_sha256,
                "mock_checkpoint_after_sha256": receipt.checkpoint_after_sha256,
                "room_case_ids_and_prompt_commitments": [
                    list(entry) for entry in receipt.room_case_ids
                ],
                "is_real_checkpoint": False,
                "is_independently_attested": False,
            },
        })
        if show_prompts:
            print(f"Cycle {cycle_index + 1}: {state.completed_rooms} synthetic rooms; "
                  "SOURCE-ONLY self-reported receipt (mock checkpoint).")
        if cycle_index == 0:
            # A deliberately fabricated FAIL demonstrates Labyrinth's
            # failure-to-regeneration path. This is not actual exam evidence.
            waiting = complete_maze(journey, receipt)
            journey = face_minotaur(
                waiting, _mock_failed_development_diagnostic(waiting),
            )
            if show_prompts:
                print("Fabricated DEV failure only: demonstrating regeneration, "
                      "NOT evaluating a model or running a private exam.")

    result = {
        "schema": "labyrinth.offline-cognitive-train-demo.v1",
        "objective_id": contract.scope.objective_id,
        "skill": "cognitive_empathy",
        "seed": seed,
        "mock_policy": policy,
        "offline": True,
        "source_only": True,
        "q6_host_authorized": False,
        "physical_training_performed": False,
        "real_model_checkpoint_observed": False,
        "independent_exam_performed": False,
        "mastery_or_graduation_authorized": False,
        "distinct_training_case_count": len(exposed_cases),
        "distinct_training_prompt_count": len(exposed_prompts),
        "cycles": cycles,
    }
    if show_prompts:
        print("\nSOURCE-ONLY receipt (no private answer keys):")
        print(json.dumps(result, sort_keys=True, indent=2))
        print(DISCLAIMER)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=107,
                        help="deterministic scenario seed (default: 107)")
    parser.add_argument("--mock-policy", choices=POLICIES, default="rotate-abc",
                        help="predefined mock response policy (never reads gold)")
    parser.add_argument("--json", action="store_true",
                        help="receipt JSON only, omit public prompt walkthrough")
    args = parser.parse_args(argv)
    try:
        result = run_demo(seed=args.seed, policy=args.mock_policy,
                          show_prompts=not args.json)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
