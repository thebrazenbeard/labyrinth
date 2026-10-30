"""TRAIN-only end-to-end taskmaster/corridor acceptance; no neural claims."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tempfile
import unittest

from labyrinth.journey import (
    begin_journey, complete_maze, face_minotaur,
)
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.minotaur_policy_learning import load_policy
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec


START = "0" * 64
AFTER = "1" * 64


def scope(*, scale: int = 3, specificity: int = 1) -> TrainingScopeSpec:
    return TrainingScopeSpec(
        objective_id="live-arithmetic-training", domain="reasoning",
        skills=("arithmetic",), scale=scale, specificity=specificity,
        max_cycles=3, seed=321,
    )


def toy_arithmetic_learner(view: object) -> str:
    """Test-only exact prompt arithmetic, not a trainable neural learner."""
    # Deliberately accepts the trainee's *prompt-only* view, not a Challenge.
    assert not hasattr(view, "answer")
    prompt = view.prompt
    match = re.search(r"Calculate (\d+) ([+-]) (\d+)\.", prompt)
    if match is not None:
        a, operation, b = match.groups()
        answer = int(a) + int(b) if operation == "+" else int(a) - int(b)
    else:
        contrast = re.search(r"Expression A is (\d+) \+ (\d+)\. Expression B", prompt)
        if contrast is not None:
            answer = int(contrast.group(2)) * 2
        else:
            workshop = re.search(r"starts with (\d+) components and (receives|uses) (\d+)", prompt)
            if workshop is None:
                raise AssertionError(f"unexpected trainer prompt {prompt!r}")
            a, verb, b = workshop.groups()
            answer = int(a) + int(b) if verb == "receives" else int(a) - int(b)
    return f"ANSWER: {answer}"


class LiveTrainingCorridorTests(unittest.TestCase):
    def test_policy_changes_real_prompt_tactic_after_observed_feedback(self) -> None:
        s = scope(scale=3)
        runner = TrainingCorridorRunner.begin(
            s, generate_maze_plan(s, 0), checkpoint_sha256=START,
        )
        prompts = []
        for index in range(s.total_chambers):
            view = runner.next_training_prompt()
            self.assertFalse(hasattr(view, "answer"))
            self.assertTrue(view.challenge_id.startswith("train-taskmaster-"))
            self.assertNotIn(view.prompt_sha256, [old.prompt_sha256 for old in prompts])
            prompts.append(view)
            outcome = runner.submit_training_response(toy_arithmetic_learner(view))
            self.assertTrue(outcome.correct)
            self.assertNotEqual(outcome.policy_before_sha256, outcome.policy_after_sha256)
        self.assertTrue(runner.is_complete)
        self.assertEqual([event.tactic for event in runner.observations],
                         ["practice", "practice", "contrast"])
        self.assertIn("Expression A is", prompts[2].prompt)
        self.assertNotIn("Expression A is", prompts[0].prompt)
        self.assertGreater(runner.observations[-1].difficulty,
                           runner.observations[0].difficulty)
        self.assertEqual(len(runner.training_registry), 3)
        self.assertEqual(len(runner.state.history), 3)
        with self.assertRaises(ValueError):
            runner.next_training_prompt()

    def test_failure_changes_following_difficulty_and_no_false_training_attestation(self) -> None:
        s = scope(scale=2, specificity=3)
        runner = TrainingCorridorRunner.begin(s, generate_maze_plan(s, 0),
                                              checkpoint_sha256=START)
        initial = runner.next_training_prompt()
        self.assertIn("Calculate", initial.prompt)
        bad = runner.submit_training_response("ANSWER: not-an-integer")
        self.assertFalse(bad.correct)
        next_view = runner.next_training_prompt()
        second = runner.submit_training_response("ANSWER: not-an-integer")
        self.assertFalse(second.correct)
        self.assertLess(second.difficulty, bad.difficulty)
        self.assertNotEqual(initial.prompt, next_view.prompt)
        self.assertEqual(runner.state.checkpoint_sha256, START)
        self.assertFalse(hasattr(runner.observations[0], "graduated"))
        while not runner.is_complete:
            runner.next_training_prompt()
            runner.submit_training_response("ANSWER: not-an-integer")
        with self.assertRaisesRegex(ValueError, "no changed checkpoint"):
            runner.self_reported_completion_receipt()

    def test_requires_validated_journey_custody_for_regeneration_and_keeps_policy(self) -> None:
        s = scope(scale=3)
        journey = begin_journey(s, checkpoint_sha256=START)
        runner = TrainingCorridorRunner.begin(s, journey.maze, checkpoint_sha256=START)
        with self.assertRaisesRegex(ValueError, "validated JourneyState custody"):
            TrainingCorridorRunner.begin(
                s, generate_maze_plan(s, 1), checkpoint_sha256=START,
            )
        for index in range(s.total_chambers):
            runner.run_training_chamber(
                toy_arithmetic_learner,
                checkpoint_after_sha256=AFTER if index == s.total_chambers - 1 else None,
            )
        before = runner.policy.model_hash
        with self.assertRaises(ValueError):
            runner.regenerate(generate_maze_plan(s, 1))

        receipt = runner.self_reported_completion_receipt()
        triples = receipt.room_case_ids
        journey = complete_maze(journey, receipt)
        exam = tuple(
            ExamCaseResult(
                case_id=f"dev-{split}-{index}", skill="arithmetic", split=split,
                score=0, baseline_score=0,
                prompt_sha256=hashlib.sha256(
                    f"novel-dev-{split}-{index}".encode("utf-8")
                ).hexdigest(),
            )
            for split in ("heldout", "transfer") for index in range(4)
        )
        journey = face_minotaur(journey, MinotaurExamReceipt(
            exam_id="dev-ordinary-failure", objective_id=s.objective_id,
            cycle_index=0, checkpoint_sha256=AFTER, baseline_sha256=START,
            evaluator_id="source-only-dev-test", cases=exam,
        ))
        self.assertEqual(journey.cycle_index, 1)
        second_cycle = runner.regenerate(journey)
        self.assertIs(second_cycle.policy, runner.policy)
        self.assertEqual(second_cycle.policy.model_hash, before)
        self.assertEqual(second_cycle.state.checkpoint_sha256, AFTER)
        self.assertEqual(
            set(second_cycle.state.forbidden_case_ids),
            {entry[1] for entry in triples} | {item.case_id for item in exam},
        )
        self.assertEqual(
            set(second_cycle.state.forbidden_prompt_sha256),
            {entry[2] for entry in triples} | {item.prompt_sha256 for item in exam},
        )
        fresh = second_cycle.next_training_prompt()
        self.assertIn("Expression A is", fresh.prompt)
        self.assertNotIn(fresh.prompt_sha256,
                         {event.feedback.prompt_sha256 for event in runner.state.history})
        second_cycle.submit_training_response(toy_arithmetic_learner(fresh))
        self.assertNotEqual(second_cycle.policy.model_hash, before)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "taskmaster-policy.json"
            saved = second_cycle.policy.save_policy(destination)
            recovered = load_policy(
                destination, scope=s,
                approved_training_cases=second_cycle.training_registry,
            )
            self.assertEqual(saved, recovered.model_hash)
            self.assertEqual(recovered.model_hash, second_cycle.policy.model_hash)

    def test_feedback_rejects_excessive_response_without_advancing(self) -> None:
        s = scope(scale=1)
        runner = TrainingCorridorRunner.begin(s, generate_maze_plan(s, 0),
                                              checkpoint_sha256=START)
        view = runner.next_training_prompt()
        before = runner.policy.model_hash
        with self.assertRaises(ValueError):
            runner.submit_training_response("x" * 4097)
        self.assertEqual(before, runner.policy.model_hash)
        self.assertEqual(runner.state.history, ())
        outcome = runner.submit_training_response(toy_arithmetic_learner(view))
        self.assertTrue(outcome.correct)

    def test_registry_admission_hash_change_is_not_scored_learning(self) -> None:
        s = scope(scale=3)
        runner = TrainingCorridorRunner.begin(s, generate_maze_plan(s, 0),
                                              checkpoint_sha256=START)
        first = runner.next_training_prompt()
        runner.submit_training_response(toy_arithmetic_learner(first))
        after_observation = runner.policy.model_hash
        next_view = runner.next_training_prompt()
        # Registering a new prompt is a policy *state* change, not training.
        self.assertNotEqual(after_observation, runner.policy.model_hash)
        self.assertEqual(len(runner.observations), 1)
        self.assertEqual(len(runner.state.history), 1)
        second = runner.submit_training_response(toy_arithmetic_learner(next_view))
        self.assertEqual(len(runner.observations), 2)
        self.assertNotEqual(second.policy_before_sha256, second.policy_after_sha256)

    def test_non_arithmetic_scope_fails_closed(self) -> None:
        s = TrainingScopeSpec(
            objective_id="empathy-is-not-auto-supported", domain="empathy",
            skills=("cognitive_empathy",), scale=1, specificity=1,
            max_cycles=1, seed=42,
        )
        with self.assertRaisesRegex(ValueError, "supports arithmetic only"):
            TrainingCorridorRunner.begin(
                s, generate_maze_plan(s, 0), checkpoint_sha256=START,
            )


if __name__ == "__main__":
    unittest.main()
