"""Controlled TRAIN-only adaptation probes for the tabular Minotaur taskmaster.

These tests compare observed routing policy outputs against frozen baselines
under a *synthetic*, operator-supplied correctness schedule. They do not
establish a pretrained neural taskmaster, a learned learner, or an examiner.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import re
import unittest

from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.minotaur_model import RoomContext, plan_training_room
from labyrinth.minotaur_policy_learning import (
    MinotaurTrainingPolicy,
    TrainCaseApproval,
    TrainRoomOutcome,
)
from labyrinth.training_scope import TrainingScopeSpec
from labyrinth.regenerative_maze import generate_maze_plan


SCOPE = TrainingScopeSpec(
    objective_id="dual-skill-corridor", domain="reasoning",
    skills=("arithmetic", "program_reasoning"),
    specificity=3, scale=2, max_cycles=3, seed=607,
)


class FrozenBaseline:
    """Unchanging, operator-installed TRAIN planning control."""

    def propose_room(self, metadata: object) -> dict[str, object]:
        return {"difficulty": 3, "tactic": "practice"}


def case(index: int, *, skill: str, tactic: str = "practice",
         objective: str = SCOPE.objective_id) -> tuple[TrainCaseApproval, TrainRoomOutcome]:
    commitment = hashlib.sha256(f"{objective}:TRAIN:case:{index}".encode()).hexdigest()
    approval = TrainCaseApproval(
        room_id=f"train-room-{index}", case_id=f"train-case-{index}",
        skill=skill, tactic=tactic, difficulty=3, prompt_sha256=commitment,
    )
    observed = TrainRoomOutcome(
        objective_id=objective, room_id=approval.room_id,
        case_id=approval.case_id, skill=approval.skill,
        tactic=approval.tactic, difficulty=approval.difficulty,
        prompt_sha256=approval.prompt_sha256, was_correct=True,
    )
    return approval, observed


def fixture() -> tuple[tuple[TrainCaseApproval, ...], tuple[TrainRoomOutcome, ...]]:
    entries = [
        case(i, skill="arithmetic", tactic="practice") for i in range(3)
    ] + [
        case(i, skill="program_reasoning", tactic="practice") for i in range(3, 6)
    ] + [
        case(i, skill="arithmetic", tactic="contrast") for i in range(6, 9)
    ] + [
        case(i, skill="arithmetic", tactic="transfer") for i in range(9, 13)
    ] + [
        case(i, skill="arithmetic", tactic="practice") for i in range(13, 16)
    ]
    return tuple(item[0] for item in entries), tuple(item[1] for item in entries)


APPROVALS, OUTCOMES = fixture()


def context(skill: str, *, cycle: int = 0) -> RoomContext:
    return RoomContext(
        objective_id=SCOPE.objective_id, domain=SCOPE.domain, skill=skill,
        room_id=f"train-cycle-{cycle}-{skill}", cycle_index=cycle,
        case_seed=808 + cycle, specificity=SCOPE.specificity, scale=SCOPE.scale,
    )


def policy() -> MinotaurTrainingPolicy:
    return MinotaurTrainingPolicy(scope=SCOPE, approved_training_cases=APPROVALS)


class MinotaurPolicyEffectivenessL8(unittest.TestCase):
    def test_same_declared_live_corridor_diverges_after_actual_train_feedback(self):
        # Exercise actual question materialization, rather than treating a
        # RoomPlan suggestion as if it already changed learner experience.
        scope = TrainingScopeSpec(
            objective_id="feedback-corridor", domain="reasoning",
            skills=("arithmetic",), specificity=1, scale=3,
            max_cycles=2, seed=321,
        )
        maze = generate_maze_plan(scope, 0)
        successful = TrainingCorridorRunner.begin(
            scope, maze, checkpoint_sha256="0" * 64)
        failing = TrainingCorridorRunner.begin(
            scope, maze, checkpoint_sha256="0" * 64)

        def solve(view: object) -> str:
            self.assertFalse(hasattr(view, "answer"))
            match = re.search(r"Calculate (\d+) ([+-]) (\d+)\.", view.prompt)
            self.assertIsNotNone(match, "first two TRAIN rooms should be practice")
            left, op, right = match.groups()
            value = int(left) + int(right) if op == "+" else int(left) - int(right)
            return f"ANSWER: {value}"

        for _ in range(2):
            good_view = successful.next_training_prompt()
            bad_view = failing.next_training_prompt()
            self.assertTrue(successful.submit_training_response(solve(good_view)).correct)
            self.assertFalse(failing.submit_training_response("NOT AN ANSWER").correct)
        good_third = successful.next_training_prompt()
        bad_third = failing.next_training_prompt()
        self.assertIn("Expression A is", good_third.prompt)
        self.assertIn("Calculate", bad_third.prompt)
        self.assertNotEqual(good_third.prompt_sha256, bad_third.prompt_sha256)
        self.assertNotEqual(successful.policy.model_hash, failing.policy.model_hash)

    def test_skill_specific_feedback_changes_train_routing_but_frozen_controls_do_not(self):
        trained = policy()
        static = FrozenBaseline()
        before = {skill: plan_training_room(context(skill), taskmaster=trained)
                  for skill in SCOPE.skills}
        frozen_before = {skill: plan_training_room(context(skill), taskmaster=static)
                         for skill in SCOPE.skills}
        no_model_before = {skill: plan_training_room(context(skill))
                           for skill in SCOPE.skills}
        for outcome in OUTCOMES[:3]:
            trained.observe(outcome)
        for outcome in OUTCOMES[3:6]:
            trained.observe(replace(outcome, was_correct=False))

        arithmetic = plan_training_room(context("arithmetic"), taskmaster=trained)
        program = plan_training_room(context("program_reasoning"), taskmaster=trained)
        self.assertEqual((before["arithmetic"].tactic,
                          before["arithmetic"].difficulty), ("practice", 3))
        self.assertEqual((before["program_reasoning"].tactic,
                          before["program_reasoning"].difficulty), ("practice", 3))
        self.assertEqual((arithmetic.tactic, arithmetic.difficulty), ("contrast", 4))
        self.assertEqual((program.tactic, program.difficulty), ("practice", 1))
        self.assertEqual({skill: plan_training_room(context(skill), taskmaster=static)
                          for skill in SCOPE.skills}, frozen_before)
        self.assertEqual({skill: plan_training_room(context(skill))
                          for skill in SCOPE.skills}, no_model_before)
        self.assertEqual(arithmetic.source, "untrusted_model_suggestion")

    def test_confronted_with_new_tactic_weakness_difficulty_falls_without_skill_leakage(self):
        trained = policy()
        for outcome in OUTCOMES[:3] + OUTCOMES[6:9]:
            trained.observe(outcome)
        mastered = plan_training_room(context("arithmetic"), taskmaster=trained)
        self.assertEqual((mastered.tactic, mastered.difficulty), ("transfer", 4))
        unaffected = plan_training_room(context("program_reasoning"), taskmaster=trained)
        for outcome in OUTCOMES[9:13]:
            trained.observe(replace(outcome, was_correct=False))
        weakened = plan_training_room(context("arithmetic"), taskmaster=trained)
        self.assertEqual((weakened.tactic, weakened.difficulty), ("transfer", 1))
        self.assertEqual(plan_training_room(context("program_reasoning"),
                                            taskmaster=trained), unaffected)
        # Old PRACTICE successes cannot obscure the new TRANSFER weakness.
        for outcome in OUTCOMES[13:16]:
            trained.observe(outcome)
        self.assertEqual(plan_training_room(context("arithmetic"),
                                            taskmaster=trained), weakened)

    def test_label_and_zero_update_controls_distinguish_feedback_causation(self):
        correct, sham, corrupt = policy(), policy(), policy()
        initial = sham.model_hash
        for outcome in OUTCOMES[:3]:
            self.assertTrue(correct.observe(outcome, mode="correct")["updated"])
            self.assertFalse(sham.observe(outcome, mode="zero_update")["updated"])
            self.assertTrue(corrupt.observe(outcome, mode="corrupt_label")["updated"])
        self.assertEqual(sham.model_hash, initial)
        self.assertNotEqual(correct.model_hash, corrupt.model_hash)
        self.assertEqual(correct.propose_room(context("arithmetic").public_metadata()),
                         {"difficulty": 4, "tactic": "contrast"})
        self.assertEqual(sham.propose_room(context("arithmetic").public_metadata()),
                         {"difficulty": 3, "tactic": "practice"})
        self.assertEqual(corrupt.propose_room(context("arithmetic").public_metadata()),
                         {"difficulty": 1, "tactic": "practice"})

    def test_regeneration_retains_state_and_new_training_admission_is_explicit(self):
        trained = policy()
        for outcome in OUTCOMES[:3]:
            trained.observe(outcome)
        prior = trained.propose_room(context("arithmetic").public_metadata())
        future_approval, future_outcome = case(50, skill="arithmetic")
        with self.assertRaises(ValueError):
            trained.observe(future_outcome)
        trained.extend_training_registry((future_approval,))
        self.assertEqual(trained.propose_room(context("arithmetic", cycle=1).public_metadata()),
                         prior)
        trained.observe(future_outcome)
        self.assertEqual(trained.propose_room(context("arithmetic", cycle=2).public_metadata()),
                         {"difficulty": 5, "tactic": "contrast"})
        with self.assertRaises(ValueError):
            trained.extend_training_registry((future_approval,))

    def test_exam_split_and_wrong_objective_cannot_steer_train_policy(self):
        trained = policy()
        original = trained.model_hash
        for forbidden_split in ("heldout", "private_final"):
            with self.assertRaises(ValueError):
                replace(OUTCOMES[0], split=forbidden_split)
        for bad in (
            replace(OUTCOMES[0], objective_id="other-objective"),
            replace(OUTCOMES[0], case_id="unapproved-case"),
        ):
            with self.assertRaises(ValueError):
                trained.observe(bad)
        self.assertEqual(trained.model_hash, original)
        wrong_scope = replace(SCOPE, objective_id="other-objective")
        other = MinotaurTrainingPolicy(scope=wrong_scope,
                                       approved_training_cases=APPROVALS)
        with self.assertRaises(ValueError):
            other.propose_room(context("arithmetic").public_metadata())
        self.assertNotEqual(trained.model_hash, other.model_hash)


if __name__ == "__main__":
    unittest.main()
