"""Synthetic feedback scheduling tests, NOT signed/physical qualification.

The existing genuine signed development fixture addresses the broad skill
`sentiment`, not the MiniLM Maze's two polarity skills. These unit tests
construct typed feedback explicitly to exercise the TRAIN-only deterministic
scheduler. They cannot prove upstream signatures, host custody, model learning,
or actual adaptation to a measured physical weakness.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import random
import unittest

from labyrinth.measured_sentiment_corridors import (
    plan_measured_sentiment_corridor,
)
from labyrinth.measured_weakness_feedback import MeasuredRemediation
from labyrinth.minotaur_development_feedback import (
    MinotaurDevelopmentFeedback, SkillDevelopmentObservation,
)
from labyrinth.regenerative_maze import generate_maze_plan
from scripts.qualify_maze_conditioned_minilm_cpu import (
    PUBLIC_DEVELOPMENT, PUBLIC_TRANSFER,
    SKILLS, default_scope, build_training_cycles,
)


def synthetic_feedback(scope, *, tactic="practice", weak="negative_polarity"):
    # This is TEST-DATA, not a verifiable signature. No production call should
    # create one by assigning `source_signature_verified=True` as done here.
    maze = generate_maze_plan(scope, 1, weak_skills=(weak,))
    remediation = MeasuredRemediation(
        maze=maze, weak_skills=(weak,), uncertain_skills=(weak,),
        coverage_by_skill=tuple((skill, 2, 2) for skill in SKILLS),
        exam_id="synthetic-development-exam", event_id="synthetic-development-event",
        from_cycle=0, to_cycle=1, roster_sha256="a" * 64,
    )
    obs = tuple(SkillDevelopmentObservation(
        skill=skill, heldout_correct=0 if skill == weak else 2,
        heldout_total=2, heldout_errors=2 if skill == weak else 0,
        transfer_correct=0 if skill == weak else 2,
        transfer_total=2, transfer_errors=2 if skill == weak else 0,
        uncertain=skill == weak, weak=skill == weak,
        suggested_train_tactic=tactic if skill == weak else None,
    ) for skill in SKILLS)
    return MinotaurDevelopmentFeedback(
        exam_id=remediation.exam_id, event_id=remediation.event_id,
        from_cycle=0, to_cycle=1,
        candidate_artifact_sha256=sha256(b"synthetic-adapter-file").hexdigest(),
        observations=obs, remediation=remediation,
        source_signature_verified=True,
    )


class MeasuredSentimentCorridorTests(unittest.TestCase):
    def setUp(self):
        self.scope = default_scope()

    def plan(self, tactic="practice", **options):
        feedback = synthetic_feedback(self.scope, tactic=tactic)
        return plan_measured_sentiment_corridor(
            self.scope, feedback, **options)

    def test_three_tactics_change_training_visits_without_changing_rooms(self):
        plans = {tactic: self.plan(tactic) for tactic in (
            "practice", "contrast", "transfer")}
        exposures = {
            tactic: tuple(tuple(row.exposure_id for row in batch)
                          for batch in plan.batches)
            for tactic, plan in plans.items()
        }
        self.assertEqual(len(set(exposures.values())), 3)
        self.assertEqual({plan.cycle.plan for plan in plans.values()},
                         {generate_maze_plan(self.scope, 1,
                                             weak_skills=("negative_polarity",))})
        self.assertEqual(plans["practice"].class_counts_per_batch[0], (3, 1))
        self.assertEqual(plans["transfer"].class_counts_per_batch[0], (3, 1))
        self.assertEqual(plans["contrast"].class_counts_per_batch[0], (2, 2))
        self.assertNotEqual(exposures["practice"][0], exposures["transfer"][0])
        first_batch_text = {
            tactic: tuple(row.prompt for row in plan.batches[0])
            for tactic, plan in plans.items()
        }
        self.assertEqual(len(set(first_batch_text.values())), 3)
        for plan in plans.values():
            self.assertEqual(plan.unique_training_exposures, 12)
            self.assertEqual(len(plan.batches), 16)
            self.assertFalse(plan.source_signature_host_attested)
            self.assertFalse(plan.model_execution_authenticated)
            self.assertFalse(plan.prior_skill_transfer_qualified)
            self.assertFalse(plan.graduation_authorized)

    def test_every_batch_contains_only_authentic_unique_train_examples(self):
        plan = self.plan("contrast")
        actual = build_training_cycles(
            self.scope, weak_cycle_one=("negative_polarity",))[1]
        by_id = {row.exposure_id: row for row in actual.examples}
        public = {prompt.casefold() for prompt, _ in
                  PUBLIC_DEVELOPMENT + PUBLIC_TRANSFER}
        visited = set()
        for batch in plan.batches:
            self.assertEqual(len(batch), 4)
            self.assertEqual(len({row.exposure_id for row in batch}), 4)
            self.assertTrue(all(row == by_id[row.exposure_id] for row in batch))
            self.assertTrue(all(row.prompt.casefold() not in public for row in batch))
            visited.update(row.exposure_id for row in batch)
        self.assertEqual(visited, set(by_id))

    def test_repeatable_no_global_rng_side_effect(self):
        random.seed(911)
        before = random.getstate()
        a = self.plan("transfer")
        b = self.plan("transfer")
        self.assertEqual(a, b)
        self.assertEqual(random.getstate(), before)

    def test_inconsistent_source_feedback_and_wrong_scope_fail_closed(self):
        feedback = synthetic_feedback(self.scope)
        invalid = (
            replace(feedback, source_signature_verified=False),
            replace(feedback, graduation_authorized=True),
            replace(feedback, evaluator_host_independence_verified=True),
            replace(feedback, model_execution_authenticated=True),
            replace(feedback, observations=feedback.observations[:-1]),
            replace(feedback, observations=tuple(replace(row, weak=False)
                  if row.weak else row for row in feedback.observations)),
            replace(feedback, observations=tuple(replace(row,
                suggested_train_tactic="private_final") if row.weak else row
                for row in feedback.observations)),
            replace(feedback, remediation=replace(feedback.remediation,
                event_id="unbound-event")),
            replace(feedback, remediation=replace(feedback.remediation,
                weak_skills=("unknown_skill",))),
            replace(feedback, remediation=replace(feedback.remediation,
                maze=generate_maze_plan(self.scope, 0))),
        )
        for bad in invalid:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan_measured_sentiment_corridor(self.scope, bad)
        wrong_scope = replace(self.scope, seed=self.scope.seed + 1)
        with self.assertRaises(ValueError):
            plan_measured_sentiment_corridor(wrong_scope, feedback)

    def test_insufficient_steps_fail_closed_instead_of_omitting_cases(self):
        with self.assertRaisesRegex(ValueError, "every TRAIN case"):
            self.plan("contrast", steps_per_cycle=4)
        for invalid in (0, 2, 17, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.plan("contrast", steps_per_cycle=invalid)


if __name__ == "__main__":
    unittest.main()
