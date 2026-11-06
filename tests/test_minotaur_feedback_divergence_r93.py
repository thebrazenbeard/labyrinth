"""Independent behavioral checks for the bounded staged Minotaur TRAIN path.

These are source-only regressions: a different staged TRAIN roster under
different learner feedback is not evidence of neural learning or transfer.
"""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from labyrinth.hf_pilot import (
    PilotConfig, _arithmetic_equivalence_key, _scoped_public_heldout,
)
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.minotaur_hf_roster_bridge import (
    collect_minotaur_hf_roster, replay_staged_roster, snapshot_staged_roster,
)
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.adaptive_corridors import offer_next_chamber
from labyrinth.minotaur_model import RoomPlan
from labyrinth.taskmaster_routing import _arithmetic_challenge
from labyrinth.training_scope import TrainingScopeSpec
from tests.test_journey_policy_runner import toy_arithmetic_learner


class MinotaurFeedbackDivergenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = TrainingScopeSpec(
            objective_id="adversarial-fifteen",
            domain="reasoning",
            skills=("arithmetic",),
            specificity=5,
            scale=5,
            max_cycles=2,
            seed=321,
        )
        self.config = PilotConfig(
            "unused-model", "unused-output", seed=31, train_count=15,
            eval_count=15, steps=15, tiers=(1,),
        )

    def _runner(self) -> TrainingCorridorRunner:
        return TrainingCorridorRunner.begin(
            self.scope,
            generate_maze_plan(self.scope, 0),
            checkpoint_sha256="0" * 64,
        )

    def test_identical_intention_and_seed_route_differently_after_feedback(self) -> None:
        successful = self._runner()
        struggling = self._runner()
        success_views = []
        struggle_views = []

        def success(view):
            self.assertFalse(hasattr(view, "answer"))
            success_views.append(view)
            return toy_arithmetic_learner(view)

        def failure(view):
            self.assertFalse(hasattr(view, "answer"))
            struggle_views.append(view)
            return "ANSWER: deliberately-invalid"

        success_roster = collect_minotaur_hf_roster(successful, self.config, success)
        struggle_roster = collect_minotaur_hf_roster(struggling, self.config, failure)
        self.assertEqual(len(success_views), 15)
        self.assertEqual(len(struggle_views), 15)
        self.assertTrue(all(item.correct for item in successful.observations))
        self.assertTrue(all(not item.correct for item in struggling.observations))

        # Source-frozen outcomes reveal a changed curriculum, not merely
        # different hashes for identical prompts.
        self.assertEqual(
            [item.tactic for item in successful.observations],
            ["practice"] * 2 + ["contrast"] * 2 + ["transfer"] * 11,
        )
        self.assertEqual(
            [item.tactic for item in struggling.observations],
            ["practice"] * 15,
        )
        self.assertNotEqual(
            [item.prompt for item in success_views],
            [item.prompt for item in struggle_views],
        )
        self.assertNotEqual(success_roster.train_cases, struggle_roster.train_cases)
        self.assertTrue(successful.is_complete)
        self.assertTrue(struggling.is_complete)
        self.assertFalse(success_roster.physically_authenticated)
        self.assertFalse(struggle_roster.graduation_authorized)

    def test_each_staged_train_path_stays_distinct_from_public_heldout(self) -> None:
        for learner in (toy_arithmetic_learner, lambda _view: "ANSWER: invalid"):
            with self.subTest(learner=learner.__name__):
                runner = self._runner()
                roster = collect_minotaur_hf_roster(runner, self.config, learner)
                train, report = snapshot_staged_roster(roster, self.config, self.scope)
                replay_scope, replay_train, exposures = replay_staged_roster(
                    report, self.config,
                )
                heldout = _scoped_public_heldout(
                    self.config, self.scope, 0, train,
                )
                self.assertEqual(replay_scope, self.scope)
                # Public source replay regenerates answer labels but omits
                # private generator metadata by contract. Compare the scored
                # exercise identity and contents without treating metadata
                # stripping as a curriculum change.
                self.assertEqual(
                    [replace(case, metadata=()) for case in train],
                    replay_train,
                )
                self.assertTrue(all(case.metadata == () for case in replay_train))
                self.assertEqual(exposures, roster.ordered_optimizer_exposures)
                self.assertEqual(len(heldout), 15)
                self.assertEqual({case.split for case in heldout}, {"eval"})
                self.assertEqual({case.split for case in train}, {"train"})
                self.assertEqual(len({case.id for case in train}), 15)
                self.assertEqual(len({case.prompt for case in train}), 15)
                self.assertEqual(len({_arithmetic_equivalence_key(case) for case in train}), 15)
                self.assertTrue(
                    {_arithmetic_equivalence_key(case) for case in train}.isdisjoint(
                        {_arithmetic_equivalence_key(case) for case in heldout}
                    )
                )
                self.assertTrue({case.id for case in train}.isdisjoint(
                    {case.id for case in heldout}
                ))
                self.assertTrue({case.prompt for case in train}.isdisjoint(
                    {case.prompt for case in heldout}
                ))
                self.assertEqual(
                    [entry["prompt_sha256"] for entry in report["train"]],
                    [hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
                     for case in train],
                )
                self.assertFalse(any("answer" in entry for entry in report["train"]))
                self.assertEqual(report["source_authority"],
                                 "SELF_REPORTED_TRAIN_ONLY_NOT_HOST_AUTHENTICATED")

    def test_weak_learner_has_a_complete_valid_corridor_despite_prompt_collision(self) -> None:
        """Valid weak-learner scope must not dead-end on a generated duplicate.

        In PR91, this benign objective label and seed deterministically produce
        a repeat on the 9th practice room after eight incorrect responses.
        The generator must fail closed on used evidence *and* deterministically
        choose a fresh bounded case for the same authorized room.
        """
        scope = TrainingScopeSpec(
            objective_id="r93-feedback-divergence", domain="reasoning",
            skills=("arithmetic",), specificity=5, scale=5, max_cycles=2,
            seed=321,
        )
        runner = TrainingCorridorRunner.begin(
            scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
        )
        config = self.config
        roster = collect_minotaur_hf_roster(
            runner, config, lambda _view: "ANSWER: deliberately-invalid",
        )
        self.assertTrue(runner.is_complete)
        self.assertEqual(len(roster.train_cases), 15)
        self.assertEqual(len({case.prompt for case in roster.train_cases}), 15)

    def test_weak_learner_roster_avoids_semantic_alias_before_hf_staging(self) -> None:
        """Do not admit 15 prompts and discover equivalent TRAIN cases too late.

        This independent deterministic fixture completes its raw corridor in
        PR91, but the final HF snapshot then rejects two equivalent arithmetic
        tasks. That late rejection prevents an optimizer from consuming the
        purportedly completed roster at all.
        """
        scope = TrainingScopeSpec(
            objective_id="r93-weak-seed-sweep", domain="reasoning",
            skills=("arithmetic",), specificity=5, scale=5, max_cycles=2,
            seed=11,
        )
        runner = TrainingCorridorRunner.begin(
            scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
        )
        roster = collect_minotaur_hf_roster(
            runner, self.config, lambda _view: "ANSWER: deliberately-invalid",
        )
        self.assertTrue(runner.is_complete)
        cases, _record = snapshot_staged_roster(roster, self.config, scope)
        self.assertEqual(len(cases), 15)
        self.assertEqual(len({_arithmetic_equivalence_key(case) for case in cases}), 15)

    def test_two_hundred_weak_learner_corridors_remain_stagable(self) -> None:
        """Regression for the earlier 90/200 weak-learner unusable rosters.

        This remains a source-corpus availability test, never an independent
        model-learning or evaluator-custody claim.
        """
        for seed in range(200):
            with self.subTest(seed=seed):
                scope = TrainingScopeSpec(
                    objective_id="r93-weak-seed-sweep", domain="reasoning",
                    skills=("arithmetic",), specificity=5, scale=5,
                    max_cycles=2, seed=seed,
                )
                runner = TrainingCorridorRunner.begin(
                    scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
                )
                roster = collect_minotaur_hf_roster(
                    runner, self.config, lambda _view: "ANSWER: deliberately-invalid",
                )
                cases, report = snapshot_staged_roster(roster, self.config, scope)
                self.assertEqual(len(cases), 15)
                self.assertEqual(len({_arithmetic_equivalence_key(c) for c in cases}), 15)
                self.assertEqual(len(report["train"]), 15)

    def test_exhausted_redraws_fail_closed_without_admission(self) -> None:
        """Even 513 explicitly sealed candidates may never be reused."""
        base = self._runner()
        assignment = offer_next_chamber(base.state).pending
        self.assertIsNotNone(assignment)
        selection = RoomPlan(
            assignment.room_id, assignment.difficulty,
            "practice", "initial_train_bootstrap", None,
        )
        seen_prompts: dict[str, str] = {}
        for attempt in range(513):
            case = _arithmetic_challenge(
                assignment.room, self.scope, selection, redraw_attempt=attempt,
            )
            digest = hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
            seen_prompts.setdefault(digest, case.id)
        sealed = replace(
            base.state,
            forbidden_case_ids=tuple(seen_prompts.values()),
            forbidden_prompt_sha256=tuple(seen_prompts),
        )
        runner = TrainingCorridorRunner(sealed)
        with self.assertRaisesRegex(ValueError, "exhausted bounded fresh"):
            runner.next_training_prompt()
        self.assertEqual(runner.state, sealed)
        self.assertEqual(runner.training_registry, ())
        self.assertIsNone(runner.policy)
        self.assertEqual(runner.observations, ())
        with self.assertRaisesRegex(ValueError, "bounded nonnegative"):
            _arithmetic_challenge(assignment.room, self.scope, selection,
                                  redraw_attempt=True)


if __name__ == "__main__":
    unittest.main()
