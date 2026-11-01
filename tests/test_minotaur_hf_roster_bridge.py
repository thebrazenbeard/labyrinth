"""Source-only adaptive Minotaur TRAIN catalog and HF visit-order contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch

from labyrinth import journey_policy_runner as runner_source
from labyrinth.hf_pilot import PilotConfig, _train_steps, _training_visit_exposures
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.minotaur_hf_roster_bridge import (
    MinotaurHfTrainRoster, collect_minotaur_hf_roster,
)
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec
from tests.test_hf_fullstate_hook_r7 import _Model, _Torch
from tests.test_journey_policy_runner import toy_arithmetic_learner


def _scope() -> TrainingScopeSpec:
    return TrainingScopeSpec(
        objective_id="hf-adaptive-arithmetic", domain="reasoning",
        skills=("arithmetic",), scale=3, specificity=1,
        max_cycles=2, seed=321,
    )


def _runner() -> TrainingCorridorRunner:
    scope = _scope()
    return TrainingCorridorRunner.begin(
        scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
    )


def _config() -> PilotConfig:
    return PilotConfig(
        "unused-model", "unused-output", seed=31,
        train_count=3, steps=5, tiers=(1,),
    )


class MinotaurHfRosterContractTests(unittest.TestCase):
    def test_feedback_really_changes_later_train_prompt_and_pinned_catalog(self) -> None:
        successful_runner = _runner()
        unsuccessful_runner = _runner()
        successful = collect_minotaur_hf_roster(
            successful_runner, _config(), toy_arithmetic_learner,
        )
        unsuccessful = collect_minotaur_hf_roster(
            unsuccessful_runner, _config(), lambda view: "ANSWER: not-integer",
        )
        self.assertTrue(all(entry.correct for entry in successful_runner.observations))
        self.assertFalse(any(entry.correct for entry in unsuccessful_runner.observations))
        self.assertEqual(successful.train_cases[0].prompt,
                         unsuccessful.train_cases[0].prompt)
        self.assertNotEqual([case.prompt for case in successful.train_cases[1:]],
                            [case.prompt for case in unsuccessful.train_cases[1:]])

    def test_adaptive_train_catalog_binds_exact_hf_visits(self) -> None:
        public_views = []

        def learner(view):
            self.assertFalse(hasattr(view, "answer"))
            public_views.append(view)
            return "ANSWER: not-integer"  # Negative TRAIN feedback changes routing.

        runner = _runner()
        roster = collect_minotaur_hf_roster(runner, _config(), learner)
        self.assertTrue(runner.is_complete)
        self.assertEqual(len(roster.train_cases), 3)
        self.assertEqual(len(runner.observations), 3)
        self.assertEqual(roster.initial_checkpoint_sha256,
                         runner.state.initial_checkpoint_sha256)
        self.assertEqual(roster.adaptive_feedback_rows, tuple(
            (event.assignment.room_id, event.feedback.case_id,
             event.feedback.prompt_sha256, event.feedback.score,
             event.feedback.checkpoint_before_sha256,
             event.feedback.checkpoint_after_sha256)
            for event in runner.state.history
        ))
        self.assertEqual([case.prompt for case in roster.train_cases],
                         [view.prompt for view in public_views])
        self.assertEqual(roster.ordered_optimizer_exposures,
                         _training_visit_exposures(
                             list(roster.train_cases), seed=31, steps=5,
                         ))
        self.assertEqual(
            tuple((case.id, hashlib.sha256(case.prompt.encode()).hexdigest())
                  for case in roster.train_cases),
            tuple((entry.case_id, entry.prompt_sha256)
                  for entry in runner.observations),
        )
        self.assertEqual({case.split for case in roster.train_cases}, {"train"})
        self.assertFalse(roster.physically_authenticated)
        self.assertFalse(roster.graduation_authorized)
        self.assertNotIn("train_cases=", repr(roster))

        for flag in ("physically_authenticated", "graduation_authorized"):
            with self.subTest(flag=flag), self.assertRaises(TypeError):
                replace(roster, **{flag: True})
            with self.subTest(flag=flag), self.assertRaises(TypeError):
                MinotaurHfTrainRoster(
                    scope_sha256=roster.scope_sha256,
                    cycle_index=roster.cycle_index,
                    train_cases=roster.train_cases,
                    ordered_optimizer_exposures=roster.ordered_optimizer_exposures,
                    **{flag: True},
                )
            with self.subTest(flag=flag), self.assertRaises(AttributeError):
                object.__setattr__(roster, flag, True)

    def test_invalid_scope_or_budget_refused_before_room_routing(self) -> None:
        for kwargs in (
            {"tiers": (2,)}, {"train_count": 2}, {"steps": 0},
            {"seed": True}, {"steps": True},
        ):
            runner = _runner()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                collect_minotaur_hf_roster(
                    runner, replace(_config(), **kwargs),
                    lambda view: "ANSWER: 0",
                )
            self.assertEqual(runner.observations, ())
            self.assertEqual(runner.training_registry, ())

    def test_incompatible_hf_scalar_resources_rejected_before_staging(self) -> None:
        for kwargs in (
            {"eval_count": 1}, {"eval_count": True},
            {"max_tokens": 1}, {"max_tokens": True},
            {"max_new_tokens": 0}, {"max_new_tokens": 33},
            {"learning_rate": float("nan")}, {"learning_rate": 0.1},
            {"lora_rank": 0}, {"lora_alpha": 65},
            {"device": "other"}, {"answer_only": 1},
        ):
            runner = _runner()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                collect_minotaur_hf_roster(
                    runner, replace(_config(), **kwargs),
                    lambda view: "ANSWER: 0",
                )
            self.assertEqual(runner.observations, ())
            self.assertEqual(runner.training_registry, ())

    def test_operator_gold_access_requires_exact_pending_view(self) -> None:
        runner = _runner()
        with self.assertRaises(ValueError):
            runner.operator_pending_train_case(None)
        public = runner.next_training_prompt()
        with self.assertRaises(ValueError):
            runner.operator_pending_train_case(replace(public, prompt="forged"))
        case = runner.operator_pending_train_case(public)
        self.assertEqual(case.prompt, public.prompt)
        self.assertEqual(case.split, "train")
        self.assertTrue(case.answer)
        runner.submit_training_response("ANSWER: 0")
        with self.assertRaises(ValueError):
            runner.operator_pending_train_case(public)

    def test_partial_or_completed_runner_cannot_restage_catalog(self) -> None:
        runner = _runner()
        runner.next_training_prompt()
        with self.assertRaisesRegex(ValueError, "fresh"):
            collect_minotaur_hf_roster(runner, _config(), lambda view: "ANSWER: 0")
        self.assertEqual(runner.observations, ())
        runner.submit_training_response("ANSWER: 0")
        with self.assertRaisesRegex(ValueError, "fresh"):
            collect_minotaur_hf_roster(runner, _config(), lambda view: "ANSWER: 0")

    def test_callback_mutation_refused_before_scored_training_observation(self) -> None:
        runner = _runner()

        def mutating_learner(view):
            object.__setattr__(view, "prompt", "a substituted TRAIN prompt")
            return "ANSWER: 0"

        with self.assertRaisesRegex(RuntimeError, "mutated TRAIN material"):
            collect_minotaur_hf_roster(runner, _config(), mutating_learner)
        self.assertEqual(runner.observations, ())

    def test_trusted_callback_can_hold_runner_but_private_case_mutation_is_refused(self) -> None:
        runner = _runner()

        def mutating_learner(view):
            # Same-process callbacks enclosing the runner are trusted; this
            # is not an untrusted-learner security boundary.
            private_case = runner.operator_pending_train_case(view)
            object.__setattr__(private_case, "tier", 2)
            return "ANSWER: 0"

        with self.assertRaisesRegex(RuntimeError, "mutated TRAIN material"):
            collect_minotaur_hf_roster(runner, _config(), mutating_learner)
        self.assertEqual(runner.observations, ())

    def test_retained_previous_case_gold_mutation_refused_before_second_score(self) -> None:
        runner = _runner()
        prior_case = []

        def mutating_prior_gold(view):
            if not prior_case:
                prior_case.append(runner.operator_pending_train_case(view))
            else:
                object.__setattr__(prior_case[0], "answer", "99999")
            return "ANSWER: 0"

        with self.assertRaisesRegex(RuntimeError, "historical TRAIN material"):
            collect_minotaur_hf_roster(runner, _config(), mutating_prior_gold)
        self.assertEqual(len(runner.observations), 1)
        self.assertFalse(runner.is_complete)

    def test_replaced_in_flight_scoring_gold_refused_before_observation(self) -> None:
        runner = _runner()

        def swapping_learner(view):
            case = runner.operator_pending_train_case(view)
            runner._in_flight = replace(
                runner._in_flight,
                challenge=replace(case, answer="99999"),
            )
            return toy_arithmetic_learner(view)

        with self.assertRaisesRegex(RuntimeError, "pending TRAIN scoring material"):
            collect_minotaur_hf_roster(runner, _config(), swapping_learner)
        self.assertEqual(runner.observations, ())

    def test_returned_roster_detached_from_live_runner_case_references(self) -> None:
        runner = _runner()
        original_cases = []

        def retaining_learner(view):
            original_cases.append(runner.operator_pending_train_case(view))
            return "ANSWER: 0"

        roster = collect_minotaur_hf_roster(runner, _config(), retaining_learner)
        stable_case = replace(roster.train_cases[0])
        stable_visits = roster.ordered_optimizer_exposures
        self.assertIsNot(original_cases[0], roster.train_cases[0])
        object.__setattr__(original_cases[0], "answer", "99999")
        object.__setattr__(original_cases[0], "prompt", "corrupted after staging")
        object.__setattr__(original_cases[0], "metadata", (("corrupted", "yes"),))
        self.assertEqual(roster.train_cases[0], stable_case)
        self.assertEqual(roster.ordered_optimizer_exposures, stable_visits)

    def test_duplicate_provider_train_case_refused_before_complete_roster(self) -> None:
        runner = _runner()
        original = runner_source._arithmetic_challenge
        first_case = []

        def duplicate_after_first(*args, **kwargs):
            generated = original(*args, **kwargs)
            if not first_case:
                first_case.append(generated)
                return generated
            return first_case[0]

        with patch.object(runner_source, "_arithmetic_challenge",
                          side_effect=duplicate_after_first):
            with self.assertRaisesRegex(ValueError, "reused/sealed evidence"):
                collect_minotaur_hf_roster(
                    runner, _config(), lambda view: "ANSWER: 0",
                )
        self.assertEqual(len(runner.observations), 1)
        self.assertFalse(runner.is_complete)

    def test_real_hf_optimizer_seam_consumes_exact_staged_visits(self) -> None:
        roster = collect_minotaur_hf_roster(
            _runner(), _config(), lambda view: "ANSWER: 0",
        )
        training = list(roster.train_cases)
        labels = [case.answer for case in training]
        fake_torch = _Torch()
        with patch("labyrinth.hf_pilot._encode_pair", return_value=([1], [1])):
            candidate = _train_steps(
                _Model(), object(), fake_torch, training, labels, _config(),
                expected_visit_exposures=roster.ordered_optimizer_exposures,
            )
            sham = _train_steps(
                _Model(), object(), fake_torch, training, labels, _config(),
                learning_rate=0.0,
                expected_visit_exposures=roster.ordered_optimizer_exposures,
            )
        self.assertEqual(tuple(candidate["visited_exposures"]),
                         roster.ordered_optimizer_exposures)
        self.assertEqual(candidate["visited_exposures"], sham["visited_exposures"])
        self.assertEqual(candidate["steps"], 5)

    def test_optimizer_refuses_reordered_or_changed_train_material(self) -> None:
        roster = collect_minotaur_hf_roster(
            _runner(), _config(), lambda view: "ANSWER: 0",
        )
        original = list(roster.train_cases)
        for catalog in (
            original[::-1],
            [replace(original[0], prompt="foreign TRAIN prompt")] + original[1:],
        ):
            with self.subTest(catalog=catalog), patch(
                "labyrinth.hf_pilot._encode_pair", return_value=([1], [1]),
            ), self.assertRaisesRegex(ValueError, "schedule differs"):
                _train_steps(
                    _Model(), object(), _Torch(), catalog,
                    [case.answer for case in catalog], _config(),
                    expected_visit_exposures=roster.ordered_optimizer_exposures,
                )


if __name__ == "__main__":
    unittest.main()
