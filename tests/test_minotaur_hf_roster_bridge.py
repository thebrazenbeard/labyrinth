"""TRAIN-only source bridge to the real HF optimizer exposure interface."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch

from labyrinth.hf_pilot import PilotConfig, _train_steps, _training_visit_exposures
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.minotaur_hf_roster_bridge import (
    MinotaurHfTrainRoster, collect_minotaur_hf_roster,
)
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.training_scope import TrainingScopeSpec
from tests.test_hf_fullstate_hook_r7 import _Model, _Torch


def _scope() -> TrainingScopeSpec:
    return TrainingScopeSpec(
        objective_id="hf-adaptive-arithmetic", domain="reasoning",
        skills=("arithmetic",), scale=3, specificity=1, max_cycles=2,
        seed=321,
    )


def _runner() -> TrainingCorridorRunner:
    scope = _scope()
    return TrainingCorridorRunner.begin(
        scope, generate_maze_plan(scope, 0), checkpoint_sha256="0" * 64,
    )


def _config(**kwargs: object) -> PilotConfig:
    return PilotConfig("unused-model", "unused-output", seed=31,
                       train_count=3, steps=5, tiers=(1,), **kwargs)


class MinotaurHfRosterContractTests(unittest.TestCase):
    def test_real_policy_generated_training_catalog_binds_hf_exposures(self) -> None:
        views = []

        def learner(view):
            self.assertFalse(hasattr(view, "answer"))
            views.append(view)
            return "ANSWER: not-integer"  # real negative feedback changes taskmasking

        runner = _runner()
        roster = collect_minotaur_hf_roster(runner, _config(), learner)
        self.assertTrue(runner.is_complete)
        self.assertEqual(len(roster.train_cases), 3)
        self.assertEqual(len(runner.observations), 3)
        self.assertEqual([case.prompt for case in roster.train_cases],
                         [view.prompt for view in views])
        self.assertEqual(roster.ordered_optimizer_exposures,
                         _training_visit_exposures(list(roster.train_cases), seed=31, steps=5))
        self.assertEqual(tuple((case.id, hashlib.sha256(case.prompt.encode()).hexdigest())
                               for case in roster.train_cases),
                         tuple((entry.case_id, entry.prompt_sha256)
                               for entry in runner.observations))
        self.assertEqual({case.split for case in roster.train_cases}, {"train"})
        self.assertFalse(roster.physically_authenticated)
        self.assertFalse(roster.graduation_authorized)

        # These are fixed non-authorizing properties, never caller-controlled
        # dataclass fields that replace() can flip into a forged PASS.
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

    def test_preflight_rejects_wrong_tier_and_catalog_size_without_routing(self) -> None:
        for kwargs in ({"tiers": (2,)}, {"train_count": 2}, {"steps": 0}):
            runner = _runner()
            # kwargs applied by replacement to avoid duplicate kw errors.
            bad = replace(_config(), **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                collect_minotaur_hf_roster(runner, bad, lambda view: "ANSWER: 0")
            self.assertEqual(runner.observations, ())
            self.assertEqual(runner.training_registry, ())

    def test_operator_challenge_access_requires_exact_pending_public_view(self) -> None:
        runner = _runner()
        with self.assertRaises(ValueError):
            runner.operator_pending_train_case(None)
        view = runner.next_training_prompt()
        with self.assertRaises(ValueError):
            runner.operator_pending_train_case(replace(view, prompt="forged"))
        case = runner.operator_pending_train_case(view)
        self.assertEqual(case.prompt, view.prompt)
        self.assertEqual(case.split, "train")
        self.assertTrue(case.answer)
        runner.submit_training_response("ANSWER: 0")
        with self.assertRaises(ValueError):
            runner.operator_pending_train_case(view)

    def test_completed_or_partial_runner_cannot_replay_new_catalog(self) -> None:
        runner = _runner()
        first = runner.next_training_prompt()
        with self.assertRaisesRegex(ValueError, "fresh"):
            collect_minotaur_hf_roster(runner, _config(), lambda view: "ANSWER: 0")
        self.assertEqual(runner.observations, ())
        runner.submit_training_response("ANSWER: 0")
        with self.assertRaisesRegex(ValueError, "fresh"):
            collect_minotaur_hf_roster(runner, _config(), lambda view: "ANSWER: 0")

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

    def test_optimizer_preflight_refuses_reordered_or_altered_train_material(self) -> None:
        roster = collect_minotaur_hf_roster(
            _runner(), _config(), lambda view: "ANSWER: 0",
        )
        original = list(roster.train_cases)
        bad_catalogs = (
            original[::-1],
            [replace(original[0], prompt="foreign TRAIN prompt")] + original[1:],
        )
        for catalog in bad_catalogs:
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
