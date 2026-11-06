"""Cross-contract source-only checks for reconciled R85 and adaptive TRAIN.

The staged roster and feedback replay share deterministic novelty redraws, but
neither source path authenticates a host, neural update, or examination result.
"""

from __future__ import annotations

from dataclasses import replace
import json
import unittest

from labyrinth.hf_adaptive_source_v1 import (
    _source_hash, arithmetic_semantic_atoms, build_adaptive_manifest,
    reconstruct_adaptive_source,
)
from labyrinth.hf_pilot import PilotConfig
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.minotaur_hf_roster_bridge import (
    collect_minotaur_hf_roster, replay_staged_roster, snapshot_staged_roster,
)
from labyrinth.minotaur_model import RoomPlan
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.taskmaster_routing import _arithmetic_challenge
from labyrinth.training_scope import TrainingScopeSpec
from tests.test_journey_policy_runner import toy_arithmetic_learner


class ReconciledAdaptiveRedrawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = TrainingScopeSpec(
            objective_id="r93-feedback-divergence", domain="reasoning",
            skills=("arithmetic",), specificity=5, scale=5,
            max_cycles=2, seed=321,
        )
        self.config = PilotConfig(
            "unused-model", "unused-output", seed=31, train_count=15,
            eval_count=15, steps=15, tiers=(1,),
        )

    def _stage(self, learner):
        runner = TrainingCorridorRunner.begin(
            self.scope, generate_maze_plan(self.scope, 0),
            checkpoint_sha256="0" * 64,
        )
        roster = collect_minotaur_hf_roster(runner, self.config, learner)
        self.assertTrue(runner.is_complete)
        self.assertEqual(len(roster.train_cases), 15)
        return runner, roster

    def test_fifteen_room_success_and_weak_routes_roundtrip_both_source_modes(self) -> None:
        for learner in (toy_arithmetic_learner, lambda _view: "ANSWER: invalid"):
            with self.subTest(learner=learner.__name__):
                runner, roster = self._stage(learner)
                manifest = build_adaptive_manifest(self.config, self.scope, roster)
                adaptive_train, heldout, visits = reconstruct_adaptive_source(
                    self.config, manifest,
                )
                staged_train, staged_public = snapshot_staged_roster(
                    roster, self.config, self.scope,
                )
                replay_scope, replay_train, staged_visits = replay_staged_roster(
                    staged_public, self.config,
                )
                self.assertEqual(adaptive_train, list(roster.train_cases))
                self.assertEqual(adaptive_train, staged_train)
                self.assertEqual(replay_scope, self.scope)
                self.assertEqual(
                    [replace(case, metadata=()) for case in adaptive_train],
                    replay_train,
                )
                self.assertEqual(visits, staged_visits)
                self.assertEqual(visits, roster.ordered_optimizer_exposures)
                self.assertEqual(
                    [row[3] for row in roster.adaptive_feedback_rows],
                    [int(observation.correct) for observation in runner.observations],
                )
                self.assertEqual(len({case.prompt for case in adaptive_train}), 15)
                source_atoms = set().union(
                    *(arithmetic_semantic_atoms(case) for case in adaptive_train)
                )
                heldout_atoms = set().union(
                    *(arithmetic_semantic_atoms(case) for case in heldout)
                )
                self.assertFalse(source_atoms & heldout_atoms)
                self.assertNotIn('"answer"', json.dumps(manifest))

    def test_weak_corridor_redraw_is_replayed_not_replaced_by_first_candidate(self) -> None:
        runner, roster = self._stage(lambda _view: "ANSWER: invalid")
        different_from_first_attempt = []
        for event, case in zip(runner.state.history, roster.train_cases, strict=True):
            room = event.assignment.room
            metadata = dict(case.metadata)
            selection = RoomPlan(
                event.assignment.room_id, int(metadata["difficulty"]),
                metadata["tactic"], metadata["planner_source"], None,
            )
            first = _arithmetic_challenge(
                room, self.scope, selection, redraw_attempt=0,
            )
            if first.id != case.id:
                different_from_first_attempt.append(case.id)
        self.assertTrue(different_from_first_attempt)
        manifest = build_adaptive_manifest(self.config, self.scope, roster)
        replay_train, _heldout, _visits = reconstruct_adaptive_source(
            self.config, manifest,
        )
        self.assertEqual(replay_train, list(roster.train_cases))

        forged = json.loads(json.dumps(manifest))
        changed = next(
            index for index, row in enumerate(forged["train_rooms"])
            if row["case_id"] == different_from_first_attempt[0]
        )
        forged["train_rooms"][changed]["case_id"] = "train-taskmaster-" + "f" * 32 + "-" + "f" * 12
        forged["source_sha256"] = _source_hash({
            key: value for key, value in forged.items() if key != "source_sha256"
        })
        with self.assertRaises(ValueError):
            reconstruct_adaptive_source(self.config, forged)

    def test_recomputed_hash_cannot_hide_nonterminal_feedback_route_forgery(self) -> None:
        for learner in (toy_arithmetic_learner, lambda _view: "ANSWER: invalid"):
            with self.subTest(learner=learner.__name__):
                _runner, roster = self._stage(learner)
                forged = build_adaptive_manifest(self.config, self.scope, roster)
                forged["adaptive_feedback_rows"][0][3] = 1 - forged["adaptive_feedback_rows"][0][3]
                forged["source_sha256"] = _source_hash({
                    key: value for key, value in forged.items() if key != "source_sha256"
                })
                with self.assertRaises(ValueError):
                    reconstruct_adaptive_source(self.config, forged)


if __name__ == "__main__":
    unittest.main()
