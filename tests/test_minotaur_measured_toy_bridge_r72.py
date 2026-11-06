"""Bounded actual-parameter Minotaur corridor integration, never graduation."""

from __future__ import annotations

import unittest
from unittest.mock import PropertyMock, patch

from labyrinth.minotaur_measured_toy_bridge_r72 import run_measured_toy_corridor
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.regenerative_maze import generate_maze_plan
from labyrinth.toy_backend import ToyLinearAdapter
from labyrinth.training_scope import TrainingScopeSpec


def scope() -> TrainingScopeSpec:
    return TrainingScopeSpec(
        objective_id="measured-toy-taskmaster", domain="reasoning",
        skills=("arithmetic",), specificity=1, scale=2, max_cycles=1, seed=321,
    )


class MeasuredToyBridgeTests(unittest.TestCase):
    def test_real_adapter_parameter_update_is_bundled_with_observed_train_corridor(self) -> None:
        declared = scope()
        learner = ToyLinearAdapter(seed=17)
        initial = learner.model_hash
        report = run_measured_toy_corridor(
            declared, generate_maze_plan(declared, 0), learner,
            epochs=8, learning_rate=0.5,
        )
        self.assertNotEqual(learner.model_hash, initial)
        self.assertEqual(report["checkpoint_before_sha256"], initial)
        self.assertEqual(report["checkpoint_after_sha256"], learner.model_hash)
        self.assertGreater(report["parameter_updates"], 0)
        self.assertEqual(len(report["observations"]), declared.total_chambers)
        self.assertEqual(len(report["training_case_ids"]), declared.total_chambers)
        self.assertEqual([event["tactic"] for event in report["observations"]],
                         ["practice", "practice"])
        self.assertTrue(all(type(event["difficulty"]) is int
                            for event in report["observations"]))
        self.assertNotEqual(report["observations"][0]["policy_before_sha256"],
                            report["observations"][0]["policy_after_sha256"])
        self.assertTrue(all(event["checkpoint_after_sha256"] == initial
                            for event in report["observations"]))
        event = report["optimizer_event"]
        self.assertEqual(event["phase"], "after_train_corridor")
        self.assertEqual(event["checkpoint_before_sha256"], initial)
        self.assertEqual(event["checkpoint_after_sha256"], learner.model_hash)
        self.assertEqual(event["parameter_updates"], report["parameter_updates"])
        transitions = report["registry_transitions"]
        self.assertEqual(len(transitions), declared.total_chambers - 1)
        self.assertEqual(transitions[0]["transition_kind"],
                         "train_registry_admission_nonlearning")
        self.assertIs(transitions[0]["train_observation"], False)
        self.assertEqual(transitions[0]["before_policy_sha256"],
                         report["observations"][0]["policy_after_sha256"])
        self.assertEqual(transitions[0]["after_policy_sha256"],
                         report["observations"][1]["policy_before_sha256"])
        self.assertEqual(transitions[0]["added_case_ids"],
                         [report["observations"][1]["case_id"]])
        self.assertEqual(report["source_status"], "MEASURED_TOY_PARAMETER_UPDATE_ONLY")
        self.assertIs(report["graduation_authorized"], False)
        self.assertIs(report["pretrained_improvement_established"], False)
        self.assertNotIn("answer", str(report).lower())

    def test_forged_train_report_without_actual_parameter_change_is_rejected(self) -> None:
        declared = scope()
        learner = ToyLinearAdapter(seed=17)
        initial = learner.model_hash
        fake = {"before_model_hash": initial, "after_model_hash": "f" * 64,
                "parameter_updates": 1000, "case_visits": 1000}
        with patch.object(ToyLinearAdapter, "train", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "measured parameter update"):
                run_measured_toy_corridor(
                    declared, generate_maze_plan(declared, 0), learner,
                    epochs=8, learning_rate=0.5,
                )
        self.assertEqual(learner.model_hash, initial)

    def test_zero_learning_rate_refused_before_any_train_room(self) -> None:
        declared = scope()
        learner = ToyLinearAdapter(seed=17)
        initial = learner.model_hash
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            run_measured_toy_corridor(
                declared, generate_maze_plan(declared, 0), learner,
                epochs=8, learning_rate=0.0,
            )
        self.assertEqual(learner.model_hash, initial)

    def test_all_scored_responses_precede_measured_optimizer_update(self) -> None:
        declared = scope()
        learner = ToyLinearAdapter(seed=17)
        initial = learner.model_hash
        scored_at = []
        original = TrainingCorridorRunner.submit_training_response

        def record(runner, response, **kwargs):
            observed = original(runner, response, **kwargs)
            scored_at.append((learner.model_hash, observed.checkpoint_after_sha256))
            return observed

        with patch.object(TrainingCorridorRunner, "submit_training_response",
                          autospec=True, side_effect=record):
            report = run_measured_toy_corridor(
                declared, generate_maze_plan(declared, 0), learner,
                epochs=8, learning_rate=0.5,
            )
        self.assertEqual(scored_at, [(initial, initial)] * declared.total_chambers)
        self.assertNotEqual(report["optimizer_event"]["checkpoint_after_sha256"], initial)

    def test_missing_registry_transition_blocks_toy_update(self) -> None:
        declared = scope()
        learner = ToyLinearAdapter(seed=17)
        initial = learner.model_hash
        with patch.object(TrainingCorridorRunner, "registry_transitions",
                          new_callable=PropertyMock, return_value=()):
            with self.assertRaisesRegex(RuntimeError, "missing non-learning registry"):
                run_measured_toy_corridor(
                    declared, generate_maze_plan(declared, 0), learner,
                    epochs=8, learning_rate=0.5,
                )
        self.assertEqual(learner.model_hash, initial)


if __name__ == "__main__":
    unittest.main()
