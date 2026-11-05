"""Versioned local successor source preflight; no neural run or graduation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

from labyrinth.hf_development_only_successor_v1 import (
    prepare_local_successor_source, verify_local_successor_source,
    DevelopmentOnlyNeuralExecutionBlocked,
)
from labyrinth.hf_resume_bridge import ResumeContext
from labyrinth.journey import begin_journey, complete_maze, face_minotaur
from labyrinth.journey_policy_runner import TrainingCorridorRunner
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.minotaur_hf_roster_bridge import collect_minotaur_hf_roster
from labyrinth.regenerative_maze import generate_maze_plan
from tests.test_minotaur_hf_roster_bridge import _config
from tests.test_journey_policy_runner import scope, toy_arithmetic_learner


START = "0" * 64
AFTER = "1" * 64


def _successor_fixture():
    selected = scope(scale=3)
    journey = begin_journey(selected, checkpoint_sha256=START)
    runner = TrainingCorridorRunner.begin(selected, journey.maze, checkpoint_sha256=START)
    prior_witnesses = []
    for index in range(selected.total_chambers):
        view = runner.next_training_prompt()
        prior_witnesses.append((view.challenge_id, view.prompt))
        runner.submit_training_response(
            toy_arithmetic_learner(view),
            checkpoint_after_sha256=AFTER if index == selected.total_chambers - 1 else None,
        )
    journey = complete_maze(journey, runner.self_reported_completion_receipt())
    cases = tuple(
        ExamCaseResult(
            case_id=f"dev-{part}-{index}", skill="arithmetic", split=part,
            score=0, baseline_score=0,
            prompt_sha256=hashlib.sha256(f"dev-{part}-{index}".encode()).hexdigest(),
        )
        for part in ("heldout", "transfer") for index in range(4)
    )
    journey = face_minotaur(journey, MinotaurExamReceipt(
        exam_id="local-development-failure", objective_id=selected.objective_id,
        cycle_index=0, checkpoint_sha256=AFTER, baseline_sha256=START,
        evaluator_id="local-development-only", cases=cases,
    ))
    successor_runner = runner.regenerate(journey)
    config = replace(_config(), train_count=selected.total_chambers)
    roster = collect_minotaur_hf_roster(
        successor_runner, config, toy_arithmetic_learner,
    )
    predecessor_report_bytes = json.dumps({
        "schema": "labyrinth.hf-lora-adaptive-source.v1",
        "candidate_lora_parameter_sha256": AFTER,
        "candidate_lora_tensor_value_sha256": AFTER,
        "training_case_ids": list(journey.attempts[0].training_case_ids),
        "training_prompt_sha256": list(journey.attempts[0].training_prompt_sha256),
        "scope_fingerprint": selected.fingerprint,
        "cycle_index": 0,
    }, sort_keys=True).encode()
    context = ResumeContext(
        scope_fingerprint=selected.fingerprint, cycle_index=1,
        predecessor_report_sha256=hashlib.sha256(predecessor_report_bytes).hexdigest(),
        predecessor_lora_value_sha256=AFTER,
        authority_ref="local-dev-no-review",
    )
    return (config, selected, journey, successor_runner, roster, context,
            predecessor_report_bytes, tuple(prior_witnesses))


class LocalAdaptiveSuccessorSourceTests(unittest.TestCase):
    def test_prepared_cycle_one_source_preserves_train_history_without_gold_or_exam_content(self):
        config, selected, journey, runner, roster, context, previous, prior = _successor_fixture()
        manifest = prepare_local_successor_source(
            config, selected, journey, runner, roster, context, previous, prior,
        )
        self.assertEqual(manifest["mode"], "DEVELOPMENT_ONLY")
        self.assertEqual(manifest["neural_execution"], "BLOCKED_PENDING_ARTIFACT_REPLAY_BRIDGE")
        self.assertEqual(manifest["cycle_index"], 1)
        self.assertEqual(len(manifest["training_rooms"]), config.train_count)
        dumped = json.dumps(manifest, sort_keys=True)
        for word in ('"answer":', '"gold":', '"private_exam"', '"graduated": true'):
            self.assertNotIn(word, dumped)
        self.assertEqual(verify_local_successor_source(
            config, selected, journey, runner, roster, context, previous, manifest,
        ), "SOURCE_PRECHECK_ONLY")

    def test_rejects_checkpoint_and_predecessor_identity_substitution(self):
        config, selected, journey, runner, roster, context, previous, prior = _successor_fixture()
        with self.assertRaisesRegex(ValueError, "predecessor report digest"):
            prepare_local_successor_source(config, selected, journey, runner,
                                           roster, context, previous + b" ", prior)
        with self.assertRaisesRegex(ValueError, "retained neural checkpoint"):
            prepare_local_successor_source(
                config, selected, journey, runner, roster,
                replace(context, predecessor_lora_value_sha256="f" * 64), previous, prior,
            )

    def test_replay_rejects_feedback_case_and_semantic_forgery_even_when_rehashed(self):
        from labyrinth.hf_development_only_successor_v1 import _fingerprint
        config, selected, journey, runner, roster, context, previous, prior = _successor_fixture()
        original = prepare_local_successor_source(
            config, selected, journey, runner, roster, context, previous, prior,
        )
        for corrupt in (
            lambda m: m["training_rooms"][0].__setitem__("score", 1 - m["training_rooms"][0]["score"]),
            lambda m: m["training_rooms"][0].__setitem__("case_id", "train-taskmaster-forged"),
            lambda m: m["training_rooms"][0].__setitem__("semantic_atoms", [["+", 1, 1]]),
            lambda m: m["sealed_development_exams"].append("leaked-final-exam"),
            lambda m: m["historical_training_prompts"][0].__setitem__("prompt", "Calculate 2 + 2."),
            lambda m: m["historical_train_semantic_atoms"][0].__setitem__(0, ["+", 1, 1]),
            lambda m: m.__setitem__("operator_policy_sha256", "f" * 64),
        ):
            candidate = json.loads(json.dumps(original))
            corrupt(candidate)
            candidate["source_sha256"] = _fingerprint(candidate)
            with self.subTest(candidate=candidate["source_sha256"]), self.assertRaises(ValueError):
                verify_local_successor_source(
                    config, selected, journey, runner, roster, context, previous, candidate,
                )

    def test_source_preflight_does_not_grant_neural_execution(self):
        from labyrinth.hf_development_only_successor_v1 import run_local_neural_successor
        config, selected, journey, runner, roster, context, previous, prior = _successor_fixture()
        source = prepare_local_successor_source(
            config, selected, journey, runner, roster, context, previous, prior,
        )
        with self.assertRaises(DevelopmentOnlyNeuralExecutionBlocked):
            run_local_neural_successor(source)


if __name__ == "__main__":
    unittest.main()
