"""Executable objective declaration -> toy learning -> Minotaur -> regeneration."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.objective_contract import ObjectiveContract, SkillTarget
from labyrinth.objective_learning_loop_r92 import (
    SCHEMA, arithmetic_objective, run_objective_learning_loop,
)
from labyrinth.toy_backend import ToyLinearAdapter


class ObjectiveLearningLoopTests(unittest.TestCase):
    def test_real_two_cycle_confront_regenerate_and_persist(self) -> None:
        objective = arithmetic_objective()
        with tempfile.TemporaryDirectory() as directory:
            result = run_objective_learning_loop(objective, checkpoint_dir=directory)
            self.assertEqual(result["schema"], SCHEMA)
            self.assertEqual(result["objective_contract_sha256"], objective.fingerprint)
            self.assertEqual(result["scope_fingerprint"], objective.scope.fingerprint)
            self.assertEqual(result["qualification"], "NOT_ESTABLISHED")
            self.assertIs(result["pretrained_improvement_established"], False)
            self.assertIs(result["graduation_authorized"], False)
            self.assertEqual(result["phase"], "development_passed_pending_private_qualification")
            first, second = result["cycles"]
            self.assertEqual([c["minotaur"]["status"] for c in result["cycles"]],
                             ["FAIL", "PASS"])
            self.assertEqual(first["minotaur"]["weak_skills"], ["arithmetic"])
            self.assertEqual(second["regenerated_from_weak_skills"], ["arithmetic"])
            self.assertNotEqual(first["training_room_ids"], second["training_room_ids"])
            self.assertNotEqual(first["checkpoint_before_sha256"], first["checkpoint_after_sha256"])
            self.assertEqual(second["checkpoint_before_sha256"], first["checkpoint_after_sha256"])
            self.assertEqual(result["checkpoint_sha256"], second["checkpoint_after_sha256"])
            self.assertGreater(second["development_post_correct"], second["development_pre_correct"])
            self.assertGreater(second["development_post_correct"], second["sham_correct"])
            self.assertEqual(first["sham_checkpoint_sha256"], result["baseline_checkpoint_sha256"])
            self.assertEqual(second["sham_checkpoint_sha256"], result["baseline_checkpoint_sha256"])
            for checkpoint in result["checkpoint_files"]:
                self.assertEqual(
                    ToyLinearAdapter.load_checkpoint(checkpoint["path"]).model_hash,
                    checkpoint["sha256"],
                )
            self.assertEqual(len(result["checkpoint_files"]), 3)
            train_ids = [case for cycle in result["cycles"] for case in cycle["training_case_ids"]]
            exam_ids = [case for cycle in result["cycles"] for case in cycle["development_case_ids"]]
            train_prompts = [sha for cycle in result["cycles"] for sha in cycle["training_prompt_sha256"]]
            exam_prompts = [sha for cycle in result["cycles"] for sha in cycle["development_prompt_sha256"]]
            self.assertEqual(len(train_ids), len(set(train_ids)))
            self.assertEqual(len(exam_ids), len(set(exam_ids)))
            self.assertTrue(set(train_ids).isdisjoint(exam_ids))
            self.assertTrue(set(train_prompts).isdisjoint(exam_prompts))
            self.assertNotIn("ANSWER:", json.dumps(result))

    def test_sham_is_real_zero_update_call_and_train_only_supervision(self) -> None:
        training_calls = []
        original = ToyLinearAdapter.train

        def capture(model, cases, **kwargs):
            training_calls.append((tuple(c.split for c in cases), kwargs.copy()))
            return original(model, cases, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ToyLinearAdapter, "train", autospec=True, side_effect=capture):
                result = run_objective_learning_loop(arithmetic_objective(),
                                                     checkpoint_dir=directory)
        self.assertEqual(len(training_calls), 4)
        self.assertTrue(all(set(splits) == {"train"} for splits, _ in training_calls))
        self.assertEqual([args["update_enabled"] for _, args in training_calls],
                         [True, False, True, False])
        self.assertTrue(all(c["optimizer"]["parameter_updates"] > 0
                            and c["sham_parameter_updates"] == 0 for c in result["cycles"]))

    def test_forged_optimizer_receipt_with_unchanged_weights_rejected(self) -> None:
        baseline = ToyLinearAdapter(seed=0).model_hash
        forged = {"before_model_hash": baseline, "after_model_hash": "f" * 64,
                  "parameter_updates": 100, "case_visits": 100}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ToyLinearAdapter, "train", return_value=forged):
                with self.assertRaisesRegex(RuntimeError, "measured toy optimizer"):
                    run_objective_learning_loop(arithmetic_objective(),
                                                checkpoint_dir=directory)

    def test_unsupported_audit_objective_cannot_be_claimed_learnable_by_toy(self) -> None:
        original = arithmetic_objective()
        target = SkillTarget(
            skill="arithmetic", focus_tags=("audit_validity_before_priority",),
            observable_behaviors=("Exclude INVALID reports before source priority.",),
            rubric_id="source_priority_audit_filter_v1",
            minimum_score_percent=75, minimum_gain_percent=10,
        )
        mislabeled = ObjectiveContract(
            original.scope, original.intention, (target,), original.excluded_scopes,
            original.forbidden_claims,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "approved arithmetic objective"):
                run_objective_learning_loop(mislabeled, checkpoint_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_writes_recoverable_checkpoint_files_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-m", "labyrinth.objective_learning_loop_r92",
                 "--checkpoint-dir", directory], capture_output=True, text=True, check=True,
            )
            receipt = json.loads(completed.stdout)
            self.assertEqual(receipt["schema"], SCHEMA)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 3)
            self.assertEqual(receipt["cycles"][1]["checkpoint_before_sha256"],
                             receipt["cycles"][0]["checkpoint_after_sha256"])


if __name__ == "__main__":
    unittest.main()
