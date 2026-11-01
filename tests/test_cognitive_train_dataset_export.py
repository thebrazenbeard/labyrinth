"""Contract and adversarial tests for operator-only labeled cognitive TRAIN data."""

from dataclasses import replace
import hashlib
import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from labyrinth.approved_cognitive_empathy_train_r90 import (
    PROVIDER_ID, prepare_cognitive_empathy_train_rooms,
)
from labyrinth.cognitive_objective_training_run import CognitiveObjectiveTrainingRun
from labyrinth.journey import AWAITING_EXAM, begin_journey, complete_maze
from labyrinth import cognitive_train_dataset_export
from tests.test_cognitive_objective_training_run import (
    cognitive_contract, development_fail_for_reroll,
)


def export_cognitive_train_jsonl(*args, **kwargs):
    return cognitive_train_dataset_export.export_cognitive_train_jsonl(*args, **kwargs)


class CognitiveTrainDatasetExportTests(TestCase):
    def test_fifteen_exact_train_only_supervised_rows_and_canonical_digest(self):
        self.assertTrue(hasattr(cognitive_train_dataset_export, "export_cognitive_train_jsonl"))
        contract = cognitive_contract(seed=1101)
        journey = begin_journey(contract.scope, checkpoint_sha256="1" * 64)
        result = export_cognitive_train_jsonl(contract, journey)
        lines = result.jsonl_bytes.splitlines()
        self.assertEqual(len(lines), 15)
        self.assertEqual(result.case_count, 15)
        self.assertEqual(result.provider_id, PROVIDER_ID)
        self.assertEqual(result.cycle_index, 0)
        self.assertEqual(result.objective_id, contract.scope.objective_id)
        self.assertEqual(result.contract_fingerprint, contract.fingerprint)
        self.assertEqual(result.sha256, hashlib.sha256(result.jsonl_bytes).hexdigest())
        self.assertTrue(result.jsonl_bytes.endswith(b"\n"))
        rows = [json.loads(line) for line in lines]
        expected = prepare_cognitive_empathy_train_rooms(contract, journey.maze)
        self.assertEqual(len({row["case_id"] for row in rows}), 15)
        self.assertEqual(len({row["semantic_sha256"] for row in rows}), 15)
        for row, source in zip(rows, expected):
            self.assertEqual(row["schema"], "labyrinth.cognitive-supervised-train-example.v1")
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["prompt"], source.trainee_view.prompt)
            self.assertEqual(row["target_choice"], source.prepared.challenge.answer)
            self.assertEqual(row["prompt_sha256"], source.trainee_view.prompt_sha256)
            self.assertEqual(row["semantic_sha256"], source.semantic_sha256)
            self.assertEqual(row["objective_contract_sha256"], contract.fingerprint)
            self.assertEqual(row["cycle_index"], 0)
            self.assertEqual(source.score_train_choice(row["target_choice"]), 100)
            self.assertEqual(set(row), {
                "schema", "split", "provider_id", "training_rubric_id",
                "objective_id", "objective_contract_sha256", "cycle_index",
                "room_id", "case_id", "prompt", "prompt_sha256",
                "semantic_sha256", "source_binding_sha256", "target_choice",
            })
        self.assertFalse(result.authorizes_host_execution)
        self.assertFalse(result.authorizes_training)
        self.assertFalse(result.authorizes_exam)
        self.assertFalse(result.authorizes_graduation)
        self.assertFalse(result.independent_exam_evidence)

    def test_export_is_deterministic_immutable_and_has_no_disk_side_effects(self):
        contract = cognitive_contract(seed=1102)
        journey = begin_journey(contract.scope, checkpoint_sha256="2" * 64)
        with tempfile.TemporaryDirectory() as path:
            before = sorted(Path(path).iterdir())
            first = export_cognitive_train_jsonl(contract, journey)
            second = export_cognitive_train_jsonl(contract, journey)
            self.assertEqual(first, second)
            self.assertEqual(sorted(Path(path).iterdir()), before)
        with self.assertRaises((AttributeError, TypeError)):
            first.jsonl_bytes = b"forged"
        with self.assertRaises((AttributeError, TypeError)):
            object.__setattr__(first, "sha256", "f" * 64)
        self.assertEqual(first.sha256, hashlib.sha256(first.jsonl_bytes).hexdigest())

    def test_different_cycles_do_not_reexport_prior_semantic_cases(self):
        contract = cognitive_contract(seed=1103)
        first = begin_journey(contract.scope, checkpoint_sha256="3" * 64)
        exported0 = export_cognitive_train_jsonl(contract, first)
        train0 = CognitiveObjectiveTrainingRun(contract, first)
        for _ in range(15):
            train0.run_training_chamber(lambda view: "A")
        after = complete_maze(first, train0.self_reported_completion_receipt(
            checkpoint_after_sha256="4" * 64,
        ))
        second = development_fail_for_reroll(after)
        exported1 = export_cognitive_train_jsonl(contract, second)
        seen0 = {json.loads(line)["semantic_sha256"] for line in exported0.jsonl_bytes.splitlines()}
        seen1 = {json.loads(line)["semantic_sha256"] for line in exported1.jsonl_bytes.splitlines()}
        self.assertEqual((exported0.case_count, exported1.case_count), (15, 15))
        self.assertFalse(seen0.intersection(seen1))
        self.assertEqual(exported1.cycle_index, 1)

    def test_operator_semantic_exclusion_and_alias_validation_are_fail_closed(self):
        contract = cognitive_contract(seed=1104)
        first = begin_journey(contract.scope, checkpoint_sha256="5" * 64)
        reserved = prepare_cognitive_empathy_train_rooms(contract, first.maze)[0].semantic_sha256
        for excluded in ((reserved,), (reserved, reserved), ("invalid",), [reserved]):
            with self.subTest(excluded=excluded), self.assertRaises(ValueError):
                export_cognitive_train_jsonl(
                    contract, first, forbidden_semantic_sha256=excluded,
                )

    def test_rejects_unapproved_scope_forged_journey_and_nontrain_phase(self):
        contract = cognitive_contract(seed=1105)
        first = begin_journey(contract.scope, checkpoint_sha256="6" * 64)
        with self.assertRaises(ValueError):
            export_cognitive_train_jsonl(contract, replace(first, phase=AWAITING_EXAM))
        with self.assertRaises(ValueError):
            export_cognitive_train_jsonl(contract, replace(first, checkpoint_sha256="a" * 64))
        other = cognitive_contract(seed=1105, domain="reasoning")
        with self.assertRaises(ValueError):
            export_cognitive_train_jsonl(other, begin_journey(other.scope, checkpoint_sha256="7" * 64))
        other = cognitive_contract(seed=1105, skill="reactive_empathy")
        with self.assertRaises(ValueError):
            export_cognitive_train_jsonl(other, begin_journey(other.scope, checkpoint_sha256="7" * 64))
        with self.assertRaises(ValueError):
            export_cognitive_train_jsonl(contract, first, provider_id="unknown")

    def test_forged_cognitive_source_answer_does_not_export_labeled_gold(self):
        contract = cognitive_contract(seed=1106)
        journey = begin_journey(contract.scope, checkpoint_sha256="8" * 64)
        genuine = prepare_cognitive_empathy_train_rooms(contract, journey.maze)
        first = genuine[0]
        incorrect = next(x for x in "ABC" if x != first.prepared.challenge.answer)
        changed_challenge = replace(first.prepared.challenge, answer=incorrect)
        forged = replace(first, prepared=replace(first.prepared, challenge=changed_challenge))
        with patch.object(CognitiveObjectiveTrainingRun, "_regenerated_roster", return_value=(forged,) + genuine[1:]):
            with self.assertRaises(ValueError):
                export_cognitive_train_jsonl(contract, journey)

    def test_mutating_caller_contract_during_export_cannot_change_dataset_provenance(self):
        contract = cognitive_contract(seed=1107)
        objective_id = contract.scope.objective_id
        fingerprint = contract.fingerprint
        journey = begin_journey(contract.scope, checkpoint_sha256="9" * 64)
        original = CognitiveObjectiveTrainingRun._regenerated_roster

        def mutate_after_verified_snapshot(run):
            rooms = original(run)
            object.__setattr__(contract.scope, "objective_id", "caller-forged-later")
            return rooms

        with patch.object(CognitiveObjectiveTrainingRun, "_regenerated_roster",
                          autospec=True, side_effect=mutate_after_verified_snapshot):
            exported = export_cognitive_train_jsonl(contract, journey)
        first_row = json.loads(exported.jsonl_bytes.splitlines()[0])
        self.assertEqual(exported.objective_id, objective_id)
        self.assertEqual(exported.contract_fingerprint, fingerprint)
        self.assertEqual(first_row["objective_id"], objective_id)
        self.assertEqual(first_row["objective_contract_sha256"], fingerprint)
