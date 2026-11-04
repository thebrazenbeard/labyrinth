"""Synthetic integration between authenticated arm claims and private holdout.

These tests do not run neural weights or prove separate evaluator custody.
"""

from dataclasses import replace
import hashlib
import unittest

from labyrinth.control_provenance import ArmManifest, commit_label_schedule, seal_arm_manifest
from labyrinth.evaluation_provenance_bridge import bind_and_evaluate
from labyrinth.independent_holdout import case_commitment, private_gold_seal
from tests.test_independent_holdout import _fixture


CUSTODY_KEY = b"test-bridge-control-custody-key-2026"
GOLD_KEY = b"test-bridge-independent-exam-gold-key-2026"


def setup():
    cases, exposure, arms, retention_before = _fixture()
    commitment = case_commitment(cases)
    correct = commit_label_schedule((("train-a", "5"), ("train-b", "8")), CUSTODY_KEY)
    corrupt = commit_label_schedule((("train-a", "8"), ("train-b", "5")), CUSTODY_KEY)
    empty = commit_label_schedule((), CUSTODY_KEY)
    prompt_map = dict(zip(exposure.case_ids, exposure.prompt_sha256))
    manifests = []
    observed = {}
    label_map = {"baseline": empty, "zero_update": correct,
                 "correct_labels": correct, "corrupted_labels": corrupt}
    for index, arm in enumerate(arms):
        report_digest = f"{index + 1:064x}"
        manifest = ArmManifest(
            experiment_id="test-cycle-2", objective_id="arithmetic",
            source_tree_sha256="e" * 64,
            base_model_sha256=arm.base_sha256,
            initial_adapter_sha256=arm.initial_adapter_sha256,
            final_adapter_sha256=arm.final_adapter_sha256,
            exam_precommit_sha256=commitment, report_sha256=report_digest,
            arm=arm.name, label_commitment=label_map[arm.name],
            training_visits=tuple(("train-room", case_id, prompt_map[case_id])
                                  for case_id in arm.visited_case_ids),
            training_structure_exposures=(tuple(zip(
                exposure.case_ids, exposure.prompt_sha256, exposure.structure_keys,
            )) if arm.name != "baseline" else ()),
            compute_steps=arm.compute_steps, optimizer_updates=arm.optimizer_updates,
        )
        manifests.append(seal_arm_manifest(manifest, CUSTODY_KEY))
        observed[arm.name] = (arm.final_adapter_sha256, report_digest)
    kwargs = dict(
        sealed_arms=tuple(manifests), arm_runs=arms,
        training_exposure=exposure, cases=cases,
        evaluator_precommitment=commitment,
        examiner_gold_seal=private_gold_seal(cases, GOLD_KEY),
        retention_before=retention_before,
        custody_key=CUSTODY_KEY, examiner_key=GOLD_KEY,
        correct_label_commitment=correct,
        corrupted_label_commitment=corrupt,
        empty_label_commitment=empty,
        observed_artifacts=observed,
        expected_experiment_id="test-cycle-2",
        expected_source_tree_sha256="e" * 64,
        expected_objective_id="arithmetic",
        expected_structure_keys=exposure.structure_keys,
    )
    return kwargs


class EvaluationProvenanceBridgeTests(unittest.TestCase):
    def test_verified_control_receipts_can_feed_evaluator_but_never_graduate(self):
        result = bind_and_evaluate(**setup())
        self.assertEqual(result.verdict.status, "PROVISIONAL")
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.physical_model_verified)
        self.assertEqual(len(result.sealed_manifest_sha256), 4)
        self.assertFalse(hasattr(result, "custody_key"))

    def test_inconsistent_scored_model_arm_is_rejected(self):
        kwargs = setup()
        kwargs["arm_runs"] = tuple(replace(arm, final_adapter_sha256="f" * 64)
                                   if arm.name == "correct_labels" else arm
                                   for arm in kwargs["arm_runs"])
        with self.assertRaisesRegex(ValueError, "arm.*checkpoint"):
            bind_and_evaluate(**kwargs)

    def test_false_external_artifact_readback_is_rejected(self):
        kwargs = setup()
        kwargs["observed_artifacts"] = dict(kwargs["observed_artifacts"])
        kwargs["observed_artifacts"]["correct_labels"] = ("f" * 64, "1" * 64)
        with self.assertRaisesRegex(ValueError, "artifact readback"):
            bind_and_evaluate(**kwargs)

    def test_training_exposure_and_structure_custody_must_match(self):
        kwargs = setup()
        kwargs["expected_structure_keys"] = ("forged-structure",)
        with self.assertRaisesRegex(ValueError, "structure custody"):
            bind_and_evaluate(**kwargs)
        kwargs = setup()
        kwargs["training_exposure"] = replace(kwargs["training_exposure"],
                                               case_ids=("wrong-case",))
        with self.assertRaisesRegex(ValueError, "exposure"):
            bind_and_evaluate(**kwargs)

    def test_replacing_both_structure_claim_and_expected_list_is_rejected(self):
        kwargs = setup()
        swapped = ("invented-novel-structure", "invented-novel-structure")
        kwargs["training_exposure"] = replace(
            kwargs["training_exposure"], structure_keys=swapped,
        )
        kwargs["expected_structure_keys"] = swapped
        # Prior bridge accepted this as independently held novelty because
        # the two attacker-controlled inputs agreed; the authenticated arm
        # manifest did not actually bind structural exposure.
        with self.assertRaisesRegex(ValueError, "authenticated.*structure"):
            bind_and_evaluate(**kwargs)

    def test_wrong_experiment_or_exam_precommit_fails(self):
        kwargs = setup()
        kwargs["expected_experiment_id"] = "other-experiment"
        with self.assertRaisesRegex(ValueError, "experiment"):
            bind_and_evaluate(**kwargs)
        kwargs = setup()
        kwargs["evaluator_precommitment"] = hashlib.sha256(b"other exam").hexdigest()
        with self.assertRaisesRegex(ValueError, "exam"):
            bind_and_evaluate(**kwargs)

    def test_wrong_key_or_reused_gold_key_is_rejected(self):
        kwargs = setup()
        kwargs["custody_key"] = GOLD_KEY
        with self.assertRaisesRegex(ValueError, "distinct"):
            bind_and_evaluate(**kwargs)
        kwargs = setup()
        kwargs["custody_key"] = b"forged-independent-control-key-2026"
        with self.assertRaisesRegex(ValueError, "authentication"):
            bind_and_evaluate(**kwargs)

    def test_relabelled_but_reserialized_seal_cannot_enter_evaluation(self):
        kwargs = setup()
        altered = list(kwargs["sealed_arms"])
        tampered = dict(altered[2])
        tampered["manifest"] = dict(
            tampered["manifest"],
            label_commitment=altered[3]["manifest"]["label_commitment"],
        )
        altered[2] = tampered
        kwargs["sealed_arms"] = tuple(altered)
        with self.assertRaisesRegex(ValueError, "authentication"):
            bind_and_evaluate(**kwargs)


if __name__ == "__main__":
    unittest.main()
