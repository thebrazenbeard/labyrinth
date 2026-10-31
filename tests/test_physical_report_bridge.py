"""Adversarial source-level tests for the public MiniLM physical-report bridge."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from labyrinth.physical_report_bridge import inspect_public_minilm_report


def digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def physical_report() -> dict:
    training = []
    for step in range(2):
        examples = [{
            "case_id": f"train-{step}-{i}",
            "prompt_sha256": digest(f"train-prompt-{step}-{i}"),
            "structure_key": f"review-template-{i % 2}",
            "target_label": i % 2,
        } for i in range(4)]
        training.append({"step": step, "examples": examples})
    predictions = {}
    for split in ("heldout", "transfer"):
        predictions[split] = [{
            "case_id": f"{split}-{i}",
            "prompt_sha256": digest(f"{split}-prompt-{i}"),
            "predicted_label": i % 2,
            "expected_label": i % 2,
            "response": str(i % 2),
        } for i in range(2)]
    arms = {}
    for arm in ("baseline", "zero_update", "correct_labels", "corrupted_labels"):
        batches = deepcopy(training) if arm != "baseline" else []
        if arm == "corrupted_labels":
            for batch in batches:
                for example in batch["examples"]:
                    example["target_label"] ^= 1
        calls = len(batches)
        changed = arm in ("correct_labels", "corrupted_labels")
        cycle = {
            "cycle": 0,
            "trainable_sha256_before": digest("initial"),
            "trainable_sha256_after": digest(arm) if changed else digest("initial"),
            "lora_sha256_before": digest("initial-lora"),
            "lora_sha256_after": digest(arm + "-lora") if changed else digest("initial-lora"),
            "optimizer_step_calls": calls,
            "optimizer_updates": calls if changed else 0,
            "compute_steps": calls,
            "forward_steps": calls,
            "backward_steps": calls,
            "training_forward_steps": calls,
            "training_backward_steps": calls,
            "training_batches": batches,
            "training_prompt_sha256": [example["prompt_sha256"]
                                      for batch in batches for example in batch["examples"]],
            "train_example_visits": 4 * calls,
            "heldout_predictions": deepcopy(predictions["heldout"]),
            "transfer_predictions": deepcopy(predictions["transfer"]),
            "heldout_accuracy": 1.0,
            "transfer_accuracy": 1.0,
        }
        arms[arm] = {
            "initial_trainable_sha256": digest("initial"),
            "initial_lora_sha256": digest("initial-lora"),
            "cycles": [cycle],
        }
    return {
        "schema": "labyrinth.minilm-cpu-development.v2",
        "qualification": "PUBLIC_DEVELOPMENT_ONLY",
        "private_final_exam": "NOT_RUN",
        "graduation_authorized": False,
        "physical_model_byte_custody": "SELF_REPORTED",
        "pretrained_origin": "CACHED_MODEL_ID_NOT_INDEPENDENTLY_ATTESTED",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "harness_sha256": digest("harness"),
        "run_id": "run-001",
        "physical_training_batch_size": 4,
        "steps_per_cycle": 2,
        "fixture_sizes": {"cycle0": 12, "cycle1": 12, "heldout": 2, "transfer": 2},
        "arms": arms,
        "heldout": {
            "correct_minus_baseline": 0.0, "correct_minus_zero": 0.0,
            "correct_minus_corrupted": 0.0,
            "correct_outperforms_all_controls": False,
        },
        "transfer": {
            "correct_minus_baseline": 0.0, "correct_minus_zero": 0.0,
            "correct_minus_corrupted": 0.0,
            "correct_outperforms_all_controls": False,
        },
    }


def synchronize_claimed_prompt_visits(report: dict) -> None:
    for arm in report["arms"].values():
        for cycle in arm["cycles"]:
            cycle["training_prompt_sha256"] = [
                row["prompt_sha256"] for batch in cycle["training_batches"]
                for row in batch["examples"]
            ]


class PhysicalReportBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "report.json"

    def inspect(self, report: dict, *, expected_sha: str | None = None):
        content = (json.dumps(report, sort_keys=True) + "\n").encode("utf-8")
        self.path.write_bytes(content)
        return inspect_public_minilm_report(
            self.path, expected_report_sha256=expected_sha or sha256(content).hexdigest(),
            expected_harness_sha256=digest("harness"), expected_run_id="run-001",
            expected_model_id="sentence-transformers/all-MiniLM-L6-v2",
        )

    def test_valid_four_arm_batch_report_remains_unqualified(self) -> None:
        result = self.inspect(physical_report())
        self.assertEqual(result.status, "PROVISIONAL")
        self.assertEqual(result.training_optimizer_steps, 2)
        self.assertEqual(result.training_example_visits, 8)
        self.assertEqual(result.prediction_case_counts, (("heldout", 2), ("transfer", 2)))
        self.assertFalse(result.physical_model_verified)
        self.assertFalse(result.fresh_process_verified)
        self.assertFalse(result.evaluator_custody_verified)
        self.assertFalse(result.formal_v1_compatible)
        self.assertFalse(result.graduation_authorized)

    def test_report_digest_must_match_bytes_read_from_disk(self) -> None:
        with self.assertRaisesRegex(ValueError, "report digest mismatch"):
            self.inspect(physical_report(), expected_sha=digest("fabricated-report"))

    def test_missing_arm_rejected(self) -> None:
        report = physical_report()
        del report["arms"]["baseline"]
        with self.assertRaisesRegex(ValueError, "four arms"):
            self.inspect(report)

    def test_training_batches_not_miscounted_as_optimizer_steps(self) -> None:
        report = physical_report()
        report["arms"]["correct_labels"]["cycles"][0]["train_example_visits"] = 2
        with self.assertRaisesRegex(ValueError, "example visits"):
            self.inspect(report)

    def test_cross_arm_training_schedule_mismatch_rejected(self) -> None:
        report = physical_report()
        report["arms"]["zero_update"]["cycles"][0]["training_batches"][0]["examples"][1]["case_id"] = "different"
        with self.assertRaisesRegex(ValueError, "matched batches"):
            self.inspect(report)

    def test_corrupted_label_arm_must_be_inverted(self) -> None:
        report = physical_report()
        report["arms"]["corrupted_labels"]["cycles"][0]["training_batches"][0]["examples"][0]["target_label"] = 0
        with self.assertRaisesRegex(ValueError, "corrupted labels"):
            self.inspect(report)

    def test_sham_cannot_claim_weight_updates(self) -> None:
        report = physical_report()
        report["arms"]["zero_update"]["cycles"][0]["optimizer_updates"] = 2
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.inspect(report)

    def test_baseline_cannot_claim_training(self) -> None:
        report = physical_report()
        report["arms"]["baseline"]["cycles"][0]["optimizer_step_calls"] = 2
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.inspect(report)

    def test_duplicate_prediction_case_ids_rejected(self) -> None:
        report = physical_report()
        report["arms"]["correct_labels"]["cycles"][0]["transfer_predictions"][1]["case_id"] = "transfer-0"
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            self.inspect(report)

    def test_prediction_response_consistency_checked(self) -> None:
        report = physical_report()
        report["arms"]["correct_labels"]["cycles"][0]["heldout_predictions"][0]["response"] = "1"
        with self.assertRaisesRegex(ValueError, "prediction response"):
            self.inspect(report)

    def test_self_reported_graduation_rejected(self) -> None:
        report = physical_report()
        report["graduation_authorized"] = True
        with self.assertRaisesRegex(ValueError, "graduation"):
            self.inspect(report)

    def test_public_development_predictions_cannot_be_private_exam(self) -> None:
        report = physical_report()
        report["private_final_exam"] = "PASS"
        with self.assertRaisesRegex(ValueError, "private exam"):
            self.inspect(report)

    def test_training_prediction_prompt_overlap_rejected(self) -> None:
        report = physical_report()
        training_sha = report["arms"]["correct_labels"]["cycles"][0]["training_batches"][0]["examples"][0]["prompt_sha256"]
        for arm in report["arms"].values():
            arm["cycles"][0]["transfer_predictions"][0]["prompt_sha256"] = training_sha
        with self.assertRaisesRegex(ValueError, "training/prediction prompt overlap"):
            self.inspect(report)

    def test_repeated_training_case_cannot_change_its_label(self) -> None:
        report = physical_report()
        for arm_name in ("zero_update", "correct_labels", "corrupted_labels"):
            rows = report["arms"][arm_name]["cycles"][0]["training_batches"]
            rows[1]["examples"][0]["case_id"] = rows[0]["examples"][0]["case_id"]
            rows[1]["examples"][0]["prompt_sha256"] = rows[0]["examples"][0]["prompt_sha256"]
            rows[1]["examples"][0]["structure_key"] = rows[0]["examples"][0]["structure_key"]
        report["arms"]["correct_labels"]["cycles"][0]["training_batches"][1]["examples"][0]["target_label"] = 1
        synchronize_claimed_prompt_visits(report)
        with self.assertRaisesRegex(ValueError, "identity/label changed"):
            self.inspect(report)

    def test_duplicate_json_keys_rejected_even_when_digest_matches(self) -> None:
        report = physical_report()
        content = json.dumps(report).replace('"schema": ', '"schema": "alternate", "schema": ', 1).encode()
        self.path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "duplicate JSON report field"):
            inspect_public_minilm_report(
                self.path, expected_report_sha256=sha256(content).hexdigest(),
                expected_harness_sha256=digest("harness"), expected_run_id="run-001",
                expected_model_id="sentence-transformers/all-MiniLM-L6-v2",
            )

    def test_report_cannot_substitute_model_identity(self) -> None:
        report = physical_report()
        report["model_id"] = "some-other-base"
        with self.assertRaisesRegex(ValueError, "model identity mismatch"):
            self.inspect(report)

    def test_inconsistent_aggregate_accuracy_rejected(self) -> None:
        report = physical_report()
        report["arms"]["corrupted_labels"]["cycles"][0]["transfer_accuracy"] = 0.5
        with self.assertRaisesRegex(ValueError, "aggregate accuracy"):
            self.inspect(report)

    def test_nonfinite_accuracy_rejected(self) -> None:
        report = physical_report()
        report["arms"]["correct_labels"]["cycles"][0]["heldout_accuracy"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite JSON value"):
            self.inspect(report)

    def test_symlink_report_rejected(self) -> None:
        report = physical_report()
        self.inspect(report)
        alternate = self.path.with_name("linked.json")
        alternate.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "not symlink"):
            inspect_public_minilm_report(
                alternate, expected_report_sha256=sha256(self.path.read_bytes()).hexdigest(),
                expected_harness_sha256=digest("harness"), expected_run_id="run-001",
                expected_model_id="sentence-transformers/all-MiniLM-L6-v2",
            )

    def test_training_prompt_cannot_use_two_case_ids(self) -> None:
        report = physical_report()
        for arm_name in ("zero_update", "correct_labels", "corrupted_labels"):
            batches = report["arms"][arm_name]["cycles"][0]["training_batches"]
            batches[1]["examples"][0]["prompt_sha256"] = batches[0]["examples"][0]["prompt_sha256"]
        synchronize_claimed_prompt_visits(report)
        with self.assertRaisesRegex(ValueError, "training prompt reused"):
            self.inspect(report)

    def test_case_id_cannot_be_rebound_to_new_prompt_in_next_cycle(self) -> None:
        report = physical_report()
        for name, arm in report["arms"].items():
            prior = arm["cycles"][0]
            next_cycle = deepcopy(prior)
            next_cycle["cycle"] = 1
            next_cycle["trainable_sha256_before"] = prior["trainable_sha256_after"]
            next_cycle["lora_sha256_before"] = prior["lora_sha256_after"]
            next_cycle["trainable_sha256_after"] = (
                digest(f"{name}-next") if name in ("correct_labels", "corrupted_labels")
                else prior["trainable_sha256_after"]
            )
            next_cycle["lora_sha256_after"] = (
                digest(f"{name}-lora-next") if name in ("correct_labels", "corrupted_labels")
                else prior["lora_sha256_after"]
            )
            for batch in next_cycle["training_batches"]:
                for example in batch["examples"]:
                    example["prompt_sha256"] = digest(f"rebound-{example['case_id']}")
            arm["cycles"].append(next_cycle)
        synchronize_claimed_prompt_visits(report)
        with self.assertRaisesRegex(ValueError, "training case identity changed between cycles"):
            self.inspect(report)

    def test_prediction_prompt_identity_cannot_change_between_cycles(self) -> None:
        report = physical_report()
        for name, arm in report["arms"].items():
            prior = arm["cycles"][0]
            next_cycle = deepcopy(prior)
            next_cycle["cycle"] = 1
            next_cycle["trainable_sha256_before"] = prior["trainable_sha256_after"]
            next_cycle["lora_sha256_before"] = prior["lora_sha256_after"]
            next_cycle["trainable_sha256_after"] = (
                digest(f"{name}-next") if name in ("correct_labels", "corrupted_labels")
                else prior["trainable_sha256_after"]
            )
            next_cycle["lora_sha256_after"] = (
                digest(f"{name}-lora-next") if name in ("correct_labels", "corrupted_labels")
                else prior["lora_sha256_after"]
            )
            next_cycle["heldout_predictions"][0]["prompt_sha256"] = digest("changed-heldout")
            arm["cycles"].append(next_cycle)
        with self.assertRaisesRegex(ValueError, "prediction fixture changed|frozen baseline predictions changed"):
            self.inspect(report)

    def test_forged_top_level_comparison_cannot_override_per_case_predictions(self) -> None:
        report = physical_report()
        report["heldout"]["correct_minus_baseline"] = 0.99
        report["heldout"]["correct_outperforms_all_controls"] = True
        with self.assertRaisesRegex(ValueError, "inconsistent heldout comparison"):
            self.inspect(report)

    def test_retention_summary_recomputed_when_second_cycle_exists(self) -> None:
        report = physical_report()
        for arm_name, arm in report["arms"].items():
            previous = arm["cycles"][0]
            next_cycle = deepcopy(previous)
            next_cycle["cycle"] = 1
            next_cycle["trainable_sha256_before"] = previous["trainable_sha256_after"]
            next_cycle["lora_sha256_before"] = previous["lora_sha256_after"]
            next_cycle["trainable_sha256_after"] = (
                digest(f"{arm_name}-next") if arm_name in ("correct_labels", "corrupted_labels")
                else previous["trainable_sha256_after"]
            )
            next_cycle["lora_sha256_after"] = (
                digest(f"{arm_name}-lora-next") if arm_name in ("correct_labels", "corrupted_labels")
                else previous["lora_sha256_after"]
            )
            arm["cycles"].append(next_cycle)
        report["retention"] = {name: 0.8 for name in report["arms"]}
        with self.assertRaisesRegex(ValueError, "inconsistent retention summary"):
            self.inspect(report)

    def test_frozen_baseline_and_zero_update_must_predict_identically(self) -> None:
        report = physical_report()
        sham = report["arms"]["zero_update"]["cycles"][0]
        sham["heldout_predictions"][0]["predicted_label"] = 1
        sham["heldout_predictions"][0]["response"] = "1"
        sham["heldout_accuracy"] = 0.5
        with self.assertRaisesRegex(ValueError, "frozen baseline and sham predictions differ"):
            self.inspect(report)

    def test_frozen_baseline_prediction_cannot_change_between_cycles(self) -> None:
        report = physical_report()
        for name, arm in report["arms"].items():
            prior = arm["cycles"][0]
            next_cycle = deepcopy(prior)
            next_cycle["cycle"] = 1
            next_cycle["trainable_sha256_before"] = prior["trainable_sha256_after"]
            next_cycle["lora_sha256_before"] = prior["lora_sha256_after"]
            next_cycle["trainable_sha256_after"] = (
                digest(f"{name}-next") if name in ("correct_labels", "corrupted_labels")
                else prior["trainable_sha256_after"]
            )
            next_cycle["lora_sha256_after"] = (
                digest(f"{name}-lora-next") if name in ("correct_labels", "corrupted_labels")
                else prior["lora_sha256_after"]
            )
            arm["cycles"].append(next_cycle)
        for name in ("baseline", "zero_update"):
            frozen = report["arms"][name]["cycles"][1]
            frozen["transfer_predictions"][0]["predicted_label"] = 1
            frozen["transfer_predictions"][0]["response"] = "1"
            frozen["transfer_accuracy"] = 0.5
        with self.assertRaisesRegex(ValueError, "frozen baseline predictions changed"):
            self.inspect(report)

    def test_declared_steps_cannot_hide_extra_optimizer_work(self) -> None:
        report = physical_report()
        report["steps_per_cycle"] = 1
        with self.assertRaisesRegex(ValueError, "declared steps per cycle"):
            self.inspect(report)

    def test_fixture_size_cannot_understate_prediction_coverage(self) -> None:
        report = physical_report()
        report["fixture_sizes"]["heldout"] = 1
        with self.assertRaisesRegex(ValueError, "prediction coverage"):
            self.inspect(report)

    def test_redundant_training_prompt_receipt_cannot_contradict_batches(self) -> None:
        report = physical_report()
        report["arms"]["correct_labels"]["cycles"][0]["training_prompt_sha256"][0] = digest("alien")
        with self.assertRaisesRegex(ValueError, "training prompt SHA sequence"):
            self.inspect(report)


if __name__ == "__main__":
    unittest.main()
