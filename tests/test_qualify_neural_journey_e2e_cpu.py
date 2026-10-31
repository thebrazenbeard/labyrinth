"""Source-only tests; synthetic reports are not physical neural experiments."""

from __future__ import annotations

import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import qualify_minilm_cpu as minilm
from scripts.qualify_neural_journey_e2e_cpu import audit_run, _expected_visits, main


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic_report(root: Path | None = None) -> dict:
    steps = 3
    arms: dict = {}
    for arm in minilm.ARMS:
        before = sha("initial-full")
        lora_before = sha("initial-lora")
        cycles = []
        for index, rows in enumerate(minilm.TRAIN):
            after = before if arm == "zero_update" else sha(f"{arm}-full-{index}")
            lora_after = lora_before if arm == "zero_update" else sha(f"{arm}-lora-{index}")
            adapter_bytes = f"{arm}-adapter-{index}".encode()
            head_bytes = f"{arm}-head-{index}".encode()
            if root is not None:
                checkpoint = root / arm / f"cycle-{index}"
                (checkpoint / "adapter").mkdir(parents=True)
                (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(adapter_bytes)
                (checkpoint / "head.safetensors").write_bytes(head_bytes)
            cycles.append({
                "cycle": index,
                "trainable_sha256_before": before,
                "trainable_sha256_after": after,
                "lora_sha256_before": lora_before,
                "lora_sha256_after": lora_after,
                "adapter_file_sha256": hashlib.sha256(adapter_bytes).hexdigest(),
                "head_file_sha256": hashlib.sha256(head_bytes).hexdigest(),
                "optimizer_updates": 0 if arm == "zero_update" else steps,
                "forward_steps": steps, "backward_steps": steps,
                "training_prompt_sha256": _expected_visits(rows, steps),
                "heldout_accuracy_before": 0.4,
                "heldout_accuracy": ({"correct": 0.7 + 0.1 * index,
                                       "zero_update": 0.5, "inverted": 0.4}[arm]),
                "transfer_accuracy": ({"correct": 0.7 + 0.1 * index,
                                        "zero_update": 0.5, "inverted": 0.3}[arm]),
            })
            before, lora_before = after, lora_after
        arms[arm] = {"initial_trainable_sha256": sha("initial-full"),
                     "initial_lora_sha256": sha("initial-lora"), "cycles": cycles}
    return {
        "schema": minilm.SCHEMA,
        "qualification": "PUBLIC_DEVELOPMENT_ONLY",
        "graduation_authorized": False,
        "private_final_exam": "NOT_RUN",
        "physical_model_byte_custody": "SELF_REPORTED",
        "pretrained_origin": "CACHED_MODEL_ID_NOT_INDEPENDENTLY_ATTESTED",
        "device": "cpu",
        "seed": 42, "steps_per_cycle": steps,
        "fixture_sizes": minilm.validate_fixtures(),
        "harness_sha256": hashlib.sha256(Path(minilm.__file__).read_bytes()).hexdigest(),
        "arms": arms,
        "heldout": minilm.comparison({arm: arms[arm]["cycles"][1]["heldout_accuracy"]
                                       for arm in minilm.ARMS}),
        "transfer": minilm.comparison({arm: arms[arm]["cycles"][1]["transfer_accuracy"]
                                        for arm in minilm.ARMS}),
        "retention": {arm: arms[arm]["cycles"][1]["heldout_accuracy"] -
                      arms[arm]["cycles"][0]["heldout_accuracy"] for arm in minilm.ARMS},
    }


class NeuralJourneyCustodyAuditTests(unittest.TestCase):
    def test_coherent_public_report_is_still_incomplete(self) -> None:
        result = audit_run(synthetic_report())
        self.assertEqual(result["result"], "HOLD_INCOMPLETE_NEURAL_JOURNEY")
        self.assertEqual(result["mapped_training_rooms"], 12)
        self.assertEqual(result["source_replay_phase"], "AWAITING_EXAM")
        self.assertEqual(result["gates"]["experience"], "UNKNOWN")
        self.assertEqual(result["gates"]["regenerate"], "NOT_RUN")
        self.assertFalse(result["graduation_authorized"])

    def test_public_advantage_does_not_prove_independent_control_benefit(self) -> None:
        result = audit_run(synthetic_report())
        self.assertEqual(result["reported_control_advantage"], {
            "heldout": "PASS", "transfer": "PASS", "joint": "PASS"})
        self.assertEqual(result["control_benefit"], "UNKNOWN")
        self.assertEqual(result["public_report_checks_meaning"],
                         "SCHEMA_EXPOSURE_LINEAGE_ARITHMETIC_ONLY_NOT_LEARNING")

    def test_equal_correct_and_zero_update_fails_reported_control_advantage(self) -> None:
        report = synthetic_report()
        report["arms"]["correct"]["cycles"][1]["heldout_accuracy"] = 0.5
        report["heldout"] = minilm.comparison({
            arm: report["arms"][arm]["cycles"][1]["heldout_accuracy"]
            for arm in minilm.ARMS})
        report["retention"]["correct"] = (
            report["arms"]["correct"]["cycles"][1]["heldout_accuracy"]
            - report["arms"]["correct"]["cycles"][0]["heldout_accuracy"])
        result = audit_run(report)
        self.assertEqual(result["public_report_checks"], "PASS")
        self.assertEqual(result["reported_control_advantage"], {
            "heldout": "FAIL", "transfer": "PASS", "joint": "FAIL"})
        self.assertEqual(result["control_benefit"], "FAIL")
        self.assertEqual(result["result"], "HOLD_INCOMPLETE_NEURAL_JOURNEY")

    def test_corrupted_label_control_dominates_transfer_and_fails_advantage(self) -> None:
        report = synthetic_report()
        report["arms"]["inverted"]["cycles"][1]["transfer_accuracy"] = 0.9
        report["transfer"] = minilm.comparison({
            arm: report["arms"][arm]["cycles"][1]["transfer_accuracy"]
            for arm in minilm.ARMS})
        result = audit_run(report)
        self.assertEqual(result["reported_control_advantage"], {
            "heldout": "PASS", "transfer": "FAIL", "joint": "FAIL"})
        self.assertEqual(result["control_benefit"], "FAIL")
        self.assertFalse(result["graduation_authorized"])

    def test_missing_actual_room_dispatch_not_upgraded_by_self_report(self) -> None:
        result = audit_run(synthetic_report())
        self.assertEqual(result["journey_room_dispatch"],
                         "UNKNOWN_POSTHOC_SOURCE_MAPPING_ONLY")
        self.assertEqual(result["development_case_scores"],
                         "MISSING_REPORT_ONLY_CONTAINS_AGGREGATES")

    def test_corrupted_exposure_is_rejected(self) -> None:
        report = synthetic_report()
        report["arms"]["correct"]["cycles"][0]["training_prompt_sha256"][0] = sha("exam-leak")
        with self.assertRaisesRegex(ValueError, "training exposure diverges"):
            audit_run(report)

    def test_replaced_successor_checkpoint_is_rejected(self) -> None:
        report = synthetic_report()
        report["arms"]["correct"]["cycles"][1]["lora_sha256_before"] = sha("replaced")
        with self.assertRaisesRegex(ValueError, "did not persist"):
            audit_run(report)

    def test_zero_update_mutation_is_rejected(self) -> None:
        report = synthetic_report()
        report["arms"]["zero_update"]["cycles"][0]["trainable_sha256_after"] = sha("mutated")
        with self.assertRaisesRegex(ValueError, "parameter mutation"):
            audit_run(report)

    def test_checkpoint_files_are_verified_and_tampering_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = synthetic_report(root)
            result = audit_run(report, run_dir=root)
            self.assertEqual(result["local_checkpoint_bytes"], "VERIFIED_HASHES")
            (root / "correct" / "cycle-1" / "head.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "saved head file digest mismatch"):
                audit_run(report, run_dir=root)

    def test_linked_checkpoint_arm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = synthetic_report(root)
            original = root / "correct"
            relocated = root / "moved_correct"
            original.rename(relocated)
            try:
                original.symlink_to(relocated, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symbolic links are not available")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                audit_run(report, run_dir=root)

    def test_fabricated_graduation_is_rejected(self) -> None:
        report = synthetic_report()
        report["graduation_authorized"] = True
        with self.assertRaisesRegex(ValueError, "non-graduating"):
            audit_run(report)

    def test_too_few_steps_cannot_claim_all_rooms(self) -> None:
        report = synthetic_report()
        report["steps_per_cycle"] = 2
        with self.assertRaisesRegex(ValueError, "at least three steps"):
            audit_run(report)

    def test_cli_returns_nonzero_hold_even_with_all_saved_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = synthetic_report(root)
            (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["--run-dir", str(root)])
            self.assertEqual(status, 3)
            result = json.loads(output.getvalue())
            self.assertEqual(result["result"], "HOLD_INCOMPLETE_NEURAL_JOURNEY")
            self.assertFalse(result["graduation_authorized"])


if __name__ == "__main__":
    unittest.main()
