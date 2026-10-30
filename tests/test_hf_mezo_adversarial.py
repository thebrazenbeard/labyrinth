"""Adversarial v4 MeZO receipt and CLI checks without heavy ML dependencies."""

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from labyrinth.hf_pilot import MEZO_SCHEMA, PilotConfig, _training_visit_indices, training_cases
from labyrinth.hf_report_audit import audit_hf_report
from tests.test_hf_report_audit import pilot_fixture


def mezo_receipt():
    report = pilot_fixture()
    report["schema"] = MEZO_SCHEMA
    report["initial_persistent_buffers_sha256"] = "f" * 64
    report["config"].update({"answer_only": False, "optimizer": "mezo", "zo_epsilon": 0.001})
    report["control_data"].update({
        "sham_optimizer_steps": 2, "sham_train_case_visits": 2,
        "sham_evaluation_before_updates": False, "sham_is_compute_matched": True,
        "answer_scaffold_supplied": False,
        "mezo_loss_forward_evaluations_per_arm": 4,
        "mezo_snapshot_restoration": True,
    })
    config = PilotConfig(
        model_path="fixture", output_dir="fixture", seed=13,
        train_count=2, eval_count=2, tiers=(1,), steps=2,
    )
    train, _ = training_cases(config)
    visits = [train[index].id for index in _training_visit_indices(train, seed=13, steps=2)]
    for arm in ("sham", "candidate", "corrupted"):
        report["training"][arm] = {
            "steps": 2, "example_visits": 2, "visited_case_ids": visits,
            "input_tokens_seen": 100, "losses": [1.5, 2.5],
            "positive_losses": [1.0, 2.0], "negative_losses": [2.0, 3.0],
            "forward_evaluations": 4, "direction_seeds": [13, 14],
        }
    return report


class MezoReceiptHostileTests(unittest.TestCase):
    def test_honest_mezo_v4_receipt_passes_internal_audit(self):
        self.assertEqual(audit_hf_report(mezo_receipt())["status"], "PASS")

    def test_mezo_v4_refuses_seed_numbers_with_float_type(self):
        report = mezo_receipt()
        report["training"]["sham"]["direction_seeds"] = [13.0, 14.0]
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "FAIL", result)
        self.assertTrue(any("direction seeds" in reason for reason in result["failures"]))

    def test_mezo_v4_refuses_wrong_method_and_nonfinite_epsilon(self):
        for method, epsilon in (("adamw", 0.001), ("mezo", float("inf")),
                                ("mezo", True)):
            report = mezo_receipt()
            report["config"].update(optimizer=method, zo_epsilon=epsilon)
            with self.subTest(method=method, epsilon=epsilon):
                self.assertEqual(audit_hf_report(report)["status"], "FAIL")

    def test_mezo_v4_refuses_missing_sham_pair_even_if_candidate_is_intact(self):
        report = mezo_receipt()
        del report["training"]["sham"]["negative_losses"]
        self.assertEqual(audit_hf_report(report)["status"], "FAIL")

    def test_mezo_v4_huge_integer_forward_loss_is_structured_failure(self):
        report = mezo_receipt()
        report["training"]["candidate"]["positive_losses"][0] = 10 ** 1000
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("MeZO paired loss" in reason for reason in result["failures"]))

    def test_cli_exposes_mezo_without_downloading_or_loading_torch(self):
        help_result = subprocess.run(
            [sys.executable, "-m", "labyrinth", "train-hf", "--help"],
            text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--optimizer {adamw,mezo}", help_result.stdout)
        self.assertIn("--zo-epsilon", help_result.stdout)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "local"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"not-actual-weights")
            output = root / "new-output"
            result = subprocess.run(
                [sys.executable, "-m", "labyrinth", "train-hf",
                 "--model-path", str(model), "--output-dir", str(output),
                 "--optimizer", "mezo", "--zo-epsilon", "0"],
                text=True, capture_output=True, timeout=15,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("zo_epsilon", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
