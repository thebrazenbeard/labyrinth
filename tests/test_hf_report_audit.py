"""Offline pilot-record consistency tests; never import ML dependencies."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from labyrinth.evaluation import score_response
from labyrinth.hf_pilot import (
    EVAL_SEED_XOR, LEGACY_SCHEMA, PilotConfig, _prompt_commitment,
    training_cases,
)
from labyrinth.maze import generate_maze
from labyrinth.runner import _source_state
from labyrinth.hf_report_audit import audit_hf_report


def pilot_fixture(*, seed=13, count=2):
    train, heldout = training_cases(PilotConfig(
        model_path="C:/fixture/base", output_dir="C:/fixture/run",
        seed=seed, train_count=count, eval_count=count,
        tiers=(1,), steps=2,
    ))
    trials = []
    for challenge in heldout:
        response = "ANSWER: 0"
        score = score_response(challenge, response)
        trials.append({
            "id": challenge.id, "correct": score.correct,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "response_preview": response, "score_reason": score.reason,
        })
    summary = {
        "correct": sum(t["correct"] for t in trials), "total": len(trials),
        "accuracy": sum(t["correct"] for t in trials) / len(trials),
        "format_valid": len(trials), "format_rate": 1.0, "trials": trials,
    }
    visits = {"steps": 2, "example_visits": 2,
              "visited_case_ids": [c.id for c in train[:2]],
              "input_tokens_seen": 200, "losses": [2.5, 2.0]}
    return {
        "schema": LEGACY_SCHEMA, "source": _source_state(),
        "config": {"model_path": "C:/fixture/base", "output_dir": "C:/fixture/run",
                   "seed": seed, "train_count": count, "eval_count": count,
                   "tiers": [1], "steps": 2, "max_tokens": 256,
                   "max_new_tokens": 16, "learning_rate": 0.0002,
                   "lora_rank": 4, "lora_alpha": 8, "device": "cpu"},
        "adaptation_kind": "local HF causal-LM PEFT LoRA supervised parameter update",
        "runtime_versions": {"python": "3.12", "torch": "2.0",
                             "transformers": "4.0", "peft": "0.1"},
        "fresh_process_reload": "NOT RUN: execute `python -m labyrinth.hf_pilot eval` separately",
        "training_case_ids": [c.id for c in train],
        "heldout_case_ids": [c.id for c in heldout],
        "heldout_public_prompt_sha256": _prompt_commitment(heldout),
        "initial_lora_parameter_sha256": "a" * 64,
        "candidate_lora_parameter_sha256": "b" * 64,
        "candidate_lora_tensor_value_sha256": "b" * 64,
        "corrupted_lora_parameter_sha256": "c" * 64,
        "base_weights_sha256": {"model.safetensors": "d" * 64},
        "model_metadata_sha256": {"config.json": "e" * 64},
        "candidate_adapter_sha256": {"adapter_model.safetensors": "f" * 64},
        "training_process_id": 1234,
        "control_data": {"corrupted_label_collisions": 0, "sham_updates": 0,
                         "sham_train_case_visits": 0,
                         "sham_evaluation_before_updates": True,
                         "sham_is_compute_matched": False},
        "training": {"candidate": dict(visits), "corrupted": dict(visits)},
        "evaluations": {arm: {**summary, "trials": [dict(t) for t in trials]}
                        for arm in ("baseline", "sham", "candidate", "corrupted")},
    }


class HfReportAuditTests(unittest.TestCase):
    def test_consistent_full_preview_scores_pass_internal_audit(self):
        result = audit_hf_report(pilot_fixture())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["checked_trials"], 8)
        self.assertFalse(result["authenticated_model_execution"])

    def test_rerolled_semantic_arithmetic_heldout_is_audited_against_actual_pilot_split(self):
        report = pilot_fixture(seed=1073, count=32)
        old_heldout = generate_maze(
            seed=1073 ^ EVAL_SEED_XOR, count=32, tiers=(1,), split="eval",
        )
        self.assertNotEqual(report["heldout_case_ids"], [case.id for case in old_heldout])
        checked = audit_hf_report(report)
        self.assertEqual(checked["status"], "PASS", checked["failures"])
        self.assertEqual(checked["checked_trials"], 128)
        self.assertFalse(checked["authenticated_model_execution"])

    def test_forged_per_trial_score_and_summary_are_rejected(self):
        report = pilot_fixture()
        trial = report["evaluations"]["candidate"]["trials"][0]
        trial.update({"correct": not trial["correct"], "score_reason": "correct"})
        summary = report["evaluations"]["candidate"]
        summary["correct"] = sum(t["correct"] for t in summary["trials"])
        summary["accuracy"] = summary["correct"] / summary["total"]
        self.assertEqual(audit_hf_report(report)["status"], "FAIL")

    def test_truncated_preview_never_gets_full_audit_pass(self):
        report = pilot_fixture()
        trial = report["evaluations"]["candidate"]["trials"][0]
        trial["response_preview"] = "ANSWER: 0" + "x" * 119
        trial["response_sha256"] = hashlib.sha256((trial["response_preview"] + "tail").encode()).hexdigest()
        self.assertEqual(audit_hf_report(report)["status"], "INCOMPLETE")

    def test_oversized_response_reason_can_be_incomplete_not_false_forgery(self):
        report = pilot_fixture()
        trial = report["evaluations"]["candidate"]["trials"][0]
        trial["response_preview"] = "X" * 128
        trial["response_sha256"] = hashlib.sha256(("X" * 9000).encode()).hexdigest()
        trial["correct"] = False
        trial["score_reason"] = "invalid_response"
        summary = report["evaluations"]["candidate"]
        summary["correct"] = sum(t["correct"] for t in summary["trials"])
        summary["accuracy"] = summary["correct"] / summary["total"]
        summary["format_valid"] = sum(t["score_reason"] in ("correct", "incorrect")
                                      for t in summary["trials"])
        summary["format_rate"] = summary["format_valid"] / summary["total"]
        self.assertEqual(audit_hf_report(report)["status"], "INCOMPLETE")

    def test_control_visit_and_heldout_membership_forgery_fail(self):
        report = pilot_fixture()
        report["training"]["corrupted"]["visited_case_ids"] = ["foreign", "foreign"]
        report["evaluations"]["sham"]["trials"][0]["id"] = "foreign"
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("visit" in x for x in result["failures"]))
        self.assertTrue(any("id" in x for x in result["failures"]))

    def test_missing_checkpoint_receipt_is_not_audit_pass(self):
        report = pilot_fixture()
        del report["candidate_adapter_sha256"]
        report["base_weights_sha256"] = {"model.safetensors": "not-a-checksum"}
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("digest" in x for x in result["failures"]))

    def test_older_source_digest_is_an_incomplete_provisional_audit(self):
        report = pilot_fixture()
        report["source"]["source_tree_sha256"] = "1" * 64
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["checked_trials"], 8)
        self.assertTrue(any("source" in x for x in result["unchecked"]))

    def test_huge_integer_loss_is_rejected_without_crashing(self):
        report = pilot_fixture()
        report["training"]["candidate"]["losses"][0] = 10 ** 1000
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("loss" in x for x in result["failures"]))

    def test_missing_pretrained_config_and_runtime_receipts_are_not_pass(self):
        report = pilot_fixture()
        del report["config"]["model_path"]
        del report["runtime_versions"]
        del report["adaptation_kind"]
        self.assertEqual(audit_hf_report(report)["status"], "FAIL")

    def test_malformed_digest_cannot_hide_behind_truncated_preview(self):
        report = pilot_fixture()
        trial = report["evaluations"]["candidate"]["trials"][0]
        trial["response_preview"] = "X" * 128
        trial["response_sha256"] = "not-a-hash"
        self.assertEqual(audit_hf_report(report)["status"], "FAIL")

    def test_unpaired_unicode_surrogate_is_a_structured_fail(self):
        report = pilot_fixture()
        report["evaluations"]["candidate"]["trials"][0]["response_preview"] = "\ud800"
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("response" in item for item in result["failures"]))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "malformed_preview.json"
            source.write_text(json.dumps(report, ensure_ascii=True), encoding="utf-8")
            checked = subprocess.run(
                [sys.executable, "-m", "labyrinth", "audit-hf-report", "--input", str(source)],
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(checked.returncode, 2)
            self.assertEqual(json.loads(checked.stdout)["status"], "FAIL")

    def test_checksum_maps_require_structural_checkpoint_filenames(self):
        cases = (
            ("base_weights_sha256", {"unrelated.json": "a" * 64}),
            ("candidate_adapter_sha256", {"adapter_config.json": "b" * 64}),
            ("model_metadata_sha256", {"unrelated.json": "c" * 64}),
        )
        for field, forged in cases:
            with self.subTest(field=field):
                report = pilot_fixture()
                report[field] = forged
                result = audit_hf_report(report)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(any(field in item for item in result["failures"]))

    def test_cli_checks_neural_pilot_without_importing_heavy_ml_deps(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "pilot_report.json"
            source.write_text(json.dumps(pilot_fixture()), encoding="utf-8")
            command = [sys.executable, "-m", "labyrinth", "audit-hf-report", "--input", str(source)]
            valid = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(json.loads(valid.stdout)["status"], "PASS")
            tampered = json.loads(source.read_text(encoding="utf-8"))
            tampered["evaluations"]["candidate"]["correct"] = 99
            source.write_text(json.dumps(tampered), encoding="utf-8")
            invalid = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(invalid.returncode, 2)
            self.assertEqual(json.loads(invalid.stdout)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
