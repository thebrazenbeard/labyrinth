"""Offline report audit regression cases: consistency is not provenance."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from labyrinth.adapters import ReplayAdapter
from labyrinth.maze import generate_maze
from labyrinth.report_audit import audit_report
from labyrinth.runner import build_training_artifact, run_experiment, run_maze


class ReportAuditTests(unittest.TestCase):
    def test_canonical_maze_report_is_internally_consistent(self):
        report = run_maze(ReplayAdapter({}, default="ANSWER: 0"), seed=17, count=2, tiers=(1,))
        audit = audit_report(report)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["checked_trials"], 2)
        self.assertEqual(audit["unverifiable_trials"], 0)
        self.assertIn("not authenticated", " ".join(audit["limitations"]).lower())

    def test_forged_perfect_report_can_pass_consistency_but_not_authenticity(self):
        # Demonstrates the evidence ceiling: no neural inference was executed.
        cases = generate_maze(seed=13, count=2, tiers=(1,), split="eval")
        fake = ReplayAdapter({case.prompt: f"ANSWER: {case.answer}" for case in cases},
                             model_id="purported-pretrained-neural-model")
        audit = audit_report(run_maze(fake, seed=13, count=2, tiers=(1,)))
        self.assertEqual(audit["status"], "PASS")
        self.assertFalse(audit["authenticated_model_execution"])

    def test_score_forgery_is_detected_even_with_matching_aggregates(self):
        report = run_maze(ReplayAdapter({}, default="ANSWER: 0"), seed=17, count=1, tiers=(1,))
        trial = report["evaluation"]["trials"][0]
        trial.update({"correct": True, "points": 1, "scoring_reason": "correct"})
        report["evaluation"].update({"points": 1, "accuracy": 1.0})
        audit = audit_report(report)
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any("score" in reason for reason in audit["failures"]))

    def test_changed_seed_and_source_revision_fail(self):
        report = run_maze(ReplayAdapter({}, default="ANSWER: 0"), seed=22, count=1, tiers=(1,))
        report["evaluation"]["trials"][0]["model_seed"] += 1
        report["source"]["source_tree_sha256"] = "0" * 64
        audit = audit_report(report)
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any("source" in reason for reason in audit["failures"]))
        self.assertTrue(any("seed" in reason for reason in audit["failures"]))

    def test_huge_numeric_accuracy_claim_is_rejected_without_exception(self):
        report = run_maze(ReplayAdapter({}, default="ANSWER: 0"), seed=22, count=1, tiers=(1,))
        report["evaluation"]["accuracy"] = 10 ** 1000
        audit = audit_report(report)
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any("accuracy" in reason for reason in audit["failures"]))

    def test_model_identity_requires_string_even_when_fields_match(self):
        report = run_maze(ReplayAdapter({}, default="ANSWER: 0"), seed=22, count=1, tiers=(1,))
        report["model_id"] = 1337
        report["adapter"]["id"] = 1337
        audit = audit_report(report)
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any("identity" in reason for reason in audit["failures"]))

    def test_truncated_preview_is_incomplete_not_verified(self):
        response = "ANSWER: 0" + " " * 5000
        report = run_maze(ReplayAdapter({}, default=response), seed=7, count=1, tiers=(1,))
        audit = audit_report(report)
        self.assertEqual(audit["status"], "INCOMPLETE")
        self.assertEqual(audit["checked_trials"], 0)
        self.assertEqual(audit["unverifiable_trials"], 1)

    def test_experiment_requires_artifact_for_full_prompt_audit(self):
        artifact = build_training_artifact(seed=8, count=2, tiers=(1, 2, 3))
        report = run_experiment(lambda: ReplayAdapter({}, default="ANSWER: 0"), seed=8,
                                eval_count=2, tiers=(1, 2, 3), artifact=artifact)
        incomplete = audit_report(report)
        self.assertEqual(incomplete["status"], "INCOMPLETE")
        self.assertEqual(incomplete["checked_trials"], 24)
        complete = audit_report(report, artifact=artifact)
        self.assertEqual(complete["status"], "PASS")
        self.assertEqual(complete["checked_trials"], 24)

        report["arms"]["shuffled"]["trials"][0]["model_prompt_sha256"] = "0" * 64
        self.assertEqual(audit_report(report, artifact=artifact)["status"], "FAIL")

    def test_wrong_artifact_is_not_accepted(self):
        artifact = build_training_artifact(seed=8, count=2, tiers=(1,))
        report = run_experiment(lambda: ReplayAdapter({}, default="ANSWER: 0"), seed=8,
                                eval_count=2, tiers=(1,), artifact=artifact)
        wrong = build_training_artifact(seed=9, count=2, tiers=(1,))
        self.assertEqual(audit_report(report, artifact=wrong)["status"], "FAIL")

    def test_training_and_control_metadata_forgery_is_detected(self):
        artifact = build_training_artifact(seed=8, count=2, tiers=(1, 2, 3))
        report = run_experiment(lambda: ReplayAdapter({}, default="ANSWER: 0"), seed=8,
                                eval_count=2, tiers=(1, 2, 3), artifact=artifact)
        report["shuffled_label_collisions"] = 999
        report["training_model_id"] = "spoofed-training-model"
        report["adaptation_kind"] = "neural-weight-update"
        report["controls"]["shuffled"] = "actually identical to candidate"
        audit = audit_report(report, artifact=artifact)
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any("shuffled" in x for x in audit["failures"]))
        self.assertTrue(any("training model" in x for x in audit["failures"]))
        self.assertTrue(any("adaptation" in x for x in audit["failures"]))

    def test_cli_audits_json_and_returns_nonzero_on_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "maze.json"
            report = run_maze(ReplayAdapter({}, default="ANSWER: 0"), seed=17, count=1, tiers=(1,))
            source.write_text(json.dumps(report), encoding="utf-8")
            command = [sys.executable, "-m", "labyrinth", "audit-report", "--input", str(source)]
            clean = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(clean.returncode, 0, clean.stderr)
            self.assertEqual(json.loads(clean.stdout)["status"], "PASS")
            report["evaluation"]["trials"][0]["response_sha256"] = "0" * 64
            source.write_text(json.dumps(report), encoding="utf-8")
            tampered = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(tampered.returncode, 2)
            self.assertEqual(json.loads(tampered.stdout)["status"], "FAIL")

    def test_cli_rejects_duplicate_keys_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "ambiguous.json"
            command = [sys.executable, "-m", "labyrinth", "audit-report", "--input", str(source)]
            source.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
            duplicate = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(duplicate.returncode, 2)
            self.assertIn("duplicate JSON field", duplicate.stderr)
            source.write_text('{"schema":NaN}', encoding="utf-8")
            nonfinite = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(nonfinite.returncode, 2)
            self.assertIn("nonfinite JSON", nonfinite.stderr)

    def test_cli_returns_nonzero_when_four_arm_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "experiment.json"
            artifact = build_training_artifact(seed=8, count=2, tiers=(1,))
            report = run_experiment(lambda: ReplayAdapter({}, default="ANSWER: 0"), seed=8,
                                    eval_count=2, tiers=(1,), artifact=artifact)
            source.write_text(json.dumps(report), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "labyrinth", "audit-report",
                                     "--input", str(source)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
