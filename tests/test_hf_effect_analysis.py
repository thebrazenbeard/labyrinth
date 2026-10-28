"""Paired HF pilot comparisons quantify effects without claiming model execution."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.hf_effect_analysis import (
    analyze_hf_report, paired_binary_effect, summarize_hf_effects,
)
from labyrinth.hf_pilot import EVAL_SEED_XOR
from labyrinth.maze import generate_maze


def trial(challenge_id, correct=False, formatted=True, response=None):
    return {
        "id": challenge_id,
        "correct": correct,
        "score_reason": "correct" if correct else "incorrect" if formatted else "invalid_format",
        "response_sha256": response or challenge_id * 4,
    }


class PairedEffectsTests(unittest.TestCase):
    def test_exact_paired_test_and_conservative_three_contrast_adjustment(self):
        ids = [str(i) for i in range(12)]
        candidate = [trial(i, correct=i not in ("10", "11")) for i in ids]
        reference = [trial(i, correct=i in ("10", "11")) for i in ids]
        effect = paired_binary_effect(candidate, reference, multiplicity=3)
        self.assertEqual(effect["candidate_wins"], 10)
        self.assertEqual(effect["reference_wins"], 2)
        self.assertEqual(effect["discordant"], 12)
        self.assertEqual(effect["paired_accuracy_delta"], 8 / 12)
        self.assertAlmostEqual(effect["one_sided_exact_p"], 79 / 4096)
        self.assertAlmostEqual(effect["bonferroni_p"], 3 * 79 / 4096)

    def test_format_recovery_separated_from_wrong_answer_recovery(self):
        candidate = [trial("a", True), trial("b", True), trial("c", False)]
        baseline = [trial("a", False, formatted=False),
                    trial("b", False, formatted=True), trial("c", False)]
        effect = paired_binary_effect(candidate, baseline, multiplicity=3)
        self.assertEqual(effect["candidate_wins"], 2)
        self.assertEqual(effect["wins_from_invalid_format"], 1)
        self.assertEqual(effect["wins_against_valid_wrong_answer"], 1)
        self.assertEqual(effect["shared_valid_format_total"], 2)
        self.assertEqual(effect["shared_valid_format_candidate_wins"], 1)

    def test_mismatched_cases_and_inconsistent_scores_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "same unique"):
            paired_binary_effect([trial("a")], [trial("b")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_binary_effect([trial("a"), trial("a")],
                                 [trial("a"), trial("a")])
        corrupt = trial("a", True)
        corrupt["score_reason"] = "invalid_format"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            paired_binary_effect([corrupt], [trial("a")])

    def test_corrupted_arm_reordering_rejected_and_invalid_audit_never_signals(self):
        ids = [str(i) for i in range(12)]
        candidate = [trial(i, True) for i in ids]
        control = [trial(i, False) for i in ids]
        arms = {"candidate": candidate, "baseline": control,
                "sham": [dict(t) for t in control],
                "corrupted": [dict(t) for t in control]}
        identity = {"base_weights_sha256": {"weights": "a"},
                    "candidate_adapter_sha256": {"adapter": "b"}}
        family = {i: "arithmetic" for i in ids}
        with self.assertRaisesRegex(ValueError, "order"):
            summarize_hf_effects({**arms, "corrupted": list(reversed(control))},
                                 challenge_kinds=family, audit_status="PASS",
                                 model_identity=identity)
        complete = summarize_hf_effects(arms, challenge_kinds=family,
                                        audit_status="PASS", model_identity=identity)
        self.assertTrue(complete["exploratory_three_control_pattern"])
        tampered = summarize_hf_effects(arms, challenge_kinds=family,
                                       audit_status="FAIL", model_identity=identity)
        self.assertFalse(tampered["exploratory_three_control_pattern"])
        self.assertEqual(tampered["reasoning_improvement"], "NOT_ESTABLISHED")

    def test_only_format_repairs_cannot_trigger_exploratory_signal(self):
        ids = [str(i) for i in range(12)]
        candidate = [trial(i, True) for i in ids]
        control = [trial(i, False, formatted=False) for i in ids]
        arms = {"candidate": candidate, **{
            arm: [dict(t) for t in control]
            for arm in ("baseline", "sham", "corrupted")
        }}
        result = summarize_hf_effects(
            arms, challenge_kinds={i: "arithmetic" for i in ids}, audit_status="PASS",
            model_identity={"base_weights_sha256": {"weights": "a"},
                            "candidate_adapter_sha256": {"adapter": "b"}},
        )
        self.assertFalse(result["exploratory_three_control_pattern"])
        self.assertEqual(result["comparisons"]["baseline"]["wins_against_valid_wrong_answer"], 0)

    def test_report_labels_adapter_as_candidate_without_claiming_reasoning(self):
        identifiers = ["eval-t1-0", "eval-t1-1"]
        arms = {
            "baseline": [trial(identifiers[0]), trial(identifiers[1])],
            "sham": [trial(identifiers[0]), trial(identifiers[1])],
            "corrupted": [trial(identifiers[0]), trial(identifiers[1])],
            "candidate": [trial(identifiers[0], True), trial(identifiers[1], True)],
        }
        summary = summarize_hf_effects(
            arms, challenge_kinds={"eval-t1-0": "arithmetic", "eval-t1-1": "arithmetic"},
            audit_status="PASS", model_identity={"base_weights_sha256": {"a": "b"},
                                                   "candidate_adapter_sha256": {"c": "d"}},
        )
        self.assertEqual(summary["artifact_role"], "Minotaur candidate adapter plus pinned backbone")
        self.assertEqual(summary["comparisons"]["baseline"]["candidate_wins"], 2)
        self.assertEqual(summary["families"]["arithmetic"]["candidate_accuracy"], 1.0)
        self.assertEqual(summary["reasoning_improvement"], "NOT_ESTABLISHED")
        self.assertEqual(summary["generalization"], "NOT_ESTABLISHED")
        self.assertEqual(summary["evaluation_provenance"], "SELF_REPORTED")
        changed_identity = {"base_weights_sha256": {"a": "b"},
                            "candidate_adapter_sha256": {"c": "changed"}}
        changed = summarize_hf_effects(
            arms, challenge_kinds={"eval-t1-0": "arithmetic", "eval-t1-1": "arithmetic"},
            audit_status="PASS", model_identity=changed_identity,
        )
        self.assertNotEqual(summary["composition_receipt_sha256"],
                            changed["composition_receipt_sha256"])

    def test_high_level_reconstructs_real_family_mapping(self):
        heldout = generate_maze(seed=7 ^ EVAL_SEED_XOR, count=2,
                                tiers=(1, 4), split="eval")
        trials = [trial(c.id, correct=False) for c in heldout]
        report = {
            "config": {"seed": 7, "eval_count": 2, "tiers": [1, 4]},
            "base_weights_sha256": {"model.safetensors": "a" * 64},
            "model_metadata_sha256": {"config.json": "b" * 64},
            "candidate_adapter_sha256": {"adapter.safetensors": "c" * 64},
            "evaluations": {
                arm: {"trials": [dict(t) for t in trials]}
                for arm in ("baseline", "sham", "candidate", "corrupted")
            },
        }
        with patch("labyrinth.hf_effect_analysis.audit_hf_report",
                   return_value={"status": "INCOMPLETE", "unchecked": ["untrusted runtime"]}):
            result = analyze_hf_report(report)
        self.assertIn("arithmetic", result["families"])
        self.assertIn("register_program", result["families"])
        self.assertEqual(result["internal_audit_status"], "INCOMPLETE")
        self.assertEqual(result["training_mechanism"], "NOT_VERIFIED_BY_THIS_ANALYSIS")
        report["evaluations"]["candidate"]["trials"][0]["id"] = "foreign"
        with self.assertRaisesRegex(ValueError, "regenerated"):
            analyze_hf_report(report)

    def test_cli_emits_unqualified_analysis_and_does_not_overwrite(self):
        heldout = generate_maze(seed=4 ^ EVAL_SEED_XOR, count=2,
                                tiers=(1,), split="eval")
        trial_list = [trial(c.id) for c in heldout]
        fake = {
            "config": {"seed": 4, "eval_count": 2, "tiers": [1]},
            "base_weights_sha256": {"model.safetensors": "a" * 64},
            "model_metadata_sha256": {"config.json": "b" * 64},
            "candidate_adapter_sha256": {"adapter.safetensors": "c" * 64},
            "evaluations": {
                arm: {"trials": [dict(t) for t in trial_list]}
                for arm in ("baseline", "sham", "candidate", "corrupted")
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_path, output_path = root / "fake.json", root / "analysis.json"
            input_path.write_text(json.dumps(fake), encoding="utf-8")
            command = [sys.executable, "-m", "labyrinth.hf_effect_analysis",
                       "--input", str(input_path), "--report", str(output_path)]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 2)
            self.assertTrue(output_path.is_file())
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output["internal_audit_status"], "FAIL")
            self.assertEqual(output["generalization"], "NOT_ESTABLISHED")
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 2)
            self.assertIn("File exists", second.stderr)


if __name__ == "__main__":
    unittest.main()
