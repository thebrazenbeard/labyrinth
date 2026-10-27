"""Independent verification for the real-neural trial analyzer.

The synthetic responses test scoring software only: they are NOT evidence of
any neural training or model improvement.
"""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from labyrinth.paired_analysis import (
    MANIFEST_SCHEMA, TRIALS_SCHEMA, _hash, _mcnemar_exact, analyze, main,
)


def _sha(payload):
    return hashlib.sha256(payload.encode()).hexdigest()


def fixture():
    cases = [
        {"case_id": "dev-1", "prompt_sha256": _sha("first"), "answer": "7",
         "template_id": "S1", "slice": "within_template", "structure_fingerprint": _sha("S1")},
        {"case_id": "dev-2", "prompt_sha256": _sha("second"), "answer": "12",
         "template_id": "S2", "slice": "structural", "structure_fingerprint": _sha("S2")},
    ]
    manifest = {"schema": MANIFEST_SCHEMA, "source_commit": "1" * 40,
                "split": "dev", "cases": cases,
                "train_case_ids": ["train-1"], "train_prompt_sha256": [_sha("training")],
                "train_structure_fingerprints": [_sha("S1")]}
    def arm(responses, weight, *, steps=2, visits=10):
        return {"checkpoint_sha256": _sha("ckpt-" + weight),
                "parameter_sha256": _sha("param-" + weight),
                "optimizer_steps": steps, "train_case_visits": visits,
                "trials": [{"case_id": case["case_id"], "prompt_sha256": case["prompt_sha256"],
                            "generation_seed": index + 300, "response": response}
                           for index, (case, response) in enumerate(zip(cases, responses))]}
    trials = {"schema": TRIALS_SCHEMA, "source_commit": "1" * 40,
              "base_model_sha256": _sha("base"),
              "base_tokenizer_sha256": _sha("tokenizer"),
              "generation_config_sha256": _sha("greedy-fixed-128"),
              "eval_manifest_sha256": _hash(manifest),
              "arms": {
                  "baseline": arm(["ANSWER: 6", "ANSWER: 12"], "base", steps=0, visits=0),
                  "trained": arm(["ANSWER: 7", "ANSWER: 12"], "trained"),
                  "sham": arm(["ANSWER: 6", "ANSWER: 12"], "base"),
                  "corrupted": arm(["ANSWER: 6", "ANSWER: 13"], "corrupt"),
              }}
    return manifest, trials


class PairedAnalysisTests(unittest.TestCase):
    def test_paired_results_are_independent_of_gold_in_report(self):
        manifest, trials = fixture()
        result = analyze(manifest, trials)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["template_count"], 2)
        self.assertEqual(result["arm_summary"]["baseline"]["correct"], 1)
        self.assertEqual(result["arm_summary"]["trained"]["accuracy"], 1)
        self.assertEqual(result["paired_vs_baseline"]["trained"]["baseline_wrong_candidate_right"], 1)
        self.assertEqual(result["paired_vs_baseline"]["trained"]["baseline_right_candidate_wrong"], 0)
        self.assertEqual(result["paired_vs_baseline"]["trained"]["delta_accuracy"], 0.5)
        self.assertEqual(result["paired_vs_baseline"]["trained"]["exact_mcnemar_p_exploratory"], 1)
        self.assertEqual(result["by"]["slice"]["structural"]["trained"]["correct"], 1)
        self.assertEqual(result["by"]["template_group"]["group_0001"]["baseline"]["correct"], 0)
        self.assertNotIn("ANSWER:", json.dumps(result))
        self.assertNotIn("\"answer\"", json.dumps(result))

    def test_invalid_format_separate_from_wrong_answer(self):
        manifest, trials = fixture()
        trials["arms"]["trained"]["trials"][0]["response"] = "The result is 7"
        report = analyze(manifest, trials)
        self.assertEqual(report["arm_summary"]["trained"]["correct"], 1)
        self.assertEqual(report["arm_summary"]["trained"]["format_valid"], 1)
        self.assertEqual(report["arm_summary"]["trained"]["format_adherence"], 0.5)

    def test_cross_arm_mismatch_fail_closed(self):
        for mutation in ("wrong-id", "wrong-prompt", "wrong-seed", "missing-trial", "wrong-budget"):
            with self.subTest(mutation=mutation):
                manifest, trials = fixture()
                bad = trials["arms"]["trained"]
                if mutation == "wrong-id":
                    bad["trials"][0]["case_id"] = "dev-other"
                elif mutation == "wrong-prompt":
                    bad["trials"][0]["prompt_sha256"] = _sha("other")
                elif mutation == "wrong-seed":
                    bad["trials"][0]["generation_seed"] += 1
                elif mutation == "missing-trial":
                    bad["trials"].pop()
                else:
                    bad["optimizer_steps"] += 1
                with self.assertRaises(ValueError):
                    analyze(manifest, trials)

    def test_split_contamination_and_sham_mutation_fail_closed(self):
        for mutation in ("id-overlap", "prompt-overlap", "structure-overlap", "changed-sham",
                         "unchanged-trained", "changed-manifest"):
            with self.subTest(mutation=mutation):
                manifest, trials = fixture()
                if mutation == "id-overlap":
                    manifest["train_case_ids"].append("dev-1")
                elif mutation == "prompt-overlap":
                    manifest["train_prompt_sha256"].append(manifest["cases"][0]["prompt_sha256"])
                elif mutation == "structure-overlap":
                    manifest["train_structure_fingerprints"].append(_sha("S2"))
                elif mutation == "changed-sham":
                    trials["arms"]["sham"]["parameter_sha256"] = _sha("not-base")
                elif mutation == "unchanged-trained":
                    trials["arms"]["trained"]["parameter_sha256"] = trials["arms"]["baseline"]["parameter_sha256"]
                else:
                    manifest["cases"][0]["answer"] = "8"
                if mutation in ("id-overlap", "prompt-overlap", "structure-overlap"):
                    trials["eval_manifest_sha256"] = _hash(manifest)
                with self.assertRaises(ValueError):
                    analyze(manifest, trials)

    def test_structural_slice_alias_cannot_bypass_overlap_check(self):
        manifest, trials = fixture()
        # Previous implementation accepted 'novel_structure' without checking
        # an overlapping train/eval structure fingerprint.
        manifest["cases"][1]["slice"] = "novel_structure"
        manifest["cases"][1]["structure_fingerprint"] = _sha("S1")
        trials["eval_manifest_sha256"] = _hash(manifest)
        with self.assertRaisesRegex(ValueError, "invalid evaluation slice"):
            analyze(manifest, trials)

        manifest["cases"][1]["slice"] = "length_shift"
        trials["eval_manifest_sha256"] = _hash(manifest)
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            analyze(manifest, trials)

    def test_gold_tainted_template_id_never_echoed(self):
        manifest, trials = fixture()
        manifest["cases"][1]["template_id"] = "ANSWER: 12"
        trials["eval_manifest_sha256"] = _hash(manifest)
        with self.assertRaisesRegex(ValueError, "template ID"):
            analyze(manifest, trials)

        # Even valid opaque IDs may encode a private answer; use only generated
        # group indexes in the report, never the caller's raw labels.
        manifest["cases"][1]["template_id"] = "GOLDANSWER12"
        trials["eval_manifest_sha256"] = _hash(manifest)
        report = analyze(manifest, trials)
        self.assertNotIn("GOLDANSWER12", json.dumps(report))
        self.assertIn("group_0002", report["by"]["template_group"])

    def test_reject_one_sided_curriculum(self):
        manifest, trials = fixture()
        trials["arms"]["uniform"] = copy.deepcopy(trials["arms"]["trained"])
        with self.assertRaisesRegex(ValueError, "uniform/adaptive"):
            analyze(manifest, trials)

    def test_adaptive_vs_uniform_direct_comparison(self):
        manifest, trials = fixture()
        trials["arms"]["uniform"] = copy.deepcopy(trials["arms"]["baseline"])
        trials["arms"]["adaptive"] = copy.deepcopy(trials["arms"]["trained"])
        for name in ("uniform", "adaptive"):
            trials["arms"][name]["optimizer_steps"] = 2
            trials["arms"][name]["train_case_visits"] = 10
        report = analyze(manifest, trials)
        self.assertEqual(report["adaptive_vs_uniform"]["delta_accuracy"], 0.5)

    def test_mcnemar_exact_small_discordant_count(self):
        self.assertEqual(_mcnemar_exact(0, 0), 1)
        self.assertEqual(_mcnemar_exact(1, 0), 1)
        self.assertEqual(_mcnemar_exact(2, 0), 0.5)
        self.assertAlmostEqual(_mcnemar_exact(10, 0), 2 / 1024)

    def test_cli_keeps_private_gold_separate_and_never_overwrites(self):
        manifest, trials = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            mp, tp, rp = [directory / name for name in ("private.json", "trials.json", "report.json")]
            mp.write_text(json.dumps(manifest))
            tp.write_text(json.dumps(trials))
            self.assertEqual(main(["--private-manifest", str(mp), "--trials", str(tp), "--report", str(rp)]), 0)
            result = json.loads(rp.read_text())
            self.assertEqual(result["case_count"], 2)
            before = rp.read_bytes()
            with self.assertRaises(SystemExit):
                main(["--private-manifest", str(mp), "--trials", str(tp), "--report", str(rp)])
            self.assertEqual(before, rp.read_bytes())


if __name__ == "__main__":
    unittest.main()
