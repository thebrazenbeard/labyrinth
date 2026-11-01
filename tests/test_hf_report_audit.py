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
    EVAL_SEED_XOR, LEGACY_SCHEMA, PRE_EXPOSURE_SCHEMA, SCHEMA, PilotConfig,
    _prompt_commitment, _training_visit_exposures, _training_visit_indices,
    derange_answers,
    training_cases,
)
from labyrinth.hf_adaptive_source_v1 import (
    ADAPTIVE_SCHEMA, build_adaptive_manifest, reconstruct_adaptive_source,
)
from labyrinth.minotaur_hf_roster_bridge import collect_minotaur_hf_roster
from labyrinth.maze import generate_maze
from labyrinth.runner import _source_state
from labyrinth.hf_report_audit import audit_hf_report as strict_audit_hf_report
from tests.test_minotaur_hf_roster_bridge import _config, _runner, _scope
from tests.test_journey_policy_runner import toy_arithmetic_learner


def audit_hf_report(report, *, allow_legacy=True):
    """The original fixture exercises historical v2 internal-consistency only."""
    return strict_audit_hf_report(report, allow_legacy=allow_legacy)


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


def matched_v3_fixture():
    """A valid historical matched-sham pilot, without in-loop exposure proof."""
    report = pilot_fixture()
    report["schema"] = "labyrinth.hf-lora-pilot.v3"
    report["initial_persistent_buffers_sha256"] = "c" * 64
    report["config"]["answer_only"] = True
    report["control_data"].update({
        "answer_scaffold_supplied": True,
        "sham_optimizer_steps": 2, "sham_train_case_visits": 2,
        "sham_evaluation_before_updates": False, "sham_is_compute_matched": True,
    })
    report["training"]["sham"] = dict(report["training"]["candidate"])
    train, _ = training_cases(PilotConfig(
        model_path="unused", output_dir="unused", seed=13,
        train_count=2, eval_count=2, tiers=(1,), steps=2,
    ))
    expected = [train[i].id for i in _training_visit_indices(train, seed=13, steps=2)]
    for arm in ("sham", "candidate", "corrupted"):
        report["training"][arm]["visited_case_ids"] = expected
    return report


def adaptive_pilot_fixture():
    """Source-only fixture: no inference or optimizer execution is claimed."""
    config = _config()
    roster = collect_minotaur_hf_roster(_runner(), config, toy_arithmetic_learner)
    manifest = build_adaptive_manifest(config, _scope(), roster)
    train, heldout, planned = reconstruct_adaptive_source(config, manifest)
    report = pilot_fixture(seed=config.seed, count=config.train_count)
    report["schema"] = ADAPTIVE_SCHEMA
    report["config"].update({
        "answer_only": config.answer_only,
        "eval_count": config.eval_count,
        "steps": config.steps,
    })
    report["initial_persistent_buffers_sha256"] = "a" * 64
    report["adaptive_train_source_v1"] = manifest
    report["training_case_ids"] = [case.id for case in train]
    report["heldout_case_ids"] = [case.id for case in heldout]
    report["heldout_public_prompt_sha256"] = _prompt_commitment(heldout)
    report["control_data"].update({
        "corrupted_label_collisions": sum(
            case.answer == label for case, label in zip(train, derange_answers(train))
        ),
        "sham_optimizer_steps": config.steps,
        "sham_train_case_visits": config.steps,
        "sham_evaluation_before_updates": False,
        "sham_is_compute_matched": True,
        "answer_scaffold_supplied": False,
    })
    visits = {
        "steps": config.steps, "example_visits": config.steps,
        "visited_case_ids": [case_id for case_id, _ in planned],
        "visited_exposures": [list(pair) for pair in planned],
        "input_tokens_seen": 200,
        "losses": [2.5] * config.steps,
    }
    report["training"] = {arm: dict(visits) for arm in ("sham", "candidate", "corrupted")}
    report["optimizer_exposure_recording"] = (
        "in-loop-completed-optimizer-step.v1:source-only")
    trials = []
    for case in heldout:
        response = "ANSWER: 0"
        score = score_response(case, response)
        trials.append({
            "id": case.id,
            "correct": score.correct,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "response_preview": response,
            "score_reason": score.reason,
        })
    summary = {
        "correct": sum(t["correct"] for t in trials),
        "total": len(trials),
        "accuracy": sum(t["correct"] for t in trials) / len(trials),
        "format_valid": len(trials),
        "format_rate": 1.0,
        "trials": trials,
    }
    report["evaluations"] = {
        arm: {**summary, "trials": [dict(t) for t in trials]}
        for arm in ("baseline", "sham", "candidate", "corrupted")
    }
    return report


class HfReportAuditTests(unittest.TestCase):
    def test_legacy_schema_cannot_downgrade_a_default_required_exposure_audit(self):
        report = matched_v3_fixture()
        # Before the versioned contract, a claim with its new source marker and
        # ledgers deleted could still return PASS merely by using the old v3.
        self.assertEqual(strict_audit_hf_report(report)["status"], "FAIL")
        compatible = strict_audit_hf_report(report, allow_legacy=True)
        self.assertEqual(compatible["status"], "PASS", compatible["failures"])
        self.assertEqual(compatible["evidence_contract"], "legacy-internal-consistency-only")
        self.assertFalse(compatible["authenticated_model_execution"])

    def test_new_version_requires_complete_in_loop_ledger_even_after_reserialization(self):
        report = matched_v3_fixture()
        report["schema"] = "labyrinth.hf-lora-pilot.v3-exposure.v1"
        report["optimizer_exposure_recording"] = (
            "in-loop-completed-optimizer-step.v1:source-only")
        train, _ = training_cases(PilotConfig(
            model_path="unused", output_dir="unused", seed=13,
            train_count=2, eval_count=2, tiers=(1,), steps=2,
        ))
        expected = [[train[i].id, hashlib.sha256(train[i].prompt.encode()).hexdigest()]
                    for i in _training_visit_indices(train, seed=13, steps=2)]
        for arm in ("sham", "candidate", "corrupted"):
            report["training"][arm]["visited_exposures"] = [list(x) for x in expected]

        self.assertEqual(strict_audit_hf_report(report)["status"], "PASS")
        tampered = json.loads(json.dumps(report))
        del tampered["optimizer_exposure_recording"]
        for arm in ("sham", "candidate", "corrupted"):
            del tampered["training"][arm]["visited_exposures"]
        self.assertEqual(strict_audit_hf_report(tampered)["status"], "FAIL")
        # A caller who expects v4 cannot accept a schema-downgrade to v3.
        tampered["schema"] = "labyrinth.hf-lora-pilot.v3"
        self.assertEqual(strict_audit_hf_report(tampered)["status"], "FAIL")
        compatible = strict_audit_hf_report(tampered, allow_legacy=True)
        self.assertEqual(compatible["status"], "PASS", compatible["failures"])
        self.assertEqual(compatible["evidence_contract"], "legacy-internal-consistency-only")

    def test_adaptive_schema_requires_replayed_public_manifest_and_all_source_visits(self):
        report = adaptive_pilot_fixture()
        checked = audit_hf_report(report)
        self.assertEqual(checked["status"], "PASS", checked)
        self.assertFalse(checked["authenticated_model_execution"])
        invalid = []
        stripped = json.loads(json.dumps(report))
        stripped.pop("adaptive_train_source_v1")
        invalid.append(stripped)
        stripped_visits = json.loads(json.dumps(report))
        for arm in ("sham", "candidate", "corrupted"):
            stripped_visits["training"][arm].pop("visited_exposures")
        invalid.append(stripped_visits)
        downgraded = json.loads(json.dumps(report))
        downgraded["schema"] = SCHEMA
        downgraded.pop("adaptive_train_source_v1")
        invalid.append(downgraded)
        mislabeled = json.loads(json.dumps(report))
        mislabeled["schema"] = SCHEMA
        invalid.append(mislabeled)
        falsely_resumed = json.loads(json.dumps(report))
        falsely_resumed["resumed_journey"] = {"cycle_index": 1}
        invalid.append(falsely_resumed)
        swapped = json.loads(json.dumps(report))
        swapped["adaptive_train_source_v1"]["train_rooms"][0]["tactic"] = "transfer"
        invalid.append(swapped)
        for attack in invalid:
            with self.subTest(schema=attack["schema"]):
                self.assertEqual(strict_audit_hf_report(attack)["status"], "FAIL")

    def test_report_audit_rejects_forged_optional_visit_exposures(self):
        report = pilot_fixture()
        train, _ = training_cases(PilotConfig(
            model_path="unused", output_dir="unused", seed=13,
            train_count=2, eval_count=2, tiers=(1,), steps=2,
        ))
        # Legacy sequential visits are different from v3's seeded schedule;
        # use the actual sequential source order for this v2 report.
        expected = [[case.id, hashlib.sha256(case.prompt.encode()).hexdigest()]
                    for case in train[:2]]
        for arm in ("candidate", "corrupted"):
            report["training"][arm]["visited_exposures"] = [list(row) for row in expected]
        self.assertEqual(audit_hf_report(report)["status"], "PASS")
        hostile_rows = (
            [["foreign-eval-id", "0" * 64] for _ in expected],
            [expected[1], expected[0]],
            [expected[0], [expected[1][0], "0" * 64]],
            expected[:1],
            ["not-an-exposure"] * 2,
            {"case": expected[0]},
        )
        for hostile in hostile_rows:
            with self.subTest(hostile=hostile):
                report["training"]["candidate"]["visited_exposures"] = hostile
                checked = audit_hf_report(report)
                self.assertEqual(checked["status"], "FAIL", checked)
                self.assertTrue(any("exposure" in item for item in checked["failures"]))

    def test_optional_visit_exposures_require_consistent_arm_coverage(self):
        report = pilot_fixture()
        train, _ = training_cases(PilotConfig(
            model_path="unused", output_dir="unused", seed=13,
            train_count=2, eval_count=2, tiers=(1,), steps=2,
        ))
        report["training"]["candidate"]["visited_exposures"] = [
            [case.id, hashlib.sha256(case.prompt.encode()).hexdigest()]
            for case in train[:2]
        ]
        checked = audit_hf_report(report)
        self.assertEqual(checked["status"], "FAIL", checked)
        self.assertTrue(any("exposure" in item for item in checked["failures"]))

    def test_v3_visit_exposures_match_regenerated_seeded_schedule(self):
        report = pilot_fixture()
        report["schema"] = PRE_EXPOSURE_SCHEMA
        report["initial_persistent_buffers_sha256"] = "c" * 64
        report["config"]["answer_only"] = True
        report["control_data"].update({
            "answer_scaffold_supplied": True,
            "sham_optimizer_steps": 2, "sham_train_case_visits": 2,
            "sham_evaluation_before_updates": False,
            "sham_is_compute_matched": True,
        })
        report["training"]["sham"] = dict(report["training"]["candidate"])
        train, _ = training_cases(PilotConfig(
            model_path="unused", output_dir="unused", seed=13,
            train_count=2, eval_count=2, tiers=(1,), steps=2,
        ))
        planned = list(_training_visit_exposures(train, seed=13, steps=2))
        for arm in ("sham", "candidate", "corrupted"):
            report["training"][arm]["visited_case_ids"] = [case_id for case_id, _ in planned]
            # In-process pilot values contain tuple pairs; serialized reports
            # contain JSON arrays. Both forms represent the same commitment.
            report["training"][arm]["visited_exposures"] = list(planned)
        self.assertEqual(audit_hf_report(report)["status"], "PASS")
        report["training"]["sham"]["visited_exposures"] = [list(pair) for pair in reversed(planned)]
        checked = audit_hf_report(report)
        self.assertEqual(checked["status"], "FAIL", checked)
        self.assertTrue(any("exposure" in item for item in checked["failures"]))

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

    def test_in_loop_optimizer_exposure_marker_requires_exact_prompt_roster(self):
        report = pilot_fixture()
        train, _ = training_cases(PilotConfig(
            model_path="unused", output_dir="unused", seed=13,
            train_count=2, eval_count=2, tiers=(1,), steps=2,
        ))
        expected = [[case.id, hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()]
                    for case in train[:2]]
        report["optimizer_exposure_recording"] = (
            "in-loop-completed-optimizer-step.v1:source-only")
        for arm in ("candidate", "corrupted"):
            report["training"][arm]["visited_exposures"] = [list(pair) for pair in expected]
        self.assertEqual(audit_hf_report(report)["status"], "PASS")
        report["training"]["candidate"]["visited_exposures"][0][1] = "0" * 64
        result = audit_hf_report(report)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("optimizer TRAIN exposure" in failure
                            for failure in result["failures"]))
        report["training"]["candidate"]["visited_exposures"] = [list(pair)
                                                                   for pair in expected]
        del report["training"]["corrupted"]["visited_exposures"]
        self.assertEqual(audit_hf_report(report)["status"], "FAIL")

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
                [sys.executable, "-m", "labyrinth", "audit-hf-report", "--input", str(source), "--allow-legacy"],
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
            command = [sys.executable, "-m", "labyrinth", "audit-hf-report", "--input", str(source), "--allow-legacy"]
            strict = subprocess.run(command[:-1], text=True, capture_output=True, timeout=30)
            self.assertEqual(strict.returncode, 2, strict.stderr)
            self.assertEqual(json.loads(strict.stdout)["status"], "FAIL")
            valid = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(json.loads(valid.stdout)["status"], "PASS")
            self.assertEqual(json.loads(valid.stdout)["evidence_contract"],
                             "legacy-internal-consistency-only")
            tampered = json.loads(source.read_text(encoding="utf-8"))
            tampered["evaluations"]["candidate"]["correct"] = 99
            source.write_text(json.dumps(tampered), encoding="utf-8")
            invalid = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(invalid.returncode, 2)
            self.assertEqual(json.loads(invalid.stdout)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
