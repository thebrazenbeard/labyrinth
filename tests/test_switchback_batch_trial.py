"""Wave16 hostile controls for source-public multi-case process comparisons."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import io
import json
import sys
import unittest
from unittest import mock

from labyrinth.switchback_circuit import make_case
from labyrinth.switchback_oracle import verify_plan
from scripts import switchback_batch_trial as batch
from scripts import switchback_model_trial as single


WINNING = ["0", "0", "WAIT", "0", "COMMIT"]
WIN_SCRIPT = ("import json,sys; json.load(sys.stdin);"
              "print(json.dumps({'kind':'PLAN','actions':['0','0','WAIT','0','COMMIT']}))")


def _config(seeds=(0, 1), *, per_case=2, total_calls=4,
            timeout=1.0, total_seconds=8.0):
    return batch.BatchConfig(tuple(seeds), tuple(11 + i for i in range(len(seeds))),
                             per_case, total_calls, timeout, total_seconds, 1024)


class SwitchbackBatchTrialTests(unittest.TestCase):
    def test_two_actual_fixed_child_cases_are_counted_and_replayable(self):
        result = batch.run_batch(_config(), batch.AdapterSpec(
            mode="fixed", fixed_actions=("0",)))
        self.assertTrue(batch.verify_batch_receipt(result))
        self.assertEqual(result["aggregate"]["precommitted_case_count"], 2)
        self.assertEqual(len(result["cases"]), 2)
        self.assertEqual([row["admission"] for row in result["cases"]],
                         ["COMPLETED", "COMPLETED"])
        self.assertEqual(result["aggregate"]["source_verified_wins"], 0)
        self.assertEqual(result["aggregate"]["verified_fraction_planned"],
                         {"numerator": 0, "denominator": 2})
        self.assertEqual(result["aggregate"]["calls_reserved"], 4)
        self.assertFalse(result["comparison_qualified"])
        self.assertFalse(result["neural_learning_verified"])
        self.assertFalse(result["model_weight_update_verified"])
        self.assertEqual(result["evidence_class"],
                         "SOURCE_PUBLIC_PRECOMMITTED_PROCESS_BATCH")

    def test_one_case_cherrypick_cannot_report_100_percent(self):
        self.assertEqual(verify_plan(make_case(0), WINNING)["verdict"], "VERIFIED_WIN")
        self.assertNotEqual(verify_plan(make_case(1), WINNING)["verdict"], "VERIFIED_WIN")
        spec = batch.AdapterSpec(mode="command",
                                 command_argv=(sys.executable, "-c", WIN_SCRIPT),
                                 explicit_authorization=True)
        result = batch.run_batch(_config(per_case=1, total_calls=2), spec)
        self.assertTrue(batch.verify_batch_receipt(result))
        self.assertEqual(result["aggregate"]["source_verified_wins"], 1)
        self.assertEqual(result["aggregate"]["verified_fraction_planned"],
                         {"numerator": 1, "denominator": 2})
        self.assertEqual([row["verified_win"] for row in result["cases"]], [True, False])
        self.assertEqual(result["aggregate"]["precommitted_case_count"], 2)

    def test_duplicate_or_incorrectly_typed_seeds_are_rejected_before_any_run(self):
        with self.assertRaises(ValueError):
            _config(seeds=(0, 0))
        with self.assertRaises(ValueError):
            batch.BatchConfig((0, True), (1, 2))
        with self.assertRaises(ValueError):
            batch.BatchConfig((0, 1), (11,))
        with self.assertRaises(ValueError):
            batch.BatchConfig(tuple(range(33)), tuple(range(33)))
        with self.assertRaises(ValueError):
            batch.BatchConfig((0,), (1,), max_total_calls=0)

    def test_precommitted_manifest_and_frozen_config_cannot_be_edited(self):
        config = _config()
        with self.assertRaises(FrozenInstanceError):
            config.case_seeds = (0,)
        spec = batch.AdapterSpec(mode="fixed", fixed_actions=())
        with self.assertRaises(FrozenInstanceError):
            spec.mode = "command"
        result = batch.run_batch(_config(seeds=(0,), per_case=1,
                                         total_calls=1, total_seconds=3), spec)
        self.assertEqual(result["manifest_sha256"], batch._sha(result["manifest"]))
        self.assertEqual(result["manifest"]["config"]["case_seeds"], (0,))
        self.assertTrue(batch.verify_batch_receipt(result))

    def test_global_call_reservation_preserves_skipped_cases_in_denominator(self):
        result = batch.run_batch(_config(seeds=(0, 1, 2), per_case=2,
                                         total_calls=3, total_seconds=12),
                                 batch.AdapterSpec(mode="fixed"))
        self.assertEqual([r["admission"] for r in result["cases"]],
                         ["COMPLETED", "SKIPPED_GLOBAL_CALL_BUDGET",
                          "SKIPPED_GLOBAL_CALL_BUDGET"])
        self.assertEqual(result["aggregate"]["calls_reserved"], 2)
        self.assertEqual(result["aggregate"]["verified_fraction_planned"]["denominator"], 3)
        self.assertTrue(batch.verify_batch_receipt(result))

    def test_global_time_reservation_preserves_partial_cohort(self):
        result = batch.run_batch(_config(seeds=(0, 1, 2), per_case=2,
                                         total_calls=6, timeout=1, total_seconds=3.1),
                                 batch.AdapterSpec(mode="fixed"))
        self.assertEqual([r["admission"] for r in result["cases"]],
                         ["COMPLETED", "SKIPPED_GLOBAL_TIME_RESERVATION",
                          "SKIPPED_GLOBAL_TIME_RESERVATION"])
        self.assertEqual(result["aggregate"]["seconds_reserved"], 3.0)
        self.assertEqual(result["aggregate"]["verified_fraction_planned"]["denominator"], 3)
        self.assertTrue(batch.verify_batch_receipt(result))

    def test_inadequate_global_budget_never_silently_filters_all_cases(self):
        result = batch.run_batch(_config(seeds=(0, 1), per_case=2,
                                         total_calls=1, total_seconds=8),
                                 batch.AdapterSpec(mode="fixed"))
        self.assertEqual(len(result["cases"]), 2)
        self.assertEqual(result["aggregate"]["calls_executed"], 0)
        self.assertEqual(result["aggregate"]["receipts_valid"], 0)
        self.assertEqual(result["aggregate"]["skipped_or_invalid"], 2)
        self.assertTrue(batch.verify_batch_receipt(result))

    def test_injected_fake_perfect_scores_rejected_without_valid_oracle_receipts(self):
        fake = {"schema": "labyrinth.switchback-real-model-trial.v1",
                "case_seed_host_only": 0, "verified_success": True,
                "trial_binding_valid": True, "source_receipt_replay_valid": True,
                "source_verdict": "VERIFIED_WIN", "turns": [],
                "process_calls": 0}
        with mock.patch.object(batch, "run_trial", return_value=fake):
            result = batch.run_batch(_config(), batch.AdapterSpec(mode="fixed"))
        self.assertEqual(result["aggregate"]["source_verified_wins"], 0)
        self.assertEqual(result["aggregate"]["verified_fraction_planned"],
                         {"numerator": 0, "denominator": 2})
        self.assertEqual([r["admission"] for r in result["cases"]],
                         ["INVALID_TRIAL_RECEIPT", "INVALID_TRIAL_RECEIPT"])
        self.assertTrue(batch.verify_batch_receipt(result))

    def test_stolen_seed_zero_receipt_cannot_be_counted_again_for_seed_one(self):
        valid = single.run_trial(
            single.TrialConfig(0, 11, "command", 1, 1., 3., 1024),
            single.CommandProcessAdapter((sys.executable, "-c", WIN_SCRIPT),
                                         generation_seed=11,
                                         authorize_command=True,
                                         timeout_seconds=1, max_output_bytes=1024))
        self.assertTrue(valid["verified_success"])
        with mock.patch.object(batch, "run_trial", return_value=valid):
            result = batch.run_batch(_config(seeds=(1,), per_case=1,
                                             total_calls=1, total_seconds=3),
                                     batch.AdapterSpec(mode="command",
                                                       command_argv=(sys.executable, "-c", WIN_SCRIPT),
                                                       explicit_authorization=True))
        self.assertEqual(result["aggregate"]["source_verified_wins"], 0)
        self.assertEqual(result["cases"][0]["admission"], "INVALID_TRIAL_RECEIPT")
        self.assertTrue(batch.verify_batch_receipt(result))

    def test_rehashed_100_percent_aggregate_is_not_replay_valid(self):
        result = batch.run_batch(_config(seeds=(0,), per_case=1,
                                         total_calls=1, total_seconds=3),
                                 batch.AdapterSpec(mode="fixed"))
        result["aggregate"]["source_verified_wins"] = 1
        result["aggregate"]["verified_fraction_planned"] = {"numerator": 1,
                                                             "denominator": 1}
        result["batch_receipt_sha256"] = batch._sha({
            key: value for key, value in result.items() if key != "batch_receipt_sha256"})
        self.assertFalse(batch.verify_batch_receipt(result))

    def test_rehashed_model_learning_claim_does_not_gain_qualification(self):
        result = batch.run_batch(_config(seeds=(0,), per_case=1,
                                         total_calls=1, total_seconds=3),
                                 batch.AdapterSpec(mode="fixed"))
        result["model_weight_update_verified"] = True
        result["batch_receipt_sha256"] = batch._sha({
            key: value for key, value in result.items() if key != "batch_receipt_sha256"})
        self.assertFalse(batch.verify_batch_receipt(result))

    def test_rehashed_skipped_case_may_not_claim_completion(self):
        result = batch.run_batch(_config(seeds=(0, 1), per_case=1,
                                         total_calls=1, total_seconds=3),
                                 batch.AdapterSpec(mode="fixed"))
        self.assertTrue(batch.verify_batch_receipt(result))
        self.assertEqual(result["cases"][1]["admission"],
                         "SKIPPED_GLOBAL_CALL_BUDGET")
        for field, replacement in (("admission", "COMPLETED"),
                                   ("reserved_calls", 1),
                                   ("reserved_seconds", 2.0)):
            forged = deepcopy(result)
            forged["cases"][1][field] = replacement
            forged["batch_receipt_sha256"] = batch._sha({
                key: value for key, value in forged.items()
                if key != "batch_receipt_sha256"})
            with self.subTest(field=field):
                self.assertFalse(batch.verify_batch_receipt(forged))

    def test_rehashed_global_aggregates_cannot_exaggerate_coverage(self):
        result = batch.run_batch(_config(seeds=(0, 1), per_case=1,
                                         total_calls=1, total_seconds=3),
                                 batch.AdapterSpec(mode="fixed"))
        self.assertTrue(batch.verify_batch_receipt(result))
        for field, replacement in (("calls_reserved", 999),
                                   ("calls_executed", 999),
                                   ("seconds_reserved", 999.0),
                                   ("skipped_or_invalid", 0)):
            forged = deepcopy(result)
            forged["aggregate"][field] = replacement
            forged["batch_receipt_sha256"] = batch._sha({
                key: value for key, value in forged.items()
                if key != "batch_receipt_sha256"})
            with self.subTest(field=field):
                self.assertFalse(batch.verify_batch_receipt(forged))

    def test_boolean_integer_equivalence_cannot_rewrite_case_or_resource_counts(self):
        result = batch.run_batch(_config(seeds=(1,), per_case=1,
                                         total_calls=1, total_seconds=3),
                                 batch.AdapterSpec(mode="fixed"))
        self.assertTrue(batch.verify_batch_receipt(result))
        for location, field in (("row", "case_seed"),
                                ("row", "reserved_calls"),
                                ("aggregate", "precommitted_case_count"),
                                ("aggregate", "calls_reserved"),
                                ("aggregate", "receipts_valid")):
            forged = deepcopy(result)
            target = (forged["cases"][0] if location == "row"
                      else forged["aggregate"])
            if target[field] != 1:
                continue
            target[field] = True
            forged["batch_receipt_sha256"] = batch._sha({
                key: value for key, value in forged.items()
                if key != "batch_receipt_sha256"})
            with self.subTest(field=field, location=location):
                self.assertFalse(batch.verify_batch_receipt(forged))

    def test_double_rehashed_non_mapping_manifest_fails_closed(self):
        source = batch.run_batch(_config(seeds=(1,), per_case=1,
                                         total_calls=1, total_seconds=3),
                                 batch.AdapterSpec(mode="fixed"))
        for replacement in (None, [], "invalid"):
            forged = deepcopy(source)
            forged["manifest"]["config"] = replacement
            forged["manifest_sha256"] = batch._sha(forged["manifest"])
            forged["batch_receipt_sha256"] = batch._sha({
                key: value for key, value in forged.items()
                if key != "batch_receipt_sha256"})
            with self.subTest(replacement=repr(replacement)):
                self.assertFalse(batch.verify_batch_receipt(forged))

    def test_rehashed_nested_child_wire_hash_cannot_impersonate_same_output(self):
        result = batch.run_batch(_config(seeds=(0,), per_case=1,
                                         total_calls=1, total_seconds=3),
                                 batch.AdapterSpec(mode="fixed"))
        self.assertTrue(batch.verify_batch_receipt(result))
        forged = deepcopy(result)
        trial = forged["cases"][0]["trial"]
        self.assertEqual(trial["turns"][0]["process_receipts"][0]["status"], "OK")
        self.assertIsNotNone(trial["turns"][0]["response_sha256"])
        trial["turns"][0]["process_receipts"][0]["response_sha256"] = "0" * 64
        trial["receipt_sha256"] = batch._sha({
            key: value for key, value in trial.items() if key != "receipt_sha256"})
        forged["batch_receipt_sha256"] = batch._sha({
            key: value for key, value in forged.items()
            if key != "batch_receipt_sha256"})
        self.assertFalse(batch.verify_batch_receipt(forged))

    def test_partial_per_case_timeout_retains_source_unknown_without_invented_win(self):
        command = (sys.executable, "-c", "import time;time.sleep(3)")
        spec = batch.AdapterSpec(mode="command", command_argv=command,
                                 explicit_authorization=True)
        result = batch.run_batch(_config(seeds=(0, 1), per_case=1,
                                         total_calls=2, timeout=.05,
                                         total_seconds=3), spec)
        self.assertEqual(len(result["cases"]), 2)
        self.assertEqual(result["aggregate"]["source_verified_wins"], 0)
        self.assertTrue(batch.verify_batch_receipt(result))
        self.assertFalse(result["generalization_qualified"])

    def test_local_command_and_ollama_without_explicit_authorization_are_rejected(self):
        with self.assertRaises(PermissionError):
            batch.AdapterSpec(mode="command", command_argv=(sys.executable,))
        with self.assertRaises(PermissionError):
            batch.AdapterSpec(mode="ollama", ollama_model="qwen-local")
        with self.assertRaises(ValueError):
            batch.AdapterSpec(mode="fixed", command_argv=(sys.executable,))

    def test_cli_fixed_batch_emits_one_replayable_receipt(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = batch.main(["--seeds", "0", "1", "--mode", "fixed",
                                 "--max-calls-per-case", "1", "--max-total-calls", "2",
                                 "--max-total-seconds", "5"])
        self.assertEqual(status, 0)
        report = json.loads(stdout.getvalue())
        self.assertTrue(batch.verify_batch_receipt(report))
        self.assertEqual(report["aggregate"]["precommitted_case_count"], 2)


if __name__ == "__main__":
    unittest.main()
