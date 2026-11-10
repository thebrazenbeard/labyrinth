"""Wave18 hostile controls for precommitted Novel Router subprocess batches."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import sys
import unittest
from unittest import mock

from labyrinth import switchback_novel_transfer as novel
from scripts import novel_router_batch_trial as batch
from scripts import novel_router_model_trial as single


WIN = novel.search(novel.make_case(0))["witness"]
WIN_JSON = json.dumps({"kind": "PLAN", "actions": WIN})
ABSTAIN = '{"kind":"ABSTAIN"}'


def _config(seeds=(0, 1), *, calls=2, total_calls=4,
            timeout=2., total_seconds=12.):
    return batch.BatchConfig(tuple(seeds),
                             tuple(index + 41 for index in range(len(seeds))),
                             calls, total_calls, timeout, total_seconds, 1024)


def _rehash(report):
    report["batch_receipt_sha256"] = batch._sha({
        key: value for key, value in report.items()
        if key != "batch_receipt_sha256"})


class NovelRouterBatchTests(unittest.TestCase):
    def test_two_real_fixture_subprocesses_replay_and_keep_full_denominator(self):
        report = batch.run_batch(_config(), single.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(report))
        self.assertEqual([row["admission"] for row in report["cases"]],
                         ["COMPLETED", "COMPLETED"])
        self.assertEqual(report["aggregate"]["precommitted_case_count"], 2)
        self.assertEqual(report["aggregate"]["source_verified_wins"], 0)
        self.assertEqual(report["aggregate"]["verified_fraction_planned"],
                         {"numerator": 0, "denominator": 2})
        self.assertEqual(report["aggregate"]["calls_reserved"], 4)
        self.assertEqual(report["aggregate"]["calls_executed"], 2)
        self.assertEqual(report["aggregate"]["receipts_valid"], 2)
        self.assertIs(report["neural_learning_verified"], False)
        self.assertIs(report["generation_seed_effect_verified"], False)
        self.assertIs(report["private_heldout_qualified"], False)
        self.assertEqual(report["report_classification"], batch.PRIVATE_LABEL)

    def test_precommitted_cherrypick_win_remains_one_of_two(self):
        self.assertEqual(novel.verify_plan(novel.make_case(0), WIN)["verdict"],
                         "VERIFIED_WIN")
        self.assertNotEqual(novel.verify_plan(novel.make_case(1), WIN)["verdict"],
                            "VERIFIED_WIN")
        report = batch.run_batch(_config(calls=1, total_calls=2,
                                         timeout=2, total_seconds=7),
                                 single.AdapterSpec(fixture_responses=(WIN_JSON,)))
        self.assertTrue(batch.verify_batch_receipt(report))
        self.assertEqual([row["verified_win"] for row in report["cases"]],
                         [True, False])
        self.assertEqual(report["aggregate"]["verified_fraction_planned"],
                         {"numerator": 1, "denominator": 2})
        self.assertFalse(report["comparison_qualified"])

    def test_duplicate_bool_seeds_and_unbounded_manifest_are_rejected(self):
        with self.assertRaises(ValueError):
            _config(seeds=(0, 0))
        with self.assertRaises(ValueError):
            batch.BatchConfig((0, True), (10, 11))
        with self.assertRaises(ValueError):
            batch.BatchConfig((0, 1), (10,))
        with self.assertRaises(ValueError):
            batch.BatchConfig(tuple(range(33)), tuple(range(33)))
        with self.assertRaises(ValueError):
            batch.BatchConfig((0,), (10,), max_calls_per_case=22)

    def test_skip_call_budget_remains_in_denominator_not_disappearing(self):
        report = batch.run_batch(_config(seeds=(0, 1, 2), calls=2,
                                         total_calls=3, total_seconds=15),
                                 single.AdapterSpec())
        self.assertEqual([row["admission"] for row in report["cases"]],
                         ["COMPLETED", "SKIPPED_GLOBAL_CALL_BUDGET",
                          "SKIPPED_GLOBAL_CALL_BUDGET"])
        self.assertEqual(report["aggregate"]["calls_reserved"], 2)
        self.assertEqual(report["aggregate"]["verified_fraction_planned"]["denominator"],
                         3)
        self.assertTrue(batch.verify_batch_receipt(report))

    def test_global_time_reservation_keeps_partial_cohort(self):
        report = batch.run_batch(_config(seeds=(0, 1, 2), calls=2,
                                         total_calls=6, total_seconds=5.1),
                                 single.AdapterSpec())
        self.assertEqual([row["admission"] for row in report["cases"]],
                         ["COMPLETED", "SKIPPED_GLOBAL_TIME_RESERVATION",
                          "SKIPPED_GLOBAL_TIME_RESERVATION"])
        self.assertEqual(report["aggregate"]["seconds_reserved"], 5.)
        self.assertTrue(batch.verify_batch_receipt(report))

    def test_when_every_case_skips_zero_valid_results_are_not_hidden(self):
        report = batch.run_batch(_config(calls=2, total_calls=1,
                                         total_seconds=12), single.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(report))
        self.assertEqual(report["aggregate"]["precommitted_case_count"], 2)
        self.assertEqual(report["aggregate"]["receipts_valid"], 0)
        self.assertEqual(report["aggregate"]["skipped_or_invalid"], 2)
        self.assertEqual(report["aggregate"]["calls_executed"], 0)

    def test_fabricated_perfect_model_result_is_rejected_without_source_receipt(self):
        fake = {"schema": single.SCHEMA, "verified_success": True,
                "source_verdict": "VERIFIED_WIN", "turns": [],
                "source_receipt_replay_valid": True}
        with mock.patch.object(batch.single, "run_trial", return_value=fake):
            report = batch.run_batch(_config(), single.AdapterSpec())
        self.assertEqual(report["aggregate"]["source_verified_wins"], 0)
        self.assertEqual([row["admission"] for row in report["cases"]],
                         ["INVALID_TRIAL_RECEIPT", "INVALID_TRIAL_RECEIPT"])
        self.assertTrue(batch.verify_batch_receipt(report))

    def test_stolen_other_seed_or_stale_generation_receipt_is_not_counted(self):
        actual = single.run_trial(single.TrialConfig(0, 41, 1, 2., 3., 1024),
                                  single.AdapterSpec(fixture_responses=(WIN_JSON,)))
        self.assertTrue(actual["verified_success"])
        for seeds, expected_gen in (((1,), 41), ((0,), 42)):
            config = batch.BatchConfig(tuple(seeds), (expected_gen,),
                                       1, 1, 2., 4., 1024)
            with mock.patch.object(batch.single, "run_trial", return_value=actual):
                report = batch.run_batch(config,
                                         single.AdapterSpec(fixture_responses=(WIN_JSON,)))
            self.assertEqual(report["aggregate"]["source_verified_wins"], 0)
            self.assertEqual(report["cases"][0]["admission"], "INVALID_TRIAL_RECEIPT")
            self.assertTrue(batch.verify_batch_receipt(report))

    def test_forged_rehashed_100_percent_aggregate_is_rejected(self):
        report = batch.run_batch(_config(seeds=(0,), calls=1,
                                         total_calls=1, total_seconds=4),
                                 single.AdapterSpec())
        report["aggregate"]["source_verified_wins"] = 1
        report["aggregate"]["verified_fraction_planned"] = {
            "numerator": 1, "denominator": 1}
        _rehash(report)
        self.assertFalse(batch.verify_batch_receipt(report))

    def test_rehashed_resource_and_row_admission_spoofs_are_rejected(self):
        actual = batch.run_batch(_config(calls=1, total_calls=1,
                                         total_seconds=5), single.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(actual))
        for scope, field, value in (("aggregate", "calls_reserved", 999),
                                    ("aggregate", "skipped_or_invalid", 0),
                                    ("aggregate", "seconds_reserved", 999.),
                                    ("row", "admission", "COMPLETED"),
                                    ("row", "reserved_calls", 1)):
            forged = deepcopy(actual)
            target = (forged["aggregate"] if scope == "aggregate"
                      else forged["cases"][1])
            target[field] = value
            _rehash(forged)
            with self.subTest(scope=scope, field=field):
                self.assertFalse(batch.verify_batch_receipt(forged))

    def test_wall_timeout_skip_cannot_be_followed_by_admitted_case(self):
        original = batch.run_batch(_config(calls=1, total_calls=2,
                                            total_seconds=7), single.AdapterSpec())
        self.assertEqual([row["admission"] for row in original["cases"]],
                         ["COMPLETED", "COMPLETED"])
        forged = deepcopy(original)
        first = forged["cases"][0]
        first.update({"admission": "SKIPPED_GLOBAL_WALL_TIMEOUT",
                      "trial": None, "trial_receipt_valid": False,
                      "verified_win": False, "reserved_calls": 0,
                      "reserved_seconds": 0.0})
        first.pop("review_reason")
        forged["aggregate"]["receipts_valid"] = 1
        forged["aggregate"]["skipped_or_invalid"] = 1
        forged["aggregate"]["calls_reserved"] = 1
        forged["aggregate"]["calls_executed"] = 1
        forged["aggregate"]["seconds_reserved"] = 3.0
        _rehash(forged)
        self.assertFalse(batch.verify_batch_receipt(forged))

    def test_bool_int_aliases_rejected_in_manifest_case_or_aggregates(self):
        original = batch.run_batch(_config(seeds=(1,), calls=1,
                                           total_calls=1, total_seconds=4),
                                   single.AdapterSpec())
        for scope, field in (("row", "case_seed"),
                             ("row", "reserved_calls"),
                             ("aggregate", "precommitted_case_count"),
                             ("aggregate", "calls_reserved"),
                             ("aggregate", "receipts_valid")):
            forged = deepcopy(original)
            target = (forged["cases"][0] if scope == "row"
                      else forged["aggregate"])
            self.assertEqual(target[field], 1)
            target[field] = True
            _rehash(forged)
            with self.subTest(scope=scope, field=field):
                self.assertFalse(batch.verify_batch_receipt(forged))

    def test_double_rehashed_bad_manifest_type_is_fail_closed_not_exception(self):
        original = batch.run_batch(_config(seeds=(0,), calls=1,
                                           total_calls=1, total_seconds=4),
                                   single.AdapterSpec())
        for value in (None, "bad", [], 0):
            forged = deepcopy(original)
            forged["manifest"]["config"] = value
            forged["manifest_sha256"] = batch._sha(forged["manifest"])
            _rehash(forged)
            self.assertFalse(batch.verify_batch_receipt(forged))

    def test_rehashed_review_reason_or_unrecognized_claim_cannot_self_certify(self):
        actual = batch.run_batch(_config(seeds=(0,), calls=1,
                                         total_calls=1, total_seconds=4),
                                 single.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(actual))
        for attack in ("reason", "manifest_claim", "aggregate_extra",
                       "negative_wall"):
            forged = deepcopy(actual)
            if attack == "reason":
                forged["cases"][0]["review_reason"] = "FORMALLY_GUARANTEED_LEARNING"
            elif attack == "manifest_claim":
                forged["manifest"]["private_heldout_qualified"] = True
                forged["manifest_sha256"] = batch._sha(forged["manifest"])
            elif attack == "aggregate_extra":
                forged["aggregate"]["graduation_eligible"] = True
            else:
                forged["aggregate"]["wall_elapsed_ms"] = -1
            _rehash(forged)
            with self.subTest(attack=attack):
                self.assertFalse(batch.verify_batch_receipt(forged))

    def test_rehashed_source_neural_and_private_claims_cannot_qualify_batch(self):
        original = batch.run_batch(_config(seeds=(0,), calls=1,
                                           total_calls=1, total_seconds=4),
                                   single.AdapterSpec())
        for field in ("neural_learning_verified", "private_heldout_qualified",
                      "model_weight_update_verified", "comparison_qualified",
                      "generation_seed_effect_verified"):
            forged = deepcopy(original)
            forged[field] = True
            _rehash(forged)
            with self.subTest(field=field):
                self.assertFalse(batch.verify_batch_receipt(forged))

    def test_rehashed_child_stdout_digest_is_crosschecked(self):
        original = batch.run_batch(_config(seeds=(0,), calls=1,
                                           total_calls=1, total_seconds=4),
                                   single.AdapterSpec())
        self.assertTrue(batch.verify_batch_receipt(original))
        forged = deepcopy(original)
        trial = forged["cases"][0]["trial"]
        trial["turns"][0]["child"]["response_sha256"] = "0"*64
        trial["receipt_sha256"] = single._sha({
            key: value for key, value in trial.items() if key != "receipt_sha256"})
        _rehash(forged)
        self.assertFalse(batch.verify_batch_receipt(forged))

    def test_opted_in_real_local_command_can_submit_a_witness(self):
        code = ("import json,sys; d=json.load(sys.stdin);"
                "assert d['generation_seed'] in (41,42);"
                "assert 'case_seed' not in str(d);"
                "print(json.dumps({'kind':'PLAN','actions':"+repr(WIN)+"}))")
        spec = single.AdapterSpec(mode="command",
                                  command_argv=(sys.executable, "-c", code),
                                  authorize_command=True)
        report = batch.run_batch(_config(calls=1, total_calls=2,
                                         timeout=2, total_seconds=7), spec)
        self.assertTrue(batch.verify_batch_receipt(report))
        self.assertEqual([row["verified_win"] for row in report["cases"]],
                         [True, False])

    def test_local_command_without_explicit_authorization_is_blocked(self):
        with self.assertRaises(PermissionError):
            single.AdapterSpec(mode="command", command_argv=(sys.executable,))
        with self.assertRaises(PermissionError):
            single.AdapterSpec(mode="command", command_argv=("python",),
                               authorize_command=True)

    def test_partial_timeout_child_stays_in_denominator_and_never_wins(self):
        spec = single.AdapterSpec(mode="command",
                                  command_argv=(sys.executable, "-c",
                                                "import time;time.sleep(5)"),
                                  authorize_command=True)
        report = batch.run_batch(_config(calls=1, total_calls=2,
                                         timeout=.05, total_seconds=3), spec)
        self.assertEqual(report["aggregate"]["precommitted_case_count"], 2)
        self.assertEqual(report["aggregate"]["source_verified_wins"], 0)
        self.assertTrue(batch.verify_batch_receipt(report))
        self.assertFalse(report["generalization_qualified"])

    def test_invalid_candidate_json_remains_unknown_not_a_false_win(self):
        spec = single.AdapterSpec(fixture_responses=('not json',))
        report = batch.run_batch(_config(seeds=(0,), calls=1,
                                         total_calls=1, total_seconds=4), spec)
        self.assertEqual(report["aggregate"]["source_verified_wins"], 0)
        self.assertTrue(batch.verify_batch_receipt(report))

    def test_default_cli_redacts_raw_candidate_seed_and_argv(self):
        canary = "NOVEL_ROUTER_PRIVATE_CANDIDATE_DO_NOT_EXPORT"
        proposal = json.dumps({"kind": "NOVEL_RULE", "description": canary})
        output = io.StringIO()
        with redirect_stdout(output):
            status = batch.main(["--seeds", "0", "1", "--max-calls-per-case", "1",
                                 "--max-total-calls", "2",
                                 "--max-total-seconds", "7",
                                 "--fixture-json", proposal])
        self.assertEqual(status, 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["schema"],
                         "labyrinth.novel-router-batch-public-summary.v1")
        self.assertIs(summary["private_receipt_omitted"], True)
        for secret in (canary, "case_seed", "generation_seeds",
                       "command_argv", "raw_stdout_base64", "source_receipt",
                       "manifest"):
            self.assertNotIn(secret, output.getvalue())

    def test_private_opt_in_cli_emits_full_replayable_receipt(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = batch.main(["--seeds", "0", "--max-calls-per-case", "1",
                                 "--max-total-calls", "1",
                                 "--max-total-seconds", "4",
                                 "--emit-private-json-stdout"])
        self.assertEqual(status, 0)
        full = json.loads(output.getvalue())
        self.assertEqual(full["manifest"]["config"]["case_seeds"], [0])
        self.assertIn("raw_stdout_base64", full["cases"][0]["trial"]["turns"][0]["child"])
        self.assertTrue(batch.verify_batch_receipt(full))

    def test_cli_refuses_fake_100_percent_even_if_run_batch_claims_success(self):
        original = batch.run_batch(_config(seeds=(0,), calls=1,
                                           total_calls=1, total_seconds=4),
                                   single.AdapterSpec())
        forged = deepcopy(original)
        forged["aggregate"]["source_verified_wins"] = 1
        forged["aggregate"]["verified_fraction_planned"] = {
            "numerator": 1, "denominator": 1}
        _rehash(forged)
        self.assertFalse(batch.verify_batch_receipt(forged))
        for extra_args in ([], ["--emit-private-json-stdout"]):
            stdout = io.StringIO()
            with mock.patch.object(batch, "run_batch", return_value=forged):
                with redirect_stdout(stdout):
                    exit_code = batch.main(["--seeds", "0", *extra_args])
            self.assertEqual(exit_code, 2)
            emitted = json.loads(stdout.getvalue())
            self.assertEqual(emitted["status"], "EVALUATOR_DISAGREEMENT")
            self.assertTrue(emitted["source_score_withheld"])
            self.assertNotIn("source_verified_wins", emitted)
            self.assertNotIn("manifest", emitted)

    def test_no_seed_no_execution_cli(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                batch.main([])

    def test_bounded_runner_error_does_not_erase_precommitted_case(self):
        with mock.patch.object(batch.single, "run_trial", side_effect=RuntimeError("boom")):
            report = batch.run_batch(_config(seeds=(0, 1), calls=1,
                                             total_calls=2, total_seconds=7),
                                     single.AdapterSpec())
        self.assertEqual(len(report["cases"]), 2)
        self.assertEqual([row["admission"] for row in report["cases"]],
                         ["RUNNER_ERROR", "RUNNER_ERROR"])
        self.assertTrue(batch.verify_batch_receipt(report))


if __name__ == "__main__":
    unittest.main()
