"""Hostile source-only fixture and subprocess controls for Novel Router Wave17."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
import sys
import unittest
from unittest import mock

from labyrinth import switchback_novel_transfer as novel
from scripts import novel_router_model_trial as runner


WIN = novel.search(novel.make_case(0))["witness"]
WIN_JSON = json.dumps({"kind": "PLAN", "actions": WIN})
ABSTAIN = '{"kind":"ABSTAIN"}'


def _config(seed=0, calls=2, timeout=2., total=6., output=4096):
    return runner.TrialConfig(seed, 73, calls, timeout, total, output)


def _spec(*responses):
    return runner.AdapterSpec(fixture_responses=tuple(responses) if responses else (ABSTAIN,))


def _rehash(value):
    value["receipt_sha256"] = runner._sha({
        key: item for key, item in value.items() if key != "receipt_sha256"})


class NovelRouterModelTrialTests(unittest.TestCase):
    def test_fixture_child_actually_executes_and_abstains_with_replay_receipt(self):
        result = runner.run_trial(_config(), _spec())
        self.assertEqual(result["terminal_reason"], "SUBMITTED")
        self.assertEqual(result["source_verdict"], "UNKNOWN")
        self.assertFalse(result["verified_success"])
        self.assertTrue(result["source_receipt_replay_valid"])
        self.assertTrue(runner.verify_trial_receipt(result))
        self.assertIsInstance(result["turns"][0]["child"]["pid"], int)
        self.assertGreater(result["turns"][0]["child"]["pid"], 0)
        for field in ("model_weight_update_verified", "private_custody",
                      "neural_learning_verified", "graduation_eligible",
                      "generalization_qualified", "model_identity_qualified"):
            self.assertIs(result[field], False)

    def test_exact_winning_fixture_plan_source_verifies_without_learning_claim(self):
        self.assertEqual(novel.verify_plan(novel.make_case(0), WIN)["verdict"],
                         "VERIFIED_WIN")
        result = runner.run_trial(_config(calls=1, total=2), _spec(WIN_JSON))
        self.assertEqual(result["terminal_reason"], "SUBMITTED")
        self.assertEqual(result["source_verdict"], "VERIFIED_WIN")
        self.assertTrue(result["verified_success"])
        self.assertTrue(runner.verify_trial_receipt(result))
        self.assertFalse(result["neural_learning_verified"])

    def test_valid_other_solution_not_rejected_for_diverging_from_shortest_witness(self):
        longer = ["TICK"] * 7 + ["ROUTE", "LOAD", "TICK", "COMMIT"]
        self.assertEqual(novel.verify_plan(novel.make_case(0), longer)["verdict"],
                         "VERIFIED_WIN")
        outcome = runner.run_trial(_config(calls=1, total=2),
                                   _spec(json.dumps({"kind":"PLAN","actions":longer})))
        self.assertTrue(outcome["verified_success"])
        self.assertTrue(runner.verify_trial_receipt(outcome))

    def test_probe_and_then_final_claim_binds_observation_transcript(self):
        result = runner.run_trial(_config(calls=2), _spec(
            '{"kind":"ACTION","action":"LOAD"}', WIN_JSON))
        self.assertEqual(result["process_calls"], 2)
        self.assertEqual(result["source_receipt"]["probe_count"], 1)
        self.assertEqual(result["source_receipt"]["probe_trace"][0]["action"], "LOAD")
        self.assertTrue(runner.verify_trial_receipt(result))

    def test_source_known_plan_can_fail_structurally_different_seed(self):
        other_seed = next(seed for seed in range(1, 30)
                          if novel.verify_plan(novel.make_case(seed), WIN)["verdict"] !=
                          "VERIFIED_WIN")
        outcome = runner.run_trial(_config(seed=other_seed, calls=1, total=2),
                                   _spec(WIN_JSON))
        self.assertFalse(outcome["verified_success"])
        self.assertTrue(runner.verify_trial_receipt(outcome))

    def test_duplicate_keys_and_trailing_json_are_terminal_invalid_output(self):
        for bad in ('{"kind":"ABSTAIN","kind":"PLAN"}',
                    '{"kind":"ABSTAIN"}{"kind":"PLAN"}',
                    'not json', '{"kind":"ACTION","action":"BOGUS"}',
                    '{"kind":"PLAN","actions":["COMMIT"],"x":1}'):
            with self.subTest(bad=bad):
                result = runner.run_trial(_config(calls=1, total=2), _spec(bad))
                self.assertEqual(result["terminal_reason"], "INVALID_MODEL_OUTPUT")
                self.assertFalse(result["verified_success"])
                self.assertTrue(runner.verify_trial_receipt(result))

    def test_frozen_limits_reject_bool_seed_and_excessive_calls(self):
        with self.assertRaises(ValueError):
            runner.TrialConfig(True)
        with self.assertRaises(ValueError):
            runner.TrialConfig(0, max_calls=22)
        with self.assertRaises(ValueError):
            runner.TrialConfig(0, max_calls=3, total_timeout_seconds=1)
        with self.assertRaises(ValueError):
            runner.TrialConfig(0, max_output_bytes=1)

    def test_explicit_command_is_opt_in_and_absolute(self):
        with self.assertRaises(PermissionError):
            runner.AdapterSpec(mode="command", command_argv=(sys.executable, "-V"))
        with self.assertRaises(PermissionError):
            runner.AdapterSpec(mode="command", command_argv=("python", "-V"),
                               authorize_command=True)
        with self.assertRaises(ValueError):
            runner.AdapterSpec(mode="fixture", command_argv=(sys.executable,))

    def test_real_explicit_local_command_receives_public_only_prompt(self):
        code = ("import json,sys; p=json.load(sys.stdin);"
                "assert p['schema']=='labyrinth.novel-router-public-turn.v1';"
                "assert p['generation_seed']==73;"
                "assert 'seed' not in p['view'];"
                "assert 'case_seed' not in str(p);"
                "assert 'case_id' not in str(p);"
                "assert 'mechanic_signature' not in str(p);"
                "print(json.dumps({'kind':'ABSTAIN'}))")
        spec = runner.AdapterSpec(mode="command", command_argv=(sys.executable,
                                   "-c", code), authorize_command=True)
        result = runner.run_trial(_config(calls=1, total=2), spec)
        self.assertEqual(result["terminal_reason"], "SUBMITTED")
        self.assertTrue(runner.verify_trial_receipt(result))
        self.assertFalse(result["verified_success"])

    def test_distinct_generation_seeds_reach_real_child_and_bind_raw_request(self):
        code = ("import sys,json; data=json.load(sys.stdin);"
                "assert data['generation_seed'] in (73,74);"
                "print(json.dumps({'kind':'ABSTAIN'}))")
        spec = runner.AdapterSpec(mode="command", command_argv=(sys.executable,
                                  "-c", code), authorize_command=True)
        first = runner.run_trial(_config(calls=1, total=2), spec)
        second = runner.run_trial(runner.TrialConfig(0, 74, 1, 2., 2., 4096), spec)
        self.assertTrue(runner.verify_trial_receipt(first))
        self.assertTrue(runner.verify_trial_receipt(second))
        self.assertNotEqual(first["turns"][0]["request_sha256"],
                            second["turns"][0]["request_sha256"])
        forged = deepcopy(first)
        forged["configuration"]["generation_seed"] = 74
        forged["configuration_sha256"] = runner._sha(forged["configuration"])
        _rehash(forged)
        self.assertFalse(runner.verify_trial_receipt(forged))

    def test_real_explicit_command_can_propose_verified_plan(self):
        code = ("import sys,json;json.load(sys.stdin);"
                "print(json.dumps({\"kind\":\"PLAN\",\"actions\":"+
                repr(WIN)+"}))")
        spec = runner.AdapterSpec(mode="command", command_argv=(sys.executable,
                                   "-c", code), authorize_command=True)
        result = runner.run_trial(_config(calls=1, total=2), spec)
        self.assertTrue(result["verified_success"])
        self.assertTrue(runner.verify_trial_receipt(result))

    def test_subprocess_timeout_is_bounded_and_not_a_false_win(self):
        spec = runner.AdapterSpec(mode="command", command_argv=(sys.executable,
                                   "-c", "import time; time.sleep(5)"),
                                  authorize_command=True)
        result = runner.run_trial(_config(calls=1, timeout=.05, total=1), spec)
        self.assertFalse(result["verified_success"])
        self.assertEqual(result["turns"][0]["child"]["status"], "TIMEOUT")
        self.assertTrue(runner.verify_trial_receipt(result))

    def test_oversize_output_is_terminal_and_never_accepted(self):
        spec = runner.AdapterSpec(mode="command", command_argv=(sys.executable,
                                   "-c", "print('X'*100000)"),
                                  authorize_command=True)
        result = runner.run_trial(_config(calls=1, timeout=1, total=2, output=128), spec)
        self.assertFalse(result["verified_success"])
        self.assertEqual(result["turns"][0]["child"]["status"], "OVERSIZE")
        self.assertTrue(runner.verify_trial_receipt(result))

    def test_repeated_probe_script_exhausts_calls_without_unearned_win(self):
        result = runner.run_trial(_config(calls=3, timeout=2, total=7),
                                  _spec(*(['{"kind":"ACTION","action":"TICK"}']*3)))
        self.assertEqual(result["terminal_reason"], "CALL_BUDGET_EXHAUSTED")
        self.assertEqual(result["process_calls"], 3)
        self.assertEqual(result["source_receipt"]["probe_count"], 3)
        self.assertFalse(result["verified_success"])
        self.assertTrue(runner.verify_trial_receipt(result))

    def test_rehashed_wire_response_sha_cannot_fake_same_child_bytes(self):
        outcome = runner.run_trial(_config(calls=1, total=2), _spec(WIN_JSON))
        self.assertTrue(runner.verify_trial_receipt(outcome))
        forged = deepcopy(outcome)
        forged["turns"][0]["child"]["response_sha256"] = "0"*64
        _rehash(forged)
        self.assertFalse(runner.verify_trial_receipt(forged))

    def test_rehashed_terminal_or_child_status_cannot_rewrite_causal_result(self):
        outcome = runner.run_trial(_config(calls=1, total=2), _spec(WIN_JSON))
        self.assertTrue(runner.verify_trial_receipt(outcome))
        for mutation in ("terminal", "child_status", "turn_status"):
            forged = deepcopy(outcome)
            if mutation == "terminal":
                forged["terminal_reason"] = "TRIAL_TIMEOUT"
                forged["verified_success"] = False
            elif mutation == "child_status":
                forged["turns"][0]["child"]["status"] = "TIMEOUT"
                forged["verified_success"] = False
            else:
                forged["turns"][0]["status"] = "MODEL_TIMEOUT"
            _rehash(forged)
            with self.subTest(mutation=mutation):
                self.assertFalse(runner.verify_trial_receipt(forged))

    def test_wrong_case_seed_with_rehashed_trial_cannot_reuse_valid_win(self):
        outcome = runner.run_trial(_config(calls=1, total=2), _spec(WIN_JSON))
        forged = deepcopy(outcome)
        forged["configuration"]["case_seed"] = 1
        forged["configuration_sha256"] = runner._sha(forged["configuration"])
        _rehash(forged)
        self.assertFalse(runner.verify_trial_receipt(forged))

    def test_rehashed_metadata_type_and_artifact_spoofs_fail_closed(self):
        authentic = runner.run_trial(_config(calls=1, total=2), _spec(ABSTAIN))
        self.assertTrue(runner.verify_trial_receipt(authentic))
        for mutation in ("source_commit", "executable", "source_files", "model_artifact"):
            fake = deepcopy(authentic)
            if mutation == "source_commit":
                fake["source_commit"] = "UNBOUND"
            elif mutation == "executable":
                fake["executable"] = "some-executable"
            elif mutation == "source_files":
                fake["source_files"]["runner"] = "not-a-hash-receipt"
            else:
                fake["model_artifact"] = {"bytes": 0, "sha256": "f" * 64}
            _rehash(fake)
            with self.subTest(mutation=mutation):
                self.assertFalse(runner.verify_trial_receipt(fake))

    def test_parser_only_mutation_cannot_turn_child_abstain_into_win(self):
        with mock.patch.object(runner, "_parse_decision", return_value={
                "kind": "PLAN", "actions": WIN}):
            outcome = runner.run_trial(_config(calls=1, total=2), _spec(ABSTAIN))
        self.assertFalse(outcome["verified_success"])
        self.assertFalse(outcome["source_receipt_replay_valid"])

    def test_shared_primary_grammar_mutation_cannot_certify_child_abstain_as_plan(self):
        # A subtler shared-scorer mutant than patching only the top parser:
        # both parser wrappers previously called one _valid_decision helper.
        with mock.patch.object(runner, "_valid_decision", return_value={
                "kind": "PLAN", "actions": WIN}):
            outcome = runner.run_trial(_config(calls=1, total=2), _spec(ABSTAIN))
        self.assertFalse(outcome["verified_success"])
        self.assertFalse(outcome["source_receipt_replay_valid"])

    def test_unauthorized_or_malformed_submitted_source_cannot_certify_win(self):
        valid = runner.run_trial(_config(seed=1, calls=1, total=2), _spec(ABSTAIN))
        stolen = deepcopy(valid["source_receipt"])
        with mock.patch.object(novel.ProbeSession, "submit", return_value=stolen):
            outcome = runner.run_trial(_config(calls=1, total=2), _spec(WIN_JSON))
        self.assertFalse(outcome["verified_success"])
        self.assertFalse(outcome["source_receipt_replay_valid"])

    def test_cli_requires_explicit_seed_and_serializes_replayable_fixture(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main([])
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = runner.main(["--seed", "0", "--max-calls", "1",
                                "--total-timeout", "2", "--fixture-json", ABSTAIN])
        self.assertEqual(code, 0)
        summary = json.loads(stream.getvalue())
        self.assertEqual(summary["schema"],
                         "labyrinth.novel-router-local-trial-summary.v1")
        self.assertIs(summary["private_receipt_omitted"], True)
        self.assertNotIn("configuration", summary)
        self.assertNotIn("adapter", summary)
        self.assertNotIn("source_receipt", summary)
        self.assertNotIn("case_seed", stream.getvalue())
        self.assertNotIn("raw_stdout_base64", stream.getvalue())
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = runner.main(["--seed", "0", "--max-calls", "1",
                                "--total-timeout", "2",
                                "--emit-private-json-stdout", "--fixture-json", ABSTAIN])
        self.assertEqual(code, 0)
        report = json.loads(stream.getvalue())
        self.assertTrue(runner.verify_trial_receipt(report))
        self.assertEqual(report["configuration"]["case_seed"], 0)
        self.assertIn("raw_stdout_base64", report["turns"][0]["child"])

    def test_default_cli_hides_raw_novel_candidate_and_command_arguments(self):
        canary = "WAVE17_PRIVATE_CANDIDATE_CONTENT_DO_NOT_EXPORT"
        candidate = json.dumps({"kind": "NOVEL_RULE", "description": canary})
        stream = io.StringIO()
        with redirect_stdout(stream):
            runner.main(["--seed", "5", "--max-calls", "1",
                         "--total-timeout", "2", "--fixture-json", candidate])
        default = stream.getvalue()
        self.assertNotIn(canary, default)
        self.assertNotIn("fixture_responses", default)
        self.assertNotIn("model_artifact", default)
        self.assertNotIn("raw_stdout_base64", default)


if __name__ == "__main__":
    unittest.main()
