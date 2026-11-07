"""Wave15: real child-process transport, synthetic no-learning source controls."""

from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

from labyrinth.minotaur_process_adapter import ProcessModelAdapter
from labyrinth.switchback_circuit import ProbeSession, make_case
from labyrinth.switchback_oracle import verify_receipt_replay
from labyrinth.switchback_oracle import search
from scripts import switchback_model_trial as trial


WINNING_PLAN = ["0", "0", "WAIT", "0", "COMMIT"]
COMMAND_WIN = ("import json,sys; r=json.load(sys.stdin); "
               "print(json.dumps({'kind':'PLAN','actions':['0','0','WAIT','0','COMMIT']}))")


def _command(source: str, *, timeout: float = 1.0,
             output: int = 1024) -> trial.CommandProcessAdapter:
    return trial.CommandProcessAdapter((sys.executable, "-c", source),
                                       timeout_seconds=timeout,
                                       max_output_bytes=output,
                                       authorize_command=True)


def _config(*, mode: str = "command", calls: int = 1,
            timeout: float = 1., total: float = 3., output: int = 1024,
            seed: int = 0) -> trial.TrialConfig:
    return trial.TrialConfig(seed, 0, mode, calls, timeout, total, output)


class SwitchbackProcessTrialTests(unittest.TestCase):
    def test_actual_command_child_can_propose_real_witness_without_learning_claim(self):
        report = trial.run_trial(_config(), _command(COMMAND_WIN))
        self.assertEqual(report["terminal_reason"], "SUBMITTED")
        self.assertEqual(report["source_verdict"], "VERIFIED_WIN")
        self.assertTrue(report["verified_success"])
        self.assertTrue(report["source_receipt_replay_valid"])
        self.assertTrue(verify_receipt_replay(report["source_receipt"]))
        self.assertFalse(report["model_neural_identity_verified"])
        self.assertFalse(report["model_weight_update_verified"])
        self.assertFalse(report["generalization_qualified"])
        self.assertEqual(len(report["turns"]), 1)
        self.assertEqual(report["turns"][0]["process_receipts"][0]["status"], "OK")
        self.assertEqual(report["process_calls"], 1)
        self.assertTrue(report["executable"]["sha256"])
        self.assertEqual(report["receipt_sha256"], sha256(trial._canonical({
            key: value for key, value in report.items() if key != "receipt_sha256"
        })).hexdigest())

    def test_public_request_only_exposes_candidate_view_even_after_probe(self):
        session = ProbeSession(make_case(0))
        first = json.loads(trial._public_request(session.view(), 0))
        self.assertEqual(first["schema"], trial.PUBLIC_SCHEMA)
        self.assertNotIn("seed", first)
        self.assertNotIn("case_id", first["view"])
        self.assertNotIn("mechanic_signature", first["view"])
        session.probe("0")
        later = json.loads(trial._public_request(session.view(), 1))
        self.assertEqual(later["view"]["observations"][0]["action"], "0")
        self.assertIn("before", later["view"]["observations"][0])
        self.assertIn("after", later["view"]["observations"][0])
        with self.assertRaises(ValueError):
            trial._public_request({**session.view(), "case_id": "PRIVATE"}, 1)
        with self.assertRaises(ValueError):
            trial._public_request({**session.view(), "case_seed": 0}, 1)
        with self.assertRaises(ValueError):
            trial._public_request({**session.view(), "before": {"family": 1}}, 1)

    def test_existing_fixed_process_adapter_charges_probes_then_abstains(self):
        adapter = ProcessModelAdapter(mode="fixed", actions=("0", "RESET"),
                                      generation_seed=0, timeout_seconds=1,
                                      max_output_bytes=1024)
        report = trial.run_trial(_config(mode="fixed", calls=3, total=4), adapter)
        self.assertEqual(report["terminal_reason"], "SUBMITTED")
        self.assertEqual(report["source_verdict"], "UNKNOWN")
        self.assertFalse(report["verified_success"])
        self.assertEqual(report["source_receipt"]["probe_count"], 2)
        self.assertTrue(verify_receipt_replay(report["source_receipt"]))
        self.assertEqual([t["process_receipts"][0]["status"] for t in report["turns"]],
                         ["OK", "OK", "OK"])
        self.assertEqual(report["mode"], "fixed")

    def test_bounded_command_invalid_json_is_terminal_without_false_win(self):
        report = trial.run_trial(_config(), _command("print('not JSON')"))
        self.assertEqual(report["terminal_reason"], "INVALID_MODEL_OUTPUT")
        self.assertFalse(report["verified_success"])
        self.assertTrue(report["source_receipt_replay_valid"])
        self.assertEqual(report["source_verdict"], "UNKNOWN")

    def test_duplicate_keys_are_illegal_even_if_one_proposal_is_valid(self):
        script = 'print(\'{"kind":"PLAN","kind":"ABSTAIN","actions":["COMMIT"]}\')'
        report = trial.run_trial(_config(), _command(script))
        self.assertEqual(report["terminal_reason"], "INVALID_MODEL_OUTPUT")
        self.assertFalse(report["verified_success"])

    def test_illegal_public_action_is_terminal_not_a_free_retry(self):
        report = trial.run_trial(_config(), _command(
            "print('{\"kind\":\"ACTION\",\"action\":\"SECRET\"}')"))
        self.assertEqual(report["terminal_reason"], "INVALID_MODEL_OUTPUT")
        self.assertEqual(len(report["turns"]), 1)
        self.assertFalse(report["verified_success"])

    def test_process_timeout_fails_closed_and_child_is_reported(self):
        source = "import time; time.sleep(2); print('late')"
        report = trial.run_trial(_config(timeout=.05, total=.2),
                                 _command(source, timeout=.05))
        self.assertEqual(report["terminal_reason"], "MODEL_TIMEOUT")
        self.assertFalse(report["verified_success"])
        self.assertEqual(report["turns"][0]["process_receipts"][0]["status"], "TIMEOUT")

    def test_stdout_overflow_is_capped_and_cannot_claim_win(self):
        source = "import sys;sys.stdout.write('X'*100000)"
        report = trial.run_trial(_config(output=512), _command(source, output=512))
        self.assertEqual(report["terminal_reason"], "MODEL_OVERSIZE")
        self.assertFalse(report["verified_success"])
        self.assertLessEqual(report["turns"][0]["process_receipts"][0][
            "response_bytes"], 512)

    def test_proposal_shape_nesting_and_python_bool_spoofs_rejected(self):
        for text in ('{"kind":"PLAN","actions":["COMMIT",true]}',
                     '{"kind":"PLAN","actions":["COMMIT"],"verified_success":true}',
                     '{"kind":"ACTION","action":false}',
                     '{"kind":"CLAIM_UNSAT","complete":true}'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    trial._parse_response(text, ["0", "1", "WAIT", "RESET"])

    def test_config_is_frozen_and_mode_seed_contract_enforced(self):
        config = _config()
        with self.assertRaises(FrozenInstanceError):
            config.case_seed = 9
        with self.assertRaises(ValueError):
            trial.TrialConfig(True)
        with self.assertRaises(ValueError):
            trial.TrialConfig(0, False)
        with self.assertRaises(ValueError):
            trial.TrialConfig(0, max_model_calls=18)
        with self.assertRaises(ValueError):
            trial.TrialConfig(0, max_model_calls=2, call_timeout_seconds=2,
                              total_timeout_seconds=3)
        with self.assertRaises(ValueError):
            trial.run_trial(_config(mode="fixed"), _command(COMMAND_WIN))

    def test_command_requires_explicit_authorization_and_absolute_executable(self):
        with self.assertRaises(PermissionError):
            trial.CommandProcessAdapter((sys.executable, "-c", COMMAND_WIN))
        with self.assertRaises(ValueError):
            trial.CommandProcessAdapter(("python3", "-c", COMMAND_WIN),
                                        authorize_command=True)
        with self.assertRaises(PermissionError):
            ProcessModelAdapter(mode="ollama", model="unverified", authorize_local_model=False)

    def test_model_file_hash_is_direct_file_read_not_model_identity_claim(self):
        path = Path(trial.__file__)
        report = trial.run_trial(_config(), _command(COMMAND_WIN), model_artifact_path=path)
        self.assertEqual(report["model_artifact"]["sha256"],
                         sha256(path.read_bytes()).hexdigest())
        self.assertFalse(report["model_neural_identity_verified"])
        self.assertEqual(report["model_artifact"]["bytes"], path.stat().st_size)

    def test_genuine_wrong_case_winning_receipt_cannot_launder_abstention(self):
        wrong_case = make_case(0)
        true_win = ProbeSession(wrong_case).submit({"kind": "PLAN", "actions": search(
            wrong_case)["witness"]})
        self.assertTrue(verify_receipt_replay(true_win))
        adapter = ProcessModelAdapter(mode="fixed", actions=(), generation_seed=0,
                                      timeout_seconds=1, max_output_bytes=1024)
        with mock.patch.object(ProbeSession, "submit", return_value=true_win):
            result = trial.run_trial(_config(mode="fixed", seed=1), adapter)
        self.assertEqual(result["terminal_reason"], "EVALUATOR_DISAGREEMENT")
        self.assertTrue(result["source_receipt_replay_valid"])
        self.assertFalse(result["trial_binding_valid"])
        self.assertFalse(result["verified_success"])
        self.assertEqual(result["source_verdict"], "EVALUATOR_DISAGREEMENT")
        self.assertEqual(result["turns"][0]["decision"], {"kind": "ABSTAIN"})

    def test_same_case_winning_receipt_cannot_launder_abstention(self):
        case = make_case(0)
        true_win = ProbeSession(case).submit({"kind": "PLAN", "actions": search(
            case)["witness"]})
        adapter = ProcessModelAdapter(mode="fixed", actions=(), generation_seed=0,
                                      timeout_seconds=1, max_output_bytes=1024)
        with mock.patch.object(ProbeSession, "submit", return_value=true_win):
            result = trial.run_trial(_config(mode="fixed"), adapter)
        self.assertTrue(result["source_receipt_replay_valid"])
        self.assertFalse(result["trial_binding_valid"])
        self.assertFalse(result["verified_success"])
        self.assertEqual(result["terminal_reason"], "EVALUATOR_DISAGREEMENT")

    def test_same_case_same_plan_but_missing_actual_probe_trace_is_not_credit(self):
        case = make_case(0)
        no_probe_win = ProbeSession(case).submit({"kind": "PLAN", "actions": WINNING_PLAN})
        source = ("import json,sys;r=json.load(sys.stdin); "
                  "print(json.dumps({'kind':'ACTION','action':'0'} if r['turn']==0 "
                  "else {'kind':'PLAN','actions':['0','0','WAIT','0','COMMIT']}))")
        adapter = _command(source)
        with mock.patch.object(ProbeSession, "submit", return_value=no_probe_win):
            result = trial.run_trial(_config(calls=2, total=3), adapter)
        self.assertEqual([t["decision"]["kind"] for t in result["turns"]],
                         ["ACTION", "PLAN"])
        self.assertTrue(result["source_receipt_replay_valid"])
        self.assertFalse(result["trial_binding_valid"])
        self.assertFalse(result["verified_success"])

    def test_single_modified_decision_parser_cannot_turn_raw_abstain_into_win(self):
        adapter = ProcessModelAdapter(mode="fixed", actions=(), generation_seed=0,
                                      timeout_seconds=1, max_output_bytes=1024)
        made_up = {"kind": "PLAN", "actions": WINNING_PLAN}
        with mock.patch.object(trial, "_parse_response", return_value=made_up):
            report = trial.run_trial(_config(mode="fixed"), adapter)
        self.assertEqual(report["terminal_reason"], "INVALID_MODEL_OUTPUT")
        self.assertFalse(report["verified_success"])
        self.assertTrue(report["trial_binding_valid"])
        self.assertEqual(report["source_verdict"], "UNKNOWN")
        self.assertIsNone(report["turns"][0]["decision"])

    def test_cli_fixed_mode_emits_single_receipt_no_model_install(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = trial.main(["--seed", "0", "--mode", "fixed", "--max-calls", "1",
                               "--fixed-actions", "0"])
        self.assertEqual(code, 2)
        parsed = json.loads(stdout.getvalue())
        self.assertEqual(parsed["schema"], trial.SCHEMA)
        self.assertFalse(parsed["verified_success"])
        self.assertEqual(parsed["mode"], "fixed")


if __name__ == "__main__":
    unittest.main()
