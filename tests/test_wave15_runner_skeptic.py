"""Independent Wave15 subprocess runner adversarial controls.

Frozen PRE-FIX author source: 9629bc44a7dc9c13c5d1822a6475c677ce460815.
Single host-function fault injections expose missing model-output and case
binding. The actual candidate remains an unchanged FIXED child process.
"""

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.minotaur_process_adapter import ProcessModelAdapter, _ABSTAIN
from labyrinth.switchback_circuit import ProbeSession, make_case
from labyrinth.switchback_oracle import search, verify_receipt_replay
from scripts import switchback_model_trial as trial


class SwitchbackRunnerSkeptic(unittest.TestCase):
    def _fixed(self, actions=()):
        return ProcessModelAdapter(
            mode="fixed", actions=actions, generation_seed=0,
            timeout_seconds=1.0, max_output_bytes=1024)

    def _config(self, seed=0, calls=1):
        return trial.TrialConfig(
            case_seed=seed, generation_seed=0, mode="fixed",
            max_model_calls=calls, call_timeout_seconds=1.0,
            total_timeout_seconds=calls + 2.0, max_output_bytes=1024)

    def _saved_win(self, seed=0):
        case = make_case(seed)
        witness = search(case)["witness"]
        self.assertIsNotNone(witness)
        receipt = ProbeSession(case).submit({"kind": "PLAN", "actions": witness})
        self.assertTrue(verify_receipt_replay(receipt))
        self.assertTrue(receipt["verified_success"])
        return receipt

    def test_other_case_replayable_win_cannot_credit_abstaining_child(self):
        stolen = self._saved_win(seed=0)
        with patch.object(ProbeSession, "submit", return_value=stolen):
            report = trial.run_trial(self._config(seed=1), self._fixed())
        self.assertEqual(report["turns"][0]["decision"], {"kind": "ABSTAIN"})
        self.assertNotEqual(report["source_receipt"]["seed"],
                            report["case_seed_host_only"])
        self.assertFalse(report["verified_success"],
                         "a genuine seed0 receipt credited a seed1 abstaining model")

    def test_same_case_stale_win_cannot_credit_abstaining_child(self):
        stolen = self._saved_win(seed=0)
        with patch.object(ProbeSession, "submit", return_value=stolen):
            report = trial.run_trial(self._config(seed=0), self._fixed())
        self.assertEqual(report["turns"][0]["decision"], {"kind": "ABSTAIN"})
        self.assertEqual(report["source_receipt"]["seed"], report["case_seed_host_only"])
        self.assertFalse(report["verified_success"],
                         "model ABSTAIN was silently replaced by stale winning PLAN")

    def test_stale_zero_probe_receipt_cannot_credit_probed_session(self):
        stolen = self._saved_win(seed=0)
        with patch.object(ProbeSession, "submit", return_value=stolen):
            report = trial.run_trial(self._config(seed=0, calls=2),
                                     self._fixed(actions=("0",)))
        self.assertEqual([turn["decision"]["kind"] for turn in report["turns"]],
                         ["ACTION", "ABSTAIN"])
        self.assertEqual(report["source_receipt"]["probe_count"], 0)
        self.assertFalse(report["verified_success"],
                         "unrelated zero-probe receipt replaced charged probe history")

    def test_model_action_must_equal_charged_probe_action(self):
        """One changed host probe action cannot credit a model's different trace."""
        path = search(make_case(0))["witness"]
        script = (
            "import json,sys; request=json.load(sys.stdin); "
            "turn=request['turn']; "
            f"answer={{'kind':'PLAN','actions':{path!r}}}; "
            "print(json.dumps({'kind':'ACTION','action':'0'} if turn==0 else answer))"
        )
        adapter = trial.CommandProcessAdapter(
            (sys.executable, "-c", script), generation_seed=0,
            timeout_seconds=1.0, max_output_bytes=1024, authorize_command=True)
        config = trial.TrialConfig(0, 0, "command", 2, 1.0, 3.0, 1024)
        actual_probe = ProbeSession.probe

        def silently_replace_action(self, action):
            return actual_probe(self, "WAIT" if action == "0" else action)

        with patch.object(ProbeSession, "probe", new=silently_replace_action):
            report = trial.run_trial(config, adapter)
        self.assertEqual([turn["decision"]["kind"] for turn in report["turns"]],
                         ["ACTION", "PLAN"])
        self.assertEqual(report["turns"][0]["decision"]["action"], "0")
        self.assertEqual(report["source_receipt"]["probe_trace"][0]["action"], "WAIT")
        self.assertFalse(report["verified_success"],
                         "model ACTION 0 credited against evaluator's WAIT trace")

    def test_single_corrupted_parser_cannot_promote_raw_abstain_to_plan(self):
        path = search(make_case(0))["witness"]
        with patch.object(trial, "_parse_response", return_value={
                "kind": "PLAN", "actions": path}):
            report = trial.run_trial(self._config(seed=0), self._fixed())
        self.assertEqual(report["turns"][0]["response_sha256"],
                         sha256(_ABSTAIN.encode("utf-8")).hexdigest())
        # PRE-FIX the host accepted the corrupt PLAN as the model decision.
        # A fail-closed correction may intentionally discard that decision.
        self.assertFalse(report["verified_success"],
                         "a single parser substitution scored an actual ABSTAIN as WIN")

    def test_strict_duplicate_nonfinite_and_multidoc_json_refusal(self):
        malformed = (
            '{"kind":"PLAN","kind":"ABSTAIN","actions":["COMMIT"]}',
            '{"kind":"PLAN","actions":["COMMIT"]}\n{"kind":"ABSTAIN"}',
            '{"kind":"ACTION","action":"0","x":NaN}',
            '{"kind":"ACTION","action":"0","x":1e999}',
            '{"kind":"PLAN","actions":["COMMIT",true]}',
        )
        for raw in malformed:
            with self.subTest(raw=raw):
                with self.assertRaises((ValueError, TypeError)):
                    trial._parse_response(raw, ["0", "1", "WAIT", "RESET"])

    def test_missing_absolute_executable_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            nonexistent = str(Path(folder) / "no-model")
            with self.assertRaises(ValueError):
                trial.CommandProcessAdapter(
                    (nonexistent,), authorize_command=True)

    def test_executable_hash_does_not_bind_unhashed_python_script(self):
        """Negative control: host provenance is intentionally unqualified."""
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "candidate.py"
            source.write_text('print(\'{"kind":"ABSTAIN"}\')\n', encoding="utf-8")
            argv = (sys.executable, str(source))
            first = trial.CommandProcessAdapter(
                argv, timeout_seconds=1.0, max_output_bytes=1024,
                authorize_command=True)
            first_hash = first.executable_hash["sha256"]
            first_argv = first.argv_sha256
            source.write_text('print(\'{"kind":"CLAIM_UNSAT"}\')\n', encoding="utf-8")
            second = trial.CommandProcessAdapter(
                argv, timeout_seconds=1.0, max_output_bytes=1024,
                authorize_command=True)
            self.assertEqual(first_hash, second.executable_hash["sha256"])
            self.assertEqual(first_argv, second.argv_sha256)
            report = trial.run_trial(
                trial.TrialConfig(0, 0, "command", 1, 1.0, 3.0, 1024),
                second)
            self.assertFalse(report["model_neural_identity_verified"])
            self.assertFalse(report["model_weight_update_verified"])
            self.assertEqual(report["turns"][0]["decision"],
                             {"kind": "CLAIM_UNSAT"})


if __name__ == "__main__":
    unittest.main()
