"""Wave16 hostile process-trial provenance and child-admission tests.

Target clean source 16f84839df869d2a89243d55baa212cf92527944.
All process failures are injected in memory; no service/model/child launched.
"""

from __future__ import annotations

from hashlib import sha256
import json
import unittest
from unittest.mock import patch

from labyrinth.minotaur_process_adapter import (
    ProcessAdapterFailure, ProcessCallReceipt, ProcessModelAdapter,
)
from labyrinth.switchback_circuit import ProbeSession, make_case
from scripts import switchback_model_trial as trial


WINNING = json.dumps({"kind": "PLAN", "actions": ["0", "0", "WAIT", "0", "COMMIT"]})


def setup():
    adapter = ProcessModelAdapter(mode="fixed", generation_seed=0,
                                  timeout_seconds=1, max_output_bytes=1024)
    config = trial.TrialConfig(0, 0, "fixed", 1, 1, 3, 1024)
    return config, adapter


def synthetic_receipt(prompt, response, *, status="OK", request_hash=None,
                      response_hash=None):
    return ProcessCallReceipt(
        schema="labyrinth.minotaur-process.v1", status=status,
        child_pid=1234, generation_seed=0, evaluator_seed_discarded=True,
        request_sha256=(request_hash or sha256(prompt.encode()).hexdigest()),
        response_sha256=(response_hash or sha256(response.encode()).hexdigest()),
        response_bytes=len(response.encode()), parent_session_frame_visible=False,
        elapsed_ms=1, resource_admission="REAPED_SLOT_RELEASED",
    )


class Wave16Destroyer(unittest.TestCase):
    def test_verified_win_requires_a_real_correlated_process_receipt(self):
        config, adapter = setup()
        with patch.object(ProcessModelAdapter, "generate", return_value=WINNING):
            report = trial.run_trial(config, adapter)
        observed = (report["verified_success"],
                    report["turns"][0]["process_receipts"],
                    report["source_verdict"])
        self.assertEqual(observed, (False, [], "EVALUATOR_DISAGREEMENT"))

    def test_timeout_receipt_cannot_be_reinterpreted_as_child_win(self):
        config, adapter = setup()

        def false_success(_self, prompt, *, seed=None):
            adapter._receipts.append(synthetic_receipt(prompt, WINNING, status="TIMEOUT"))
            return WINNING

        with patch.object(ProcessModelAdapter, "generate", new=false_success):
            report = trial.run_trial(config, adapter)
        self.assertFalse(report["verified_success"])
        self.assertNotEqual(report["terminal_reason"], "SUBMITTED")

    def test_response_hash_mismatch_blocks_winning_receipt(self):
        config, adapter = setup()

        def false_success(_self, prompt, *, seed=None):
            adapter._receipts.append(synthetic_receipt(
                prompt, WINNING, response_hash="0" * 64))
            return WINNING

        with patch.object(ProcessModelAdapter, "generate", new=false_success):
            report = trial.run_trial(config, adapter)
        self.assertFalse(report["verified_success"])
        self.assertFalse(report["trial_binding_valid"])

    def test_private_fields_refused_before_candidate_transport(self):
        case = make_case(0)
        view = ProbeSession(case).view()
        for poisoned in (
                {**view, "seed": case.seed},
                {**view, "case_id": case.case_id},
                {**view, "mechanic_signature": case.mechanic_signature},
                {**view, "observations": [{"action": "WAIT", "valid": True,
                    "applied": True, "before": "000", "after": "000",
                    "private": {"seed": case.seed}}]},
        ):
            with self.subTest(keys=set(poisoned)):
                with self.assertRaises(ValueError):
                    trial._public_request(poisoned, 0)

    def test_repeated_timeout_without_receipts_never_retries_or_wins(self):
        config, adapter = setup()
        calls = []

        def timeout(_self, _prompt, *, seed=None):
            calls.append(seed)
            raise ProcessAdapterFailure("TIMEOUT")

        with patch.object(ProcessModelAdapter, "generate", new=timeout):
            report = trial.run_trial(config, adapter)
        self.assertEqual(calls, [None])
        self.assertEqual(report["terminal_reason"], "MODEL_TIMEOUT")
        self.assertFalse(report["verified_success"])
        self.assertEqual(len(report["turns"]), 1)

    def test_raw_wire_payload_cannot_smuggle_scoring_fields(self):
        public_actions = ["0", "1", "WAIT", "RESET"]
        for raw in (
                '{"kind":"PLAN","actions":["0"],"verified_success":true}',
                '{"kind":"ACTION","action":"0","case_id":"spoof"}',
                '{"kind":"CLAIM_UNSAT","source_commit":"forged"}',
                '{"kind":"PLAN","actions":["COMMIT"],"kind":"ABSTAIN"}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    trial._independent_wire_decision(raw, public_actions, max_bytes=1024)


if __name__ == "__main__":
    unittest.main()
