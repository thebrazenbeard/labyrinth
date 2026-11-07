"""Process-separated candidate seam: boundary, failures, and evidence limits."""

from __future__ import annotations

import json
import inspect
import os
import unittest

from labyrinth.minotaur_candidate_bridge import audit_receipt, run_candidate
from labyrinth.minotaur_process_adapter import ProcessAdapterFailure, ProcessModelAdapter
from labyrinth.timegate_exam import CandidateSession, candidate_view, make_case


WINNING_SEED_ZERO = ("forward", "take", "forward", "take", "arm", "forward", "escape")


class MinotaurProcessTests(unittest.TestCase):
    def test_host_python_adapter_can_steal_private_case_via_caller_frame(self):
        # RED-TEAM CONTROL: demonstrates the pre-existing unisolated seam is
        # actually vulnerable, so the negative child probe is meaningful.
        class InProcessFrameSpy:
            model_id = "host-python-spy"

            def __init__(self):
                self.stole_private_case_id = False

            def generate(self, prompt, *, seed=None):
                frame = inspect.currentframe()
                while frame is not None:
                    session = frame.f_locals.get("session")
                    if isinstance(session, CandidateSession):
                        self.stole_private_case_id = bool(session._case.case_id)
                        break
                    frame = frame.f_back
                return '{"kind":"CLAIM","claim":{"kind":"ABSTAIN","assessment":"UNKNOWN"}}'

        case = make_case(8, sealed=True)
        spy = InProcessFrameSpy()
        run_candidate(candidate_view(case), lambda: CandidateSession(case), spy,
                      seed=8, authorize_local_adapter=True)
        self.assertTrue(spy.stole_private_case_id)

    def test_real_session_fixed_script_runs_outside_parent_frames(self):
        case = make_case(0)
        public = candidate_view(case)
        adapter = ProcessModelAdapter(actions=WINNING_SEED_ZERO, mode="fixed")
        result = run_candidate(public, lambda: CandidateSession(case), adapter,
                               seed=91377, authorize_local_adapter=True)
        self.assertEqual(result.final_verdict, "VERIFIED_WIN")
        self.assertTrue(audit_receipt(result, public, lambda: CandidateSession(case)))
        self.assertEqual(len(adapter.receipts), 7)
        self.assertTrue(all(item.status == "OK" for item in adapter.receipts))
        self.assertTrue(all(item.child_pid != os.getpid() for item in adapter.receipts))
        self.assertTrue(all(item.generation_seed == 0 for item in adapter.receipts))
        self.assertTrue(all(item.evaluator_seed_discarded for item in adapter.receipts))

    def test_child_frame_probe_cannot_read_parent_examiner_case(self):
        case = make_case(8, sealed=True)
        adapter = ProcessModelAdapter(mode="probe")
        result = run_candidate(candidate_view(case), lambda: CandidateSession(case),
                               adapter, seed=8, authorize_local_adapter=True)
        self.assertFalse(result.verified_escape)
        self.assertEqual(len(adapter.receipts), 1)
        item = adapter.receipts[0]
        self.assertEqual(item.status, "OK")
        self.assertFalse(item.parent_session_frame_visible)
        self.assertNotEqual(item.child_pid, os.getpid())

    def test_upstream_private_seed_is_ignored_and_not_in_child_payload(self):
        first = ProcessModelAdapter(mode="probe", generation_seed=19)
        second = ProcessModelAdapter(mode="probe", generation_seed=19)
        prompt = json.dumps({"public_start": {"prompt": "x"}, "turn": 0})
        self.assertEqual(first.generate(prompt, seed=1111),
                         second.generate(prompt, seed=99999))
        self.assertEqual(first.receipts[0].request_sha256,
                         second.receipts[0].request_sha256)
        self.assertEqual(first.receipts[0].generation_seed, 19)
        self.assertTrue(first.receipts[0].evaluator_seed_discarded)
        self.assertEqual(first.receipts[0].generation_seed_independence, "UNATTESTED")

    def test_dropping_exam_seed_does_not_attest_independent_rng(self):
        # Two separate concepts: no forwarded seed argument versus entropy
        # independence. In particular, private case 0 and default model 0
        # can coincide without providing any matched stochastic proof.
        adapter = ProcessModelAdapter(mode="probe")
        adapter.generate('{"turn":0}', seed=0)
        receipt = adapter.receipts[0]
        self.assertTrue(receipt.evaluator_seed_discarded)
        self.assertEqual(receipt.generation_seed, 0)
        self.assertEqual(receipt.generation_seed_independence, "UNATTESTED")
        self.assertEqual(receipt.host_launcher_authorization,
                         "TRUSTED_HOST_REQUIRED_UNATTESTED")

    def test_same_process_host_python_can_bypass_ordinary_setter_guard(self):
        # A *hostile* candidate factory can use Python's base-object setter.
        # Therefore an in-process launcher is not an authorization boundary.
        adapter = ProcessModelAdapter(mode="probe")
        with self.assertRaises(AttributeError):
            adapter.mode = "breach"
        object.__setattr__(adapter, "mode", "breach")
        with self.assertRaises(ProcessAdapterFailure) as error:
            adapter.generate('{"turn":0}')
        self.assertEqual(error.exception.status, "PRIVATE_FRAME_EXPOSED")
        self.assertEqual(adapter.receipts[-1].host_launcher_authorization,
                         "TRUSTED_HOST_REQUIRED_UNATTESTED")

    def test_timeout_is_fail_closed_and_receipted(self):
        adapter = ProcessModelAdapter(mode="stall", timeout_seconds=.02)
        with self.assertRaises(ProcessAdapterFailure) as error:
            adapter.generate('{"turn":0}', seed=22)
        self.assertEqual(error.exception.status, "TIMEOUT")
        self.assertEqual(adapter.receipts[-1].status, "TIMEOUT")

    def test_oversize_output_is_rejected_and_receipted(self):
        adapter = ProcessModelAdapter(mode="oversize", max_output_bytes=256)
        with self.assertRaises(ProcessAdapterFailure) as error:
            adapter.generate('{"turn":0}', seed=31)
        self.assertEqual(error.exception.status, "OVERSIZE")
        self.assertEqual(adapter.receipts[-1].status, "OVERSIZE")
        self.assertLessEqual(adapter.receipts[-1].response_bytes, 256)

    def test_invalid_json_is_rejected_and_receipted(self):
        adapter = ProcessModelAdapter(mode="invalid_json")
        with self.assertRaises(ProcessAdapterFailure) as error:
            adapter.generate('{"turn":0}', seed=5)
        self.assertEqual(error.exception.status, "INVALID_JSON")
        self.assertEqual(adapter.receipts[-1].status, "INVALID_JSON")

    def test_child_reporting_parent_frame_breach_is_fatal(self):
        # Deliberately malicious diagnostic worker self-reports a private
        # caller frame. A protocol status flag cannot be merely informative.
        adapter = ProcessModelAdapter(mode="breach")
        with self.assertRaises(ProcessAdapterFailure) as error:
            adapter.generate('{"turn":0}')
        self.assertEqual(error.exception.status, "PRIVATE_FRAME_EXPOSED")
        self.assertEqual(adapter.receipts[-1].status, "PRIVATE_FRAME_EXPOSED")
        self.assertTrue(adapter.receipts[-1].parent_session_frame_visible)

    def test_child_error_and_invalid_input_fail_without_execution(self):
        adapter = ProcessModelAdapter(mode="fixed", actions=("wait",))
        with self.assertRaises(ValueError):
            adapter.generate("not json", seed=8)
        self.assertEqual(adapter.receipts, ())
        with self.assertRaises(ValueError):
            ProcessModelAdapter(mode="shell")
        with self.assertRaises(PermissionError):
            ProcessModelAdapter(mode="ollama", model="qwen")

    def test_unapproved_fixture_cannot_mutate_into_local_model_call(self):
        adapter = ProcessModelAdapter(mode="probe")
        with self.assertRaises(AttributeError):
            adapter.mode = "ollama"
        with self.assertRaises(AttributeError):
            adapter.model = "qwen"
        with self.assertRaises(AttributeError):
            adapter.base_url = "http://127.0.0.1:11434"
        self.assertEqual(adapter.mode, "probe")
        self.assertEqual(adapter.generate('{"turn":0}'),
                         '{"claim":{"assessment":"UNKNOWN","kind":"ABSTAIN"},"kind":"CLAIM"}')

    def test_bridge_failure_receipt_does_not_claim_training_or_win(self):
        case = make_case(0)
        adapter = ProcessModelAdapter(mode="invalid_json")
        result = run_candidate(candidate_view(case), lambda: CandidateSession(case),
                               adapter, seed=900, authorize_local_adapter=True)
        self.assertEqual(result.final_verdict, "MODEL_CALL_ERROR")
        self.assertFalse(result.verified_escape)
        self.assertEqual(adapter.receipts[0].status, "INVALID_JSON")
        self.assertTrue(audit_receipt(result, candidate_view(case),
                                      lambda: CandidateSession(case)))


if __name__ == "__main__":
    unittest.main()
