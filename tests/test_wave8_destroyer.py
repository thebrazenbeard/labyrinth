"""Independent RED attacks against the PR152 candidate-summary receipt boundary.

These tests demand rejection of malformed, self-forged host receipts. They
cannot establish cryptographic provenance; same-process Python remains trusted.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace
from io import BytesIO
from threading import Event
import hashlib
import inspect
import json
import time
import unittest
from unittest.mock import patch

from labyrinth.minotaur_cross_exam import ExamSpec, run_cross_exam
from labyrinth.minotaur_process_adapter import ProcessAdapterFailure, ProcessModelAdapter
from labyrinth.oracle_chamber_exam import make_case as make_oracle_case
from labyrinth.minotaur_trusted_launcher import (
    ProcessCandidateConfig, TrustedCaseReceipt, candidate_public_summary,
    run_trusted_cross_exam,
)


class UntrustedReceiptAttack(unittest.TestCase):
    def test_legacy_callback_default_denied_before_private_case_construction(self):
        class MaliciousFactory:
            def __call__(self, _public):
                raise AssertionError("HOST_CALLBACK_ENTERED_PRIVATE_EVALUATOR")

        with patch("labyrinth.minotaur_cross_exam.make_public_case",
                   side_effect=AssertionError("PRIVATE_CASE_CONSTRUCTED")) as maker:
            with self.assertRaises(PermissionError):
                run_cross_exam((ExamSpec("oracle", 739, sealed=True),),
                               candidate_factory=MaliciousFactory(),
                               authorize_local_adapter=True)
        maker.assert_not_called()

    def test_opted_in_host_factory_still_steals_exam_secret_40_of_40(self):
        class HostLeakCommit:
            model_id = "host-callback-attack:no-learning"

            def __init__(self, answer):
                self.answer = answer

            def generate(self, _prompt, *, seed=None):
                return json.dumps({"kind": "CLAIM", "claim": {
                    "kind": "COMMIT", "secret": self.answer}})

        stolen = []

        def attacker(_public):
            spec = inspect.currentframe().f_back.f_locals["spec"]
            stolen.append(spec.seed)
            return HostLeakCommit(make_oracle_case(
                spec.seed, variant=spec.variant, sealed=spec.sealed).secret)

        results = run_cross_exam(
            tuple(ExamSpec("oracle", seed, sealed=True) for seed in range(40)),
            candidate_factory=attacker, authorize_local_adapter=True,
            trust_host_python_factory=True)
        self.assertEqual(len(stolen), 40)
        self.assertEqual(sum(case.candidate.final_verdict == "VERIFIED_WIN"
                             for case in results), 40)
        self.assertTrue(all(len(case.candidate.turns) == 1 for case in results))

    def test_deeply_nested_public_json_rejected_as_bounded_input_error(self):
        adapter = ProcessModelAdapter(mode="probe")
        payload = '{"a":' + '[' * 10000 + '0' + ']' * 10000 + '}'
        self.assertLess(len(payload), 64000)
        with patch("labyrinth.minotaur_process_adapter.subprocess.Popen",
                   side_effect=AssertionError("CHILD_LAUNCH_REACHED")) as child:
            with self.assertRaises(ValueError):
                adapter.generate(payload)
        child.assert_not_called()

    def test_wall_deadline_not_extended_by_blocked_writer_cleanup(self):
        release = Event()

        class BlockedInput:
            def write(self, data):
                release.wait(4)

            def flush(self):
                pass

            def close(self):
                pass

        class ExitedChild:
            pid = 1049
            returncode = 0
            stdin = BlockedInput()
            stdout = BytesIO(b"")

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        adapter = ProcessModelAdapter(mode="probe", timeout_seconds=.01)
        begin = time.monotonic()
        try:
            with patch("labyrinth.minotaur_process_adapter.subprocess.Popen",
                       return_value=ExitedChild()):
                with self.assertRaises(ProcessAdapterFailure):
                    adapter.generate('{"turn":0}')
        finally:
            release.set()
        elapsed_ms = (time.monotonic() - begin) * 1000
        # Allow generous scheduling overhead but not the hard-coded 1s join.
        self.assertLess(elapsed_ms, 500)

    def test_numeric_exponent_overflow_rejected_before_ipc_launch(self):
        adapter = ProcessModelAdapter(mode="probe")
        # `parse_constant` rejects literal Infinity but Python's default
        # float parser still maps a finite JSON exponent to float('inf').
        with patch("labyrinth.minotaur_process_adapter.subprocess.Popen",
                   side_effect=AssertionError("CHILD_LAUNCH_REACHED")) as child:
            with self.assertRaises(ValueError):
                adapter.generate('{"turn":1e999}')
        child.assert_not_called()

    def test_even_exact_dataclass_receipt_types_can_be_resigned_by_host_without_attestation(self):
        genuine = run_trusted_cross_exam(
            (ExamSpec("oracle", 739, sealed=True),),
            ProcessCandidateConfig(mode="probe", generation_seed=17))[0]
        self.assertNotEqual(genuine.cross_case.candidate.final_verdict, "VERIFIED_WIN")
        fake_run = replace(genuine.cross_case.candidate,
                           final_verdict="VERIFIED_WIN", verified_escape=True)
        fields = asdict(fake_run)
        fields.pop("receipt_sha256")
        self_signed = hashlib.sha256(json.dumps(
            fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        fake_run = replace(fake_run, receipt_sha256=self_signed)
        fake_case = replace(genuine.cross_case, candidate=fake_run)
        fake_host = replace(genuine, cross_case=fake_case)
        self.assertIs(type(fake_host), TrustedCaseReceipt)
        # This PASS-as-exploit distinguishes self-hash checks from authority:
        # the attacker can recompute an unkeyed hash with no private witness.
        result = candidate_public_summary((fake_host,))[0]
        self.assertEqual(result["host_reported_candidate_verdict"], "VERIFIED_WIN")
        self.assertTrue(result["host_reported_candidate_escape"])
        self.assertEqual(result["evidence_status"], "UNATTESTED_HOST_SELF_HASH")

    def test_forged_nested_case_object_must_not_project_verified_win(self):
        fake = TrustedCaseReceipt(
            cross_case=SimpleNamespace(
                family="oracle",
                candidate=SimpleNamespace(final_verdict="VERIFIED_WIN",
                                          verified_escape=True),
                baseline_escape_verified=True,
            ),
            process_calls=(), candidate_kind="DETERMINISTIC_ZERO_UPDATE_CONTROL",
            model_seed_source="HOST_CSPRNG_REQUEST_UNATTESTED",
        )
        # Current defect: no evaluation was run, yet the public projector
        # returns an apparent independently verified escape.
        with self.assertRaises((TypeError, ValueError)):
            candidate_public_summary((fake,))

    def test_mutated_candidate_kind_must_not_reintroduce_private_seed(self):
        genuine = run_trusted_cross_exam(
            (ExamSpec("oracle", 739, sealed=True),),
            ProcessCandidateConfig(mode="probe", generation_seed=17))[0]
        # dataclasses.replace preserves the exact required receipt type,
        # but nested arbitrary data should not cross the public projection.
        forged = replace(genuine, candidate_kind={"seed": genuine.cross_case.seed})
        with self.assertRaises((TypeError, ValueError)):
            candidate_public_summary((forged,))


if __name__ == "__main__":
    unittest.main()
