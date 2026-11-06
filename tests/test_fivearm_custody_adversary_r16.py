"""Independent R16 hostile probes of provisional five-arm capture custody.

All signing keys and port state here are synthetic. A green test is never
evidence of a deployed append-only host, physical run, or final examination.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib
import json
import unittest

from labyrinth.fivearm_capture_boundary_r15 import (
    CaptureBoundaryError, CaptureHead,
    chained_head_sha256,
)


class FiveArmCustodyAdversaryR16(unittest.TestCase):
    def setUp(self):
        baseline = importlib.import_module("test_fivearm_capture_boundary_r15")
        self.subject = baseline.FiveArmCaptureBoundaryTests(
            "test_complete_per_run_dual_signed_chain_stays_source_only"
        )
        self.subject.setUp()
        self.addCleanup(self.subject.doCleanups)

    def _resign_synthetic_chain(self, *, altered_run: str,
                                altered_report_sha256: str) -> None:
        """Rebuild a fully valid fake host chain after altered signed capture."""
        s = self.subject
        previous = s.pins.initial_head_sha256
        sequence = s.pins.initial_sequence
        for run, (old_permit, old_capture) in s.port.records.items():
            sequence += 1
            permit = replace(old_permit, sequence=sequence,
                             previous_head_sha256=previous,
                             new_head_sha256="0" * 64, signature_hex="0" * 128)
            permit = replace(permit, new_head_sha256=chained_head_sha256(permit))
            permit = replace(permit,
                             signature_hex=s.issuer.sign(permit.signed_payload()).hex())
            previous = permit.new_head_sha256
            sequence += 1
            capture = replace(old_capture, sequence=sequence,
                              previous_head_sha256=previous,
                              permit_sha256=sha256(permit.signed_payload()).hexdigest(),
                              report_sha256=(altered_report_sha256 if run == altered_run
                                             else old_capture.report_sha256),
                              new_head_sha256="0" * 64, signature_hex="0" * 128)
            capture = replace(capture, new_head_sha256=chained_head_sha256(capture))
            capture = replace(
                capture, signature_hex=s.capturer.sign(capture.signed_payload()).hex())
            previous = capture.new_head_sha256
            s.port.records[run] = (permit, capture)
        s.port.current = CaptureHead(s.pins.provider_id, s.pins.campaign_id,
                                     sequence, previous)

    def test_external_preoutcome_witness_transport_exception_is_typed_denial(self):
        """An unavailable external verifier must not escape as an arbitrary exception."""
        def offline(_payload, _plan):
            raise ConnectionError("synthetic witness transport offline")

        with self.assertRaises(CaptureBoundaryError):
            self.subject.verify(verify_external_preoutcome_witness=offline)

    def test_permit_and_capture_role_keys_cannot_be_interchanged(self):
        s = self.subject
        first = next(iter(s.port.records))
        permit, capture = s.port.records[first]
        s.port.records[first] = (
            replace(permit, signature_hex=s.capturer.sign(permit.signed_payload()).hex()),
            capture,
        )
        with self.assertRaises(CaptureBoundaryError):
            s.verify()
        s.port.records[first] = (
            permit,
            replace(capture, signature_hex=s.issuer.sign(capture.signed_payload()).hex()),
        )
        with self.assertRaises(CaptureBoundaryError):
            s.verify()

    def test_host_listing_duplicate_or_non_tuple_cannot_hide_unregistered_trial(self):
        s = self.subject
        original = s.port.list_run_names
        runs = original(s.pins.campaign_id)
        s.port.list_run_names = lambda _: runs + (runs[0],)
        with self.assertRaises(CaptureBoundaryError):
            s.verify()
        s.port.list_run_names = lambda _: list(runs)
        with self.assertRaises(CaptureBoundaryError):
            s.verify()

    def test_cross_run_permit_and_capture_substitution_denied(self):
        s = self.subject
        first, second = list(s.port.records)[:2]
        original = s.port.records[first]
        s.port.records[first] = s.port.records[second]
        with self.assertRaises(CaptureBoundaryError):
            s.verify()
        s.port.records[first] = (original[0], s.port.records[second][1])
        with self.assertRaises(CaptureBoundaryError):
            s.verify()

    def test_signed_receipts_reject_duplicated_or_non_initial_attempt(self):
        s = self.subject
        first = next(iter(s.port.records))
        original = s.port.records[first]
        permit, capture = original
        s.port.records[first] = (replace(permit, attempt_number=2), capture)
        with self.assertRaises(CaptureBoundaryError):
            s.verify()
        s.port.records[first] = (permit, replace(capture, attempt_id="other"))
        with self.assertRaises(CaptureBoundaryError):
            s.verify()

    def test_fully_resigned_wrong_report_digest_is_still_refused(self):
        s = self.subject
        run = next(iter(s.port.records))
        self._resign_synthetic_chain(altered_run=run, altered_report_sha256="0" * 64)
        with self.assertRaisesRegex(CaptureBoundaryError, "offline five-arm audit"):
            s.verify()

    def test_fully_resigned_report_missing_public_split_is_refused(self):
        s = self.subject
        run = next(iter(s.port.records))
        report = s.fixture.root / "runs" / run / "report.json"
        payload = json.loads(report.read_text(encoding="utf-8"))
        del payload["arms"]["correct"]["cycles"][-1]["public_transfer_accuracy"]
        report.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        self._resign_synthetic_chain(altered_run=run,
                                     altered_report_sha256=sha256(report.read_bytes()).hexdigest())
        with self.assertRaisesRegex(CaptureBoundaryError, "offline five-arm audit"):
            s.verify()

    def test_invalid_host_head_identity_and_boolean_sequence_fail(self):
        s = self.subject
        old = s.port.current
        for substituted in (
            replace(old, provider_id="foreign"),
            replace(old, campaign_id="foreign"),
            replace(old, sequence=True),
            replace(old, head_sha256="0" * 64),
        ):
            with self.subTest(substituted=substituted):
                s.port.current = substituted
                with self.assertRaises(CaptureBoundaryError):
                    s.verify()

    def test_source_verifier_does_not_promote_untrusted_witness_to_custody(self):
        s = self.subject
        result = s.verify(verify_external_preoutcome_witness=lambda *_: True)
        self.assertEqual(result.status, "FIVE_ARM_CAPTURE_SOURCE_ONLY")
        self.assertFalse(result.independent_custody_verified)
        self.assertFalse(result.physical_model_verified)
        self.assertFalse(result.graduation_authorized)

    def test_post_audit_checkpoint_race_is_not_an_atomic_snapshot(self):
        """Bounded source audit cannot own file bytes after its last filesystem read."""
        s = self.subject
        first = next(iter(s.port.records))
        checkpoint = (s.fixture.root / "runs" / first / "correct" / "cycle-0"
                      / "adapter" / "adapter_model.safetensors")
        old_read_head = s.port.read_head
        calls = 0

        def mutate_at_final_remote_read(campaign_id):
            nonlocal calls
            calls += 1
            if calls == 3:
                checkpoint.write_bytes(checkpoint.read_bytes() + b"RACE-AFTER-READ")
            return old_read_head(campaign_id)

        s.port.read_head = mutate_at_final_remote_read
        result = s.verify()
        self.assertEqual(result.status, "FIVE_ARM_CAPTURE_SOURCE_ONLY")
        self.assertFalse(result.independent_custody_verified)
        # The same corrupted checkpoint is detected on the *next* source audit.
        with self.assertRaises(CaptureBoundaryError):
            s.verify()


if __name__ == "__main__":
    unittest.main()
