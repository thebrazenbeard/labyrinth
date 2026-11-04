"""Source-only signed capture boundary; local signers do not attest real custody."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import importlib
import unittest

from labyrinth.fivearm_capture_boundary_r15 import (
    CaptureBoundaryError, CaptureHead, CapturePins, CaptureReceipt,
    PermitReceipt, chained_head_sha256, verify_captured_fivearm_campaign,
)


class LocalCapturePort:
    def __init__(self):
        self.records = {}
        self.current = None

    def read_head(self, campaign_id):
        return self.current

    def read_run_receipts(self, campaign_id, run_name):
        return self.records[run_name]

    def list_run_names(self, campaign_id):
        return tuple(sorted(self.records))


class FiveArmCaptureBoundaryTests(unittest.TestCase):
    def setUp(self):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        except ImportError:
            self.skipTest("cryptography Ed25519 unavailable; verifier fails closed")
        fixtures = importlib.import_module("test_fivearm_campaign_admission_r13")
        self.fixture = fixtures.FiveArmCampaignAdmissionTests(
            "test_complete_five_arm_grid_reloads_every_child_and_stays_provisional")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.issuer = Ed25519PrivateKey.generate()
        self.capturer = Ed25519PrivateKey.generate()
        self.foreign = Ed25519PrivateKey.generate()
        self.pub = lambda private: private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.port = LocalCapturePort()
        self.initial_head = "1" * 64
        self.pins = CapturePins(
            provider_id="independent-capture-provider", campaign_id="fivearm-campaign-1",
            plan_sha256=self.fixture.expected_plan_sha256,
            initial_sequence=40, initial_head_sha256=self.initial_head,
            permit_public_key=self.pub(self.issuer),
            capture_public_key=self.pub(self.capturer),
        )
        previous = self.initial_head
        seq = 40
        for job in self.fixture.plan["manifest"]["jobs"]:
            run = job["run_name"]
            attempt = "attempt:" + run
            seq += 1
            permit = PermitReceipt(
                provider_id=self.pins.provider_id, campaign_id=self.pins.campaign_id,
                run_name=run, plan_sha256=self.pins.plan_sha256,
                attempt_id=attempt, attempt_number=1,
                sequence=seq, previous_head_sha256=previous,
                new_head_sha256="0" * 64, signature_hex="0" * 128,
            )
            permit = replace(permit, new_head_sha256=chained_head_sha256(permit))
            permit = replace(permit, signature_hex=self.issuer.sign(permit.signed_payload()).hex())
            previous = permit.new_head_sha256
            seq += 1
            capture = CaptureReceipt(
                provider_id=self.pins.provider_id, campaign_id=self.pins.campaign_id,
                run_name=run, attempt_id=attempt,
                permit_sha256=sha256(permit.signed_payload()).hexdigest(),
                report_sha256=self.fixture.report_hashes[run],
                sequence=seq, previous_head_sha256=previous,
                new_head_sha256="0" * 64, signature_hex="0" * 128,
            )
            capture = replace(capture, new_head_sha256=chained_head_sha256(capture))
            capture = replace(capture, signature_hex=self.capturer.sign(capture.signed_payload()).hex())
            previous = capture.new_head_sha256
            self.port.records[run] = (permit, capture)
        self.port.current = CaptureHead(self.pins.provider_id, self.pins.campaign_id,
                                        seq, previous)

    def verify(self, **overrides):
        args = dict(
            spec=self.fixture.spec, campaign_root=self.fixture.root,
            pins=self.pins, port=self.port,
            expected_runner_sha256=self.fixture.plan["manifest"]["runner_sha256"],
            expected_protocol_sha256=self.fixture.plan["manifest"]["protocol_sha256"],
            expected_pretrained_base_sha256="f" * 64,
            preoutcome_witness=self.fixture.witness,
            verify_external_preoutcome_witness=lambda witness, plan_sha: (
                witness == self.fixture.witness and plan_sha == self.fixture.expected_plan_sha256),
        )
        args.update(overrides)
        return verify_captured_fivearm_campaign(**args)

    def test_complete_per_run_dual_signed_chain_stays_source_only(self):
        result = self.verify()
        self.assertEqual(result.status, "FIVE_ARM_CAPTURE_SOURCE_ONLY")
        self.assertEqual(result.captured_runs, 6)
        self.assertFalse(result.independent_custody_verified)
        self.assertFalse(result.physical_model_verified)
        self.assertFalse(result.graduation_authorized)
        self.assertEqual(result.audit.run_count, 6)

    def test_missing_or_duplicate_run_seal_fails_closed(self):
        first = next(iter(self.port.records))
        saved = self.port.records.pop(first)
        with self.assertRaises(CaptureBoundaryError):
            self.verify()
        self.port.records[first] = (saved[0], saved[1], saved[1])
        with self.assertRaises(CaptureBoundaryError):
            self.verify()

    def test_bad_signature_and_posthoc_report_selection_fail(self):
        first = next(iter(self.port.records))
        permit, capture = self.port.records[first]
        altered = replace(capture, report_sha256="0" * 64)
        self.port.records[first] = (permit, altered)
        with self.assertRaises(CaptureBoundaryError):
            self.verify()
        self.port.records[first] = (permit, replace(
            capture, signature_hex=self.foreign.sign(capture.signed_payload()).hex()))
        with self.assertRaises(CaptureBoundaryError):
            self.verify()

    def test_attempt_replay_or_changed_pregrant_binding_fails(self):
        first, second = list(self.port.records)[:2]
        other_permit, other_capture = self.port.records[second]
        permit, capture = self.port.records[first]
        self.port.records[second] = (replace(other_permit, attempt_id=permit.attempt_id), other_capture)
        with self.assertRaises(CaptureBoundaryError):
            self.verify()
        self.port.records[second] = (other_permit, other_capture)
        self.port.records[first] = (replace(permit, plan_sha256="0" * 64), capture)
        with self.assertRaises(CaptureBoundaryError):
            self.verify()

    def test_rollback_and_unavailable_authority_readback_fail(self):
        original = self.port.current
        self.port.current = replace(original, sequence=original.sequence - 1)
        with self.assertRaises(CaptureBoundaryError):
            self.verify()
        self.port.current = None
        with self.assertRaises(CaptureBoundaryError):
            self.verify()

    def test_forked_sequence_and_unregistered_extra_trial_fail(self):
        first, second = list(self.port.records)[:2]
        permit, capture = self.port.records[second]
        prior_permit = self.port.records[first][0]
        self.port.records[second] = (
            replace(permit, sequence=prior_permit.sequence), capture)
        with self.assertRaises(CaptureBoundaryError):
            self.verify()
        self.port.records[second] = (permit, capture)
        self.port.records["extra-after-selection"] = (permit, capture)
        with self.assertRaises(CaptureBoundaryError):
            self.verify()

    def test_tampered_report_after_signed_capture_fails_source_audit(self):
        first = next(iter(self.port.records))
        report = self.fixture.root / "runs" / first / "report.json"
        report.write_bytes(report.read_bytes() + b" ")
        with self.assertRaisesRegex(CaptureBoundaryError, "offline five-arm audit"):
            self.verify()

    def test_preoutcome_plan_witness_is_still_mandatory(self):
        with self.assertRaises(CaptureBoundaryError):
            self.verify(preoutcome_witness=b"wrong")

    def test_host_readback_torn_between_reads_fails_closed(self):
        old_read_head = self.port.read_head
        calls = 0

        def drifting_head(campaign_id):
            nonlocal calls
            calls += 1
            head = old_read_head(campaign_id)
            return head if calls == 1 else replace(head, sequence=head.sequence + 1)

        self.port.read_head = drifting_head
        with self.assertRaises(CaptureBoundaryError):
            self.verify()

    def test_new_trial_during_offline_audit_invalidates_capture_head(self):
        original = self.port.read_head
        calls = 0

        def late_trial(campaign_id):
            nonlocal calls
            calls += 1
            head = original(campaign_id)
            return head if calls < 3 else replace(head, sequence=head.sequence + 1)

        self.port.read_head = late_trial
        with self.assertRaisesRegex(CaptureBoundaryError, "changed during source audit"):
            self.verify()

    def test_remote_port_unavailable_does_not_fallback_to_caller_hashes(self):
        def offline(_campaign_id):
            raise ConnectionError("synthetic external host unreachable")

        self.port.read_head = offline
        with self.assertRaisesRegex(CaptureBoundaryError, "readback unavailable"):
            self.verify()

    def test_untrusted_or_missing_capture_signer_is_not_independence(self):
        with self.assertRaises(CaptureBoundaryError):
            self.verify(pins=replace(self.pins, capture_public_key=self.pub(self.foreign)))
        with self.assertRaises(CaptureBoundaryError):
            self.verify(pins=replace(self.pins, permit_public_key=b""))
        with self.assertRaises(CaptureBoundaryError):
            self.verify(pins=replace(self.pins, capture_public_key=self.pins.permit_public_key))


if __name__ == "__main__":
    unittest.main()
