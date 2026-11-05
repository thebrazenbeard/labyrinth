"""Source-only external authority contract and adversarial fake-host tests.

The fake port represents a separately held monotonic service for API
behavior, NOT evidence that a real host key, one-shot store or lease exists.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from unittest import TestCase
from unittest.mock import patch

from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityConflict, ExternalAuthorityError, ExternalAuthorityHead,
    ExternalAuthorityOutcomeUnknown, ExternalMonotonicReservationClient,
    ExternalReservationReceipt, ExternalReservationRequest, next_head_digest,
)


_ZERO = "0" * 64
_PUBLIC_TEST_KEY = b"x" * 32


def request(**changes) -> ExternalReservationRequest:
    values = dict(
        provider_id="independent-host-one", run_id="run-a",
        event_id="signed-development-event", exam_id="dev-exam-01",
        candidate_artifact_sha256="a" * 64,
        scope_fingerprint="b" * 64,
        exam_lineage_sha256="c" * 64,
        prior_checkpoint_sha256="d" * 64,
        to_cycle=2, expected_epoch=4, expected_fence=7,
        expected_commit_index=0, expected_head_sha256=_ZERO,
        operation_nonce="reserved-on-host-01",
    )
    values.update(changes)
    return ExternalReservationRequest(**values)


class FakeExternalHost:
    """Stateful external-API test double, not production authority."""

    def __init__(self):
        self.head = request().expected_head()
        self.pinned_intent = request()
        self.consumed_events: set[tuple[str, str]] = set()
        self.consumed_exams: set[tuple[str, str, str, int]] = set()
        self.reserve_calls = 0
        self.raise_after_commit = False
        self.fail_post_readback = False
        self.fail_pre_readback = False
        self.mutator = None
        self.last_receipt = None
        self._has_reserved = False

    def read_head(self, run_id: str) -> ExternalAuthorityHead:
        if self.fail_pre_readback or (self.fail_post_readback and self._has_reserved):
            raise ConnectionError("external host read failed")
        if run_id != self.head.run_id:
            raise ExternalAuthorityConflict("unknown host run")
        return self.head

    def reserve_once(self, intent: ExternalReservationRequest) -> ExternalReservationReceipt:
        self.reserve_calls += 1
        if intent.expected_head() != self.head:
            raise ExternalAuthorityConflict("host atomic CAS rejected stale head")
        event_key = (intent.run_id, intent.event_id)
        exam_key = (intent.run_id, intent.exam_id, intent.scope_fingerprint,
                    intent.to_cycle)
        if event_key in self.consumed_events or exam_key in self.consumed_exams:
            raise ExternalAuthorityConflict("host event or exam already consumed")
        # This pin is held by the fake *external* host. The client-side copy
        # below is a separate preflight invariant, not a substitute for this
        # independent server-side decision. Neither is real deployment proof.
        pinned_fields = (
            "run_id", "event_id", "exam_id", "candidate_artifact_sha256",
            "scope_fingerprint", "exam_lineage_sha256",
            "prior_checkpoint_sha256", "to_cycle", "operation_nonce",
        )
        if any(getattr(intent, field) != getattr(self.pinned_intent, field)
               for field in pinned_fields):
            raise ExternalAuthorityConflict("external host-pinned intent mismatch")
        receipt = ExternalReservationReceipt(
            provider_id=intent.provider_id, run_id=intent.run_id,
            event_id=intent.event_id,
            reservation_id=f"reservation-{self.reserve_calls}",
            request_sha256=intent.digest(), epoch=intent.expected_epoch,
            fence=intent.expected_fence,
            commit_index=intent.expected_commit_index + 1,
            previous_head_sha256=intent.expected_head_sha256,
            new_head_sha256=next_head_digest(
                intent, f"reservation-{self.reserve_calls}"),
            signature_hex="f" * 128,
        )
        self.consumed_events.add(event_key)
        self.consumed_exams.add(exam_key)
        self.head = receipt.post_head()
        self.last_receipt = receipt
        self._has_reserved = True
        if self.raise_after_commit:
            raise TimeoutError("ACK lost after external durable commit")
        return self.mutator(receipt) if self.mutator else receipt


def source_client(port: FakeExternalHost, *, pinned_request=None) -> ExternalMonotonicReservationClient:
    return ExternalMonotonicReservationClient(
        port=port, provider_id="independent-host-one",
        pinned_request=request() if pinned_request is None else pinned_request,
        pinned_ed25519_public_key=_PUBLIC_TEST_KEY)


class ExternalMonotonicAuthoritySourceTests(TestCase):
    @staticmethod
    def _reserve(client, intent):
        # The fake host uses a dummy signature. Mock only the cryptographic
        # primitive; binding/CAS/readback checks stay genuine source code.
        with patch("labyrinth.external_monotonic_authority_contract._verify_ed25519") as verify:
            result = client.reserve(intent)
            verify.assert_called_once()
            return result

    def test_one_shot_external_cas_and_replay_denial(self):
        port = FakeExternalHost()
        client = source_client(port)
        intent = request()
        receipt = self._reserve(client, intent)
        self.assertEqual(receipt.request_sha256, intent.digest())
        self.assertEqual(port.head, receipt.post_head())
        self.assertEqual(port.reserve_calls, 1)
        # A repeated signed receipt is not a second authority decision:
        # client calls the external host, which has already advanced.
        with self.assertRaisesRegex(ExternalAuthorityConflict, "head mismatch"):
            self._reserve(client, intent)
        self.assertEqual(port.reserve_calls, 1)

    def test_reissue_same_exam_under_new_event_still_conflicts(self):
        port = FakeExternalHost()
        client = source_client(port)
        self._reserve(client, request())
        second = request(
            event_id="new-event-same-exam", operation_nonce="second-nonce",
            expected_commit_index=1,
            expected_head_sha256=port.head.head_sha256)
        with self.assertRaisesRegex(ExternalAuthorityConflict, "already consumed"):
            self._reserve(source_client(port, pinned_request=second), second)

    def test_stale_local_snapshot_cannot_roll_back_external_head(self):
        port = FakeExternalHost()
        client = source_client(port)
        stale_trainer_snapshot = dict(checkpoint_sha256="d" * 64,
                                      saved_head=request().expected_head())
        self._reserve(client, request())
        # Restoring trainer bytes does not restore the independently held
        # port. This ONLY tests the model of a real separate monotonic host.
        self.assertEqual(stale_trainer_snapshot["saved_head"].commit_index, 0)
        self.assertEqual(port.head.commit_index, 1)
        with self.assertRaisesRegex(ExternalAuthorityConflict, "head mismatch"):
            self._reserve(client, request())

    def test_cloned_external_host_state_is_an_explicit_antirollback_falsifier(self):
        # Material LIMIT: if *external provider state and signing identity*
        # can also be cloned/rolled back, two apparently correct receipts are
        # issued for the same request. A provider must anchor state outside
        # the attacker's snapshot domain; source checks cannot establish it.
        first = FakeExternalHost()
        reverted_clone = FakeExternalHost()
        a = self._reserve(source_client(first), request())
        b = self._reserve(source_client(reverted_clone), request())
        self.assertEqual(a.request_sha256, b.request_sha256)
        self.assertEqual(a.commit_index, b.commit_index)
        self.assertEqual(first.head.head_sha256, reverted_clone.head.head_sha256)
        self.assertEqual(first.reserve_calls + reverted_clone.reserve_calls, 2)

    def test_wrong_candidate_scope_checkpoint_exam_or_fence_fails_before_commit(self):
        for changes in (
            {"candidate_artifact_sha256": "e" * 64},
            {"scope_fingerprint": "e" * 64},
            {"prior_checkpoint_sha256": "e" * 64},
            {"exam_lineage_sha256": "e" * 64},
            {"expected_fence": 8},
            {"expected_epoch": 5},
            {"expected_commit_index": 1},
            {"expected_head_sha256": "e" * 64},
        ):
            port = FakeExternalHost()
            with self.subTest(changes=changes):
                # Different field values may be a valid new intended
                # operation, but host-pinned intent MUST not be replaced
                # by an arbitrary caller. Here compare to pinned request.
                forged = replace(request(), **changes)
                self.assertNotEqual(forged.digest(), request().digest())
                with self.assertRaises(ExternalAuthorityConflict):
                    self._reserve(source_client(port), forged)
                self.assertEqual(port.reserve_calls, 0)
                self.assertEqual(port.head.commit_index, 0)

    def test_independent_fake_host_rejects_foreign_valid_shape_intent(self):
        # Even a caller able to create a second locally pinned client cannot
        # cause the external host to accept a candidate/scope rewrite.
        forged = request(candidate_artifact_sha256="e" * 64,
                         scope_fingerprint="f" * 64)
        port = FakeExternalHost()
        with self.assertRaisesRegex(ExternalAuthorityConflict, "host-pinned"):
            self._reserve(source_client(port, pinned_request=forged), forged)
        self.assertEqual(port.head.commit_index, 0)
        self.assertFalse(port.consumed_events)

    def test_signed_receipt_cannot_be_rebound_to_other_candidate_or_event(self):
        for field, value in (
            ("request_sha256", "e" * 64),
            ("provider_id", "foreign-host"),
            ("run_id", "foreign-run"),
            ("event_id", "foreign-event"),
            ("fence", 8),
            ("epoch", 5),
            ("commit_index", 2),
            ("previous_head_sha256", "e" * 64),
            ("new_head_sha256", "e" * 64),
        ):
            port = FakeExternalHost()
            port.mutator = lambda receipt, name=field, changed=value: replace(
                receipt, **{name: changed})
            with self.subTest(field=field):
                with self.assertRaises(ExternalAuthorityOutcomeUnknown):
                    self._reserve(source_client(port), request())

    def test_host_loss_of_ack_is_unknown_and_no_automatic_retry(self):
        port = FakeExternalHost()
        port.raise_after_commit = True
        client = source_client(port)
        with self.assertRaisesRegex(ExternalAuthorityOutcomeUnknown, "unknown"):
            self._reserve(client, request())
        self.assertEqual(port.reserve_calls, 1)
        self.assertEqual(port.head.commit_index, 1)
        with self.assertRaises(ExternalAuthorityConflict):
            self._reserve(client, request())

    def test_postcommit_readback_loss_is_unknown(self):
        port = FakeExternalHost()
        port.fail_post_readback = True
        with self.assertRaisesRegex(ExternalAuthorityOutcomeUnknown, "readback unavailable"):
            self._reserve(source_client(port), request())
        self.assertEqual(port.head.commit_index, 1)

    def test_bad_signature_and_forbidden_claims_fail_closed(self):
        port = FakeExternalHost()
        port.mutator = lambda receipt: replace(receipt, signature_hex="bad")
        with self.assertRaisesRegex(ExternalAuthorityOutcomeUnknown, "unverified"):
            self._reserve(source_client(port), request())
        for claim in ("graduation_authorized", "model_execution_authenticated"):
            port = FakeExternalHost()
            port.mutator = lambda receipt, key=claim: replace(receipt, **{key: True})
            with self.subTest(claim=claim), self.assertRaisesRegex(
                ExternalAuthorityOutcomeUnknown, "unverified"):
                self._reserve(source_client(port), request())

    def test_inputs_are_strictly_typed_and_public_key_is_pinned(self):
        for changed in (
            {"event_id": "../oops"},
            {"exam_id": ""},
            {"candidate_artifact_sha256": "g" * 64},
            {"expected_epoch": True},
            {"expected_fence": -1},
            {"operation_nonce": "newline\nspoof"},
            {"to_cycle": 0},
            {"to_cycle": 2**63},
        ):
            with self.subTest(changed=changed), self.assertRaises(ExternalAuthorityError):
                replace(request(), **changed).validate()
        with self.assertRaisesRegex(ExternalAuthorityError, "public key"):
            ExternalMonotonicReservationClient(
                port=FakeExternalHost(), provider_id="independent-host-one",
                pinned_request=request(),
                pinned_ed25519_public_key=b"short")
        with self.assertRaises(ExternalAuthorityConflict):
            source_client(FakeExternalHost()).reserve(
                request(provider_id="another-provider"))

    def test_real_ed25519_verifier_when_dependency_available_or_fails_closed(self):
        import labyrinth.external_monotonic_authority_contract as authority
        payload = b"receipt domain\n"
        if importlib.util.find_spec("cryptography") is None:
            with self.assertRaisesRegex(ExternalAuthorityError, "cryptography"):
                authority._verify_ed25519(_PUBLIC_TEST_KEY, payload, b"x" * 64)
            return
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        signer = Ed25519PrivateKey.generate()
        key = signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        authority._verify_ed25519(key, payload, signer.sign(payload))
        with self.assertRaisesRegex(ExternalAuthorityError, "signature"):
            authority._verify_ed25519(key, payload + b"tampered", signer.sign(payload))

    def test_real_ed25519_signature_validates_full_external_reservation(self):
        if importlib.util.find_spec("cryptography") is None:
            # This environment cannot establish even signature authenticity.
            # The client must not return a successful reservation receipt.
            with self.assertRaises(ExternalAuthorityOutcomeUnknown):
                source_client(FakeExternalHost()).reserve(request())
            return
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        signer = Ed25519PrivateKey.generate()  # Test-host only; never trainer custody.

        class SignedTestHost(FakeExternalHost):
            def reserve_once(self, intent):
                receipt = super().reserve_once(intent)
                return replace(receipt, signature_hex=signer.sign(
                    receipt.signed_payload()).hex())

        key = signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        host = SignedTestHost()
        client = ExternalMonotonicReservationClient(
            port=host, provider_id="independent-host-one",
            pinned_request=request(), pinned_ed25519_public_key=key)
        receipt = client.reserve(request())
        self.assertEqual(receipt.post_head(), host.head)
        self.assertEqual(receipt.commit_index, 1)
        self.assertFalse(receipt.graduation_authorized)
        with self.assertRaises(ExternalAuthorityConflict):
            client.reserve(request())
