"""Deterministic SOURCE-CONTRACT tests; the fake is not remote authority."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import hmac
from threading import Barrier, Lock
import unittest

from labyrinth.atomic_artifact_publication import (
    ArtifactPublicationError,
    PublicationDenied,
    PublicationIntent,
    PublicationOutcomeUnknown,
    PublicationReceipt,
    SinkConflict,
    canonical_receipt_payload,
    publish_with_sink,
)


_KEY = b"test fixture only; never a remotely managed signature"
_OLD = "a" * 64


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


class InMemorySinkTestDouble:
    """Atomic within a Python lock ONLY; no durability or host isolation."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.sink_id = "test-sink"
        self.run_id = "test-run"
        self.epoch = 1
        self.fence = 0
        self.current_sha256 = _OLD
        self.staged: dict[str, bytes] = {}
        self.public: dict[str, bytes] = {}
        self.operations: set[str] = set()
        self.transaction_number = 0
        self.fail_after_commit = False
        self.corrupt_receipt = False

    def stage(self, object_id: str, payload: bytes) -> None:
        # Test fixture models sink-managed staging, not a worker-readable path.
        self.staged[object_id] = payload

    def advance_worker_fence(self, *, epoch: int, fence: int) -> None:
        with self.lock:
            self.epoch = epoch
            self.fence = fence

    def commit_if_current(self, intent: PublicationIntent) -> PublicationReceipt:
        with self.lock:
            if (intent.sink_id != self.sink_id
                    or intent.run_id != self.run_id
                    or intent.epoch != self.epoch
                    or intent.fence != self.fence
                    or intent.expected_current_sha256 != self.current_sha256
                    or intent.operation_id in self.operations
                    or intent.artifact_id in self.public):
                raise SinkConflict("atomic predecessor mismatch")
            payload = self.staged.get(intent.staged_object_id)
            if payload is None or _sha(payload) != intent.staged_sha256:
                raise SinkConflict("sink bytes differ from the declared digest")

            # This critical section stands in for a **remote durable atomic**
            # state-and-visibility transaction; it does not implement one.
            self.public[intent.artifact_id] = payload
            self.current_sha256 = intent.staged_sha256
            self.operations.add(intent.operation_id)
            self.transaction_number += 1
            provisional = PublicationReceipt(
                intent, f"test-tx-{self.transaction_number}",
                self.current_sha256, True, "unsigned",
            )
            signature = hmac.new(
                _KEY, canonical_receipt_payload(provisional), sha256,
            ).hexdigest()
            receipt = replace(provisional, signature_hex=signature)
            if self.corrupt_receipt:
                receipt = replace(receipt, signature_hex="forged")
            if self.fail_after_commit:
                raise ConnectionError("simulated partition after remote commit")
            return receipt


def _verify(payload: bytes, signature: str, sink_id: str) -> bool:
    if sink_id != "test-sink":
        return False
    expected = hmac.new(_KEY, payload, sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _intent(payload: bytes, *, operation_id: str = "op-1",
            artifact_id: str = "artifact-1", staged_object_id: str = "stage-1",
            epoch: int = 1, fence: int = 0,
            expected_current_sha256: str = _OLD,
            run_id: str = "test-run") -> PublicationIntent:
    return PublicationIntent(
        sink_id="test-sink", operation_id=operation_id,
        run_id=run_id, artifact_id=artifact_id,
        staged_object_id=staged_object_id,
        epoch=epoch, fence=fence,
        expected_current_sha256=expected_current_sha256,
        staged_sha256=_sha(payload),
    )


class AtomicPublicationContractTests(unittest.TestCase):
    def test_matching_fence_and_current_digest_publish_atomically_in_fake(self) -> None:
        sink = InMemorySinkTestDouble()
        data = b"new immutable checkpoint"
        sink.stage("stage-1", data)
        intent = _intent(data)
        receipt = publish_with_sink(
            sink, intent, expected_sink_id="test-sink",
            verify_receipt_signature=_verify,
        )
        self.assertTrue(receipt.durable)
        self.assertEqual(receipt.current_sha256, _sha(data))
        self.assertEqual(sink.current_sha256, _sha(data))
        self.assertEqual(sink.public["artifact-1"], data)

    def test_stale_A_after_B_advances_fence_must_not_become_visible(self) -> None:
        sink = InMemorySinkTestDouble()
        a_data = b"worker A stale"
        b_data = b"worker B current"
        sink.stage("stage-1", a_data)
        sink.stage("stage-B", b_data)
        a = _intent(a_data)

        # A has an arbitrarily recent read-only preflight. B wins authority
        # advance BEFORE A's sink-owned commit; A's old fence is invalid.
        sink.advance_worker_fence(epoch=2, fence=1)
        with self.assertRaises(PublicationDenied):
            publish_with_sink(
                sink, a, expected_sink_id="test-sink",
                verify_receipt_signature=_verify,
            )
        self.assertEqual(sink.public, {})
        self.assertEqual(sink.current_sha256, _OLD)

        b = _intent(b_data, operation_id="op-B", artifact_id="artifact-B",
                    staged_object_id="stage-B", epoch=2, fence=1)
        publish_with_sink(
            sink, b, expected_sink_id="test-sink",
            verify_receipt_signature=_verify,
        )
        self.assertEqual(sink.public, {"artifact-B": b_data})

    def test_two_workers_racing_on_same_predecessor_have_one_winner(self) -> None:
        sink = InMemorySinkTestDouble()
        a_data, b_data = b"candidate A", b"candidate B"
        sink.stage("stage-1", a_data)
        sink.stage("stage-2", b_data)
        intents = [
            _intent(a_data),
            _intent(b_data, operation_id="op-2", artifact_id="artifact-2",
                    staged_object_id="stage-2"),
        ]
        gate = Barrier(2)

        def race(intent: PublicationIntent) -> str:
            gate.wait()
            try:
                publish_with_sink(
                    sink, intent, expected_sink_id="test-sink",
                    verify_receipt_signature=_verify,
                )
            except PublicationDenied:
                return "DENIED"
            return "PUBLISHED"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(race, intents))
        self.assertCountEqual(outcomes, ["PUBLISHED", "DENIED"])
        self.assertEqual(len(sink.public), 1)
        self.assertEqual(len(sink.operations), 1)

    def test_ambiguous_partition_after_commit_is_hold_not_safe_retry(self) -> None:
        sink = InMemorySinkTestDouble()
        data = b"committed before lost acknowledgment"
        sink.stage("stage-1", data)
        sink.fail_after_commit = True
        intent = _intent(data)
        with self.assertRaisesRegex(PublicationOutcomeUnknown, "reconcile"):
            publish_with_sink(
                sink, intent, expected_sink_id="test-sink",
                verify_receipt_signature=_verify,
            )
        self.assertEqual(sink.public["artifact-1"], data)
        self.assertEqual(sink.current_sha256, _sha(data))
        sink.fail_after_commit = False
        with self.assertRaises(PublicationDenied):
            publish_with_sink(
                sink, intent, expected_sink_id="test-sink",
                verify_receipt_signature=_verify,
            )

    def test_signature_tampering_after_commit_is_unknown_not_success(self) -> None:
        sink = InMemorySinkTestDouble()
        data = b"trusted bytes"
        sink.stage("stage-1", data)
        sink.corrupt_receipt = True
        with self.assertRaisesRegex(PublicationOutcomeUnknown, "unauthenticated"):
            publish_with_sink(
                sink, _intent(data), expected_sink_id="test-sink",
                verify_receipt_signature=_verify,
            )
        self.assertEqual(sink.public["artifact-1"], data)

    def test_staged_digest_mismatch_denies_without_visibility(self) -> None:
        sink = InMemorySinkTestDouble()
        sink.stage("stage-1", b"actual")
        with self.assertRaises(PublicationDenied):
            publish_with_sink(
                sink, _intent(b"fake digest"), expected_sink_id="test-sink",
                verify_receipt_signature=_verify,
            )
        self.assertFalse(sink.public)

    def test_cross_run_or_wrong_sink_identity_is_rejected(self) -> None:
        sink = InMemorySinkTestDouble()
        sink.stage("stage-1", b"actual")
        with self.assertRaises(PublicationDenied):
            publish_with_sink(
                sink, _intent(b"actual", run_id="another-run"),
                expected_sink_id="test-sink", verify_receipt_signature=_verify,
            )
        with self.assertRaises(ArtifactPublicationError):
            publish_with_sink(
                sink, _intent(b"actual"),
                expected_sink_id="untrusted-sink", verify_receipt_signature=_verify,
            )

    def test_duplicate_operation_id_is_rejected_at_sink(self) -> None:
        sink = InMemorySinkTestDouble()
        sink.stage("stage-1", b"one")
        intent = _intent(b"one")
        publish_with_sink(
            sink, intent, expected_sink_id="test-sink",
            verify_receipt_signature=_verify,
        )
        sink.stage("stage-2", b"two")
        duplicate = _intent(
            b"two", operation_id="op-1", artifact_id="artifact-2",
            staged_object_id="stage-2",
            expected_current_sha256=_sha(b"one"),
        )
        with self.assertRaises(PublicationDenied):
            publish_with_sink(
                sink, duplicate, expected_sink_id="test-sink",
                verify_receipt_signature=_verify,
            )
        self.assertEqual(list(sink.public), ["artifact-1"])

    def test_invalid_identity_digest_bool_epoch_and_unchanged_digest(self) -> None:
        data = b"valid"
        base = _intent(data)
        variants = [
            replace(base, operation_id="../escape"),
            replace(base, staged_object_id="C:\\Temp\\worker"),
            replace(base, staged_sha256="A" * 64),
            replace(base, expected_current_sha256=base.staged_sha256),
            replace(base, epoch=True),
            replace(base, fence=-1),
        ]
        for invalid in variants:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ArtifactPublicationError):
                    invalid.validate()

    def test_bad_verifier_is_outcome_unknown_even_with_visible_artifact(self) -> None:
        sink = InMemorySinkTestDouble()
        data = b"untrusted verifier"
        sink.stage("stage-1", data)
        with self.assertRaises(PublicationOutcomeUnknown):
            publish_with_sink(
                sink, _intent(data), expected_sink_id="test-sink",
                verify_receipt_signature=lambda *_: False,
            )
        self.assertEqual(sink.public["artifact-1"], data)


if __name__ == "__main__":
    unittest.main()
