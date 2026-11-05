"""Source-level contract probes; FakeAuthority is NOT a deployed anchor."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
from threading import Lock, Thread
import unittest

from labyrinth import monotonic_anchor_protocol as anchor


REQUEST_KEY = b"request-secret-independent-custodian-00001"
AUTHORITY_KEY = b"authority-receipt-secret-independent-0001"
GENESIS = anchor.AnchorState("journal-alpha", 0, "a" * 64,
                             None, None, 0, None, "GENESIS")
RUNNING = anchor.AnchorState("journal-alpha", 1, "b" * 64,
                             "run-alpha", "event-alpha", 0, "1" * 64, "RUNNING")
PROMOTED = anchor.AnchorState("journal-alpha", 2, "c" * 64,
                              "run-alpha", "event-alpha", 1, "2" * 64, "RUNNING")
COMPLETED = anchor.AnchorState("journal-alpha", 3, "d" * 64,
                               "run-alpha", "event-alpha", 2, "2" * 64, "COMPLETED")


class FakeIndependentAuthority:
    """Test double acting as a separate, durable CAS and HMAC authority."""

    def __init__(self):
        self.lock = Lock()
        self.current = GENESIS
        self.last_op = "genesis"
        self.used_ids: set[str] = set()
        self.online = True
        self.drop_ack_after_commit = False
        self.partition_after_commit = False
        self.corrupt_ack = False
        self.replay_snapshot: anchor.AuthorityReceipt | None = None

    @staticmethod
    def signed_receipt(operation_id: str,
                       state: anchor.AnchorState) -> anchor.AuthorityReceipt:
        signature = hmac.new(
            AUTHORITY_KEY,
            anchor._RECEIPT_DOMAIN + anchor._receipt_bytes(operation_id, state),
            sha256,
        ).hexdigest()
        return anchor.AuthorityReceipt(operation_id, state, signature)

    def read(self, authority_id: str) -> anchor.AuthorityReceipt:
        if not self.online:
            raise ConnectionError("independent control plane disconnected")
        if authority_id != self.current.authority_id:
            raise anchor.AnchorError("unknown authority")
        with self.lock:
            return (self.replay_snapshot or
                    self.signed_receipt(self.last_op, self.current))

    def compare_and_swap(self, operation: anchor.SignedAnchorOperation,
                         ) -> anchor.AuthorityReceipt:
        if not self.online:
            raise ConnectionError("independent control plane disconnected")
        anchor.verify_operation(operation, request_key=REQUEST_KEY)
        with self.lock:
            if operation.operation_id in self.used_ids:
                raise anchor.AnchorError("remote duplicate operation ID")
            if operation.before != self.current:
                raise anchor.AnchorError("remote CAS stale predecessor")
            self.current = operation.after
            self.last_op = operation.operation_id
            self.used_ids.add(operation.operation_id)
            receipt = self.signed_receipt(self.last_op, self.current)
        if self.drop_ack_after_commit:
            raise TimeoutError("ACK lost after durable remote state update")
        if self.partition_after_commit:
            self.online = False
        if self.corrupt_ack:
            return replace(receipt, authority_signature_hex="f" * 64)
        return receipt


class LocalProjection:
    def __init__(self):
        self.state = GENESIS
        self.lock = Lock()
        self.fail_write = False
        self.write_calls = 0

    def read(self) -> anchor.AnchorState:
        with self.lock:
            return self.state

    def cas(self, expected: anchor.AnchorState, proposed: anchor.AnchorState) -> bool:
        with self.lock:
            self.write_calls += 1
            if self.fail_write:
                raise OSError("local disk unavailable after remote commit")
            if self.state != expected:
                return False
            self.state = proposed
            return True


def signed(operation_id: str, before: anchor.AnchorState,
           after: anchor.AnchorState) -> anchor.SignedAnchorOperation:
    return anchor.sign_operation(
        operation_id, before, after, request_key=REQUEST_KEY,
        authority_receipt_key=AUTHORITY_KEY,
    )


def advance(remote: FakeIndependentAuthority, local: LocalProjection,
            operation: anchor.SignedAnchorOperation) -> anchor.AuthorityReceipt:
    return anchor.advance_remote_then_project(
        remote, operation, receipt_key=AUTHORITY_KEY,
        local_read=local.read, local_compare_and_swap=local.cas,
    )


class MonotonicAnchorContractTests(unittest.TestCase):
    def test_run_reservation_checkpoint_promotion_and_completion(self):
        remote, local = FakeIndependentAuthority(), LocalProjection()
        for operation, expected in (
            (signed("reserve-1", GENESIS, RUNNING), RUNNING),
            (signed("promote-1", RUNNING, PROMOTED), PROMOTED),
            (signed("complete-1", PROMOTED, COMPLETED), COMPLETED),
        ):
            receipt = advance(remote, local, operation)
            self.assertEqual(receipt.state, expected)
            self.assertEqual(local.read(), expected)
            self.assertEqual(remote.current, expected)
        with self.assertRaisesRegex(anchor.AnchorError, "completed run"):
            signed("reopen", COMPLETED,
                   replace(COMPLETED, epoch=4, journal_sha256="e" * 64))

    def test_network_partition_fails_closed_before_local_effect(self):
        remote, local = FakeIndependentAuthority(), LocalProjection()
        remote.online = False
        with self.assertRaises(anchor.AnchorUnavailable):
            advance(remote, local, signed("reserve-offline", GENESIS, RUNNING))
        self.assertEqual(remote.current, GENESIS)
        self.assertEqual(local.read(), GENESIS)
        self.assertEqual(local.write_calls, 0)

    def test_remote_commit_then_local_crash_remains_spent(self):
        remote, local = FakeIndependentAuthority(), LocalProjection()
        local.fail_write = True
        operation = signed("reserve-crash", GENESIS, RUNNING)
        with self.assertRaisesRegex(anchor.AnchorReconciliationRequired,
                                    "remote authority advanced"):
            advance(remote, local, operation)
        self.assertEqual(remote.current, RUNNING)
        self.assertEqual(local.read(), GENESIS)
        local.fail_write = False
        with self.assertRaises(anchor.AnchorReconciliationRequired):
            advance(remote, local, operation)
        self.assertEqual(local.read(), GENESIS)
        self.assertEqual(remote.used_ids, {"reserve-crash"})

    def test_lost_remote_ack_is_unknown_committed_not_a_retry(self):
        remote, local = FakeIndependentAuthority(), LocalProjection()
        remote.drop_ack_after_commit = True
        operation = signed("reserve-lost-ack", GENESIS, RUNNING)
        with self.assertRaisesRegex(anchor.AnchorReconciliationRequired,
                                    "outcome unknown"):
            advance(remote, local, operation)
        self.assertEqual(remote.current, RUNNING)
        self.assertEqual(local.read(), GENESIS)
        with self.assertRaises(anchor.AnchorReconciliationRequired):
            advance(remote, local, operation)

    def test_partition_after_remote_ack_before_local_projection_holds(self):
        remote, local = FakeIndependentAuthority(), LocalProjection()
        remote.partition_after_commit = True
        with self.assertRaisesRegex(anchor.AnchorReconciliationRequired,
                                    "readback unavailable"):
            advance(remote, local, signed("commit-then-partition", GENESIS, RUNNING))
        self.assertEqual(remote.current, RUNNING)
        self.assertEqual(local.read(), GENESIS)
        self.assertEqual(local.write_calls, 0)

    def test_two_concurrent_reservations_only_one_remote_owner(self):
        remote, local = FakeIndependentAuthority(), LocalProjection()
        successes: list[str] = []
        rejected: list[Exception] = []

        def worker(index: int) -> None:
            operation = signed(f"event-{index}", GENESIS, RUNNING)
            try:
                advance(remote, local, operation)
                successes.append(operation.operation_id)
            except anchor.AnchorError as exc:
                rejected.append(exc)

        threads = [Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(rejected), 7)
        self.assertEqual(len(remote.used_ids), 1)
        self.assertEqual(remote.current, RUNNING)
        self.assertEqual(local.read(), RUNNING)

    def test_stale_worker_rejected_after_fence_advances(self):
        remote, local = FakeIndependentAuthority(), LocalProjection()
        advance(remote, local, signed("reserve", GENESIS, RUNNING))
        self.assertEqual(anchor.require_current_worker_fence(
            remote, authority_id=RUNNING.authority_id,
            receipt_key=AUTHORITY_KEY, minimum_epoch=1,
            run_id=RUNNING.run_id, event_id=RUNNING.event_id,
            fence=0, checkpoint_sha256=RUNNING.checkpoint_sha256,
        ), RUNNING)
        advance(remote, local, signed("checkpoint-2", RUNNING, PROMOTED))
        for wrong in ((0, RUNNING.checkpoint_sha256),
                      (1, RUNNING.checkpoint_sha256)):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(
                anchor.AnchorError, "stale or cross-run worker"):
                anchor.require_current_worker_fence(
                    remote, authority_id=RUNNING.authority_id,
                    receipt_key=AUTHORITY_KEY, minimum_epoch=2,
                    run_id=RUNNING.run_id, event_id=RUNNING.event_id,
                    fence=wrong[0], checkpoint_sha256=wrong[1],
                )

    def test_unauthenticated_receipt_or_request_never_writes_locally(self):
        remote, local = FakeIndependentAuthority(), LocalProjection()
        remote.corrupt_ack = True
        with self.assertRaisesRegex(anchor.AnchorReconciliationRequired,
                                    "receipt untrusted"):
            advance(remote, local, signed("reserve-corrupt", GENESIS, RUNNING))
        self.assertEqual(remote.current, RUNNING)
        self.assertEqual(local.read(), GENESIS)
        self.assertEqual(local.write_calls, 0)
        operation = signed("new-op", RUNNING, PROMOTED)
        forged = replace(operation, request_signature_hex="f" * 64)
        with self.assertRaisesRegex(anchor.AnchorError, "signature mismatch"):
            anchor.verify_operation(forged, request_key=REQUEST_KEY)

    def test_old_valid_signed_snapshot_rejected_given_protected_epoch_floor(self):
        remote, local = FakeIndependentAuthority(), LocalProjection()
        old = remote.read(GENESIS.authority_id)
        advance(remote, local, signed("reserve", GENESIS, RUNNING))
        remote.replay_snapshot = old
        with self.assertRaisesRegex(anchor.AnchorError, "stale or mismatched"):
            anchor.read_anchor(remote, authority_id=GENESIS.authority_id,
                               receipt_key=AUTHORITY_KEY, minimum_epoch=1)
        # If the epoch floor is also rolled back, a once-valid signature alone
        # cannot establish freshness: the external service must be independent.
        self.assertEqual(anchor.read_anchor(
            remote, authority_id=GENESIS.authority_id,
            receipt_key=AUTHORITY_KEY, minimum_epoch=0), GENESIS)

    def test_transition_rejects_reused_run_and_backward_checkpoint(self):
        with self.assertRaisesRegex(anchor.AnchorError, "run ownership"):
            signed("switch-run", RUNNING,
                   replace(PROMOTED, run_id="run-other"))
        with self.assertRaisesRegex(anchor.AnchorError, "new checkpoint"):
            signed("same-checkpoint", RUNNING,
                   replace(PROMOTED, checkpoint_sha256=RUNNING.checkpoint_sha256))
        with self.assertRaisesRegex(anchor.AnchorError, "epoch"):
            signed("version-replay", RUNNING,
                   replace(PROMOTED, epoch=RUNNING.epoch))
        with self.assertRaisesRegex(anchor.AnchorError, "must differ"):
            anchor.sign_operation("same-secret", GENESIS, RUNNING,
                                  request_key=REQUEST_KEY,
                                  authority_receipt_key=REQUEST_KEY)


if __name__ == "__main__":
    unittest.main()
