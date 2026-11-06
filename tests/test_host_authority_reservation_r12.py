"""Local source-only authorization tests; FakeAnchor is NOT independent hosting."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
from pathlib import Path
import shutil
import sqlite3
import tempfile
from threading import Lock, Thread
import unittest

from labyrinth import host_authority_reservation_r12 as auth


PROVIDER = b"provider-access-key-unique-000000000001"
STATE = b"host-journal-secret-unique-00000000001"
RECEIPT = b"host-receipt-secret-unique-00000000001"
ANCHOR_RECEIPT = b"external-signature-unique-00000000001"
ANCHOR_ACTION = b"external-action-secret-unique-00000001"
SECRETS = dict(provider_key=PROVIDER, state_key=STATE,
               host_receipt_key=RECEIPT, anchor_receipt_key=ANCHOR_RECEIPT,
               anchor_request_key=ANCHOR_ACTION)
BINDING = auth.ProviderBinding("authority-one", "host-one", "provider-alpha",
                               sha256(PROVIDER).hexdigest(), "f" * 64)


class FakeAnchor:
    """Single-lock synthetic *remote* whose state is not in the local database."""

    def __init__(self, binding: auth.ProviderBinding = BINDING):
        self.lock = Lock()
        self.current = self._signed(binding.sha256, 0, "0" * 64)
        self.fail_before = False
        self.fail_after = False
        self.forge_reply = False
        self.read_forge = False
        self.calls = 0

    @staticmethod
    def _signed(identity: str, epoch: int, history: str) -> auth.EpochReceipt:
        plain = auth.EpochReceipt(identity, epoch, history, "")
        return replace(plain, signature_hex=auth._mac(
            ANCHOR_RECEIPT, auth._ANCHOR, plain.signed_bytes()))

    def read(self, binding_sha256: str) -> auth.EpochReceipt:
        with self.lock:
            out = self.current
        if self.read_forge:
            return replace(out, signature_hex="0" * 64)
        return out

    def compare_and_swap(self, before: auth.EpochReceipt, after: auth.EpochReceipt,
                         event_id: str, proposal_sha256: str,
                         request_mac_hex: str) -> auth.EpochReceipt:
        if self.fail_before:
            raise TimeoutError("remote unavailable before CAS")
        command = auth._json([before.binding_sha256, before.epoch,
                              before.history_sha256, after.epoch,
                              after.history_sha256, event_id, proposal_sha256])
        if not hmac.compare_digest(request_mac_hex, auth._mac(
                ANCHOR_ACTION, auth._ADVANCE, command)):
            raise ValueError("remote authentication failed")
        with self.lock:
            self.calls += 1
            if (before != self.current
                    or after.binding_sha256 != before.binding_sha256
                    or after.epoch != before.epoch + 1
                    or after.history_sha256 != auth._history(
                        before.history_sha256, event_id, proposal_sha256)):
                raise ValueError("external CAS stale or changed history")
            self.current = self._signed(after.binding_sha256, after.epoch,
                                        after.history_sha256)
            out = self.current
        if self.fail_after:
            raise TimeoutError("remote commit succeeded but reply lost")
        if self.forge_reply:
            return replace(out, signature_hex="0" * 64)
        return out


def proposal(event_id: str, fence: int, cycle: int = 1) -> bytes:
    return auth._json({
        "schema": "labyrinth.signed-journey-proposal.v1",
        "event_id": event_id, "exam_id": "exam-alpha",
        "scope_sha256": "a" * 64, "contract_sha256": "b" * 64,
        "journey_sha256": "c" * 64, "learner_checkpoint_sha256": "d" * 64,
        "signed_binding_sha256": "e" * 64,
        "training_roster_sha256": "f" * 64,
        "prior_cycle": cycle - 1, "regenerated_cycle": cycle,
        "expected_host_fence": fence, "weak_skills": ["arithmetic"],
        "rooms": [{"room_id": "room-1", "skill": "arithmetic", "case_seed": 8}],
        "answerless_train_pairs": [{"challenge_id": "case-1",
                                    "prompt_sha256": "1" * 64,
                                    "skill": "arithmetic"}],
    })


class HostAuthorityReservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "host-authority.sqlite"
        self.anchor = FakeAnchor()
        self.host = auth.HostProviderReservationService(
            self.db, BINDING, external=self.anchor, **SECRETS)
        self.host.initialize()

    def request(self, event: str, epoch: int, data: bytes | None = None
                ) -> tuple[auth.ProviderRequest, bytes]:
        if data is None:
            data = proposal(event, epoch + 1)
        return (auth.sign_provider_request(BINDING, provider_key=PROVIDER,
                                           event_id=event, proposal_bytes=data,
                                           expected_epoch=epoch), data)

    def test_two_successive_epochs_are_durable_and_replay_spent(self) -> None:
        req1, data1 = self.request("event-one", 0)
        result = self.host.reserve(req1, data1)
        self.assertEqual(result.fence, 1)
        self.assertTrue(self.host.verify_host_receipt(result, result.signed_bytes))
        self.assertEqual(self.host.reconcile("event-one"), result)
        with self.assertRaises(auth.HostAuthorityError):
            self.host.reserve(req1, data1)
        # A fresh handle does not get a fresh signer, event budget or identity.
        reopened = auth.HostProviderReservationService(
            self.db, BINDING, external=self.anchor, **SECRETS)
        req2, data2 = self.request("event-two", 1)
        second = reopened.reserve(req2, data2)
        self.assertEqual(second.fence, 2)
        self.assertEqual(self.anchor.current.epoch, 2)
        with self.assertRaisesRegex(auth.HostAuthorityError, "existing"):
            reopened.initialize()

    def test_changed_host_provider_policy_or_key_cannot_claim_original_database(self) -> None:
        req, data = self.request("event-one", 0)
        for replacement in (
            replace(BINDING, provider_id="provider-evil"),
            replace(BINDING, authority_id="other-authority"),
            replace(BINDING, host_id="host-two"),
            replace(BINDING, policy_sha256="e" * 64),
        ):
            with self.subTest(replacement=replacement):
                wrong = auth.HostProviderReservationService(
                    self.db, replacement, external=self.anchor, **SECRETS)
                with self.assertRaises(auth.HostAuthorityError):
                    wrong.reserve(req, data)
        with self.assertRaises(auth.HostAuthorityError):
            auth.HostProviderReservationService(
                self.db, BINDING, external=self.anchor,
                **{**SECRETS, "provider_key": b"wrong-provider-key-00000000000000001"})

    def test_provider_request_cannot_retarget_a_different_host_or_policy(self) -> None:
        # Same provider secret and authority name, but a separately provisioned
        # host/policy must NOT accept an admission signed for host one.
        req, data = self.request("event-one", 0)
        for suffix, changed in (
            ("host", replace(BINDING, host_id="host-other")),
            ("policy", replace(BINDING, policy_sha256="1" * 64)),
        ):
            with self.subTest(suffix=suffix):
                service = auth.HostProviderReservationService(
                    self.root / f"different-{suffix}.sqlite", changed,
                    external=FakeAnchor(changed), **SECRETS)
                service.initialize()
                with self.assertRaises(auth.HostAuthorityError):
                    service.reserve(req, data)

    def test_host_receipt_signature_is_cryptographically_policy_bound(self) -> None:
        req, data = self.request("event-one", 0)
        receipt = self.host.reserve(req, data)
        changed = replace(BINDING, policy_sha256="9" * 64)
        another = auth.HostProviderReservationService(
            self.root / "other-policy.sqlite", changed,
            external=FakeAnchor(changed), **SECRETS)
        another.initialize()
        self.assertFalse(another.verify_host_receipt(receipt, receipt.signed_bytes))

    def test_signature_and_proposal_contamination_fail_before_remote_effect(self) -> None:
        req, data = self.request("event-one", 0)
        forged = replace(req, signature_hex="0" * 64)
        changed = data.replace(b"exam-alpha", b"exam-omega")
        for candidate, raw in ((forged, data), (req, changed)):
            with self.subTest(candidate=candidate):
                with self.assertRaises(auth.HostAuthorityError):
                    self.host.reserve(candidate, raw)
        malformed = auth._json({"schema": "labyrinth.signed-journey-proposal.v1",
                                "event_id": "event-one", "expected_host_fence": 1})
        unsigned, _ = self.request("event-one", 0, malformed)
        with self.assertRaises(auth.HostAuthorityError):
            self.host.reserve(unsigned, malformed)
        self.assertEqual(self.anchor.calls, 0)

    def test_old_valid_db_snapshot_refused_against_newer_anchor(self) -> None:
        old = self.root / "old-valid.sqlite"
        shutil.copyfile(self.db, old)
        req, data = self.request("event-one", 0)
        self.host.reserve(req, data)
        # Distinct-file old snapshot has a valid original state MAC.
        rollback = self.root / "rolled-back.sqlite"
        shutil.copyfile(old, rollback)
        stale = auth.HostProviderReservationService(
            rollback, BINDING, external=self.anchor, **SECRETS)
        with self.assertRaisesRegex(auth.HostAuthorityHold, "rollback/reconcile"):
            stale.reserve(req, data)
        self.assertEqual(self.anchor.calls, 1)
        self.assertEqual(self.anchor.current.epoch, 1)

    def test_remote_commit_then_lost_ack_requires_manual_reconciliation(self) -> None:
        req, data = self.request("event-one", 0)
        self.anchor.fail_after = True
        with self.assertRaisesRegex(auth.HostAuthorityHold, "outcome unknown"):
            self.host.reserve(req, data)
        self.anchor.fail_after = False
        self.assertEqual(self.anchor.current.epoch, 1)
        with self.assertRaises(auth.HostAuthorityHold):
            self.host.reserve(req, data)
        with self.assertRaises(auth.HostAuthorityHold):
            self.host.reconcile("event-one")

    def test_remote_partition_before_commit_is_fail_closed(self) -> None:
        req, data = self.request("event-one", 0)
        self.anchor.fail_before = True
        with self.assertRaises(auth.HostAuthorityHold):
            self.host.reserve(req, data)
        self.assertEqual(self.anchor.current.epoch, 0)
        self.assertIsNone(self.host.reconcile("event-one"))

    def test_remote_commit_then_local_sqlite_abort_never_replays(self) -> None:
        # A local disk/constraint failure happens strictly AFTER remote CAS.
        with sqlite3.connect(self.db) as db:
            db.execute("""CREATE TRIGGER reject_event BEFORE INSERT ON consumed
                BEGIN SELECT RAISE(ABORT, 'simulated disk fault'); END""")
        req, data = self.request("event-one", 0)
        with self.assertRaisesRegex(auth.HostAuthorityHold, "local projection failed"):
            self.host.reserve(req, data)
        self.assertEqual(self.anchor.current.epoch, 1)
        with self.assertRaisesRegex(auth.HostAuthorityHold, "rollback/reconcile"):
            self.host.reserve(req, data)
        with self.assertRaises(auth.HostAuthorityHold):
            self.host.reconcile("event-one")

    def test_wrong_fence_and_noncanonical_proposals_never_reach_anchor(self) -> None:
        wrong_fence = proposal("event-one", 12)
        req, _ = self.request("event-one", 0, wrong_fence)
        with self.assertRaises(auth.HostAuthorityError):
            self.host.reserve(req, wrong_fence)
        canonical = proposal("event-one", 1)
        altered_format = canonical[:-1] + b" "
        req2, _ = self.request("event-one", 0, altered_format)
        with self.assertRaises(auth.HostAuthorityError):
            self.host.reserve(req2, altered_format)
        self.assertEqual(self.anchor.calls, 0)

    def test_answer_leakage_cannot_hide_inside_signed_train_views(self) -> None:
        import json

        raw = json.loads(proposal("event-one", 1))
        raw["answerless_train_pairs"][0]["answer"] = "4"
        poisoned = auth._json(raw)
        signed, _ = self.request("event-one", 0, poisoned)
        with self.assertRaises(auth.HostAuthorityError):
            self.host.reserve(signed, poisoned)
        self.assertEqual(self.anchor.calls, 0)

    def test_external_signature_forgery_is_denied_without_event_consumption(self) -> None:
        self.anchor.read_forge = True
        req, data = self.request("event-one", 0)
        with self.assertRaises(auth.HostAuthorityHold):
            self.host.reserve(req, data)
        self.anchor.read_forge = False
        self.assertEqual(self.anchor.calls, 0)
        self.anchor.forge_reply = True
        with self.assertRaises(auth.HostAuthorityHold):
            self.host.reserve(req, data)
        self.assertEqual(self.anchor.current.epoch, 1)
        with self.assertRaises(auth.HostAuthorityHold):
            self.host.reconcile("event-one")

    def test_two_concurrent_providers_same_epoch_only_one_commits(self) -> None:
        req1, data1 = self.request("event-one", 0)
        req2, data2 = self.request("event-two", 0)
        results, failures = [], []

        def worker(request, data):
            try:
                results.append(self.host.reserve(request, data))
            except auth.HostAuthorityError as exc:
                failures.append(exc)

        threads = [Thread(target=worker, args=(req1, data1)),
                   Thread(target=worker, args=(req2, data2))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(self.anchor.current.epoch, 1)

    def test_sqlite_tampering_and_receipt_rebinding_fail_closed(self) -> None:
        req, data = self.request("event-one", 0)
        receipt = self.host.reserve(req, data)
        self.assertFalse(self.host.verify_host_receipt(
            replace(receipt, signature_hex="0" * 64), receipt.signed_bytes))
        self.assertFalse(self.host.verify_host_receipt(
            receipt, receipt.signed_bytes + b"extra"))
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE authority SET epoch=0")
        with self.assertRaisesRegex(auth.HostAuthorityError, "seal"):
            self.host.reconcile("event-one")

    def test_rotation_of_only_one_inprocess_key_is_detected(self) -> None:
        req, data = self.request("event-one", 0)
        self.host._provider_key = b"rotated-provider-key-000000000000001"
        with self.assertRaisesRegex(auth.HostAuthorityError, "changed after binding"):
            self.host.reserve(req, data)


if __name__ == "__main__":
    unittest.main()
