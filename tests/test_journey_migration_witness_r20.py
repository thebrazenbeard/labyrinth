"""Source-only legacy migration preflight; synthetic anchor is not a host."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
from pathlib import Path
import sqlite3
import tempfile
import unittest

from labyrinth import host_authority_reservation_r12 as old
from labyrinth.journey_migration_witness_r20 import (
    LegacyMigrationHold,
    inspect_legacy_source_inventory,
)


PROVIDER = b"migration-provider-key-unique-000000001"
STATE = b"migration-host-state-key-unique-00001"
RECEIPT = b"migration-host-receipt-key-unique-001"
ANCHOR_RECEIPT = b"migration-anchor-receipt-key-unique1"
ANCHOR_ACTION = b"migration-anchor-action-key-unique-01"
SECRETS = dict(provider_key=PROVIDER, state_key=STATE,
               host_receipt_key=RECEIPT, anchor_receipt_key=ANCHOR_RECEIPT,
               anchor_request_key=ANCHOR_ACTION)
BINDING = old.ProviderBinding(
    "auth-one", "host-one", "provider-one",
    sha256(PROVIDER).hexdigest(), "f" * 64)


class FakeAnchor:
    """Ephemeral test adapter, NOT independent anti-rollback authority."""

    @staticmethod
    def signed(epoch: int, history: str) -> old.EpochReceipt:
        identity = old.EpochReceipt(BINDING.sha256, epoch, history, "")
        return replace(identity, signature_hex=old._mac(
            ANCHOR_RECEIPT, old._ANCHOR, identity.signed_bytes()))

    def __init__(self) -> None:
        self.current = self.signed(0, "0" * 64)

    def read(self, _binding_sha256: str) -> old.EpochReceipt:
        return self.current

    def compare_and_swap(self, before: old.EpochReceipt, after: old.EpochReceipt,
                         event_id: str, proposal_sha256: str,
                         request_mac_hex: str) -> old.EpochReceipt:
        payload = old._json([
            before.binding_sha256, before.epoch, before.history_sha256,
            after.epoch, after.history_sha256, event_id, proposal_sha256,
        ])
        assert hmac.compare_digest(
            request_mac_hex, old._mac(ANCHOR_ACTION, old._ADVANCE, payload))
        assert self.current == before
        assert after.epoch == before.epoch + 1
        assert after.history_sha256 == old._history(
            before.history_sha256, event_id, proposal_sha256)
        self.current = self.signed(after.epoch, after.history_sha256)
        return self.current


def proposal(event_id: str, exam_id: str, fence: int) -> bytes:
    return old._json({
        "schema": "labyrinth.signed-journey-proposal.v1",
        "event_id": event_id, "exam_id": exam_id,
        "scope_sha256": "a" * 64, "contract_sha256": "b" * 64,
        "journey_sha256": "c" * 64, "learner_checkpoint_sha256": "d" * 64,
        "signed_binding_sha256": "e" * 64,
        "training_roster_sha256": "f" * 64,
        "prior_cycle": fence - 1, "regenerated_cycle": fence,
        "expected_host_fence": fence, "weak_skills": ["reasoning"],
        "rooms": [{"room_id": "room-one", "skill": "reasoning", "case_seed": 1}],
        "answerless_train_pairs": [{"challenge_id": "case-one",
                                     "prompt_sha256": "1" * 64,
                                     "skill": "reasoning"}],
    })


class MigrationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "authority.sqlite"
        self.anchor = FakeAnchor()
        self.service = old.HostProviderReservationService(
            self.path, BINDING, external=self.anchor, **SECRETS)
        self.service.initialize()

    def reserve(self, event_id: str, exam_id: str, epoch: int) -> None:
        data = proposal(event_id, exam_id, epoch + 1)
        request = old.sign_provider_request(
            BINDING, provider_key=PROVIDER, event_id=event_id,
            proposal_bytes=data, expected_epoch=epoch)
        self.service.reserve(request, data)

    def test_empty_source_inventory_is_deterministic_but_not_migration(self) -> None:
        inventory = inspect_legacy_source_inventory(self.service)
        self.assertEqual(inventory.epoch, 0)
        self.assertEqual(inventory.events, ())
        self.assertEqual(inventory.history_sha256, "0" * 64)
        self.assertEqual(inventory.missing_identity_fields,
                         ("exam_id", "run_id", "proposal_bytes"))
        self.assertFalse(inventory.migration_authorized)
        self.assertFalse(inventory.independent_host_custody_verified)
        self.assertFalse(inventory.exam_or_run_uniqueness_verified)
        self.assertEqual(inventory.inventory_sha256,
                         inspect_legacy_source_inventory(self.service).inventory_sha256)
        with self.assertRaisesRegex(LegacyMigrationHold, "omit exam_id, run_id"):
            inventory.require_authorized_migration()

    def test_event_inventory_survives_source_restart_exactly(self) -> None:
        self.reserve("event-one", "exam-private", 0)
        self.reserve("event-two", "exam-second", 1)
        first = inspect_legacy_source_inventory(self.service)
        self.assertEqual(first.epoch, 2)
        self.assertEqual([e.event_id for e in first.events],
                         ["event-one", "event-two"])
        self.assertEqual([e.epoch for e in first.events], [1, 2])
        self.assertEqual(first.history_sha256, self.anchor.current.history_sha256)
        reopened = old.HostProviderReservationService(
            self.path, BINDING, external=self.anchor, **SECRETS)
        second = inspect_legacy_source_inventory(reopened)
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(first.inventory_sha256, second.inventory_sha256)
        self.assertNotIn(b"exam-private", first.canonical_bytes())
        self.assertNotIn(b"exam-second", first.canonical_bytes())
        self.assertNotIn(b"run_id\":\"", first.canonical_bytes())
        with self.assertRaises(LegacyMigrationHold):
            second.require_authorized_migration()

    def test_modified_legacy_event_digest_is_rejected_by_source_hmac(self) -> None:
        self.reserve("event-one", "exam-one", 0)
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE consumed SET proposal_sha=? WHERE event_id=?",
                       ("0" * 64, "event-one"))
        with self.assertRaises(LegacyMigrationHold):
            inspect_legacy_source_inventory(self.service)

    def test_rollback_of_source_anchor_is_refused(self) -> None:
        self.reserve("event-one", "exam-one", 0)
        self.anchor.current = self.anchor.signed(0, "0" * 64)
        with self.assertRaisesRegex(LegacyMigrationHold, "disagree"):
            inspect_legacy_source_inventory(self.service)

    def test_added_unverified_column_does_not_make_legacy_complete(self) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("ALTER TABLE consumed ADD COLUMN run_id TEXT")
        with self.assertRaisesRegex(LegacyMigrationHold, "schema differs"):
            inspect_legacy_source_inventory(self.service)


if __name__ == "__main__":
    unittest.main()
