"""Adversarial tests of R12 legacy inventory's deliberately narrow trust claim.

These synthetic attackers can replace Python provider adapters and/or possess
the R12 in-process HMAC keys.  Passing tests illustrate why a source inventory
must never be promoted to an independently authenticated migration witness.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
from pathlib import Path
import sqlite3
import tempfile
import unittest

from labyrinth import host_authority_reservation_r12 as legacy
from labyrinth.journey_migration_witness_r20 import (
    LegacyMigrationHold,
    inspect_legacy_source_inventory,
)


PROVIDER = b"r21-provider-source-key-unique-00000001"
STATE = b"r21-state-source-key-unique-000000001"
RECEIPT = b"r21-host-receipt-key-unique-0000001"
ANCHOR_RECEIPT = b"r21-anchor-receipt-key-unique-001"
ANCHOR_ACTION = b"r21-anchor-action-key-unique-0001"
SECRETS = dict(
    provider_key=PROVIDER, state_key=STATE, host_receipt_key=RECEIPT,
    anchor_receipt_key=ANCHOR_RECEIPT, anchor_request_key=ANCHOR_ACTION,
)
BINDING = legacy.ProviderBinding(
    "r21-authority", "r21-host", "r21-provider",
    sha256(PROVIDER).hexdigest(), "f" * 64,
)
ZERO = "0" * 64


class SyntheticAnchor:
    """Replaceable same-key synthetic authority; not a host-owned epoch."""

    @staticmethod
    def signed(epoch: int, history: str) -> legacy.EpochReceipt:
        unsigned = legacy.EpochReceipt(BINDING.sha256, epoch, history, "")
        return replace(unsigned, signature_hex=legacy._mac(
            ANCHOR_RECEIPT, legacy._ANCHOR, unsigned.signed_bytes()))

    def __init__(self) -> None:
        self.current = self.signed(0, ZERO)

    def read(self, _binding_sha256: str) -> legacy.EpochReceipt:
        return self.current

    def compare_and_swap(self, before: legacy.EpochReceipt,
                         after: legacy.EpochReceipt, event_id: str,
                         proposal_sha256: str, request_mac_hex: str
                         ) -> legacy.EpochReceipt:
        command = legacy._json([
            before.binding_sha256, before.epoch, before.history_sha256,
            after.epoch, after.history_sha256, event_id, proposal_sha256,
        ])
        assert hmac.compare_digest(
            request_mac_hex,
            legacy._mac(ANCHOR_ACTION, legacy._ADVANCE, command),
        )
        assert self.current == before
        assert after.epoch == before.epoch + 1
        assert after.history_sha256 == legacy._history(
            before.history_sha256, event_id, proposal_sha256)
        self.current = self.signed(after.epoch, after.history_sha256)
        return self.current


class FlappingAnchor(SyntheticAnchor):
    """Signed authority readbacks change during one inventory transaction."""

    def __init__(self) -> None:
        super().__init__()
        self.read_count = 0

    def read(self, binding_sha256: str) -> legacy.EpochReceipt:
        self.read_count += 1
        if self.read_count > 1:
            return self.signed(1, "a" * 64)
        return super().read(binding_sha256)


def proposal(event_id: str, exam_id: str, fence: int) -> bytes:
    return legacy._json({
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


class LegacyInventoryRedTeam(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "authority.sqlite"
        self.anchor = SyntheticAnchor()
        self.service = legacy.HostProviderReservationService(
            self.path, BINDING, external=self.anchor, **SECRETS)
        self.service.initialize()

    def reserve(self, event_id: str, exam_id: str, epoch: int) -> None:
        data = proposal(event_id, exam_id, epoch + 1)
        request = legacy.sign_provider_request(
            BINDING, provider_key=PROVIDER, event_id=event_id,
            proposal_bytes=data, expected_epoch=epoch)
        self.service.reserve(request, data)

    def test_distinct_events_reusing_exam_are_not_identifiable_as_duplicates(self) -> None:
        self.reserve("event-one", "exam-reused", 0)
        self.reserve("event-two", "exam-reused", 1)
        inventory = inspect_legacy_source_inventory(self.service)
        self.assertEqual(inventory.epoch, 2)
        self.assertEqual(tuple(row.event_id for row in inventory.events),
                         ("event-one", "event-two"))
        self.assertNotIn(b"exam-reused", inventory.canonical_bytes())
        self.assertFalse(inventory.exam_or_run_uniqueness_verified)
        with self.assertRaises(LegacyMigrationHold):
            inventory.require_authorized_migration()

    def test_rebound_same_key_authority_is_source_consistent_not_host_custody(self) -> None:
        self.reserve("event-one", "exam-one", 0)
        first = inspect_legacy_source_inventory(self.service)
        approved_anchor = self.anchor
        forged_anchor = SyntheticAnchor()
        forged_anchor.current = forged_anchor.signed(1, first.history_sha256)
        # R25 rejects normal post-construction rebinds. A same-interpreter
        # adversary remains able to corrupt both the port and its source-only
        # pin, proving this is *not* independently administered host custody.
        object.__setattr__(self.service, "external", forged_anchor)
        object.__setattr__(self.service, "_pinned_external", forged_anchor)
        second = inspect_legacy_source_inventory(self.service)
        self.assertEqual(first.inventory_sha256, second.inventory_sha256)
        self.assertIsNot(self.service.external, approved_anchor)
        self.assertFalse(second.independent_host_custody_verified)
        with self.assertRaises(LegacyMigrationHold):
            second.require_authorized_migration()

    def test_whole_local_db_and_synthetic_anchor_rollback_is_not_host_detected(self) -> None:
        genesis_db = self.path.read_bytes()
        genesis = inspect_legacy_source_inventory(self.service)
        self.reserve("event-one", "exam-one", 0)
        advanced = inspect_legacy_source_inventory(self.service)
        self.assertNotEqual(genesis.inventory_sha256, advanced.inventory_sha256)
        # Synthetic test only: attacker has write authority over BOTH stores.
        self.path.write_bytes(genesis_db)
        self.anchor.current = self.anchor.signed(0, ZERO)
        rolled_back = inspect_legacy_source_inventory(self.service)
        self.assertEqual(genesis.inventory_sha256, rolled_back.inventory_sha256)
        self.assertFalse(rolled_back.independent_host_custody_verified)
        with self.assertRaises(LegacyMigrationHold):
            rolled_back.require_authorized_migration()

    def test_attacker_with_all_python_keys_can_reseal_modified_event(self) -> None:
        self.reserve("event-one", "exam-one", 0)
        forged_proposal_sha = "9" * 64
        forged_history = legacy._history(ZERO, "event-one", forged_proposal_sha)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "UPDATE consumed SET proposal_sha=?,host_signature=? WHERE event_id=?",
                (forged_proposal_sha,
                 self.service._host_signature("event-one", forged_proposal_sha, 1),
                 "event-one"),
            )
            conn.execute(
                "UPDATE authority SET history_sha=?,state_mac=? WHERE id=1",
                (forged_history, self.service._seal(1, forged_history)),
            )
        self.anchor.current = self.anchor.signed(1, forged_history)
        inventory = inspect_legacy_source_inventory(self.service)
        self.assertEqual(inventory.events[0].proposal_sha256, forged_proposal_sha)
        self.assertTrue(inventory.source_hmac_checked)
        self.assertFalse(inventory.independent_host_custody_verified)
        with self.assertRaises(LegacyMigrationHold):
            inventory.require_authorized_migration()

    def test_changed_signed_anchor_readback_refuses_source_inventory(self) -> None:
        flapping = FlappingAnchor()
        object.__setattr__(self.service, "external", flapping)
        object.__setattr__(self.service, "_pinned_external", flapping)
        with self.assertRaisesRegex(LegacyMigrationHold, "disagree"):
            inspect_legacy_source_inventory(self.service)

    def test_forged_inventory_flags_cannot_call_migration_authorizer(self) -> None:
        source = inspect_legacy_source_inventory(self.service)
        attacker_flags = replace(
            source, missing_identity_fields=(), source_hmac_checked=True,
            source_anchor_readback_checked=True,
            independent_host_custody_verified=True,
            exam_or_run_uniqueness_verified=True,
            migration_authorized=True,
        )
        self.assertNotEqual(source.canonical_bytes(), attacker_flags.canonical_bytes())
        with self.assertRaisesRegex(LegacyMigrationHold, "independent complete-history"):
            attacker_flags.require_authorized_migration()


if __name__ == "__main__":
    unittest.main()
