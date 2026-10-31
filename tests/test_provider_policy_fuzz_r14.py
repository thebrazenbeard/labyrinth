"""Adversarial source-only provider/Journey contract tests.

All signatures and epoch CAS receipts come from synthetic fixture keys and
an in-process FakeAnchor. No independently administered host or GPU is used.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from labyrinth import host_authority_reservation_r12 as authority
from test_host_authority_reservation_r12 import (
    BINDING, FakeAnchor, PROVIDER, SECRETS, proposal,
)
from test_provider_journey_adapter_r13 import ProviderJourneyAdapterTests


class ProviderPolicyFuzzR14(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.anchor = FakeAnchor()
        self.db = self.root / "authority.sqlite"
        self.host = authority.HostProviderReservationService(
            self.db, BINDING, external=self.anchor, **SECRETS,
        )
        self.host.initialize()

    def _signed(self, record: dict, event: str = "event-one", epoch: int = 0):
        raw = authority._json(record)
        request = authority.sign_provider_request(
            BINDING, provider_key=PROVIDER, event_id=event,
            proposal_bytes=raw, expected_epoch=epoch,
        )
        return request, raw

    def test_signed_weakness_and_room_view_permutation_matrix_does_not_consume(self):
        base = json.loads(proposal("event-one", 1))

        def alter(key: str, value):
            clone = json.loads(json.dumps(base))
            clone[key] = value
            return clone

        variants = [
            alter("weak_skills", []),
            alter("weak_skills", ["arithmetic", "arithmetic"]),
            alter("weak_skills", [False]),
            alter("weak_skills", "arithmetic"),
        ]
        mismatch = json.loads(json.dumps(base))
        mismatch["rooms"][0]["skill"] = "other_skill"
        variants.append(mismatch)
        mismatch = json.loads(json.dumps(base))
        mismatch["weak_skills"] = ["arithmetic", "program_reasoning"]
        mismatch["answerless_train_pairs"][0]["skill"] = "program_reasoning"
        variants.append(mismatch)
        duplicate = json.loads(json.dumps(base))
        duplicate["rooms"].append({"room_id": "room-2", "skill": "arithmetic", "case_seed": 9})
        duplicate["answerless_train_pairs"].append({
            "challenge_id": "case-2", "prompt_sha256": "1" * 64,  # repeated digest
            "skill": "arithmetic",
        })
        variants.append(duplicate)
        repeated_challenge = json.loads(json.dumps(duplicate))
        repeated_challenge["answerless_train_pairs"][1]["prompt_sha256"] = "2" * 64
        repeated_challenge["answerless_train_pairs"][1]["challenge_id"] = "case-1"
        variants.append(repeated_challenge)
        repeated_room = json.loads(json.dumps(duplicate))
        repeated_room["answerless_train_pairs"][1]["prompt_sha256"] = "2" * 64
        repeated_room["rooms"][1]["room_id"] = "room-1"
        variants.append(repeated_room)
        mismatch = json.loads(json.dumps(base))
        mismatch["rooms"].append({"room_id": "room-2", "skill": "arithmetic", "case_seed": 9})
        variants.append(mismatch)  # no corresponding TRAIN view
        mismatch = json.loads(json.dumps(base))
        mismatch["weak_skills"] = ["arithmetic", "program_reasoning"]
        mismatch["rooms"].append({"room_id": "room-2", "skill": "program_reasoning", "case_seed": 9})
        mismatch["answerless_train_pairs"].append({
            "challenge_id": "case-2", "prompt_sha256": "2" * 64,
            "skill": "program_reasoning",
        })
        mismatch["answerless_train_pairs"].reverse()  # same set, wrong positional mapping
        variants.append(mismatch)
        extra = json.loads(json.dumps(base))
        extra["answerless_train_pairs"][0]["answer"] = "secret"
        variants.append(extra)
        protected = json.loads(json.dumps(base))
        protected["graduation_authorized"] = True
        variants.append(protected)
        protected = json.loads(json.dumps(base))
        protected["decision"] = "GRADUATE"
        variants.append(protected)
        for index, variant in enumerate(variants):
            with self.subTest(index=index):
                request, raw = self._signed(variant)
                with self.assertRaises(authority.HostAuthorityError):
                    self.host.reserve(request, raw)
                self.assertEqual(self.anchor.calls, 0)

    def test_signed_provider_request_is_not_usable_under_changed_host_or_policy(self):
        raw = proposal("event-one", 1)
        request = authority.sign_provider_request(
            BINDING, provider_key=PROVIDER, event_id="event-one",
            proposal_bytes=raw, expected_epoch=0,
        )
        for replacement in (
            replace(BINDING, host_id="another-host"),
            replace(BINDING, policy_sha256="0" * 64),
        ):
            with self.subTest(replacement=replacement):
                other = authority.HostProviderReservationService(
                    self.root / f"other-{replacement.sha256[:8]}.sqlite",
                    replacement, external=FakeAnchor(replacement), **SECRETS,
                )
                other.initialize()
                with self.assertRaises(authority.HostAuthorityError):
                    other.reserve(request, raw)
        self.assertEqual(self.anchor.calls, 0)

    def test_existing_database_hardlink_added_after_constructor_is_denied(self):
        alias = self.root / "late-alias.sqlite"
        other = authority.HostProviderReservationService(
            alias, BINDING, external=self.anchor, **SECRETS,
        )
        os.link(self.db, alias)
        request, raw = self._signed(json.loads(proposal("event-one", 1)))
        with self.assertRaises(authority.HostAuthorityError):
            other.reserve(request, raw)
        self.assertEqual(self.anchor.calls, 0)

    def test_old_signed_host_receipt_denied_when_anchor_readback_forged(self):
        request, raw = self._signed(json.loads(proposal("event-one", 1)))
        receipt = self.host.reserve(request, raw)
        self.assertTrue(self.host.verify_host_receipt(receipt, receipt.signed_bytes))
        self.anchor.read_forge = True
        with self.assertRaises(authority.HostAuthorityError):
            self.host.reconcile("event-one")

    def test_old_host_receipt_cannot_authorize_after_authenticated_epoch_rollback(self):
        request, raw = self._signed(json.loads(proposal("event-one", 1)))
        receipt = self.host.reserve(request, raw)
        self.assertEqual(self.host.reconcile("event-one"), receipt)
        # Synthetic hostile remote returned a cryptographically valid *old*
        # signed genesis receipt. Source cannot prevent rollback of both real
        # stores, but it must refuse this divergent external/local lineage.
        self.anchor.current = self.anchor._signed(BINDING.sha256, 0, "0" * 64)
        with self.assertRaises(authority.HostAuthorityHold):
            self.host.reconcile("event-one")
        self.assertEqual(self.anchor.calls, 1)

    def test_changed_event_and_partial_remote_epoch_authority_do_not_commit(self):
        payload = json.loads(proposal("event-one", 1))
        request, raw = self._signed(payload, event="event-two")
        with self.assertRaises(authority.HostAuthorityError):
            self.host.reserve(request, raw)
        self.assertEqual(self.anchor.calls, 0)
        request, raw = self._signed(payload)
        self.anchor.read_forge = True
        with self.assertRaises(authority.HostAuthorityHold):
            self.host.reserve(request, raw)
        self.anchor.read_forge = False
        self.assertEqual(self.anchor.calls, 0)
        self.anchor.forge_reply = True
        with self.assertRaises(authority.HostAuthorityHold):
            self.host.reserve(request, raw)
        # CAS actually happened but the forged acknowledgment cannot graduate
        # into a usable event; repeating is fenced by the remote epoch.
        self.assertEqual(self.anchor.current.epoch, 1)
        self.anchor.forge_reply = False
        with self.assertRaises(authority.HostAuthorityHold):
            self.host.reserve(request, raw)

    def test_adapter_rejects_substituted_training_view_under_pinned_roster(self):
        """A valid signer must not spend the event for a different question."""
        fixture = ProviderJourneyAdapterTests(
            "test_real_signed_journey_calls_provider_then_remote_cas_and_host_verifier"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        captured: list[bytes] = []

        def capture(raw: bytes):
            captured.append(raw)
            raise RuntimeError("capture before external reservation")

        with self.assertRaisesRegex(RuntimeError, "capture"):
            fixture.prepare(host_reserve_once=capture)
        self.assertEqual(len(captured), 1)
        original = json.loads(captured[0])
        altered = json.loads(json.dumps(original))
        altered["answerless_train_pairs"][0]["prompt_sha256"] = "0" * 64
        self.assertNotEqual(altered["answerless_train_pairs"],
                            original["answerless_train_pairs"])
        try:
            accepted = fixture.adapter.host_reserve_once(authority._json(altered))
        except authority.HostAuthorityError:
            accepted = None
        self.assertEqual(
            fixture.anchor.current.epoch, 0,
            "substituted TRAIN prompt spent one host epoch under the unchanged roster pin",
        )
        self.assertIsNone(accepted)


if __name__ == "__main__":
    unittest.main()
