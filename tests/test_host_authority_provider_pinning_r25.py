"""R25 source-only provider identity tripwires; NOT physical host attestation.

The synthetic FakeAnchor shares a trust domain with the client. These tests
prove that the service rejects ordinary replacement and detects selected
post-construction mutation before returning authorization, not that a trainer
who controls Python objects, keys, processes or disks cannot forge a provider.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest

from labyrinth import host_authority_reservation_r12 as auth
from test_host_authority_reservation_r12 import (
    BINDING, FakeAnchor, PROVIDER, SECRETS, proposal,
)


class RebindingAfterCommitAnchor(FakeAnchor):
    """Simulate a callback mutating the source object's authority after CAS."""

    host: auth.HostProviderReservationService | None = None

    def compare_and_swap(self, before, after, event_id, proposal_sha256,
                         request_mac_hex):
        reply = super().compare_and_swap(
            before, after, event_id, proposal_sha256, request_mac_hex)
        assert self.host is not None
        # Bypass ordinary __setattr__ intentionally. This can happen inside a
        # same-process adversary; the source must still fail closed at readback.
        object.__setattr__(self.host, "external", FakeAnchor())
        return reply


class ProviderPinningTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = Path(temp.name) / "authority.sqlite"
        self.binding = replace(BINDING)
        self.anchor = FakeAnchor(self.binding)
        self.host = auth.HostProviderReservationService(
            self.db, self.binding, external=self.anchor, **SECRETS)
        self.host.initialize()

    def _request(self):
        body = proposal("event-one", 1)
        return auth.sign_provider_request(
            self.binding, provider_key=PROVIDER, event_id="event-one",
            proposal_bytes=body, expected_epoch=0), body

    def test_direct_rebinding_of_provider_binding_and_db_refused(self) -> None:
        for name, new in (
            ("external", FakeAnchor(self.binding)),
            ("binding", replace(self.binding, host_id="other-host")),
            ("db_path", self.db.parent / "other.sqlite"),
            ("_pinned", False),
        ):
            with self.subTest(field=name):
                with self.assertRaisesRegex(auth.HostAuthorityError, "immutable"):
                    setattr(self.host, name, new)
        req, body = self._request()
        self.assertEqual(self.host.reserve(req, body).fence, 1)

    def test_signing_key_replacement_fails_before_remote_cas(self) -> None:
        self.host._host_key = b"evil-host-key-000000000000000000000001"
        req, body = self._request()
        with self.assertRaisesRegex(auth.HostAuthorityError, "key changed"):
            self.host.reserve(req, body)
        self.assertEqual(self.anchor.calls, 0)

    def test_low_level_provider_swap_fails_closed_before_remote_cas(self) -> None:
        intruder = FakeAnchor(self.binding)
        object.__setattr__(self.host, "external", intruder)
        req, body = self._request()
        with self.assertRaisesRegex(auth.HostAuthorityHold, "provider or binding changed"):
            self.host.reserve(req, body)
        self.assertEqual(self.anchor.calls, 0)
        self.assertEqual(intruder.calls, 0)

    def test_low_level_binding_mutation_fails_before_remote_cas(self) -> None:
        object.__setattr__(self.binding, "policy_sha256", "0" * 64)
        body = proposal("event-one", 1)
        # The signed request is intentionally still in the formerly pinned
        # provider domain; no signer/trainer action may retarget the service.
        req = auth.sign_provider_request(
            BINDING, provider_key=PROVIDER, event_id="event-one",
            proposal_bytes=body, expected_epoch=0)
        with self.assertRaisesRegex(auth.HostAuthorityHold, "provider or binding changed"):
            self.host.reserve(req, body)
        self.assertEqual(self.anchor.calls, 0)

    def test_pointer_rebound_inside_remote_read_is_detected(self) -> None:
        genuine_read = self.anchor.read

        def swapping_read(sha):
            result = genuine_read(sha)
            object.__setattr__(self.host, "external", FakeAnchor(self.binding))
            return result

        self.anchor.read = swapping_read
        req, body = self._request()
        with self.assertRaises(auth.HostAuthorityHold):
            self.host.reserve(req, body)
        self.assertEqual(self.anchor.calls, 0)

    def test_rebind_after_remote_commit_never_returns_training_receipt(self) -> None:
        anchor = RebindingAfterCommitAnchor(self.binding)
        host = auth.HostProviderReservationService(
            self.db, self.binding, external=anchor, **SECRETS)
        anchor.host = host
        req, body = self._request()
        with self.assertRaisesRegex(auth.HostAuthorityHold, "provider or binding changed"):
            host.reserve(req, body)
        # The external event is spent but local projection remains genesis:
        # this is a HOLD needing independent reconciliation, not a retry.
        self.assertEqual(anchor.current.epoch, 1)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT epoch FROM authority").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
