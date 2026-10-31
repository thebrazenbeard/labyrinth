"""Host-authority adversarial tests, not evidence of a deployed host service.

The adversary is an authenticated but scope-hostile provider, a filesystem
aliasing actor, or a concurrently executing client. FakeAnchor is independent
only as an in-process test double; no external epoch authority was probed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from threading import Barrier, Thread
import unittest

from labyrinth import host_authority_reservation_r12 as auth
from tests.test_host_authority_reservation_r12 import (
    BINDING, FakeAnchor, PROVIDER, SECRETS,
    proposal,
)


class HostAuthorityRedteamR13(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.anchor = FakeAnchor()
        self.db = self.root / "authority.sqlite"
        self.service = auth.HostProviderReservationService(
            self.db, BINDING, external=self.anchor, **SECRETS)
        self.service.initialize()

    def _request(self, event: str, epoch: int, data: bytes | None = None):
        payload = data if data is not None else proposal(event, epoch + 1)
        request = auth.sign_provider_request(
            BINDING, provider_key=PROVIDER, event_id=event,
            proposal_bytes=payload, expected_epoch=epoch)
        return request, payload

    def test_symlinked_grandparent_does_not_pass_database_path_guard(self) -> None:
        """A host store path may have a non-symlink immediate parent but aliased ancestor."""
        real = self.root / "storage-real"
        child = real / "authority"
        child.mkdir(parents=True)
        alias = self.root / "storage-link"
        alias.symlink_to(real, target_is_directory=True)
        through_alias = alias / "authority" / "new-host.sqlite"
        self.assertFalse(through_alias.parent.is_symlink())  # Threat precondition.
        with self.assertRaises(auth.HostAuthorityError):
            auth.HostProviderReservationService(
                through_alias, BINDING, external=FakeAnchor(), **SECRETS)

    def test_existing_authority_database_hardlink_is_not_an_admissible_alias(self) -> None:
        """A second pathname for the same DB can bypass pathname-level isolation."""
        alias = self.root / "authority-hardlink.sqlite"
        os.link(self.db, alias)
        self.assertGreater(alias.stat().st_nlink, 1)
        with self.assertRaises(auth.HostAuthorityError):
            auth.HostProviderReservationService(
                alias, BINDING, external=self.anchor, **SECRETS)

    def test_authenticated_provider_cannot_request_train_outside_weak_skill(self) -> None:
        """Policy SHA binding must not become merely an opaque string in admission."""
        record = json.loads(proposal("event-one", 1))
        record["rooms"][0]["skill"] = "unauthorized"
        record["answerless_train_pairs"][0]["skill"] = "unauthorized"
        data = auth._json(record)
        signed, payload = self._request("event-one", 0, data)
        with self.assertRaises(auth.HostAuthorityError):
            self.service.reserve(signed, payload)
        self.assertEqual(self.anchor.calls, 0)

    def test_answerless_challenges_cannot_alias_identical_prompts(self) -> None:
        """Distinct case labels do not imply distinct training exposure content."""
        record = json.loads(proposal("event-one", 1))
        duplicate = dict(record["answerless_train_pairs"][0])
        duplicate["challenge_id"] = "case-two"
        record["answerless_train_pairs"].append(duplicate)
        data = auth._json(record)
        signed, payload = self._request("event-one", 0, data)
        with self.assertRaises(auth.HostAuthorityError):
            self.service.reserve(signed, payload)
        self.assertEqual(self.anchor.calls, 0)

    def test_room_and_view_skills_must_match_before_spending_epoch(self) -> None:
        record = json.loads(proposal("event-one", 1))
        record["weak_skills"].append("program_reasoning")
        record["answerless_train_pairs"][0]["skill"] = "program_reasoning"
        request, payload = self._request("event-one", 0, auth._json(record))
        with self.assertRaises(auth.HostAuthorityError):
            self.service.reserve(request, payload)
        self.assertEqual(self.anchor.calls, 0)

    def test_db_path_rechecked_if_alias_appears_after_binding(self) -> None:
        alias = self.root / "created-after-constructor.sqlite"
        new = auth.HostProviderReservationService(
            alias, BINDING, external=self.anchor, **SECRETS)
        os.link(self.db, alias)
        request, payload = self._request("event-one", 0)
        with self.assertRaises(auth.HostAuthorityError):
            new.reserve(request, payload)
        self.assertEqual(self.anchor.calls, 0)

    def test_two_independent_service_handles_race_on_one_fenced_epoch(self) -> None:
        """An actual SQLite race must produce one commit and one denial."""
        other = auth.HostProviderReservationService(
            self.db, BINDING, external=self.anchor, **SECRETS)
        left = self._request("event-left", 0)
        right = self._request("event-right", 0)
        start = Barrier(3)
        successes: list[auth.HostLoopReservation] = []
        failures: list[Exception] = []

        def worker(service, args):
            start.wait()
            try:
                successes.append(service.reserve(*args))
            except Exception as exc:
                failures.append(exc)

        one = Thread(target=worker, args=(self.service, left))
        two = Thread(target=worker, args=(other, right))
        one.start()
        two.start()
        start.wait()
        one.join(timeout=10)
        two.join(timeout=10)
        self.assertFalse(one.is_alive() or two.is_alive())
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], auth.HostAuthorityError)
        self.assertEqual(self.anchor.calls, 1)
        self.assertEqual(self.anchor.current.epoch, 1)

    def test_consumed_event_cannot_be_resigned_at_later_epoch(self) -> None:
        first, body = self._request("event-one", 0)
        spent = self.service.reserve(first, body)
        self.assertTrue(self.service.verify_host_receipt(spent, spent.signed_bytes))
        replay, replay_body = self._request("event-one", 1, proposal("event-one", 2))
        with self.assertRaises(auth.HostAuthorityError):
            self.service.reserve(replay, replay_body)
        self.assertEqual(self.anchor.current.epoch, 1)


if __name__ == "__main__":
    unittest.main()
