"""Q8 trust-domain falsification; source tests are not host attestations.

Demonstrate which *observable* invariant an independently held monotonic
witness must enforce, and why local SQLite transaction tests alone cannot
establish it. No network, host, or protected signing identity is used here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.labyrinth_sqlite_authority.sqlite_reservation_port import SQLiteReservationPort
from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityConflict,
    ExternalAuthorityError,
    ExternalMonotonicReservationClient,
    ExternalReservationRequest,
)


def _intent() -> ExternalReservationRequest:
    return ExternalReservationRequest(
        provider_id="held-witness", run_id="journey-1",
        event_id="dev-event-1", exam_id="dev-exam-1",
        candidate_artifact_sha256="a" * 64,
        scope_fingerprint="b" * 64,
        exam_lineage_sha256="c" * 64,
        prior_checkpoint_sha256="d" * 64,
        to_cycle=2, expected_epoch=1, expected_fence=4,
        expected_commit_index=0, expected_head_sha256="0" * 64,
        operation_nonce="nonce-1",
    )


class WitnessBoundarySourceTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.intent = _intent()
        self.held = self.port(self.root / "held.sqlite3")
        self.held.operator_bootstrap_run(self.intent.expected_head())
        self.held.operator_stage_intent(self.intent)

    def port(self, path: Path, *, signer=None, key=None) -> SQLiteReservationPort:
        return SQLiteReservationPort(
            db_path=path, provider_id="held-witness",
            sign_bytes=self.private_key.sign if signer is None else signer,
            pinned_ed25519_public_key=self.public_key if key is None else key,
        )

    def client(self, port: SQLiteReservationPort):
        return ExternalMonotonicReservationClient(
            port=port, provider_id=self.intent.provider_id,
            pinned_request=self.intent,
            pinned_ed25519_public_key=self.public_key,
        )

    def _clone(self, name: str) -> SQLiteReservationPort:
        with self.held._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        cloned = self.root / f"{name}.sqlite3"
        shutil.copyfile(self.root / "held.sqlite3", cloned)
        return self.port(cloned)

    def test_separate_cloned_ledgers_and_signer_both_authorize_same_event(self):
        # Red adversarial result, expressed as a passing falsification test.
        first, second = self._clone("clone-a"), self._clone("clone-b")
        receipts = [self.client(p).reserve(self.intent) for p in (first, second)]
        self.assertEqual([r.commit_index for r in receipts], [1, 1])
        self.assertEqual([p.read_head("journey-1").commit_index for p in (first, second)], [1, 1])

    def test_two_isolated_clients_using_one_shared_witness_one_wins(self):
        # The clients may have cloned *trainer* state; only their witness is shared.
        first = self.client(self.held)
        second = self.client(self.held)
        accepted = first.reserve(self.intent)
        self.assertEqual(accepted.commit_index, 1)
        with self.assertRaises(ExternalAuthorityConflict):
            second.reserve(self.intent)
        self.assertEqual(self.held.read_head("journey-1"), accepted.post_head())

    def test_rewinding_even_shared_witness_restores_replay_vulnerability(self):
        stale_backup = self._clone("stale-backup")
        self.client(self.held).reserve(self.intent)
        # Authority becomes forkable again if its storage and identity can roll back.
        second_receipt = self.client(stale_backup).reserve(self.intent)
        self.assertEqual(second_receipt.commit_index, 1)
        self.assertEqual(self.held.read_head("journey-1").commit_index, 1)

    def test_distinct_signer_cannot_impersonate_pinned_witness(self):
        impostor = self._clone("impostor")
        impostor_key = Ed25519PrivateKey.generate()
        wrong = self.port(self.root / "impostor.sqlite3", signer=impostor_key.sign)
        with self.assertRaises(ExternalAuthorityError):
            wrong.reserve_once(self.intent)
        self.assertEqual(impostor.read_head("journey-1").commit_index, 0)

    def test_forged_candidate_is_rejected_even_by_the_shared_witness(self):
        altered = replace(self.intent, candidate_artifact_sha256="e" * 64)
        with self.assertRaises(ExternalAuthorityConflict):
            self.held.reserve_once(altered)
        self.assertEqual(self.held.read_head("journey-1").commit_index, 0)


if __name__ == "__main__":
    unittest.main()
