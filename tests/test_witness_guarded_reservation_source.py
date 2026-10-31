"""Source-only external-witness checks; no protected host attestation."""

from dataclasses import replace
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Lock
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.labyrinth_sqlite_authority.sqlite_reservation_port import SQLiteReservationPort
from experiments.labyrinth_sqlite_authority.witness_guarded_reservation import (
    WitnessCommit, WitnessGuardedReservationPort, receipt_digest,
)
from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityConflict, ExternalAuthorityError,
    ExternalAuthorityOutcomeUnknown, ExternalMonotonicReservationClient,
    ExternalReservationRequest,
)


def _public(private: Ed25519PrivateKey) -> bytes:
    return private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _intent() -> ExternalReservationRequest:
    return ExternalReservationRequest(
        provider_id="monotonic-host", run_id="run-one",
        event_id="event-one", exam_id="exam-one",
        candidate_artifact_sha256="a" * 64,
        scope_fingerprint="b" * 64,
        exam_lineage_sha256="c" * 64,
        prior_checkpoint_sha256="d" * 64,
        to_cycle=2, expected_epoch=1, expected_fence=5,
        expected_commit_index=0, expected_head_sha256="0" * 64,
        operation_nonce="nonce-one",
    )


class TestOnlySharedWitness:
    """NOT durable, independently hosted, uncloneable, or qualified."""

    def __init__(self, intent: ExternalReservationRequest,
                 signing_key: Ed25519PrivateKey):
        self.intent = intent
        self.signing_key = signing_key
        self.head = intent.expected_head()
        self.consumed = set()
        self._lock = Lock()
        self.bad_signature = False
        self.bad_receipt_digest = False

    def read_head(self, run_id):
        if run_id != self.intent.run_id:
            raise ExternalAuthorityConflict("unknown run")
        with self._lock:
            return self.head

    def commit_receipt_once(self, request, receipt):
        with self._lock:
            if (request != self.intent or self.head != request.expected_head()
                    or request.event_id in self.consumed):
                raise ExternalAuthorityConflict("witness fence/event used")
            unsigned = WitnessCommit(
                witness_id="independent-witness", provider_id=request.provider_id,
                run_id=request.run_id, event_id=request.event_id,
                exam_id=request.exam_id, operation_nonce=request.operation_nonce,
                request_sha256=request.digest(),
                local_receipt_sha256=("f" * 64 if self.bad_receipt_digest
                                      else receipt_digest(receipt)),
                epoch=receipt.epoch, fence=receipt.fence,
                commit_index=receipt.commit_index,
                previous_head_sha256=receipt.previous_head_sha256,
                new_head_sha256=receipt.new_head_sha256,
            )
            signature = self.signing_key.sign(unsigned.signed_payload()).hex()
            self.head = receipt.post_head()
            self.consumed.add(request.event_id)
            return replace(unsigned, signature_hex=("0" * 128 if self.bad_signature
                                                     else signature))


class WitnessGuardedSourceTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.request = _intent()
        self.local_key = Ed25519PrivateKey.generate()
        self.local_public = _public(self.local_key)
        self.witness_key = Ed25519PrivateKey.generate()
        self.witness_public = _public(self.witness_key)
        self.source_db = self.root / "source.sqlite3"
        self.local = self._port(self.source_db)
        self.local.operator_bootstrap_run(self.request.expected_head())
        self.local.operator_stage_intent(self.request)
        self.witness = TestOnlySharedWitness(self.request, self.witness_key)

    def _port(self, path):
        return SQLiteReservationPort(
            db_path=path, provider_id=self.request.provider_id,
            sign_bytes=self.local_key.sign,
            pinned_ed25519_public_key=self.local_public,
        )

    def _clone_local(self, name):
        with self.local._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target = self.root / f"{name}.sqlite3"
        shutil.copyfile(self.source_db, target)
        return self._port(target)

    def _client(self, local, witness=None):
        wrapped = WitnessGuardedReservationPort(
            local_proposal=local,
            witness=self.witness if witness is None else witness,
            witness_id="independent-witness",
            provider_id=self.request.provider_id,
            pinned_local_public_key=self.local_public,
            pinned_witness_public_key=self.witness_public,
        )
        return ExternalMonotonicReservationClient(
            port=wrapped,
            provider_id=self.request.provider_id,
            pinned_request=self.request,
            pinned_ed25519_public_key=self.local_public,
        )

    def test_two_cloned_local_ledgers_use_one_shared_external_witness(self):
        local_one, local_two = self._clone_local("clone-a"), self._clone_local("clone-b")
        receipt = self._client(local_one).reserve(self.request)
        self.assertEqual(receipt.commit_index, 1)
        with self.assertRaises(ExternalAuthorityConflict):
            self._client(local_two).reserve(self.request)
        self.assertEqual(local_two.read_head(self.request.run_id).commit_index, 0)
        self.assertEqual(self.witness.read_head(self.request.run_id), receipt.post_head())

    def test_bad_independently_signed_ack_is_outcome_unknown_not_retriable(self):
        self.witness.bad_signature = True
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            self._client(self.local).reserve(self.request)
        self.assertEqual(self.local.read_head(self.request.run_id).commit_index, 1)
        self.assertEqual(self.witness.read_head(self.request.run_id).commit_index, 1)
        with self.assertRaises(ExternalAuthorityConflict):
            self._client(self.local).reserve(self.request)

    def test_signed_ack_for_another_receipt_is_never_accepted(self):
        self.witness.bad_receipt_digest = True
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            self._client(self.local).reserve(self.request)
        self.assertEqual(self.witness.read_head(self.request.run_id).commit_index, 1)

    def test_earlier_unwitnessed_local_commit_is_not_treated_as_safe_retry(self):
        # A direct local SQLite commit is not an authorized external release.
        self.local.reserve_once(self.request)
        self.assertEqual(self.local.read_head(self.request.run_id).commit_index, 1)
        self.assertEqual(self.witness.read_head(self.request.run_id).commit_index, 0)
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            self._client(self.local).reserve(self.request)

    def test_distinct_witness_public_identity_required(self):
        with self.assertRaisesRegex(ExternalAuthorityError, "differ"):
            WitnessGuardedReservationPort(
                local_proposal=self.local, witness=self.witness,
                witness_id="independent-witness",
                provider_id=self.request.provider_id,
                pinned_local_public_key=self.local_public,
                pinned_witness_public_key=self.local_public,
            )

    def test_cloning_witness_storage_and_signer_still_allows_replay(self):
        # This is a deliberate FALSIFIER, not a passing physical Q8 result.
        local_a, local_b = self._clone_local("clone-a"), self._clone_local("clone-b")
        witness_a = TestOnlySharedWitness(self.request, self.witness_key)
        witness_b = TestOnlySharedWitness(self.request, self.witness_key)
        first = self._client(local_a, witness_a).reserve(self.request)
        second = self._client(local_b, witness_b).reserve(self.request)
        self.assertEqual((first.commit_index, second.commit_index), (1, 1))
        self.assertEqual(witness_a.read_head(self.request.run_id).commit_index, 1)
        self.assertEqual(witness_b.read_head(self.request.run_id).commit_index, 1)

    def test_primitive_receipt_types_refuse_proxy_values(self):
        class FakeReceipt:
            def signed_payload(self):
                return b"forged"

        with self.assertRaises(ExternalAuthorityError):
            receipt_digest(FakeReceipt())


if __name__ == "__main__":
    unittest.main()
