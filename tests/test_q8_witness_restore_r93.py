"""Negative Q8 source-CAS tests with a deliberately untrusted SQLite toy port.

The signing keys and database are test-owned. Nothing below is proof of a
separate operator, hardware monotonicity, independent witness or Q8 PASS.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.external_monotonic_authority_contract import ExternalAuthorityHead
from labyrinth.q8_witness_restore_r93 import (
    Q8PlasticChange, Q8WitnessPins, Q8WitnessRecord, Q8SourceRejection,
    Q8OutcomeUnknown, Q8SourceCASObservation, probe_q8_source_cas,
)


def _pub(private):
    return private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _digest(data):
    return hashlib.sha256(data.encode("ascii")).hexdigest()


class ToySQLitePort:
    """Intentionally INSECURE, co-owned test signer and DB; never deploy."""

    def __init__(self, path: Path, service, witness):
        self.path = path
        self.service, self.witness = service, witness
        self.crash_after_commit = False
        self.tamper_after = False
        self.reuse_journal_generation = False
        self.invocations = 0
        with self._open() as db:
            db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, witness TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS used (event_id TEXT PRIMARY KEY)")
            if db.execute("SELECT count(*) FROM state").fetchone()[0] == 0:
                genesis = self._signed(Q8WitnessRecord(
                    host_id="host01", run_id="run01", epoch=3, fence=9,
                    commit_index=10, head_sha256="a" * 64,
                    journal_generation=5, journal_sha256="b" * 64,
                    witness_counter=30, previous_witness_sha256="0" * 64,
                    event_id="genesis", state_change_sha256="0" * 64,
                    service_signature_hex="0" * 128,
                    witness_signature_hex="0" * 128))
                db.execute("INSERT INTO state (id,witness) VALUES (1,?)", (self._json(genesis),))

    @staticmethod
    def _json(record):
        return json.dumps(asdict(record), sort_keys=True)

    def _open(self):
        return sqlite3.connect(str(self.path), timeout=2.0)

    def _signed(self, record):
        payload = record.signed_payload()
        return replace(record,
                       service_signature_hex=self.service.sign(
                           b"labyrinth.q8.r93.service-head.v1\0" + payload).hex(),
                       witness_signature_hex=self.witness.sign(
                           b"labyrinth.q8.r93.witness-head.v1\0" + payload).hex())

    def read_witness_record(self, run_id):
        with self._open() as db:
            row = db.execute("SELECT witness FROM state WHERE id=1").fetchone()[0]
        record = Q8WitnessRecord(**json.loads(row))
        if self.tamper_after:
            return replace(record, journal_generation=max(1, record.journal_generation - 1))
        return record

    def read_service_head(self, run_id):
        record = self.read_witness_record(run_id)
        return ExternalAuthorityHead("service01", record.run_id, record.epoch,
                                     record.fence, record.commit_index, record.head_sha256)

    def compare_and_swap_once(self, change, expected):
        self.invocations += 1
        with self._open() as db:
            db.execute("BEGIN IMMEDIATE")
            current = Q8WitnessRecord(**json.loads(
                db.execute("SELECT witness FROM state WHERE id=1").fetchone()[0]))
            if current != expected:
                raise ValueError("stale compare-and-swap predecessor")
            db.execute("INSERT INTO used (event_id) VALUES (?)", (change.event_id,))
            nxt = self._signed(replace(current,
                commit_index=current.commit_index + 1,
                head_sha256=_digest(current.head_sha256 + change.digest()),
                journal_sha256=_digest(current.journal_sha256 + change.digest()),
                journal_generation=(current.journal_generation
                    if self.reuse_journal_generation
                    else current.journal_generation + 1),
                witness_counter=current.witness_counter + 1,
                previous_witness_sha256=current.digest(),
                event_id=change.event_id, state_change_sha256=change.digest()))
            db.execute("UPDATE state SET witness=? WHERE id=1", (self._json(nxt),))
            db.commit()
        if self.crash_after_commit:
            raise OSError("test crash after durable commit before ACK")
        return nxt


class Q8ExternalWitnessSourceTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix="q8-r93-")
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "toy.sqlite3"
        self.service = Ed25519PrivateKey.generate()
        self.witness = Ed25519PrivateKey.generate()
        self.port = ToySQLitePort(self.db, self.service, self.witness)
        self.change = Q8PlasticChange(
            run_id="run01", event_id="event01", target_id="adapter01",
            prior_sha256="c" * 64, successor_sha256="d" * 64,
            evidence_sha256="e" * 64, authority_lease_sha256="f" * 64)
        self.pins = self._pins()

    def _pins(self):
        record = self.port.read_witness_record("run01")
        return Q8WitnessPins(
            host_id="host01", provider_id="service01", run_id="run01",
            service_public_key=_pub(self.service), witness_public_key=_pub(self.witness),
            minimum_witness_counter=record.witness_counter,
            minimum_witness_digest=record.digest())

    def test_real_toy_sqlite_cas_and_two_signatures_are_source_only(self):
        result = probe_q8_source_cas(port=self.port, pins=self.pins, change=self.change)
        self.assertIsInstance(result, Q8SourceCASObservation)
        self.assertEqual(result.service_commit_index, 11)
        self.assertFalse(result.physical_q8_verified)
        self.assertFalse(result.graduation_authorized)
        self.assertFalse(result.deployment_authorized)
        self.assertEqual(self.port.invocations, 1)

    def test_preflight_witness_rollback_refuses_before_any_cas(self):
        replace_pins = replace(self.pins, minimum_witness_counter=35)
        with self.assertRaisesRegex(Q8SourceRejection, "rollback"):
            probe_q8_source_cas(port=self.port, pins=replace_pins, change=self.change)
        self.assertEqual(self.port.invocations, 0)

    def test_fresh_process_reopen_preserves_consumption_and_journal_head(self):
        probe_q8_source_cas(port=self.port, pins=self.pins, change=self.change)
        reopened = ToySQLitePort(self.db, self.service, self.witness)
        self.assertEqual(reopened.read_witness_record("run01").commit_index, 11)
        with self.assertRaisesRegex(Q8OutcomeUnknown, "reconcile"):
            probe_q8_source_cas(port=reopened, pins=self._pins(),
                change=replace(self.change, prior_sha256="0" * 64,
                               successor_sha256="1" * 64))
        self.assertEqual(reopened.read_witness_record("run01").commit_index, 11)

    def test_lost_ack_after_committed_cas_is_unknown_not_retryable(self):
        self.port.crash_after_commit = True
        with self.assertRaisesRegex(Q8OutcomeUnknown, "reconcile"):
            probe_q8_source_cas(port=self.port, pins=self.pins, change=self.change)
        self.assertEqual(ToySQLitePort(self.db, self.service, self.witness)
                         .read_witness_record("run01").commit_index, 11)
        with self.assertRaisesRegex(Q8SourceRejection, "rollback"):
            probe_q8_source_cas(port=self.port, pins=self.pins, change=self.change)
        self.assertEqual(self.port.invocations, 1)

    def test_crash_reopen_rollback_clone_detected_by_external_highwatermark(self):
        backup = Path(self.tmp.name) / "pre.sqlite3"
        shutil.copyfile(self.db, backup)
        probe_q8_source_cas(port=self.port, pins=self.pins, change=self.change)
        highwater = self._pins()
        # Simulate a local host administrator overwriting the service volume.
        shutil.copyfile(backup, self.db)
        reopened = ToySQLitePort(self.db, self.service, self.witness)
        with self.assertRaisesRegex(Q8SourceRejection, "rollback"):
            probe_q8_source_cas(port=reopened, pins=highwater,
                change=replace(self.change, event_id="event02"))
        self.assertEqual(reopened.invocations, 0)

    def test_malicious_one_role_key_cannot_claim_two_independent_signers(self):
        forged = replace(self.pins, witness_public_key=self.pins.service_public_key)
        with self.assertRaisesRegex(Q8SourceRejection, "distinct"):
            probe_q8_source_cas(port=self.port, pins=forged, change=self.change)
        self.assertEqual(self.port.invocations, 0)

    def test_foreign_witness_signer_rejected_before_cas(self):
        foreign = Ed25519PrivateKey.generate()
        with self.assertRaisesRegex(Q8SourceRejection, "preflight"):
            probe_q8_source_cas(port=self.port,
                pins=replace(self.pins, witness_public_key=_pub(foreign)),
                change=self.change)
        self.assertEqual(self.port.invocations, 0)

    def test_post_construct_pin_mutation_cannot_become_authority(self):
        object.__setattr__(self.pins, "witness_public_key", self.pins.service_public_key)
        with self.assertRaisesRegex(Q8SourceRejection, "distinct"):
            probe_q8_source_cas(port=self.port, pins=self.pins, change=self.change)

    def test_unbound_state_family_or_successor_is_refused_preflight(self):
        for bad in (replace(self.change, state_family="identity"),
                    replace(self.change, reversible=False),
                    replace(self.change, successor_sha256=self.change.prior_sha256)):
            with self.subTest(bad=bad):
                with self.assertRaises(Q8SourceRejection):
                    probe_q8_source_cas(port=self.port, pins=self.pins, change=bad)
        self.assertEqual(self.port.invocations, 0)

    def test_changed_signed_record_is_rejected_before_cas(self):
        self.port.tamper_after = True
        with self.assertRaisesRegex(Q8SourceRejection, "preflight"):
            probe_q8_source_cas(port=self.port, pins=self.pins, change=self.change)
        self.assertEqual(self.port.invocations, 0)

    def test_mismatched_service_provider_is_rejected_before_cas(self):
        with self.assertRaisesRegex(Q8SourceRejection, "invalid service head"):
            probe_q8_source_cas(port=self.port,
                pins=replace(self.pins, provider_id="foreign"), change=self.change)
        self.assertEqual(self.port.invocations, 0)

    def test_post_commit_unsigned_observation_mutation_is_unknown(self):
        class Malicious(ToySQLitePort):
            def compare_and_swap_once(inner, change, expected):
                real = super().compare_and_swap_once(change, expected)
                return replace(real, journal_sha256="d" * 64)
        malformed = Malicious(self.db, self.service, self.witness)
        with self.assertRaisesRegex(Q8OutcomeUnknown, "reconcile"):
            probe_q8_source_cas(port=malformed, pins=self.pins, change=self.change)
        self.assertEqual(malformed.read_witness_record("run01").commit_index, 11)

    def test_signed_but_nonadvancing_journal_generation_is_unknown(self):
        self.port.reuse_journal_generation = True
        with self.assertRaisesRegex(Q8OutcomeUnknown, "reconcile"):
            probe_q8_source_cas(port=self.port, pins=self.pins, change=self.change)
        self.assertEqual(self.port.read_witness_record("run01").commit_index, 11)

    def test_port_mutating_intent_before_commit_cannot_rebind_expected_change(self):
        class MutatingPort(ToySQLitePort):
            def compare_and_swap_once(inner, change, expected):
                # Port crosses the trust boundary. The caller passed a frozen
                # dataclass, but object.__setattr__ bypasses frozen=True.
                object.__setattr__(change, "successor_sha256", "1" * 64)
                return super().compare_and_swap_once(change, expected)

        malicious = MutatingPort(self.db, self.service, self.witness)
        with self.assertRaisesRegex(Q8OutcomeUnknown, "reconcile"):
            probe_q8_source_cas(port=malicious, pins=self.pins, change=self.change)
        # The forged-port operation may have committed: never blindly retry.
        self.assertEqual(malicious.read_witness_record("run01").commit_index, 11)
        # Calling code's original intent object is not overwritten.
        self.assertEqual(self.change.successor_sha256, "d" * 64)

    def test_port_mutating_expected_head_cannot_forge_zero_advance_as_cas(self):
        class ForgingPort(ToySQLitePort):
            def compare_and_swap_once(inner, change, expected):
                original = inner.read_witness_record("run01")
                # Tamper with the exact caller-supplied predecessor object.
                # Current signed state remains index=10, counter=30, gen=5.
                object.__setattr__(expected, "commit_index", 9)
                object.__setattr__(expected, "witness_counter", 29)
                object.__setattr__(expected, "journal_generation", 4)
                # Forge a new dual-signed record without incrementing index,
                # counter or generation, but satisfy mutated predecessor math.
                fake = inner._signed(replace(original,
                    head_sha256="e" * 64, journal_sha256="f" * 64,
                    previous_witness_sha256=expected.digest(),
                    event_id=change.event_id,
                    state_change_sha256=change.digest()))
                with inner._open() as db:
                    db.execute("UPDATE state SET witness=? WHERE id=1", (inner._json(fake),))
                return fake

        malicious = ForgingPort(self.db, self.service, self.witness)
        with self.assertRaisesRegex(Q8OutcomeUnknown, "reconcile"):
            probe_q8_source_cas(port=malicious, pins=self.pins, change=self.change)
        self.assertEqual(malicious.read_witness_record("run01").commit_index, 10)

    def test_result_flags_cannot_be_mutated_to_authority(self):
        observation = probe_q8_source_cas(port=self.port, pins=self.pins, change=self.change)
        with self.assertRaises(AttributeError):
            object.__setattr__(observation, "physical_q8_verified", True)
        with self.assertRaises(TypeError):
            Q8SourceCASObservation("0" * 64, 11, physical_q8_verified=True)
