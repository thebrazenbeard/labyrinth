"""Adversarial SOURCE tests. Same UID/host: NEVER independent custody proof."""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path
import os
import socket
import struct
from tempfile import TemporaryDirectory
import threading
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.q6_durable_slot_authority_r120 import (
    Q6DurableConflict, Q6DurableError, Q6DurableOutcomeUnknown,
    Q6DurableHead, Q6DurableSlotAuthority, verify_receipt,
)
from labyrinth.q6_governed_campaign_contract import (
    Q6CampaignReservationIntent, Q6CampaignReservationReceipt,
    build_q6_campaign_manifest, next_q6_head_digest,
)
from labyrinth.q6_stratified_toy_study_r93 import construct_q6_cohorts
from labyrinth.q6_durable_slot_unix_r120 import Q6DurableUnixService, SCHEMA as UNIX_SCHEMA
from labyrinth.authority_unix_service_r22 import _canonical as _wire_json, _recv_frame


class _SameProcessFakeWitness:
    """Hostile simulator only: NOT an independent durable host witness."""

    def __init__(self, head):
        self.head = head
        self.refuse = False
        self.ack_lost = False
        self.attempts = 0

    def read_head(self):
        return self.head

    def compare_and_advance(self, expected, new):
        self.attempts += 1
        if self.refuse or self.head != expected:
            return False
        self.head = new
        if self.ack_lost:
            raise ConnectionError("same-process simulated lost remote witness ACK")
        return True


class Q6DurableSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_q6_campaign_manifest(
            construct_q6_cohorts(seed=7, per_profile=2), seed=7,
            per_profile=2, epochs_stage_a=2, epochs_stage_b=2)

    def setUp(self):
        self.tmp = TemporaryDirectory(prefix="q6-durable-r120-")
        self.addCleanup(self.tmp.cleanup)
        self.database = Path(self.tmp.name) / "operator.sqlite"
        self.signing_seed = bytes.fromhex("41" * 32)
        self.campaign_signer = Ed25519PrivateKey.from_private_bytes(bytes.fromhex("42" * 32))
        self.campaign_key = self.campaign_signer.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.roles = {"owner": "operator", "trainer1": "trainer", "eval1": "evaluator"}
        self.service = self._open(create=True)
        self.receipt = self._campaign("run-one")
        self.service.provision_campaign(
            manifest=self.manifest, receipt=self.receipt, principal="owner")
        self.train = next(case for case in self.manifest.cases if case.partition == "train_a")
        self.eval = next(case for case in self.manifest.cases if case.partition == "heldout")
        self.slot = self.service.address(
            run_id="run-one", partition="train_a", case_id=self.train.case_id,
            arm="trained", phase="after_a", visit=0)
        self.eval_slot = self.service.address(
            run_id="run-one", partition="heldout", case_id=self.eval.case_id,
            arm="trained", phase="after_a", visit=0)

    def _open(self, *, create=False, hook=None, **kwargs):
        args = dict(database_path=self.database, signing_seed=self.signing_seed,
                    campaign_public_key=self.campaign_key, principal_roles=self.roles,
                    create=create, after_commit_hook=hook,
                    allow_unwitnessed_source_simulation=True)
        args.update(kwargs)
        return Q6DurableSlotAuthority(**args)

    def _campaign(self, run_id):
        intent = Q6CampaignReservationIntent(
            provider_id="q6-operator", run_id=run_id,
            campaign_id="campaign-" + run_id,
            manifest_sha256=self.manifest.sha256,
            candidate_checkpoint_sha256="a" * 64,
            operation_nonce="nonce-" + run_id,
            expected_epoch=4, expected_fence=9, expected_commit_index=11,
            expected_head_sha256="b" * 64)
        receipt = Q6CampaignReservationReceipt(
            provider_id=intent.provider_id, run_id=intent.run_id,
            campaign_id=intent.campaign_id, reservation_id="reservation-" + run_id,
            request_sha256=intent.sha256, manifest_sha256=intent.manifest_sha256,
            epoch=intent.expected_epoch, fence=intent.expected_fence,
            commit_index=intent.expected_commit_index + 1,
            previous_head_sha256=intent.expected_head_sha256,
            new_head_sha256=next_q6_head_digest(intent, "reservation-" + run_id),
            status="CAMPAIGN_RESERVED_SOURCE_ONLY", signature_hex="")
        return replace(receipt, signature_hex=self.campaign_signer.sign(
            receipt.signed_payload()).hex())

    def _begin(self, slot=None, **kwargs):
        return self.service.begin_exposure(
            slot_id=(slot or self.slot).slot_id,
            expected_head=kwargs.pop("expected_head", self.service.read_head()),
            principal=kwargs.pop("principal", "trainer1"), **kwargs)

    def _complete(self, slot=None, **kwargs):
        return self.service.complete_exposure(
            slot_id=(slot or self.slot).slot_id,
            expected_head=kwargs.pop("expected_head", self.service.read_head()),
            principal=kwargs.pop("principal", "trainer1"),
            result_sha256=kwargs.pop("result_sha256", "c" * 64),
            execution_sha256=kwargs.pop("execution_sha256", "d" * 64),
            checkpoint_sha256=kwargs.pop("checkpoint_sha256", "e" * 64),
            completion_nonce=kwargs.pop("completion_nonce", "once-one"), **kwargs)

    def _start_witnessed(self):
        """New isolated production-mode DB, with only an IN-PROCESS fake witness."""
        self.database = Path(self.tmp.name) / "witnessed.sqlite"
        self.service = self._open(create=True,
                                  allow_unwitnessed_source_simulation=False)
        witness = _SameProcessFakeWitness(self.service.read_head())
        self.service = self._open(external_witness=witness,
                                  allow_unwitnessed_source_simulation=False)
        self.service.provision_campaign(manifest=self.manifest,
                                        receipt=self.receipt, principal="owner")
        self.assertEqual(witness.head, self.service.read_head())
        return witness

    def test_provision_is_signed_monotonic_event_and_no_prompt_storage(self):
        self.assertEqual(self.service.read_head().commit_index, 1)
        self.assertEqual(self.service.reconcile(self.slot.slot_id, principal="trainer1").state,
                         "RESERVED")
        with sqlite3.connect(self.database) as db:
            raw = " ".join(str(item) for row in db.execute(
                "SELECT receipt_json FROM journal") for item in row)
        self.assertNotIn("gold", raw.lower())
        self.assertFalse(self.service.read_head().source_only_unqualified is False)

    def test_begin_is_durable_before_ack_and_irreversible_across_restart(self):
        before = self.service.read_head()
        receipt = self._begin()
        verify_receipt(receipt, self.service.public_key)
        self.assertEqual(receipt.previous_head_sha256, before.head_sha256)
        self.service = self._open()
        status = self.service.reconcile(self.slot.slot_id, principal="trainer1")
        self.assertEqual(status.state, "BEGUN")
        self.assertEqual(status.begin_receipt, receipt)
        with self.assertRaises(Q6DurableConflict):
            self._begin()

    def test_lost_begin_ack_reconciles_without_resending_reveal(self):
        def crash(event):
            if event == "BEGIN":
                raise OSError("peer lost signed ACK")
        self.service = self._open(hook=crash)
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self._begin()
        self.service = self._open()
        status = self.service.reconcile(self.slot.slot_id, principal="trainer1")
        self.assertEqual(status.state, "BEGUN")
        verify_receipt(status.begin_receipt, self.service.public_key)
        with self.assertRaises(Q6DurableConflict):
            self._begin()

    def test_complete_idempotent_after_ack_loss_but_changed_result_refused(self):
        self._begin()
        def crash(event):
            if event == "COMPLETE":
                raise BrokenPipeError("peer lost completion ACK")
        self.service = self._open(hook=crash)
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self._complete()
        self.service = self._open()
        before = self.service.read_head()
        completed = self.service.reconcile(self.slot.slot_id, principal="trainer1")
        self.assertEqual(completed.state, "COMPLETED")
        self.assertEqual(self._complete(expected_head=before), completed.complete_receipt)
        self.assertEqual(self.service.read_head(), before)
        with self.assertRaises(Q6DurableConflict):
            self._complete(result_sha256="f" * 64)

    def test_stale_head_and_duplicate_slot_before_any_second_exposure(self):
        stale = self.service.read_head()
        self._begin()
        with self.assertRaises(Q6DurableConflict):
            self._begin(expected_head=stale)
        with self.assertRaises(Q6DurableConflict):
            self._complete(expected_head=stale)
        self._complete()
        self.assertEqual(self.service.read_head().commit_index, 3)

    def test_eval_principal_is_not_trainer_and_neither_is_operator(self):
        with self.assertRaises(Q6DurableConflict):
            self._begin(slot=self.eval_slot)
        with self.assertRaises(Q6DurableConflict):
            self._begin(principal="owner")
        with self.assertRaises(Q6DurableConflict):
            self._begin(principal="client-claimed-owner")
        with self.assertRaises(Q6DurableConflict):
            self.service.reconcile(self.eval_slot.slot_id, principal="trainer1")
        eval_receipt = self._begin(slot=self.eval_slot, principal="eval1")
        self.assertEqual(eval_receipt.principal, "eval1")
        verify_receipt(eval_receipt, self.service.public_key)

    def test_unauthorized_campaign_provision_and_cross_run_semantic_replay(self):
        next_receipt = self._campaign("run-two")
        head = self.service.read_head()
        with self.assertRaises(Q6DurableConflict):
            self.service.provision_campaign(
                manifest=self.manifest, receipt=next_receipt, principal="trainer1")
        with self.assertRaises(Q6DurableConflict):
            self.service.provision_campaign(
                manifest=self.manifest, receipt=next_receipt, principal="owner")
        self.assertEqual(head, self.service.read_head())
        with self.assertRaises(Q6DurableConflict):
            self.service.provision_campaign(
                manifest=self.manifest, receipt=self.receipt, principal="owner")

    def test_forged_external_campaign_signature_refused_before_provision(self):
        forged = replace(self._campaign("run-two"), signature_hex="a" * 128)
        head = self.service.read_head()
        with self.assertRaises((Q6DurableOutcomeUnknown, Q6DurableConflict)):
            self.service.provision_campaign(
                manifest=self.manifest, receipt=forged, principal="owner")
        self.assertEqual(head, self.service.read_head())

    def test_signed_receipt_tamper_and_persisted_db_tamper_fail_closed(self):
        receipt = self._begin()
        with self.assertRaises(Q6DurableConflict):
            verify_receipt(replace(receipt, slot_id="0" * 64), self.service.public_key)
        with self.assertRaises(Q6DurableConflict):
            verify_receipt(replace(receipt, graduation_authorized=True), self.service.public_key)
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE slots SET state='RESERVED' WHERE slot_id=?", (self.slot.slot_id,))
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self.service.read_head()
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self._open()

    def test_mutated_journal_signature_fails_closed(self):
        self._begin()
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE journal SET receipt_json='{}' WHERE commit_index=2")
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self.service.reconcile(self.slot.slot_id, principal="trainer1")

    def test_local_rollback_only_detected_with_external_witness(self):
        snapshot = Path(self.tmp.name) / "pre-begin.snapshot"
        with sqlite3.connect(self.database) as live, sqlite3.connect(snapshot) as backup:
            live.backup(backup)
        prior = self.service.read_head()
        self._begin()
        high_water = self.service.read_head()
        self.assertNotEqual(prior, high_water)
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self.service.assert_external_head_matches(prior)
        self.service.assert_external_head_matches(high_water)
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self.service.assert_external_head_matches(None)
        # Complete local DB snapshot restore can be internally consistent;
        # only the separate external HIGH-WATER value detects this rollback.
        with sqlite3.connect(snapshot) as backup, sqlite3.connect(self.database) as live:
            backup.backup(live)
        self.service = self._open()
        self.assertEqual(self.service.read_head(), prior)
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self.service.assert_external_head_matches(high_water)

    def test_two_separate_service_instances_compete_for_one_begin(self):
        import threading
        competitor = self._open()
        barrier = threading.Barrier(2)
        head = self.service.read_head()
        def go(service):
            barrier.wait(timeout=3)
            try:
                return service.begin_exposure(slot_id=self.slot.slot_id,
                                              expected_head=head, principal="trainer1")
            except Q6DurableConflict:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(go, service) for service in (self.service, competitor)]
            events = [future.result() for future in futures]
        self.assertEqual(sum(value is not None for value in events), 1)
        self.assertEqual(self.service.read_head().commit_index, head.commit_index + 1)
        self.assertEqual(self.service.reconcile(self.slot.slot_id, principal="trainer1").state,
                         "BEGUN")

    def test_production_facing_begin_fails_closed_without_external_witness(self):
        self._start_witnessed()
        source_head = self.service.read_head()
        self.service = self._open(allow_unwitnessed_source_simulation=False)
        with self.assertRaisesRegex(Q6DurableOutcomeUnknown, "independently administered"):
            self._begin(expected_head=source_head)
        self.assertEqual(self.service.read_head(), source_head)
        self.assertEqual(self.service.reconcile(self.slot.slot_id, principal="trainer1").state,
                         "RESERVED")

    def test_witnessed_begin_advances_anchor_before_ack_and_detects_rollback(self):
        witness = self._start_witnessed()
        snapshot = Path(self.tmp.name) / "witnessed-pre-begin.sqlite"
        with sqlite3.connect(self.database) as live, sqlite3.connect(snapshot) as backup:
            live.backup(backup)
        attempts = witness.attempts
        self._begin()
        anchored = self.service.read_head()
        self.assertEqual(witness.head, anchored)
        self.assertEqual(witness.attempts, attempts + 1)
        # The previously valid SQLite signed prefix has been restored.
        with sqlite3.connect(snapshot) as backup, sqlite3.connect(self.database) as live:
            backup.backup(live)
        self.service = self._open(external_witness=witness,
                                  allow_unwitnessed_source_simulation=False)
        self.assertEqual(self.service.reconcile(self.slot.slot_id, principal="trainer1").state,
                         "RESERVED")
        with self.assertRaisesRegex(Q6DurableOutcomeUnknown, "witness differs"):
            self._begin()
        self.assertNotEqual(self.service.read_head(), anchored)
        self.assertEqual(witness.head, anchored)

    def test_postcommit_witness_refusal_denies_begin_ack_and_is_recoverable(self):
        witness = self._start_witnessed()
        witness.refuse = True
        self.service = self._open(external_witness=witness,
                                  allow_unwitnessed_source_simulation=False)
        with self.assertRaisesRegex(Q6DurableOutcomeUnknown, "refused CAS"):
            self._begin()
        self.assertEqual(self.service.reconcile(self.slot.slot_id,
                                                principal="trainer1").state, "BEGUN")
        self.assertNotEqual(self.service.read_head(), witness.head)
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self._begin()
        witness.refuse = False
        head = self.service.reconcile_external_anchor(principal="owner")
        self.assertEqual(head, witness.head)
        self.assertEqual(witness.attempts, 3)  # PROVISION, refused BEGIN, recovery
        with self.assertRaises(Q6DurableConflict):
            self._begin()  # no second BEGIN after witness recovery

    def test_witness_cas_commits_then_loses_ack_no_second_prompt(self):
        witness = self._start_witnessed()
        witness.ack_lost = True
        self.service = self._open(external_witness=witness,
                                  allow_unwitnessed_source_simulation=False)
        with self.assertRaisesRegex(Q6DurableOutcomeUnknown, "ACK unknown"):
            self._begin()
        self.assertEqual(self.service.read_head(), witness.head)
        self.assertEqual(self.service.reconcile_external_anchor(principal="owner"),
                         witness.head)
        with self.assertRaises(Q6DurableConflict):
            self._begin()

    def test_witness_ahead_or_divergent_refuses_before_any_commit(self):
        witness = self._start_witnessed()
        old = self.service.read_head()
        witness.head = Q6DurableHead(old.commit_index + 1, "f" * 64)
        self.service = self._open(external_witness=witness,
                                  allow_unwitnessed_source_simulation=False)
        with self.assertRaises(Q6DurableOutcomeUnknown):
            self._begin()
        self.assertEqual(self.service.read_head(), old)
        self.assertEqual(witness.attempts, 1)  # only earlier PROVISION

    def test_restart_rejects_missing_db_wrong_signer_or_role_roster(self):
        with self.assertRaises(Q6DurableConflict):
            self._open(create=True)
        with self.assertRaises(Q6DurableConflict):
            self._open(signing_seed=b"x" * 32)
        with self.assertRaises(Q6DurableConflict):
            self._open(allow_unwitnessed_source_simulation=False)
        prior_roles = self.roles
        self.roles = {"owner": "operator", "attacker": "trainer"}
        try:
            with self.assertRaises(Q6DurableConflict):
                self._open()
        finally:
            self.roles = prior_roles

    def test_unix_peer_credential_binding_and_exact_request_shape(self):
        witness = self._start_witnessed()
        socket_path = Path(self.tmp.name) / "q6.sock"
        uid = os.getuid()
        with Q6DurableUnixService(socket_path=socket_path, core=self.service,
                                  principal_by_uid={uid: "trainer1"}) as server:
            worker = threading.Thread(target=server.serve_forever,
                                      kwargs={"poll_interval": 0.01})
            worker.start()
            try:
                def request(body):
                    payload = _wire_json(body)
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.connect(str(socket_path))
                        client.sendall(struct.pack("!I", len(payload)) + payload)
                        return _recv_frame(client)

                head = self.service.read_head()
                good = {"schema": UNIX_SCHEMA, "op": "begin",
                        "slot_id": self.slot.slot_id,
                        "expected_head": vars(head)}
                forged = dict(good, principal="owner")
                self.assertEqual(request(forged)["error"], "BAD_REQUEST")
                self.assertEqual(request(dict(good, op="provision"))["error"], "BAD_REQUEST")
                result = request(good)
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["value"]["principal"], "trainer1")
                self.assertEqual(request(good)["error"], "CONFLICT")
                recovered = request({"schema": UNIX_SCHEMA, "op": "reconcile",
                                     "slot_id": self.slot.slot_id})
                self.assertTrue(recovered["ok"])
                self.assertEqual(recovered["value"]["state"], "BEGUN")
                self.assertEqual(recovered["value"]["begin_receipt"], result["value"])
                self.assertEqual(request({"schema": UNIX_SCHEMA, "op": "read_head"})["value"]
                                 ["commit_index"], 2)
                self.assertEqual(witness.head, self.service.read_head())
                self.assertEqual(server.dispatch(uid + 70000, good)["error"], "FORBIDDEN")
            finally:
                server.shutdown()
                worker.join(timeout=5)

    def test_unix_socket_refuses_operator_identity_and_unsafe_directory(self):
        socket_path = Path(self.tmp.name) / "q6.sock"
        with self.assertRaisesRegex(Exception, "witness CAS"):
            Q6DurableUnixService(socket_path=socket_path, core=self.service,
                                 principal_by_uid={os.getuid(): "trainer1"})
        self._start_witnessed()
        with self.assertRaises(Exception):
            Q6DurableUnixService(socket_path=socket_path, core=self.service,
                                 principal_by_uid={os.getuid(): "owner"})
        self.assertFalse(socket_path.exists())


if __name__ == "__main__":
    unittest.main()
