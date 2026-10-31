"""Real Unix-socket/process integration for the source-only authority prototype.

These Linux tests exercise peer-credential extraction and signed transport
behavior across processes.  They do NOT establish a privileged deployment:
the fixture's server and trainer intentionally run under the same Unix UID,
and the signer, database and operator roster are all fixture-controlled.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import socket
import stat
import struct
import sys
from tempfile import TemporaryDirectory
import time
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.authority_service_core_r20 import (
    HostProvisionedJourney, SourceOnlyJourneyAuthorityCore,
)
from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityConflict, ExternalAuthorityOutcomeUnknown,
    ExternalMonotonicReservationClient,
)
from labyrinth.journey_external_authority_contract_r17 import (
    JourneyExternalAuthorityBinding, build_journey_external_request,
)


def _binding(run_id: str, event_id: str, exam_id: str) -> JourneyExternalAuthorityBinding:
    return JourneyExternalAuthorityBinding(
        provider_id="provider-a", host_id="host-a", run_id=run_id,
        event_id=event_id, exam_id=exam_id, scope_sha256="1" * 64,
        objective_contract_sha256="2" * 64, exam_lineage_sha256="3" * 64,
        journey_sha256="4" * 64, signed_binding_sha256="5" * 64,
        learner_checkpoint_sha256="6" * 64, prior_training_roster_sha256="7" * 64,
        next_training_allocation_sha256="8" * 64, host_policy_sha256="9" * 64,
        source_tree_sha256="a" * 64, fullstate_binding_sha256="b" * 64,
        consumed_event_ledger_root_sha256="c" * 64,
        migration_witness_sha256="d" * 64, previous_cycle=0, next_cycle=1,
    )


def _pin(binding: JourneyExternalAuthorityBinding) -> HostProvisionedJourney:
    return HostProvisionedJourney(
        binding=binding, authenticated_principal_id="trainer-a",
        operator_pinned_ledger_root_sha256="c" * 64,
        operator_pinned_migration_witness_sha256="d" * 64,
        initial_authority_epoch=1, initial_fence=7,
    )


def _server_process(socket_path: str, database_path: str, signing_seed: bytes,
                    roster: tuple[HostProvisionedJourney, ...],
                    principal_by_uid: dict[int, str], ready, stop, errors) -> None:
    """Fixture subprocess: no user-supplied principal crosses the wire."""
    try:
        from labyrinth.authority_unix_service_r22 import UnixAuthorityService

        core = SourceOnlyJourneyAuthorityCore(
            database_path=Path(database_path), provider_id="provider-a",
            service_signing_key=signing_seed, provisioned_journeys=roster,
        )
        with UnixAuthorityService(
            socket_path=Path(socket_path), core=core,
            principal_by_uid=principal_by_uid,
        ) as server:
            server.timeout = 0.10
            ready.set()
            while not stop.is_set():
                server.handle_request()
    except Exception as exc:
        errors.put(f"{type(exc).__name__}: {exc}")
        ready.set()


@unittest.skipUnless(sys.platform.startswith("linux") and hasattr(socket, "SO_PEERCRED"),
                     "Linux SO_PEERCRED Unix socket integration required")
class UnixAuthorityProcessE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.socket_path = self.root / "authority.sock"
        self.database_path = self.root / "authority.sqlite3"
        self.a = _binding("run-a", "event-a", "exam-a")
        self.b = _binding("run-b", "event-b", "exam-a")
        self.roster = (_pin(self.a), _pin(self.b))
        signer = Ed25519PrivateKey.generate()
        self.private_seed = signer.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        self.public_key = signer.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw,
        )
        self.parent_core = SourceOnlyJourneyAuthorityCore(
            database_path=self.database_path, provider_id="provider-a",
            service_signing_key=self.private_seed,
            provisioned_journeys=self.roster,
        )
        self.ctx = multiprocessing.get_context("fork")
        self.proc = None
        self.stop_event = None
        self.addCleanup(self._stop)

    def _start(self, *, authorized: bool = True) -> None:
        self.assertIsNone(self.proc)
        ready = self.ctx.Event()
        self.stop_event = self.ctx.Event()
        errors = self.ctx.Queue()
        uids = ({os.getuid(): "trainer-a"} if authorized
                else {os.getuid() + 1: "trainer-a"})
        self.proc = self.ctx.Process(
            target=_server_process,
            args=(str(self.socket_path), str(self.database_path),
                  self.private_seed, self.roster, uids,
                  ready, self.stop_event, errors),
            daemon=True,
        )
        self.proc.start()
        self.assertTrue(ready.wait(4.0), "Unix service process failed to reach readiness")
        if not errors.empty():
            self.fail(f"Unix service startup failure: {errors.get()}")
        self.assertTrue(self.proc.is_alive(), "Unix service exited during startup")
        self.assertTrue(self.socket_path.exists(), "Unix service never bound its socket")

    def _stop(self) -> None:
        if self.proc is None:
            return
        self.stop_event.set()
        self.proc.join(timeout=3.0)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=3.0)
            self.fail("Unix test service did not shut down on request")
        self.proc = None
        self.stop_event = None

    def _request(self, binding: JourneyExternalAuthorityBinding, nonce: str):
        return build_journey_external_request(
            binding, self.parent_core.read_head(binding.run_id),
            operation_nonce=nonce,
        )

    def _port(self, binding: JourneyExternalAuthorityBinding, request, key=None):
        from labyrinth.authority_unix_client_r22 import UnixAuthorityPort
        return UnixAuthorityPort(
            socket_path=self.socket_path,
            pinned_binding=binding, pinned_request=request,
            pinned_ed25519_public_key=(self.public_key if key is None else key),
            timeout_seconds=1.5,
        )

    def _raw(self, data: dict[str, object], *, read_reply: bool = True):
        body = json.dumps(data, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True).encode("ascii")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(2.0)
            peer.connect(str(self.socket_path))
            peer.sendall(struct.pack("!I", len(body)) + body)
            if not read_reply:
                return None
            n = struct.unpack("!I", self._recv_exact(peer, 4))[0]
            return json.loads(self._recv_exact(peer, n).decode("ascii"))

    @staticmethod
    def _recv_exact(peer: socket.socket, remaining: int) -> bytes:
        chunks = []
        while remaining:
            part = peer.recv(remaining)
            if not part:
                raise AssertionError("unexpected truncated source transport response")
            chunks.append(part)
            remaining -= len(part)
        return b"".join(chunks)

    def test_separate_process_roundtrip_verifies_signed_cas_and_receipt(self):
        self._start()
        self.assertTrue(stat.S_ISSOCK(self.socket_path.lstat().st_mode))
        self.assertEqual(stat.S_IMODE(self.socket_path.lstat().st_mode), 0o600)
        req = self._request(self.a, "operation-a")
        port = self._port(self.a, req)
        client = ExternalMonotonicReservationClient(
            port=port, provider_id="provider-a", pinned_request=req,
            pinned_ed25519_public_key=self.public_key,
        )
        receipt = client.reserve(req)
        self.assertEqual(receipt.request_sha256, req.digest())
        self.assertEqual(receipt.commit_index, 1)
        self.assertFalse(receipt.model_execution_authenticated)
        self.assertFalse(receipt.graduation_authorized)
        self.assertEqual(port.read_head("run-a"), receipt.post_head())
        self.assertEqual(port.reconcile_committed_receipt(), receipt)

    def test_wire_principal_spoof_is_rejected_before_reservation(self):
        self._start()
        req = self._request(self.a, "operation-a")
        reply = self._raw({
            "schema": "labyrinth.authority-unix-transport.v1", "op": "reserve",
            "principal": "admin", "binding": asdict(self.a),
            "request": asdict(req),
        })
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["error"], "BAD_REQUEST")
        self.assertEqual(self._port(self.a, req).read_head("run-a").commit_index, 0)

    def test_unprovisioned_unix_uid_is_forbidden_independently_of_json(self):
        self._start(authorized=False)
        req = self._request(self.a, "operation-a")
        reply = self._raw({
            "schema": "labyrinth.authority-unix-transport.v1", "op": "reserve",
            "binding": asdict(self.a), "request": asdict(req),
        })
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["error"], "FORBIDDEN")
        self.assertEqual(self.parent_core.read_head("run-a").commit_index, 0)

    def test_lost_ack_reconciles_after_graceful_service_restart_without_replay(self):
        self._start()
        req = self._request(self.a, "operation-a")
        self._raw({"schema": "labyrinth.authority-unix-transport.v1",
                   "op": "reserve", "binding": asdict(self.a),
                   "request": asdict(req)}, read_reply=False)
        for _ in range(30):
            if self.parent_core.read_head("run-a").commit_index == 1:
                break
            time.sleep(0.05)
        self.assertEqual(self.parent_core.read_head("run-a").commit_index, 1)
        self._stop()
        self._start()
        port = self._port(self.a, req)
        receipt = port.reconcile_committed_receipt()
        self.assertEqual(receipt.request_sha256, req.digest())
        self.assertEqual(port.read_head("run-a"), receipt.post_head())
        with self.assertRaises(ExternalAuthorityConflict):
            ExternalMonotonicReservationClient(
                port=port, provider_id="provider-a", pinned_request=req,
                pinned_ed25519_public_key=self.public_key,
            ).reserve(req)

    def test_concurrent_clients_cannot_reuse_exam_across_distinct_runs(self):
        self._start()
        requests = (self._request(self.a, "operation-a"),
                    self._request(self.b, "operation-b"))
        ports = (self._port(self.a, requests[0]), self._port(self.b, requests[1]))

        def attempt(index: int):
            try:
                return ExternalMonotonicReservationClient(
                    port=ports[index], provider_id="provider-a",
                    pinned_request=requests[index],
                    pinned_ed25519_public_key=self.public_key,
                ).reserve(requests[index])
            except ExternalAuthorityConflict:
                return "CONFLICT"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(attempt, (0, 1)))
        self.assertEqual(sum(result == "CONFLICT" for result in results), 1)
        self.assertEqual(sum(result != "CONFLICT" for result in results), 1)
        self.assertEqual(self.parent_core.read_head("run-a").commit_index
                         + self.parent_core.read_head("run-b").commit_index, 1)

    def test_wrong_public_key_fails_closed_after_commit(self):
        self._start()
        req = self._request(self.a, "operation-a")
        other = Ed25519PrivateKey.generate().public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw,
        )
        port = self._port(self.a, req, key=other)
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            port.reserve_once(req)
        self.assertEqual(self.parent_core.read_head("run-a").commit_index, 1)
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            port.reconcile_committed_receipt()


if __name__ == "__main__":
    unittest.main()
