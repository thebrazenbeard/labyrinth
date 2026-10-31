"""Real AF_UNIX socket protocol tests, not independent host qualification."""

from dataclasses import asdict
from pathlib import Path
import json
import socketserver
import struct
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.authority_service_core_r20 import (
    AuthorityCoreConflict, HostProvisionedJourney, SourceOnlyJourneyAuthorityCore,
)
from labyrinth.authority_unix_client_r22 import (
    SCHEMA, UnixAuthorityPort, MAX_FRAME_BYTES,
)
from labyrinth.external_monotonic_authority_contract import (
    ExternalAuthorityConflict, ExternalAuthorityOutcomeUnknown,
    ExternalMonotonicReservationClient,
)
from labyrinth.journey_external_authority_contract_r17 import (
    JourneyExternalAuthorityBinding, build_journey_external_request,
)


def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _binding():
    return JourneyExternalAuthorityBinding(
        provider_id="provider-a", host_id="host-a", run_id="run-a",
        event_id="event-a", exam_id="exam-a", scope_sha256="1" * 64,
        objective_contract_sha256="2" * 64, exam_lineage_sha256="3" * 64,
        journey_sha256="4" * 64, signed_binding_sha256="5" * 64,
        learner_checkpoint_sha256="6" * 64, prior_training_roster_sha256="7" * 64,
        next_training_allocation_sha256="8" * 64, host_policy_sha256="9" * 64,
        source_tree_sha256="a" * 64, fullstate_binding_sha256="b" * 64,
        consumed_event_ledger_root_sha256="c" * 64,
        migration_witness_sha256="d" * 64, previous_cycle=0, next_cycle=1,
    )


def _recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return bytes(data)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        n = struct.unpack("!I", _recv_exact(self.request, 4))[0]
        if not 0 < n <= MAX_FRAME_BYTES:
            return
        raw = _recv_exact(self.request, n)
        record = json.loads(raw.decode("ascii"))
        server = self.server
        server.requests.append(record)
        if raw != _canonical(record):
            return
        if record.get("schema") != SCHEMA or "principal_id" in record:
            result = {"schema": SCHEMA, "ok": False, "error": "BAD_REQUEST"}
        else:
            try:
                op = record["op"]
                if op == "read_head":
                    value = asdict(server.core.read_head(record["run_id"]))
                elif op == "reserve":
                    if set(record) != {"schema", "op", "binding", "request"}:
                        raise ValueError("unrecognized client fields")
                    b = JourneyExternalAuthorityBinding(**record["binding"])
                    from labyrinth.external_monotonic_authority_contract import ExternalReservationRequest
                    req = ExternalReservationRequest(**record["request"])
                    value = asdict(server.core.reserve_journey(
                        b, req, authenticated_principal_id="trainer-a"))
                    if server.drop_next_reserve_ack:
                        server.drop_next_reserve_ack = False
                        return
                elif op == "read_receipt":
                    receipt = server.core.read_receipt(
                        record["run_id"], record["event_id"], record["operation_nonce"])
                    value = asdict(receipt) if receipt is not None else None
                else:
                    raise ValueError("unknown operation")
                result = {"schema": SCHEMA, "ok": True, "value": value}
            except AuthorityCoreConflict:
                result = {"schema": SCHEMA, "ok": False, "error": "CONFLICT"}
            except ValueError:
                result = {"schema": SCHEMA, "ok": False, "error": "BAD_REQUEST"}
        if server.override_next_response is not None:
            raw_reply = server.override_next_response
            server.override_next_response = None
        else:
            raw_reply = _canonical(result)
        self.request.sendall(struct.pack("!I", len(raw_reply)) + raw_reply)
        if server.trailing_bytes:
            self.request.sendall(server.trailing_bytes)
            server.trailing_bytes = b""


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False


class UnixAuthorityClientR22Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sock = self.root / "host.socket"
        self.private = Ed25519PrivateKey.generate()
        self.seed = self.private.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption())
        self.public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.binding = _binding()
        self.core = SourceOnlyJourneyAuthorityCore(
            database_path=self.root / "authority.sqlite3", provider_id="provider-a",
            service_signing_key=self.seed,
            provisioned_journeys=(HostProvisionedJourney(
                self.binding, "trainer-a", "c" * 64, "d" * 64, 1, 7),),
        )
        self.request = build_journey_external_request(
            self.binding, self.core.read_head("run-a"), operation_nonce="operation-a")
        self.server = _UnixServer(str(self.sock), _Handler)
        self.server.core = self.core
        self.server.requests = []
        self.server.drop_next_reserve_ack = False
        self.server.override_next_response = None
        self.server.trailing_bytes = b""
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.stopped = False
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        if self.stopped:
            return
        self.stopped = True
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def port(self, *, key=None):
        return UnixAuthorityPort(
            socket_path=self.sock, pinned_binding=self.binding,
            pinned_request=self.request,
            pinned_ed25519_public_key=self.public if key is None else key,
            timeout_seconds=0.8,
        )

    def test_public_client_reserves_one_signed_event_without_principal_on_wire(self):
        port = self.port()
        user = ExternalMonotonicReservationClient(
            port=port, provider_id="provider-a", pinned_request=self.request,
            pinned_ed25519_public_key=self.public)
        receipt = user.reserve(self.request)
        self.assertEqual(receipt.commit_index, 1)
        self.assertEqual(port.reconcile_committed_receipt(), receipt)
        self.assertEqual(self.core.read_head("run-a").commit_index, 1)
        self.assertTrue(all("principal_id" not in request for request in self.server.requests))
        reserve = next(request for request in self.server.requests if request["op"] == "reserve")
        self.assertEqual(reserve["binding"], asdict(self.binding))
        self.assertEqual(reserve["request"], asdict(self.request))

    def test_lost_ack_does_not_replay_reserve_and_read_only_reconciles(self):
        self.server.drop_next_reserve_ack = True
        port = self.port()
        user = ExternalMonotonicReservationClient(
            port=port, provider_id="provider-a", pinned_request=self.request,
            pinned_ed25519_public_key=self.public)
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            user.reserve(self.request)
        self.assertEqual(self.core.read_head("run-a").commit_index, 1)
        receipt = port.reconcile_committed_receipt()
        self.assertEqual(receipt.request_sha256, self.request.digest())
        self.assertEqual(sum(r["op"] == "reserve" for r in self.server.requests), 1)

    def test_missing_receipt_cannot_be_treated_as_permission_or_reservation_retry(self):
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            self.port().reconcile_committed_receipt()
        self.assertEqual(self.core.read_head("run-a").commit_index, 0)
        self.assertFalse(any(r["op"] == "reserve" for r in self.server.requests))

    def test_untrusted_signer_after_commit_is_unknown_not_success(self):
        foreign = Ed25519PrivateKey.generate().public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        port = self.port(key=foreign)
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            port.reserve_once(self.request)
        self.assertEqual(self.core.read_head("run-a").commit_index, 1)
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            port.reconcile_committed_receipt()

    def test_unpinned_cross_run_or_modified_request_fails_before_transport(self):
        from dataclasses import replace
        port = self.port()
        with self.assertRaises(ExternalAuthorityConflict):
            port.read_head("run-b")
        with self.assertRaises(ExternalAuthorityConflict):
            port.reserve_once(replace(self.request, operation_nonce="unexpected"))
        self.assertEqual(self.server.requests, [])

    def test_malformed_duplicate_response_and_trailing_data_fail_closed(self):
        self.server.override_next_response = (
            b'{"ok":true,"ok":true,"schema":"labyrinth.authority-unix-transport.v1","value":null}')
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            self.port().read_head("run-a")
        self.server.trailing_bytes = b"X"
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            self.port().read_head("run-a")
        self.server.override_next_response = _canonical({
            "schema": SCHEMA, "ok": False, "error": [],
        })
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            self.port().read_head("run-a")

    def test_transport_unavailable_is_unknown_and_does_not_fallback(self):
        self._stop_server()
        with self.assertRaises(ExternalAuthorityOutcomeUnknown):
            self.port().read_head("run-a")


if __name__ == "__main__":
    unittest.main()
