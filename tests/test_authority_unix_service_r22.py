"""Real Linux UDS SOURCE transport probes, not privileged-host qualification."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import stat
import struct
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.authority_service_core_r20 import (
    HostProvisionedJourney, SourceOnlyJourneyAuthorityCore,
)
from labyrinth.authority_unix_service_r22 import (
    MAX_FRAME_BYTES, OPERATOR_SCHEMA, SCHEMA, STATUS, UnixAuthorityService,
    UnixAuthorityTransportError, load_operator_service,
)
from labyrinth.journey_external_authority_contract_r17 import (
    JourneyExternalAuthorityBinding, build_journey_external_request,
)


def _binding(run: str, event: str, exam: str):
    return JourneyExternalAuthorityBinding(
        provider_id="provider-a", host_id="host-a", run_id=run,
        event_id=event, exam_id=exam, scope_sha256="1" * 64,
        objective_contract_sha256="2" * 64, exam_lineage_sha256="3" * 64,
        journey_sha256="4" * 64, signed_binding_sha256="5" * 64,
        learner_checkpoint_sha256="6" * 64,
        prior_training_roster_sha256="7" * 64,
        next_training_allocation_sha256="8" * 64,
        host_policy_sha256="9" * 64, source_tree_sha256="a" * 64,
        fullstate_binding_sha256="b" * 64,
        consumed_event_ledger_root_sha256="c" * 64,
        migration_witness_sha256="d" * 64,
        previous_cycle=0, next_cycle=1,
    )


def _pin(binding, principal="trainer-a"):
    return HostProvisionedJourney(
        binding=binding, authenticated_principal_id=principal,
        operator_pinned_ledger_root_sha256="c" * 64,
        operator_pinned_migration_witness_sha256="d" * 64,
        initial_authority_epoch=1, initial_fence=7,
    )


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _exact(sock, count):
    output = b""
    while len(output) < count:
        chunk = sock.recv(count - len(output))
        if not chunk:
            raise AssertionError("socket truncated response")
        output += chunk
    return output


@unittest.skipUnless(os.name == "posix" and hasattr(socket, "SO_PEERCRED"),
                     "Linux SO_PEERCRED required")
class UnixServiceRealPeerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.sock = self.root / "authority.sock"
        self.db = self.root / "authority.sqlite3"
        signer = Ed25519PrivateKey.generate()
        self.key = signer.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption())
        self.public = signer.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.a = _binding("run-a", "event-a", "exam-a")
        self.b = _binding("run-b", "event-b", "exam-b")
        self.server = None
        self.thread = None

    def _core(self, *pins):
        return SourceOnlyJourneyAuthorityCore(
            database_path=self.db, provider_id="provider-a",
            service_signing_key=self.key, provisioned_journeys=tuple(pins))

    def _start(self, core, principal_by_uid=None):
        if principal_by_uid is None:
            principal_by_uid = {os.getuid(): "trainer-a"}
        self.server = UnixAuthorityService(
            socket_path=self.sock, core=core, principal_by_uid=principal_by_uid)
        self.thread = Thread(target=self.server.serve_forever,
                             kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.assertEqual(stat.S_IMODE(self.sock.stat().st_mode), 0o600)
        self.assertEqual(self.server.qualification, STATUS)
        self.addCleanup(self._stop)

    def _stop(self):
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3.0)
        self.server = None
        self.thread = None

    def _send(self, message=None, *, raw=None, oversize_length=None):
        body = raw if raw is not None else _encode(message)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect(str(self.sock))
            declared = len(body) if oversize_length is None else oversize_length
            client.sendall(struct.pack("!I", declared) + body)
            size = struct.unpack("!I", _exact(client, 4))[0]
            self.assertLessEqual(size, MAX_FRAME_BYTES)
            response = json.loads(_exact(client, size).decode("ascii"))
            try:
                self.assertEqual(client.recv(1), b"")
            except ConnectionResetError:
                # Rejecting an oversized declaration leaves unread input;
                # closing may reset that malformed connection after reply.
                if oversize_length is None:
                    raise
        return response

    def _request(self, core, binding, nonce="operation-a"):
        return build_journey_external_request(
            binding, core.read_head(binding.run_id), operation_nonce=nonce)

    def _reserve(self, binding, request):
        return {"schema": SCHEMA, "op": "reserve",
                "binding": asdict(binding), "request": asdict(request)}

    def test_real_peer_credentials_gate_full_preimage_and_signed_receipt(self):
        core = self._core(_pin(self.a))
        req = self._request(core, self.a)
        self._start(core)
        initial = self._send({"schema": SCHEMA, "op": "read_head", "run_id": "run-a"})
        self.assertTrue(initial["ok"])
        self.assertEqual(initial["value"]["commit_index"], 0)
        reserved = self._send(self._reserve(self.a, req))
        self.assertTrue(reserved["ok"], reserved)
        self.assertEqual(reserved["value"]["request_sha256"], req.digest())
        self.assertFalse(reserved["value"]["graduation_authorized"])
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from labyrinth.external_monotonic_authority_contract import ExternalReservationReceipt
        receipt = ExternalReservationReceipt(**reserved["value"])
        Ed25519PublicKey.from_public_bytes(self.public).verify(
            bytes.fromhex(receipt.signature_hex), receipt.signed_payload())
        after = self._send({"schema": SCHEMA, "op": "read_head", "run_id": "run-a"})
        self.assertEqual(after["value"]["commit_index"], 1)
        retrieved = self._send({"schema": SCHEMA, "op": "read_receipt",
                                "run_id": "run-a", "event_id": "event-a",
                                "operation_nonce": "operation-a"})
        self.assertEqual(retrieved, reserved)
        self.assertEqual(self._send(self._reserve(self.a, req)), reserved)

    def test_peer_uid_is_os_derived_and_json_principal_is_not_accepted(self):
        core = self._core(_pin(self.a))
        req = self._request(core, self.a)
        self._start(core)
        forged = self._reserve(self.a, req)
        forged["authenticated_principal_id"] = "trainer-a"
        self.assertEqual(self._send(forged)["error"], "BAD_REQUEST")
        self.assertEqual(core.read_head("run-a").commit_index, 0)
        self._stop()
        self._start(core, principal_by_uid={os.getuid() + 1000: "trainer-a"})
        refused = self._send(self._reserve(self.a, req))
        self.assertEqual(refused["error"], "FORBIDDEN")
        self.assertEqual(core.read_head("run-a").commit_index, 0)

    def test_cross_run_readback_requires_server_provisioned_principal(self):
        core = self._core(_pin(self.a), _pin(self.b, principal="trainer-b"))
        req = self._request(core, self.b, nonce="operation-b")
        self._start(core)
        self.assertEqual(self._send({"schema": SCHEMA, "op": "read_head",
                                     "run_id": "run-b"})["error"], "FORBIDDEN")
        self.assertEqual(self._send(self._reserve(self.b, req))["error"], "FORBIDDEN")
        self.assertEqual(core.read_head("run-b").commit_index, 0)

    def test_malformed_frame_duplicate_keys_and_unknown_methods_fail_closed(self):
        core = self._core(_pin(self.a))
        self._start(core)
        self.assertEqual(self._send(raw=b'{"schema":"x","schema":"x"}')["error"],
                         "BAD_REQUEST")
        self.assertEqual(self._send(raw=b'{ "schema":"x" }')["error"], "BAD_REQUEST")
        self.assertEqual(self._send(raw=b"{}", oversize_length=MAX_FRAME_BYTES + 1)["error"],
                         "BAD_REQUEST")
        self.assertEqual(self._send({"schema": SCHEMA, "op": "execute_command"})["error"],
                         "BAD_REQUEST")
        self.assertEqual(core.read_head("run-a").commit_index, 0)

    def test_same_socket_concurrent_duplicates_and_restart_reconcile(self):
        core = self._core(_pin(self.a))
        req = self._request(core, self.a)
        self._start(core)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self._send(self._reserve(self.a, req)),
                                    range(6)))
        self.assertTrue(all(row == results[0] for row in results))
        self.assertEqual(core.read_head("run-a").commit_index, 1)
        self._stop()
        reopened = self._core(_pin(self.a))
        self._start(reopened)
        recovered = self._send({"schema": SCHEMA, "op": "read_receipt",
                                "run_id": "run-a", "event_id": "event-a",
                                "operation_nonce": "operation-a"})
        self.assertEqual(recovered, results[0])

    def test_existing_socket_symlink_or_public_parent_refused(self):
        core = self._core(_pin(self.a))
        symlink_target = self.root / "target"
        self.sock.symlink_to(symlink_target)
        with self.assertRaises(UnixAuthorityTransportError):
            UnixAuthorityService(socket_path=self.sock, core=core,
                                 principal_by_uid={os.getuid(): "trainer-a"})
        self.sock.unlink()
        self.root.chmod(0o777)
        with self.assertRaises(UnixAuthorityTransportError):
            UnixAuthorityService(socket_path=self.sock, core=core,
                                 principal_by_uid={os.getuid(): "trainer-a"})

    def test_explicit_operator_group_socket_mode_requires_private_group_parent(self):
        core = self._core(_pin(self.a))
        with self.assertRaises(UnixAuthorityTransportError):
            UnixAuthorityService(socket_path=self.sock, core=core,
                                 principal_by_uid={os.getuid(): "trainer-a"},
                                 authorized_group_gid=os.getgid())
        self.root.chmod(0o750)
        server = UnixAuthorityService(
            socket_path=self.sock, core=core,
            principal_by_uid={os.getuid(): "trainer-a"},
            authorized_group_gid=os.getgid())
        try:
            self.assertEqual(stat.S_IMODE(self.sock.stat().st_mode), 0o660)
            self.assertEqual(self.sock.stat().st_gid, os.getgid())
        finally:
            server.server_close()
        self.assertFalse(self.sock.exists())

    def test_operator_private_config_and_signing_key_load_are_explicit(self):
        provision = _pin(self.a)
        config = {"schema": OPERATOR_SCHEMA, "provider_id": "provider-a",
                  "principal_by_uid": {str(os.getuid()): "trainer-a"},
                  "journeys": [asdict(provision)]}
        config_path = self.root / "operator.json"
        config_path.write_bytes(_encode(config))
        config_path.chmod(0o600)
        signing_path = self.root / "signing-seed.bin"
        signing_path.write_bytes(self.key)
        signing_path.chmod(0o600)
        service = load_operator_service(
            config_path=config_path, signing_key_path=signing_path,
            database_path=self.db, socket_path=self.sock)
        try:
            self.assertEqual(service.qualification, STATUS)
            self.assertEqual(stat.S_IMODE(self.sock.stat().st_mode), 0o600)
        finally:
            service.server_close()
        self.assertFalse(self.sock.exists())
        signing_path.chmod(0o644)
        with self.assertRaises(UnixAuthorityTransportError):
            load_operator_service(config_path=config_path, signing_key_path=signing_path,
                                  database_path=self.db, socket_path=self.sock)
        signing_path.chmod(0o600)
        config_path.unlink()
        config_path.symlink_to(signing_path)
        with self.assertRaises(UnixAuthorityTransportError):
            load_operator_service(config_path=config_path, signing_key_path=signing_path,
                                  database_path=self.db, socket_path=self.sock)

    def test_operator_config_and_database_writable_parent_are_refused(self):
        config_dir = self.root / "operator"
        data_dir = self.root / "database"
        config_dir.mkdir(mode=0o700)
        data_dir.mkdir(mode=0o700)
        config_path = config_dir / "operator.json"
        signing_path = config_dir / "signing-key.bin"
        config = {"schema": OPERATOR_SCHEMA, "provider_id": "provider-a",
                  "principal_by_uid": {str(os.getuid()): "trainer-a"},
                  "journeys": [asdict(_pin(self.a))]}
        config_path.write_bytes(_encode(config))
        signing_path.write_bytes(self.key)
        config_path.chmod(0o600)
        signing_path.chmod(0o600)
        database = data_dir / "authority.sqlite3"
        data_dir.chmod(0o777)
        with self.assertRaises(UnixAuthorityTransportError):
            load_operator_service(config_path=config_path, signing_key_path=signing_path,
                                  database_path=database, socket_path=self.sock)
        data_dir.chmod(0o700)
        config_dir.chmod(0o777)
        with self.assertRaises(UnixAuthorityTransportError):
            load_operator_service(config_path=config_path, signing_key_path=signing_path,
                                  database_path=database, socket_path=self.sock)
        config_dir.chmod(0o700)
        alias = self.root / "operator-alias"
        alias.symlink_to(config_dir, target_is_directory=True)
        with self.assertRaises(UnixAuthorityTransportError):
            load_operator_service(config_path=alias / "operator.json",
                                  signing_key_path=signing_path,
                                  database_path=database, socket_path=self.sock)


if __name__ == "__main__":
    unittest.main()
