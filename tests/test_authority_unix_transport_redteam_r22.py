"""Independent Unix authority boundary attacks against the exact R22 transport.

These are local SOURCE tests. The process runs with one sandbox UID and cannot
establish independently administered host accounts, key custody or rollback.
"""

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
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from labyrinth.authority_service_core_r20 import (
    HostProvisionedJourney, SourceOnlyJourneyAuthorityCore,
)
from labyrinth.authority_unix_service_r22 import (
    OPERATOR_SCHEMA, SCHEMA, UnixAuthorityService, UnixAuthorityTransportError,
    load_operator_service,
)
from labyrinth.journey_external_authority_contract_r17 import (
    JourneyExternalAuthorityBinding, build_journey_external_request,
)


def _binding(run: str, event: str, exam: str) -> JourneyExternalAuthorityBinding:
    return JourneyExternalAuthorityBinding(
        provider_id="provider-a", host_id="host-a", run_id=run,
        event_id=event, exam_id=exam, scope_sha256="1" * 64,
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


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    while count:
        chunk = sock.recv(count)
        if not chunk:
            raise AssertionError("Unix response closed before declared bytes")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


@unittest.skipUnless(os.name == "posix" and hasattr(socket, "SO_PEERCRED"),
                     "Linux Unix peer credentials required")
class UnixAuthorityHostileR22(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root.chmod(0o700)
        self.socket_path = self.root / "authority.sock"
        self.db = self.root / "authority.sqlite3"
        signer = Ed25519PrivateKey.generate()
        self.seed = signer.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption())
        self.a = _binding("run-a", "event-a", "exam-shared")
        self.b = _binding("run-b", "event-b", "exam-shared")
        self.server = None
        self.thread = None

    def _core(self, *bindings):
        return SourceOnlyJourneyAuthorityCore(
            database_path=self.db, provider_id="provider-a",
            service_signing_key=self.seed,
            provisioned_journeys=tuple(_pin(b) for b in bindings),
        )

    def _start(self, core, *, principals=None):
        if principals is None:
            principals = {os.getuid(): "trainer-a"}
        self.server = UnixAuthorityService(
            socket_path=self.socket_path, core=core, principal_by_uid=principals)
        self.thread = Thread(target=self.server.serve_forever,
                             kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)

    def _stop(self):
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3.0)
        self.server = None

    def _wire(self, data):
        return {"schema": SCHEMA, "op": "reserve",
                "binding": asdict(data[0]), "request": asdict(data[1])}

    def _exchange(self, message, *, raw=None):
        body = raw if raw is not None else _canonical(message)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(3.0)
            connection.connect(str(self.socket_path))
            connection.sendall(struct.pack("!I", len(body)) + body)
            length = struct.unpack("!I", _exact(connection, 4))[0]
            response = json.loads(_exact(connection, length).decode("ascii"))
        return response

    def _operator_inputs(self, parent: Path):
        parent.mkdir(exist_ok=True)
        config = parent / "operator.json"
        seed = parent / "signing-seed.bin"
        body = {"schema": OPERATOR_SCHEMA, "provider_id": "provider-a",
                "principal_by_uid": {str(os.getuid()): "trainer-a"},
                "journeys": [asdict(_pin(self.a))]}
        config.write_bytes(_canonical(body))
        config.chmod(0o600)
        seed.write_bytes(self.seed)
        seed.chmod(0o600)
        return config, seed

    def test_no_claimed_principal_can_override_kernel_uid(self):
        core = self._core(self.a)
        request = build_journey_external_request(
            self.a, core.read_head("run-a"), operation_nonce="operation-a")
        self._start(core, principals={os.getuid() + 1000: "trainer-a"})
        forged = self._wire((self.a, request))
        forged["authenticated_principal_id"] = "trainer-a"
        response = self._exchange(forged)
        self.assertEqual(response["error"], "FORBIDDEN")
        self.assertEqual(core.read_head("run-a").commit_index, 0)

    def test_one_real_socket_database_refuses_cross_run_exam_reuse(self):
        core = self._core(self.a, self.b)
        req_a = build_journey_external_request(
            self.a, core.read_head("run-a"), operation_nonce="operation-a")
        req_b = build_journey_external_request(
            self.b, core.read_head("run-b"), operation_nonce="operation-b")
        self._start(core)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(self._exchange,
                                    (self._wire((self.a, req_a)),
                                     self._wire((self.b, req_b)))))
        self.assertEqual(sum(row["ok"] is True for row in results), 1)
        self.assertEqual(sum(row.get("error") == "CONFLICT" for row in results), 1)
        self.assertEqual(core.read_head("run-a").commit_index +
                         core.read_head("run-b").commit_index, 1)

    def test_duplicate_json_and_wrong_typed_binding_do_not_reserve(self):
        core = self._core(self.a)
        req = build_journey_external_request(
            self.a, core.read_head("run-a"), operation_nonce="operation-a")
        self._start(core)
        duplicate = b'{"op":"read_head","op":"reserve","schema":"' + SCHEMA.encode() + b'"}'
        self.assertEqual(self._exchange(None, raw=duplicate)["error"], "BAD_REQUEST")
        malformed = self._wire((self.a, req))
        malformed["binding"]["run_id"] = ["run-a"]
        self.assertEqual(self._exchange(malformed)["error"], "BAD_REQUEST")
        self.assertEqual(core.read_head("run-a").commit_index, 0)

    def test_operator_loader_must_refuse_world_writable_database_parent(self):
        """RED: a 0600 seed does not protect SQLite in a writable directory."""
        config, seed = self._operator_inputs(self.root)
        unsafe = self.root / "untrusted-database-parent"
        unsafe.mkdir(mode=0o777)
        unsafe.chmod(0o777)
        loaded = None
        try:
            loaded = load_operator_service(
                config_path=config, signing_key_path=seed,
                database_path=unsafe / "authority.sqlite3",
                socket_path=self.socket_path)
        except UnixAuthorityTransportError:
            return
        finally:
            if loaded is not None:
                loaded.server_close()
        self.fail("operator loader accepted world-writable authority DB parent")

    def test_operator_loader_must_refuse_world_writable_config_parent(self):
        """RED: private file mode is insufficient when the parent is writable."""
        unsafe = self.root / "untrusted-config-parent"
        unsafe.mkdir(mode=0o777)
        unsafe.chmod(0o777)
        config, seed = self._operator_inputs(unsafe)
        loaded = None
        try:
            loaded = load_operator_service(
                config_path=config, signing_key_path=seed,
                database_path=self.db, socket_path=self.socket_path)
        except UnixAuthorityTransportError:
            return
        finally:
            if loaded is not None:
                loaded.server_close()
        self.fail("operator loader accepted operator files in world-writable directory")


if __name__ == "__main__":
    unittest.main()
