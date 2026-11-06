"""Mock host-broker protocol tests; no real socket service or GPU lease."""

from hashlib import sha256
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import socket
import stat
import struct
import time

from labyrinth.trainingday_host_broker_dryrun_v1 import (
    HostBrokerUnavailable, LocalDevelopmentIntent, inspect_owner_broker_dryrun,
)


SOCKET_PATH = Path("/run/labyrinth-owner/lease.sock")


def intent():
    return LocalDevelopmentIntent(
        source_head="a" * 40, model_state_sha256="b" * 64,
        training_plan_sha256="c" * 64, experiment_id="local_saga_1",
    )


def record(path, *, writable=False, owner=0):
    return SimpleNamespace(
        st_mode=(stat.S_IFSOCK if path == SOCKET_PATH else stat.S_IFDIR)
        | ((0o777 if writable else 0o711)
           if path != SOCKET_PATH else 0o666),
        st_uid=owner, st_dev=42, st_ino=84,
    )


class FakeBrokerSocket:
    """An intentionally untrusted mock, never host-authorization evidence."""

    def __init__(self, *, peer_uid=0, decision="OWNER_APPROVED_DEV_ONLY",
                 tamper_nonce=False, expired=False):
        self.peer_uid = peer_uid
        self.decision = decision
        self.tamper_nonce = tamper_nonce
        self.expired = expired
        self.reply = b""
        self.pos = 0
        self.sent = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def settimeout(self, _seconds):
        pass

    def connect(self, path):
        if path != str(SOCKET_PATH):
            raise AssertionError("wrong broker route")

    def getsockopt(self, level, option, size):
        if (level, option, size) != (socket.SOL_SOCKET, socket.SO_PEERCRED, 12):
            raise AssertionError("kernel peer credentials were not requested")
        return struct.pack("3i", 777, self.peer_uid, 0)

    def sendall(self, payload):
        length = int.from_bytes(payload[:4], "big")
        request = json.loads(payload[4:].decode("utf-8"))
        if length != len(payload) - 4:
            raise AssertionError("noncanonical wire frame")
        self.sent = request
        data = {
            "schema": request["schema"],
            "nonce": "0" * 64 if self.tamper_nonce else request["nonce"],
            "intent_sha256": sha256(json.dumps(
                request["intent"], ensure_ascii=True, sort_keys=True,
                separators=(",", ":")).encode("ascii")).hexdigest(),
            "lease_id": "lease_dev_1", "broker_uid": self.peer_uid,
            "lease_expires_unix": 1 if self.expired else int(time.time()) + 60,
            "decision": self.decision,
        }
        packed = json.dumps(data).encode("utf-8")
        self.reply = len(packed).to_bytes(4, "big") + packed

    def recv(self, count):
        result = self.reply[self.pos:self.pos + count]
        self.pos += len(result)
        return result


class HostBrokerDryRunTests(TestCase):
    def _run(self, peer=None, *, path_writable=False, owner=0):
        peer = peer or FakeBrokerSocket()
        def fake_stat(path):
            return record(path, writable=path_writable, owner=owner)
        with mock.patch("os.getuid", return_value=1000), (
                mock.patch.object(Path, "lstat", fake_stat)), (
                mock.patch("socket.socket", return_value=peer)):
            return inspect_owner_broker_dryrun(
                broker_socket=SOCKET_PATH, expected_owner_uid=0, intent=intent())

    def test_privileged_peer_and_nonce_bound_request_give_dryrun_only(self):
        peer = FakeBrokerSocket()
        result = self._run(peer)
        self.assertEqual(result.state, "PEERCRED_BOUND_HOST_BROKER_DECLARATION_ONLY")
        self.assertFalse(result.physical_training_authorized)
        self.assertFalse(result.gpu_lease_enforced_proven)
        self.assertFalse(result.owner_approval_proven)
        self.assertFalse(result.independent_review_required_for_development)
        self.assertEqual(peer.sent["intent"]["budget_usd"], 0)
        self.assertEqual(peer.sent["intent"]["max_optimizer_steps"], 1)
        self.assertEqual(len(peer.sent["nonce"]), 64)
        with self.assertRaisesRegex(HostBrokerUnavailable, "cannot escalate"):
            replace(result, physical_training_authorized=True)

    def test_no_owner_broker_endpoint_is_not_an_automatic_grant(self):
        with self.assertRaises(HostBrokerUnavailable):
            inspect_owner_broker_dryrun(
                broker_socket=Path("/definitely/nonexistent/labyrinth.sock"),
                expected_owner_uid=0, intent=intent())

    def test_no_self_issued_same_uid_broker_can_grant(self):
        with mock.patch("os.getuid", return_value=0):
            with self.assertRaisesRegex(HostBrokerUnavailable, "self-authorize"):
                inspect_owner_broker_dryrun(
                    broker_socket=SOCKET_PATH, expected_owner_uid=0, intent=intent())

    def test_wrong_peer_uid_denied(self):
        with self.assertRaisesRegex(HostBrokerUnavailable, "kernel credentials"):
            self._run(FakeBrokerSocket(peer_uid=1001))

    def test_nonce_replay_denied(self):
        with self.assertRaisesRegex(HostBrokerUnavailable, "not bound"):
            self._run(FakeBrokerSocket(tamper_nonce=True))

    def test_untrusted_parent_directory_denied(self):
        with self.assertRaisesRegex(HostBrokerUnavailable, "directory ancestry"):
            self._run(path_writable=True)

    def test_expired_or_denied_broker_result_fails_closed(self):
        for peer in (FakeBrokerSocket(expired=True),
                     FakeBrokerSocket(decision="DENIED")):
            with self.subTest(peer=peer), self.assertRaisesRegex(
                HostBrokerUnavailable, "not bound"
            ):
                self._run(peer)

    def test_invalid_lease_intent_rejected_before_transport(self):
        with self.assertRaisesRegex(HostBrokerUnavailable, "bounded"):
            LocalDevelopmentIntent(
                source_head="a" * 40, model_state_sha256="b" * 64,
                training_plan_sha256="c" * 64, experiment_id="bad",
                budget_usd=1,
            )


if __name__ == "__main__":
    import unittest
    unittest.main()
