"""Local host-broker lease *probe*, never physical TRAIN authorization.

This probes an external, privileged WSL/Linux AF_UNIX host broker using kernel
peer credentials and a nonce-bound DEV_ONLY declaration. Merely receiving a
grant does not prove that a GPU lease is enforced or authorize this Python
process to invoke a trainer. No broker server is installed by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import struct
import time
from typing import Any


SCHEMA = "labyrinth.trainingday-owner-broker-dryrun.v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEAD = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,100}\Z", re.ASCII)
_MAX_PACKET = 8192


class HostBrokerUnavailable(ValueError):
    """No qualifying privileged host-owned development broker observed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


@dataclass(frozen=True)
class LocalDevelopmentIntent:
    source_head: str
    model_state_sha256: str
    training_plan_sha256: str
    experiment_id: str
    operation: str = "trainingday.lowmem_experiment.one_step_kernel"
    data_scope: str = "PUBLIC_TRAIN_ONLY"
    budget_usd: int = 0
    max_optimizer_steps: int = 1

    def __post_init__(self) -> None:
        if (type(self.source_head) is not str or not _HEAD.fullmatch(self.source_head)
                or type(self.model_state_sha256) is not str
                or not _SHA.fullmatch(self.model_state_sha256)
                or type(self.training_plan_sha256) is not str
                or not _SHA.fullmatch(self.training_plan_sha256)
                or type(self.experiment_id) is not str
                or not _ID.fullmatch(self.experiment_id)
                or type(self.operation) is not str
                or self.operation != "trainingday.lowmem_experiment.one_step_kernel"
                or type(self.data_scope) is not str
                or self.data_scope != "PUBLIC_TRAIN_ONLY"
                or type(self.budget_usd) is not int or self.budget_usd != 0
                or type(self.max_optimizer_steps) is not int
                or self.max_optimizer_steps != 1):
            raise HostBrokerUnavailable("invalid bounded $0 local developmental intent")


@dataclass(frozen=True)
class HostBrokerDryRunReceipt:
    broker_uid: int
    broker_pid: int
    lease_id: str
    intent_sha256: str
    response_sha256: str
    state: str = "PEERCRED_BOUND_HOST_BROKER_DECLARATION_ONLY"
    physical_training_authorized: bool = False
    gpu_lease_enforced_proven: bool = False
    owner_approval_proven: bool = False
    independent_review_required_for_development: bool = False

    def __post_init__(self) -> None:
        if (type(self.broker_uid) is not int or self.broker_uid != 0
                or type(self.broker_pid) is not int or self.broker_pid <= 0
                or type(self.lease_id) is not str or not _ID.fullmatch(self.lease_id)
                or type(self.intent_sha256) is not str
                or not _SHA.fullmatch(self.intent_sha256)
                or type(self.response_sha256) is not str
                or not _SHA.fullmatch(self.response_sha256)
                or self.state != "PEERCRED_BOUND_HOST_BROKER_DECLARATION_ONLY"
                or any(getattr(self, name) is not False for name in (
                    "physical_training_authorized", "gpu_lease_enforced_proven",
                    "owner_approval_proven", "independent_review_required_for_development"))):
            raise HostBrokerUnavailable("dry-run evidence cannot escalate physical authority")


def _socket_identity(path: Path, expected_uid: int) -> tuple[int, int]:
    if (not isinstance(path, Path) or not path.is_absolute()
            or ".." in path.parts):
        raise HostBrokerUnavailable("canonical absolute broker socket required")
    if type(expected_uid) is not int or expected_uid != 0:
        raise HostBrokerUnavailable("WSL dry-run requires a root-owned host broker")
    parts = [path, *path.parents]
    info = None
    for index, part in enumerate(parts):
        try:
            record = part.lstat()
        except OSError as exc:
            raise HostBrokerUnavailable("broker socket path unavailable") from exc
        if stat.S_ISLNK(record.st_mode):
            raise HostBrokerUnavailable("broker socket ancestry may not be symlinked")
        if record.st_uid != expected_uid:
            raise HostBrokerUnavailable("broker socket ancestry has untrusted owner")
        if index == 0:
            if not stat.S_ISSOCK(record.st_mode):
                raise HostBrokerUnavailable("broker endpoint must be a Unix socket")
            # Socket write mode allows clients to CONNECT, not to replace
            # its protected directory entry. Denying it would make a
            # root-only socket unusable by nonroot development clients.
            info = record
        else:
            if not stat.S_ISDIR(record.st_mode):
                raise HostBrokerUnavailable("broker endpoint ancestor not a directory")
            if record.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise HostBrokerUnavailable("broker directory ancestry is group/world writable")
    if info is None:
        raise HostBrokerUnavailable("broker socket identity unavailable")
    return info.st_dev, info.st_ino


def _read_exact(sock: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        part = sock.recv(count - len(data))
        if not part:
            raise HostBrokerUnavailable("host broker closed a partial response")
        data.extend(part)
    return bytes(data)


def inspect_owner_broker_dryrun(
    *, broker_socket: Path, expected_owner_uid: int,
    intent: LocalDevelopmentIntent,
) -> HostBrokerDryRunReceipt:
    """Observe a privileged local broker response without granting execution.

    A genuine deployment must provide and protect the broker independently
    of the calling training Python process. Host-owned broker installation,
    actual GPU lease enforcement, owner interaction and provenance of weights
    cannot be certified from this client response.
    """
    if type(intent) is not LocalDevelopmentIntent:
        raise HostBrokerUnavailable("typed development intent required")
    LocalDevelopmentIntent.__post_init__(intent)
    if (not hasattr(os, "getuid") or not hasattr(socket, "SO_PEERCRED")
            or not hasattr(socket, "AF_UNIX")):
        raise HostBrokerUnavailable("Linux/WSL kernel peer credential route required")
    if expected_owner_uid == os.getuid():
        raise HostBrokerUnavailable("caller cannot self-authorize as its own broker")
    before = _socket_identity(broker_socket, expected_owner_uid)
    nonce = secrets.token_hex(32)
    request = {"schema": SCHEMA, "nonce": nonce,
               "intent": asdict(intent)}
    intent_digest = sha256(_canonical(request["intent"])).hexdigest()
    payload = _canonical(request)
    if len(payload) > _MAX_PACKET:
        raise HostBrokerUnavailable("lease request exceeds bounded protocol")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(3.0)
            conn.connect(str(broker_socket))
            peer = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            pid, uid, _gid = struct.unpack("3i", peer)
            if uid != expected_owner_uid or pid <= 0:
                raise HostBrokerUnavailable("host broker kernel credentials disagree")
            conn.sendall(len(payload).to_bytes(4, "big") + payload)
            response_bytes_count = int.from_bytes(_read_exact(conn, 4), "big")
            if not 1 <= response_bytes_count <= _MAX_PACKET:
                raise HostBrokerUnavailable("invalid bounded broker response size")
            raw_response = _read_exact(conn, response_bytes_count)
    except OSError as exc:
        raise HostBrokerUnavailable("host-owned lease broker unavailable") from exc
    if _socket_identity(broker_socket, expected_owner_uid) != before:
        raise HostBrokerUnavailable("host broker socket replaced during probe")
    try:
        reply = json.loads(raw_response.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise HostBrokerUnavailable("broker returned invalid lease JSON") from exc
    if (type(reply) is not dict or set(reply) != {
                "schema", "nonce", "intent_sha256", "lease_id",
                "broker_uid", "lease_expires_unix", "decision"}
            or reply["schema"] != SCHEMA or reply["nonce"] != nonce
            or reply["intent_sha256"] != intent_digest
            or type(reply["lease_id"]) is not str or not _ID.fullmatch(reply["lease_id"])
            or type(reply["broker_uid"]) is not int or reply["broker_uid"] != uid
            or type(reply["lease_expires_unix"]) is not int
            or not int(time.time()) <= reply["lease_expires_unix"] <= int(time.time()) + 120
            or reply["decision"] != "OWNER_APPROVED_DEV_ONLY"):
        raise HostBrokerUnavailable("broker grant not bound to the current request")
    return HostBrokerDryRunReceipt(
        broker_uid=uid, broker_pid=pid, lease_id=reply["lease_id"],
        intent_sha256=intent_digest,
        response_sha256=sha256(raw_response).hexdigest(),
    )


__all__ = [
    "HostBrokerUnavailable", "LocalDevelopmentIntent",
    "HostBrokerDryRunReceipt", "inspect_owner_broker_dryrun",
]
