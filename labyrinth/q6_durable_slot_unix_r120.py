"""SOURCE-ONLY Q6 durable-slot Unix socket frontend using Linux SO_PEERCRED.

Trusted principals come exclusively from the operator's UID mapping, never
client JSON. Only begin/complete/read operations are exposed; no campaign
provisioning, prompt/gold disclosure, model execution or credential APIs.
This runnable adapter does not establish separate real host accounts.
"""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import socket
import socketserver
import stat
from types import MappingProxyType
from typing import Mapping

from .authority_unix_service_r22 import (
    UnixAuthorityTransportError, _peer_uid, _recv_frame,
    _send_frame, _require_keys, _require_private_parent,
)
from .q6_durable_slot_authority_r120 import (
    Q6DurableConflict, Q6DurableError, Q6DurableHead,
    Q6DurableOutcomeUnknown, Q6DurableSlotAuthority, _hash, _id,
)


SCHEMA = "labyrinth.q6-durable-unix.source-r120.v1"
STATUS = "SOURCE_ONLY_NO_INDEPENDENT_HOST_OR_PROMPT_CUSTODY"


class _Q6Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(5.0)
        try:
            uid = _peer_uid(self.request)
            request = _recv_frame(self.request)
            result = self.server.dispatch(uid, request)
        except (OSError, TimeoutError, UnixAuthorityTransportError):
            result = {"schema": SCHEMA, "ok": False, "error": "BAD_REQUEST"}
        except Exception:
            result = {"schema": SCHEMA, "ok": False, "error": "INTERNAL"}
        try:
            _send_frame(self.request, result)
        except (OSError, TimeoutError, UnixAuthorityTransportError):
            pass


class Q6DurableUnixService(socketserver.ThreadingMixIn,
                           socketserver.UnixStreamServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = False
    request_queue_size = 16
    qualification = STATUS

    def __init__(self, *, socket_path: Path, core: Q6DurableSlotAuthority,
                 principal_by_uid: Mapping[int, str],
                 authorized_group_gid: int | None = None):
        if os.name != "posix" or not hasattr(socket, "SO_PEERCRED"):
            raise UnixAuthorityTransportError("Linux SO_PEERCRED required")
        if (not isinstance(socket_path, Path) or not socket_path.is_absolute()
                or not isinstance(core, Q6DurableSlotAuthority)
                or len(os.fsencode(socket_path)) > 100 or os.path.lexists(socket_path)):
            raise UnixAuthorityTransportError("unsafe Q6 Unix socket path or service core")
        if core._unwitnessed_simulation or core._witness is None:
            raise UnixAuthorityTransportError(
                "Q6 Unix exposure service requires external witness CAS; simulation forbidden")
        _require_private_parent(socket_path, label="Q6 operator socket")
        parent = socket_path.parent
        owner = parent.stat()
        if (parent.is_symlink() or owner.st_uid != os.geteuid()
                or owner.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            raise UnixAuthorityTransportError("Q6 socket directory not privately owned")
        if authorized_group_gid is not None and (
                type(authorized_group_gid) is not int or authorized_group_gid < 0
                or owner.st_gid != authorized_group_gid
                or not owner.st_mode & stat.S_IXGRP):
            raise UnixAuthorityTransportError("operator group not provisioned")
        if not isinstance(principal_by_uid, Mapping) or not principal_by_uid:
            raise UnixAuthorityTransportError("operator UID roster required")
        roster = {}
        for uid, principal in principal_by_uid.items():
            if type(uid) is not int or uid < 0 or uid > 2**32 - 1:
                raise UnixAuthorityTransportError("invalid peer UID")
            _id(principal, "authenticated peer")
            if principal not in core._roles or core._roles[principal] == "operator":
                raise UnixAuthorityTransportError("socket cannot expose operator role")
            roster[uid] = principal
        self._principals = MappingProxyType(roster)
        self._core = core
        self._path = socket_path
        self._socket_mode = 0o660 if authorized_group_gid is not None else 0o600
        self._bound_stat = None
        try:
            super().__init__(str(socket_path), _Q6Handler, bind_and_activate=True)
            if authorized_group_gid is not None:
                os.chown(socket_path, -1, authorized_group_gid)
            socket_path.chmod(self._socket_mode)
            info = socket_path.lstat()
            if (not stat.S_ISSOCK(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != self._socket_mode):
                raise UnixAuthorityTransportError("Q6 socket mode invalid")
            self._bound_stat = (info.st_dev, info.st_ino)
        except BaseException:
            try:
                super().server_close()
            except Exception:
                pass
            if os.path.lexists(socket_path) and not socket_path.is_symlink():
                try:
                    if stat.S_ISSOCK(socket_path.lstat().st_mode):
                        socket_path.unlink()
                except OSError:
                    pass
            raise

    def server_close(self) -> None:
        super().server_close()
        if self._bound_stat is not None:
            try:
                info = self._path.lstat()
                if stat.S_ISSOCK(info.st_mode) and (
                        info.st_dev, info.st_ino) == self._bound_stat:
                    self._path.unlink()
            except FileNotFoundError:
                pass

    def dispatch(self, uid: int, data: dict[str, object]) -> dict[str, object]:
        def failure(code: str) -> dict[str, object]:
            return {"schema": SCHEMA, "ok": False, "error": code}
        principal = self._principals.get(uid)
        if principal is None:
            return failure("FORBIDDEN")
        try:
            if type(data) is not dict or data.get("schema") != SCHEMA:
                raise Q6DurableError("invalid transport schema")
            op = data.get("op")
            if op == "read_head":
                _require_keys(data, frozenset(("schema", "op")))
                value = self._core.read_head()
            elif op == "reconcile":
                _require_keys(data, frozenset(("schema", "op", "slot_id")))
                value = self._core.reconcile(_hash(data["slot_id"], "slot"),
                                             principal=principal)
            elif op in ("begin", "complete"):
                keys = {"schema", "op", "slot_id", "expected_head"}
                if op == "complete":
                    keys |= {"result_sha256", "execution_sha256",
                             "checkpoint_sha256", "completion_nonce"}
                _require_keys(data, frozenset(keys))
                raw_head = _require_keys(data["expected_head"], frozenset(
                    Q6DurableHead.__dataclass_fields__))
                head = Q6DurableHead(**raw_head)
                if (type(head.commit_index) is not int or head.commit_index < 0
                        or head.source_only_unqualified is not True):
                    raise Q6DurableError("invalid expected host head")
                _hash(head.head_sha256, "expected head")
                shared = dict(slot_id=_hash(data["slot_id"], "slot"),
                              expected_head=head, principal=principal)
                if op == "begin":
                    value = self._core.begin_exposure(**shared)
                else:
                    value = self._core.complete_exposure(
                        **shared, result_sha256=data["result_sha256"],
                        execution_sha256=data["execution_sha256"],
                        checkpoint_sha256=data["checkpoint_sha256"],
                        completion_nonce=data["completion_nonce"])
            else:
                raise Q6DurableError("operation not exposed")
            return {"schema": SCHEMA, "ok": True, "value": asdict(value)}
        except Q6DurableConflict:
            return failure("CONFLICT")
        except Q6DurableOutcomeUnknown:
            return failure("OUTCOME_UNKNOWN")
        except (Q6DurableError, UnixAuthorityTransportError, ValueError,
                TypeError, KeyError):
            return failure("BAD_REQUEST")
        except Exception:
            return failure("INTERNAL")
