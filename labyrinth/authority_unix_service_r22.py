"""Linux Unix-socket front end for the SOURCE-ONLY Journey authority core.

This is a runnable transport prototype, not an installed host authority.  The
server takes its UID-to-principal roster from operator-side configuration and
derives the caller's UID from ``SO_PEERCRED``; no claimed principal is accepted
from a client request.  A real deployment must independently isolate its Unix
account, signing key, database, config and rollback domain from the trainer.
An administrator or trainer running under the service UID can defeat that
boundary; the source tests here do not qualify host custody or anti-rollback.

Each connection carries one 4-byte big-endian length followed by one exact
canonical ASCII JSON object (<=1 MiB); the server replies once and closes.
There is deliberately no pickle, shell command, generic file RPC, remote
listener, model execution, or graduation endpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import socketserver
import stat
import struct
from types import MappingProxyType
from typing import Mapping

from .authority_service_core_r20 import (
    AuthorityCoreConflict, AuthorityCoreError, AuthorityCoreOutcomeUnknown,
    HostProvisionedJourney, SourceOnlyJourneyAuthorityCore,
)
from .external_monotonic_authority_contract import (
    ExternalReservationRequest, _identifier,
)
from .journey_external_authority_contract_r17 import JourneyExternalAuthorityBinding


SCHEMA = "labyrinth.authority-unix-transport.v1"
OPERATOR_SCHEMA = "labyrinth.authority-unix-operator-provision.v1"
STATUS = "SOURCE_ONLY_NO_PRIVILEGED_HOST_DEPLOYMENT"
MAX_FRAME_BYTES = 1_048_576
_CRED_STRUCT = struct.Struct("3i")
_SIZE_STRUCT = struct.Struct("!I")
_ERRORS = frozenset(("BAD_REQUEST", "FORBIDDEN", "CONFLICT", "OUTCOME_UNKNOWN", "INTERNAL"))


class UnixAuthorityTransportError(ValueError):
    """A bounded, exact-source Unix transport operation was refused."""


def _require_private_parent(path: Path, *, label: str) -> None:
    """Reject replaceable components along an operator path's entire ancestry.

    An immediately private directory is insufficient if another user can
    rename it through a writable grandparent. All ancestors must be owned by
    the server UID or root and must not be group/other writable. The sole
    exception is a *non-immediate*, root-owned sticky directory (such as
    ``/tmp``): sticky unlink semantics protect a private child owned by the
    server from other UIDs. This is not same-UID or administrator isolation.

    These source checks cannot make path resolution atomic against a process
    already privileged to administer the server's directories.
    """
    if not isinstance(path, Path) or not path.is_absolute():
        raise UnixAuthorityTransportError(f"{label} requires absolute operator path")
    try:
        for depth, parent in enumerate(path.parents):
            info = parent.lstat()
            if (not stat.S_ISDIR(info.st_mode)
                    or parent.resolve(strict=True) != parent
                    or info.st_uid not in (os.geteuid(), 0)):
                raise UnixAuthorityTransportError(
                    f"{label} directory ancestry not controlled by server/root")
            if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                # Never allow direct operator-file/socket/DB parents to be
                # writable. Only a root-owned sticky *ancestor* has the
                # cross-UID unlink protection needed for private children.
                if (depth == 0 or info.st_uid != 0
                        or not info.st_mode & stat.S_ISVTX):
                    raise UnixAuthorityTransportError(
                        f"{label} directory ancestry writable by another UID")
    except UnixAuthorityTransportError:
        raise
    except (OSError, ValueError) as exc:
        raise UnixAuthorityTransportError(f"{label} parent unavailable") from exc


def _read_operator_file(path: Path, *, max_bytes: int) -> bytes:
    """Read an owned private regular file without following links.

    These checks reduce accidental exposure during SOURCE tests.  They cannot
    prove that a trainer cannot administer the server account or filesystem.
    """
    if (not isinstance(path, Path) or not path.is_absolute()
            or not hasattr(os, "O_NOFOLLOW")):
        raise UnixAuthorityTransportError("absolute Linux private-file path required")
    _require_private_parent(path, label="operator file")
    try:
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) & 0o077
                or before.st_nlink != 1 or not 0 < before.st_size <= max_bytes):
            raise UnixAuthorityTransportError("operator file ownership/mode/size refused")
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(path, flags)
        try:
            after = os.fstat(fd)
            if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                raise UnixAuthorityTransportError("operator file changed during open")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(max_bytes + 1)
            if len(data) != before.st_size or len(data) > max_bytes:
                raise UnixAuthorityTransportError("operator file changed during read")
            return data
        finally:
            os.close(fd)
    except UnixAuthorityTransportError:
        raise
    except (OSError, ValueError) as exc:
        raise UnixAuthorityTransportError("operator file unavailable") from exc


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise UnixAuthorityTransportError("duplicate JSON key")
        result[name] = value
    return result


def _reject_constant(value: str) -> object:
    raise UnixAuthorityTransportError("nonfinite JSON value")


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise UnixAuthorityTransportError("uncanonicalizable JSON value") from exc


def _decode_exact(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise UnixAuthorityTransportError("invalid JSON frame size")
    try:
        value = json.loads(payload.decode("ascii"),
                           object_pairs_hook=_object_pairs,
                           parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnixAuthorityTransportError("noncanonical JSON frame") from exc
    if type(value) is not dict or _canonical(value) != payload:
        raise UnixAuthorityTransportError("JSON must be exact canonical object")
    return value


def _read_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    while length:
        chunk = sock.recv(length)
        if not chunk:
            raise UnixAuthorityTransportError("truncated Unix frame")
        chunks.append(chunk)
        length -= len(chunk)
    return b"".join(chunks)


def _recv_frame(sock: socket.socket) -> dict[str, object]:
    length = _SIZE_STRUCT.unpack(_read_exact(sock, _SIZE_STRUCT.size))[0]
    if not 0 < length <= MAX_FRAME_BYTES:
        raise UnixAuthorityTransportError("oversized or empty Unix frame")
    return _decode_exact(_read_exact(sock, length))


def _send_frame(sock: socket.socket, reply: dict[str, object]) -> None:
    body = _canonical(reply)
    if len(body) > MAX_FRAME_BYTES:
        raise UnixAuthorityTransportError("Unix response exceeds frame limit")
    sock.sendall(_SIZE_STRUCT.pack(len(body)) + body)


def _require_keys(value: object, names: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != names:
        raise UnixAuthorityTransportError("unexpected Unix request fields")
    return value


def _construct_exact(cls, value: object):
    data = _require_keys(value, frozenset(cls.__dataclass_fields__))
    obj = cls(**data)
    obj.validate()
    return obj


def _peer_uid(sock: socket.socket) -> int:
    if not hasattr(socket, "SO_PEERCRED") or not hasattr(socket, "AF_UNIX"):
        raise UnixAuthorityTransportError("Linux peer-credential API unavailable")
    raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _CRED_STRUCT.size)
    if len(raw) != _CRED_STRUCT.size:
        raise UnixAuthorityTransportError("Linux peer credentials incomplete")
    pid, uid, gid = _CRED_STRUCT.unpack(raw)
    if pid <= 0 or uid < 0 or gid < 0:
        raise UnixAuthorityTransportError("Linux peer credentials invalid")
    return uid


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(5.0)
        try:
            uid = _peer_uid(self.request)
            incoming = _recv_frame(self.request)
            reply = self.server.dispatch(uid, incoming)
        except (OSError, TimeoutError, UnixAuthorityTransportError):
            reply = {"schema": SCHEMA, "ok": False, "error": "BAD_REQUEST"}
        except Exception:
            # Never disclose a Python traceback or operator configuration.
            reply = {"schema": SCHEMA, "ok": False, "error": "INTERNAL"}
        try:
            _send_frame(self.request, reply)
        except (OSError, TimeoutError, UnixAuthorityTransportError):
            pass


class UnixAuthorityService(socketserver.ThreadingMixIn,
                           socketserver.UnixStreamServer):
    """One-request Unix transport around a separately provisioned service core.

    An operator must create an owned/private socket directory, separate the
    server principal from the trainer, securely load the signing key and
    provision complete Journeys *before* constructing the core.  Constructor
    arguments are not remotely supplied.  Same-UID trainer/service or root
    compromise defeats the intended physical separation.
    """

    allow_reuse_address = False
    daemon_threads = False
    block_on_close = True
    request_queue_size = 16
    qualification = STATUS

    def __init__(self, *, socket_path: Path,
                 core: SourceOnlyJourneyAuthorityCore,
                 principal_by_uid: Mapping[int, str],
                 authorized_group_gid: int | None = None):
        if os.name != "posix" or not hasattr(socket, "SO_PEERCRED"):
            raise UnixAuthorityTransportError("Linux SO_PEERCRED required")
        if (not isinstance(socket_path, Path) or not socket_path.is_absolute()
                or not isinstance(core, SourceOnlyJourneyAuthorityCore)):
            raise UnixAuthorityTransportError("absolute socket Path and source authority core required")
        if len(os.fsencode(socket_path)) > 100 or os.path.lexists(socket_path):
            raise UnixAuthorityTransportError("socket path unsafe or already occupied")
        _require_private_parent(socket_path, label="operator socket")
        parent = socket_path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise UnixAuthorityTransportError("socket parent must be an existing directory")
        parent_stat = parent.stat()
        if (parent_stat.st_uid != os.geteuid()
                or parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            raise UnixAuthorityTransportError("socket parent not private to server owner")
        # A real separate trainer UID cannot connect to a 0600 socket in a
        # 0700 directory.  A host operator may provision a *non-writable*
        # 0750 group-traversable directory and a 0660 socket instead.  The
        # server still authenticates exact peer UID; group access alone does
        # not confer a permitted Journey principal.  Local tests do not prove
        # that OS accounts or group membership are independently administered.
        if authorized_group_gid is not None:
            if (type(authorized_group_gid) is not int
                    or authorized_group_gid < 0
                    or parent_stat.st_gid != authorized_group_gid
                    or not parent_stat.st_mode & stat.S_IXGRP):
                raise UnixAuthorityTransportError("operator socket group not provisioned")
        socket_mode = 0o660 if authorized_group_gid is not None else 0o600
        if (not isinstance(principal_by_uid, Mapping) or not principal_by_uid):
            raise UnixAuthorityTransportError("operator UID principal roster required")
        copied: dict[int, str] = {}
        for uid, principal in principal_by_uid.items():
            if type(uid) is not int or uid < 0:
                raise UnixAuthorityTransportError("invalid authorized peer UID")
            try:
                _identifier(principal, "operator pinned principal")
            except ValueError as exc:
                raise UnixAuthorityTransportError("invalid authorized principal") from exc
            copied[uid] = principal
        self._principals = MappingProxyType(copied)
        self._run_principals = MappingProxyType({
            run_id: record.authenticated_principal_id
            for run_id, record in core._roster.items()
        })
        self._authority = core
        self._socket_path = socket_path
        self._socket_mode = socket_mode
        self._bound_stat: tuple[int, int] | None = None
        try:
            super().__init__(str(socket_path), _Handler, bind_and_activate=True)
            if authorized_group_gid is not None:
                os.chown(socket_path, -1, authorized_group_gid)
            socket_path.chmod(socket_mode)
            socket_stat = socket_path.lstat()
            if (not stat.S_ISSOCK(socket_stat.st_mode)
                    or stat.S_IMODE(socket_stat.st_mode) != socket_mode
                    or (authorized_group_gid is not None
                        and socket_stat.st_gid != authorized_group_gid)):
                raise UnixAuthorityTransportError("socket mode not restricted")
            self._bound_stat = (socket_stat.st_dev, socket_stat.st_ino)
        except BaseException:
            try:
                super().server_close()
            except Exception:
                pass
            # Constructor owns this initially absent path. Never chase links.
            if os.path.lexists(socket_path) and not socket_path.is_symlink():
                try:
                    if stat.S_ISSOCK(socket_path.lstat().st_mode):
                        socket_path.unlink()
                except OSError:
                    pass
            raise

    def server_close(self) -> None:
        super().server_close()
        if self._bound_stat is None:
            return
        try:
            info = self._socket_path.lstat()
            if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == self._bound_stat:
                self._socket_path.unlink()
        except FileNotFoundError:
            pass

    def dispatch(self, uid: int, data: dict[str, object]) -> dict[str, object]:
        def failure(error: str) -> dict[str, object]:
            assert error in _ERRORS
            return {"schema": SCHEMA, "ok": False, "error": error}

        principal = self._principals.get(uid)
        if principal is None:
            return failure("FORBIDDEN")
        try:
            if (data.get("schema") != SCHEMA or type(data.get("op")) is not str):
                raise UnixAuthorityTransportError("wrong transport schema or operation")
            operation = data["op"]
            if operation == "reserve":
                _require_keys(data, frozenset(("schema", "op", "binding", "request")))
                binding = _construct_exact(JourneyExternalAuthorityBinding, data["binding"])
                request = _construct_exact(ExternalReservationRequest, data["request"])
                if self._run_principals.get(binding.run_id) != principal:
                    return failure("FORBIDDEN")
                value = self._authority.reserve_journey(
                    binding, request, authenticated_principal_id=principal)
            elif operation in ("read_head", "read_receipt"):
                required = ({"schema", "op", "run_id"} if operation == "read_head" else
                            {"schema", "op", "run_id", "event_id", "operation_nonce"})
                _require_keys(data, frozenset(required))
                run_id = _identifier(data["run_id"], "run")
                if self._run_principals.get(run_id) != principal:
                    return failure("FORBIDDEN")
                if operation == "read_head":
                    value = self._authority.read_head(run_id)
                else:
                    event_id = _identifier(data["event_id"], "event")
                    nonce = _identifier(data["operation_nonce"], "operation nonce")
                    value = self._authority.read_receipt(run_id, event_id, nonce)
            else:
                raise UnixAuthorityTransportError("unsupported authority operation")
            return {"schema": SCHEMA, "ok": True,
                    "value": asdict(value) if value is not None else None}
        except (AuthorityCoreConflict,):
            return failure("CONFLICT")
        except AuthorityCoreOutcomeUnknown:
            return failure("OUTCOME_UNKNOWN")
        except (AuthorityCoreError, UnixAuthorityTransportError, ValueError, TypeError, KeyError):
            return failure("BAD_REQUEST")
        except Exception:
            return failure("INTERNAL")


def load_operator_service(*, config_path: Path, signing_key_path: Path,
                          database_path: Path, socket_path: Path,
                          authorized_group_gid: int | None = None) -> UnixAuthorityService:
    """Build an explicitly SOURCE-ONLY server from private operator files.

    The JSON provisioning file contains only *declared* Journey/legacy pins;
    no code here authenticates a historical migration witness.  The signing
    seed must be a separate 0600 binary file of exactly 32 bytes.  Do not put
    either file under the trainer's administration or regard a successful
    constructor as proof of OS principal separation.
    """
    config = _decode_exact(_read_operator_file(config_path,
                                               max_bytes=MAX_FRAME_BYTES))
    _require_keys(config, frozenset(("schema", "provider_id", "principal_by_uid",
                                     "journeys")))
    if config["schema"] != OPERATOR_SCHEMA:
        raise UnixAuthorityTransportError("wrong operator provisioning schema")
    if (type(config["principal_by_uid"]) is not dict
            or not config["principal_by_uid"]
            or type(config["journeys"]) is not list
            or not config["journeys"]):
        raise UnixAuthorityTransportError("operator UID and Journey rosters required")
    uid_roster: dict[int, str] = {}
    for raw_uid, principal in config["principal_by_uid"].items():
        if (type(raw_uid) is not str or not raw_uid.isascii()
                or not raw_uid.isdecimal() or len(raw_uid) > 10
                or (raw_uid.startswith("0") and raw_uid != "0")):
            raise UnixAuthorityTransportError("invalid canonical operator UID")
        uid = int(raw_uid)
        if uid > 2**32 - 1:
            raise UnixAuthorityTransportError("operator UID exceeds Linux range")
        uid_roster[uid] = principal
    journeys = []
    for value in config["journeys"]:
        record = _require_keys(value, frozenset(HostProvisionedJourney.__dataclass_fields__))
        binding = _construct_exact(JourneyExternalAuthorityBinding, record["binding"])
        provision = HostProvisionedJourney(
            binding=binding,
            authenticated_principal_id=record["authenticated_principal_id"],
            operator_pinned_ledger_root_sha256=record["operator_pinned_ledger_root_sha256"],
            operator_pinned_migration_witness_sha256=record["operator_pinned_migration_witness_sha256"],
            initial_authority_epoch=record["initial_authority_epoch"],
            initial_fence=record["initial_fence"],
        )
        provision.validate()
        journeys.append(provision)
    if (not isinstance(database_path, Path) or not database_path.is_absolute()
            or database_path.is_symlink()):
        raise UnixAuthorityTransportError("absolute nonsymlink operator database path required")
    _require_private_parent(database_path, label="operator database")
    signing_seed = _read_operator_file(signing_key_path, max_bytes=32)
    if len(signing_seed) != 32:
        raise UnixAuthorityTransportError("Ed25519 operator signing seed must be 32 bytes")
    try:
        core = SourceOnlyJourneyAuthorityCore(
            database_path=database_path,
            provider_id=config["provider_id"],
            service_signing_key=signing_seed,
            provisioned_journeys=tuple(journeys),
        )
    except (AuthorityCoreError, ValueError, TypeError) as exc:
        raise UnixAuthorityTransportError("operator source authority provisioning refused") from exc
    return UnixAuthorityService(
        socket_path=socket_path, core=core,
        principal_by_uid=uid_roster,
        authorized_group_gid=authorized_group_gid,
    )


def main(argv: list[str] | None = None) -> int:
    """Explicit opt-in source exercise; not a qualified host deployment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-only-test-service", action="store_true",
                        help="Acknowledge this is NOT an independently qualified authority")
    parser.add_argument("--operator-config", type=Path, required=True)
    parser.add_argument("--signing-key-file", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--authorized-group-gid", type=int)
    args = parser.parse_args(argv)
    if not args.source_only_test_service:
        parser.error("explicit --source-only-test-service is required")
    server = load_operator_service(
        config_path=args.operator_config.absolute(),
        signing_key_path=args.signing_key_file.absolute(),
        database_path=args.database.absolute(),
        socket_path=args.socket.absolute(),
        authorized_group_gid=args.authorized_group_gid,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
