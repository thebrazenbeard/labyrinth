"""Fail-closed Unix-socket client for a separately operated Journey authority.

The client transports a full, operator-pinned R17 Journey binding and an exact
reservation request.  It never sends a principal claim: a qualifying server
must derive its peer principal from OS credentials.  Responses are validated
against an independently pinned Ed25519 public key and reservation context.

This is SOURCE-ONLY transport code, not evidence that a socket endpoint,
provisioning roster, key or database is outside trainer control.  A lost ACK
must be reconciled with a read-only receipt lookup; no automatic reserve retry
or graduation/model-execution permission is provided.
"""

from __future__ import annotations

from dataclasses import asdict
import hmac
import json
from pathlib import Path
import socket
import stat
import struct
from typing import Any

from .external_monotonic_authority_contract import (
    ExternalAuthorityConflict,
    ExternalAuthorityError,
    ExternalAuthorityHead,
    ExternalAuthorityOutcomeUnknown,
    ExternalReservationReceipt,
    ExternalReservationRequest,
    _identifier,
    _verify_ed25519,
    next_head_digest,
)
from .journey_external_authority_contract_r17 import (
    JourneyExternalAuthorityBinding,
    verify_journey_external_request,
)


SCHEMA = "labyrinth.authority-unix-transport.v1"
MAX_FRAME_BYTES = 1_048_576
_ERRORS = frozenset(("BAD_REQUEST", "FORBIDDEN", "CONFLICT",
                     "OUTCOME_UNKNOWN", "INTERNAL"))


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ExternalAuthorityError("noncanonical Unix authority message") from exc


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate Unix authority JSON field")
        result[key] = value
    return result


def _strict_object(data: bytes) -> dict[str, object]:
    try:
        parsed = json.loads(data.decode("ascii"), object_pairs_hook=_no_duplicate_keys,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if type(parsed) is not dict or _canonical(parsed) != data:
            raise ValueError("noncanonical Unix authority response")
        return parsed
    except (UnicodeError, ValueError, TypeError, ExternalAuthorityError) as exc:
        raise ExternalAuthorityOutcomeUnknown(
            "malformed or noncanonical Unix authority response") from exc


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise ExternalAuthorityOutcomeUnknown("truncated Unix authority response")
        result.extend(chunk)
    return bytes(result)


class UnixAuthorityPort:
    """One pinned reservation carried over one-frame-per-connection AF_UNIX.

    The public key, full binding, and request MUST arrive from a host-managed
    provisioner.  The Python object cannot authenticate that provenance.  The
    socket path must be host controlled: checks here are defensive, not an
    independent filesystem or process-identity attestation.
    """

    def __init__(self, *, socket_path: Path,
                 pinned_binding: JourneyExternalAuthorityBinding,
                 pinned_request: ExternalReservationRequest,
                 pinned_ed25519_public_key: bytes,
                 timeout_seconds: float = 3.0):
        if (not isinstance(socket_path, Path) or not socket_path.is_absolute()
                or type(pinned_binding) is not JourneyExternalAuthorityBinding
                or type(pinned_request) is not ExternalReservationRequest
                or type(pinned_ed25519_public_key) is not bytes
                or len(pinned_ed25519_public_key) != 32
                or type(timeout_seconds) not in (float, int)
                or not 0.01 <= timeout_seconds <= 30):
            raise ExternalAuthorityError("absolute socket path and pinned authority inputs required")
        pinned_binding.validate()
        pinned_request.validate()
        try:
            verify_journey_external_request(
                pinned_binding, pinned_request.expected_head(), pinned_request,
                operation_nonce=pinned_request.operation_nonce)
        except ValueError as exc:
            raise ExternalAuthorityConflict("request differs from pinned full Journey preimage") from exc
        self._socket_path = socket_path
        self._binding = pinned_binding
        self._request = pinned_request
        self._key = pinned_ed25519_public_key
        self._timeout = float(timeout_seconds)

    def _exchange(self, payload: dict[str, object]) -> object:
        message = {"schema": SCHEMA, **payload}
        encoded = _canonical(message)
        if not 0 < len(encoded) <= MAX_FRAME_BYTES:
            raise ExternalAuthorityError("oversized Unix authority request")
        try:
            # No real identity guarantee: the socket directory must be
            # protected by the independently operated host service.
            if self._socket_path.is_symlink():
                raise ExternalAuthorityOutcomeUnknown("Unix authority socket path is a symlink")
            if self._socket_path.exists() and not stat.S_ISSOCK(self._socket_path.stat().st_mode):
                raise ExternalAuthorityOutcomeUnknown("Unix authority endpoint is not a socket")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self._timeout)
                connection.connect(str(self._socket_path))
                connection.sendall(struct.pack("!I", len(encoded)) + encoded)
                length = struct.unpack("!I", _recv_exact(connection, 4))[0]
                if not 0 < length <= MAX_FRAME_BYTES:
                    raise ExternalAuthorityOutcomeUnknown("invalid Unix authority response length")
                response = _strict_object(_recv_exact(connection, length))
                if connection.recv(1) != b"":
                    raise ExternalAuthorityOutcomeUnknown("trailing Unix authority response bytes")
        except ExternalAuthorityOutcomeUnknown:
            raise
        except (OSError, ValueError, struct.error, TimeoutError) as exc:
            raise ExternalAuthorityOutcomeUnknown("Unix authority transport outcome unknown") from exc
        if response.get("schema") != SCHEMA or type(response.get("ok")) is not bool:
            raise ExternalAuthorityOutcomeUnknown("untyped Unix authority response")
        if response["ok"]:
            if set(response) != {"schema", "ok", "value"}:
                raise ExternalAuthorityOutcomeUnknown("unexpected Unix authority success fields")
            return response["value"]
        if (set(response) != {"schema", "ok", "error"}
                or type(response["error"]) is not str
                or response["error"] not in _ERRORS):
            raise ExternalAuthorityOutcomeUnknown("unexpected Unix authority error fields")
        if response["error"] in ("BAD_REQUEST", "FORBIDDEN", "CONFLICT"):
            raise ExternalAuthorityConflict("Unix authority refused request")
        raise ExternalAuthorityOutcomeUnknown("Unix authority outcome unknown")

    @staticmethod
    def _typed(value: object, target: type, expected_fields: set[str]) -> Any:
        if type(value) is not dict or set(value) != expected_fields:
            raise ExternalAuthorityOutcomeUnknown("wrong Unix authority result schema")
        try:
            record = target(**value)
            if target is ExternalAuthorityHead:
                record.validate()
            else:
                record.signed_payload()
            return record
        except (TypeError, ValueError, ExternalAuthorityError) as exc:
            raise ExternalAuthorityOutcomeUnknown("invalid Unix authority result fields") from exc

    def read_head(self, run_id: str) -> ExternalAuthorityHead:
        _identifier(run_id, "run")
        if run_id != self._request.run_id:
            raise ExternalAuthorityConflict("cross-run head access refused")
        value = self._exchange({"op": "read_head", "run_id": run_id})
        head = self._typed(value, ExternalAuthorityHead,
                           set(ExternalAuthorityHead.__dataclass_fields__))
        if head.provider_id != self._request.provider_id or head.run_id != run_id:
            raise ExternalAuthorityOutcomeUnknown("foreign Unix authority head")
        return head

    def _verified_receipt(self, value: object) -> ExternalReservationReceipt:
        receipt = self._typed(value, ExternalReservationReceipt,
                              set(ExternalReservationReceipt.__dataclass_fields__))
        request = self._request
        if (receipt.provider_id != request.provider_id
                or receipt.run_id != request.run_id
                or receipt.event_id != request.event_id
                or not hmac.compare_digest(receipt.request_sha256, request.digest())
                or receipt.epoch != request.expected_epoch
                or receipt.fence != request.expected_fence
                or receipt.commit_index != request.expected_commit_index + 1
                or receipt.previous_head_sha256 != request.expected_head_sha256
                or receipt.new_head_sha256 != next_head_digest(request, receipt.reservation_id)
                or type(receipt.signature_hex) is not str
                or len(receipt.signature_hex) != 128):
            raise ExternalAuthorityOutcomeUnknown("Unix authority receipt differs from pinned intent")
        try:
            _verify_ed25519(self._key, receipt.signed_payload(),
                            bytes.fromhex(receipt.signature_hex))
        except (ValueError, ExternalAuthorityError) as exc:
            raise ExternalAuthorityOutcomeUnknown("Unix authority receipt signature untrusted") from exc
        return receipt

    def reserve_once(self, request: ExternalReservationRequest) -> ExternalReservationReceipt:
        if type(request) is not ExternalReservationRequest or request != self._request:
            raise ExternalAuthorityConflict("reservation differs from pinned request")
        value = self._exchange({
            "op": "reserve", "binding": asdict(self._binding),
            "request": asdict(self._request),
        })
        return self._verified_receipt(value)

    def reconcile_committed_receipt(self) -> ExternalReservationReceipt:
        """Read durable receipt after UNKNOWN; NEVER retransmit the reservation.

        A missing receipt, stale/advanced head, or failed readback means the
        outcome remains UNKNOWN.  Even success is source verification, not a
        qualified host permit or permission to graduate.
        """
        request = self._request
        value = self._exchange({
            "op": "read_receipt", "run_id": request.run_id,
            "event_id": request.event_id, "operation_nonce": request.operation_nonce,
        })
        if value is None:
            raise ExternalAuthorityOutcomeUnknown(
                "no independently confirmed receipt; do not retry reservation")
        receipt = self._verified_receipt(value)
        observed = self.read_head(request.run_id)
        if observed != receipt.post_head():
            raise ExternalAuthorityOutcomeUnknown(
                "recovery head differs from signed reservation; do not replay")
        return receipt
