"""Source-only mutually authenticated Q6 remote-witness client.

The remote endpoint, server key, private TLS client credentials and append-only
storage are NOT provisioned by this module. A same-process fake implementing
``Q6IndependentMonotonicWitness`` can still bypass a caller's trust boundary;
only independent operator custody of the witness and this client configuration
can close that gap. No prompt release is authorized by these source checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import http.client
import json
from pathlib import Path
import re
import secrets
import ssl
from urllib.parse import urlsplit

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .q6_durable_slot_authority_r120 import (
    Q6DurableHead, Q6DurableOutcomeUnknown, _canonical,
)


SCHEMA = "labyrinth.q6-independent-witness.source-r121.v1"
_SIGNATURE_DOMAIN = b"labyrinth.q6-independent-witness-reply.source-r121.v1\0"
_PATH = "/q6/witness/r121"
_MAX_RESPONSE = 4096
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_NONCE = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)


def _head_wire(head: Q6DurableHead) -> dict[str, object]:
    if (type(head) is not Q6DurableHead or type(head.commit_index) is not int
            or head.commit_index < 0 or type(head.head_sha256) is not str
            or not _HEX64.fullmatch(head.head_sha256)
            or head.source_only_unqualified is not True):
        raise Q6DurableOutcomeUnknown("malformed Q6 witness head")
    return {"commit_index": head.commit_index, "head_sha256": head.head_sha256}


def _from_wire(value: object) -> Q6DurableHead:
    if type(value) is not dict or set(value) != {"commit_index", "head_sha256"}:
        raise Q6DurableOutcomeUnknown("malformed signed witness head")
    index, digest = value["commit_index"], value["head_sha256"]
    if (type(index) is not int or index < 0 or type(digest) is not str
            or not _HEX64.fullmatch(digest)):
        raise Q6DurableOutcomeUnknown("invalid signed witness head fields")
    return Q6DurableHead(index, digest)


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Q6DurableOutcomeUnknown("duplicate witness response JSON key")
        result[key] = value
    return result


def verify_signed_reply(*, request: dict[str, object], reply_bytes: bytes,
                        pinned_signer_public_key: bytes) -> tuple[bool, Q6DurableHead]:
    """Verify strict, nonce-bound independent witness receipt; no authority grant."""
    if (type(reply_bytes) is not bytes or len(reply_bytes) > _MAX_RESPONSE
            or type(pinned_signer_public_key) is not bytes
            or len(pinned_signer_public_key) != 32):
        raise Q6DurableOutcomeUnknown("malformed witness signature inputs")
    if (type(request) is not dict or set(request) !=
            {"schema", "operation", "nonce", "expected", "new"}
            or request["schema"] != SCHEMA
            or request["operation"] not in ("READ", "CAS")
            or type(request["nonce"]) is not str
            or not _NONCE.fullmatch(request["nonce"])):
        raise Q6DurableOutcomeUnknown("malformed witness request challenge")
    try:
        reply = json.loads(reply_bytes.decode("ascii"),
                           object_pairs_hook=_no_duplicate_keys)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise Q6DurableOutcomeUnknown("noncanonical witness response") from exc
    if (type(reply) is not dict
            or set(reply) != {"schema", "operation", "nonce", "request_sha256",
                              "ok", "head", "signature_hex"}
            or reply["schema"] != SCHEMA
            or reply["operation"] != request["operation"]
            or reply["nonce"] != request["nonce"]
            or reply["request_sha256"] != sha256(_canonical(request)).hexdigest()
            or type(reply["ok"]) is not bool
            or type(reply["signature_hex"]) is not str
            or not re.fullmatch(r"[0-9a-f]{128}", reply["signature_hex"], re.ASCII)):
        raise Q6DurableOutcomeUnknown("witness reply not bound to this request")
    head = _from_wire(reply["head"])
    payload = {key: value for key, value in reply.items() if key != "signature_hex"}
    if _canonical(reply) != reply_bytes:
        raise Q6DurableOutcomeUnknown("witness response not canonical")
    try:
        Ed25519PublicKey.from_public_bytes(pinned_signer_public_key).verify(
            bytes.fromhex(reply["signature_hex"]),
            _SIGNATURE_DOMAIN + _canonical(payload),
        )
    except Exception as exc:
        raise Q6DurableOutcomeUnknown("independent witness signature invalid") from exc
    if request["operation"] == "READ":
        if reply["ok"] is not True or request["expected"] is not None or request["new"] is not None:
            raise Q6DurableOutcomeUnknown("invalid signed READ response")
    else:
        if (type(request["expected"]) is not dict or type(request["new"]) is not dict
                or _head_wire(_from_wire(request["expected"])) != request["expected"]
                or _head_wire(_from_wire(request["new"])) != request["new"]):
            raise Q6DurableOutcomeUnknown("invalid signed CAS challenge")
        if reply["ok"] is True and reply["head"] != request["new"]:
            raise Q6DurableOutcomeUnknown("witness ACK does not match new durable head")
    return reply["ok"], head


@dataclass(frozen=True)
class Q6WitnessMTLSConfiguration:
    endpoint: str
    ca_cert: Path
    client_cert: Path
    client_key: Path
    pinned_server_certificate_sha256: str
    pinned_signer_public_key: bytes
    timeout_seconds: float = 5.0


class Q6SignedMTLSRemoteWitness:
    """Transport adapter for independent externally governed witness service.

    The operator must control configuration and TLS credentials outside the
    trainer's rollback, filesystem and execution domains. Merely constructing
    this Python object is NOT host custody or physical witness attestation.
    """

    source_only_unqualified = True
    independent_host_attested = False

    def __init__(self, config: Q6WitnessMTLSConfiguration) -> None:
        if type(config) is not Q6WitnessMTLSConfiguration:
            raise Q6DurableOutcomeUnknown("typed witness TLS config required")
        parsed = urlsplit(config.endpoint)
        if (parsed.scheme != "https" or parsed.hostname is None
                or parsed.username is not None or parsed.password is not None
                or parsed.path != _PATH or parsed.query or parsed.fragment
                or not _HEX64.fullmatch(config.pinned_server_certificate_sha256)
                or type(config.pinned_signer_public_key) is not bytes
                or len(config.pinned_signer_public_key) != 32
                or type(config.timeout_seconds) not in (int, float)
                or not 0.5 <= config.timeout_seconds <= 30):
            raise Q6DurableOutcomeUnknown("insecure or unpinned remote witness endpoint")
        for path in (config.ca_cert, config.client_cert, config.client_key):
            if not isinstance(path, Path) or not path.is_absolute() or not path.is_file():
                raise Q6DurableOutcomeUnknown("operator witness TLS credentials unavailable")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.load_verify_locations(cafile=str(config.ca_cert))
        context.load_cert_chain(str(config.client_cert), str(config.client_key))
        self._context = context
        self._config = config
        self._hostname = parsed.hostname
        self._port = parsed.port or 443

    def _roundtrip(self, request: dict[str, object]) -> tuple[bool, Q6DurableHead]:
        conn = http.client.HTTPSConnection(self._hostname, port=self._port,
                                           timeout=self._config.timeout_seconds,
                                           context=self._context)
        try:
            conn.connect()
            peer = conn.sock.getpeercert(binary_form=True) if conn.sock else None
            if (type(peer) is not bytes or sha256(peer).hexdigest() !=
                    self._config.pinned_server_certificate_sha256):
                raise Q6DurableOutcomeUnknown("witness TLS peer certificate pin mismatch")
            conn.request("POST", _PATH, body=_canonical(request), headers={
                "Content-Type": "application/json", "Accept": "application/json",
            })
            response = conn.getresponse()
            if (response.status != 200 or
                    response.getheader("Content-Type", "").split(";", 1)[0] != "application/json"):
                raise Q6DurableOutcomeUnknown("witness HTTPS response not authenticated OK")
            reply = response.read(_MAX_RESPONSE + 1)
            return verify_signed_reply(request=request, reply_bytes=reply,
                                       pinned_signer_public_key=self._config.pinned_signer_public_key)
        except Q6DurableOutcomeUnknown:
            raise
        except Exception as exc:
            raise Q6DurableOutcomeUnknown("remote witness outcome unknown; no prompt release") from exc
        finally:
            conn.close()

    def read_head(self) -> Q6DurableHead:
        request = {"schema": SCHEMA, "operation": "READ", "nonce": secrets.token_hex(16),
                   "expected": None, "new": None}
        accepted, head = self._roundtrip(request)
        if accepted is not True:
            raise Q6DurableOutcomeUnknown("signed witness READ refused")
        return head

    def compare_and_advance(self, expected: Q6DurableHead, new: Q6DurableHead) -> bool:
        if new.commit_index != expected.commit_index + 1:
            raise Q6DurableOutcomeUnknown("witness CAS must advance one global event")
        request = {"schema": SCHEMA, "operation": "CAS", "nonce": secrets.token_hex(16),
                   "expected": _head_wire(expected), "new": _head_wire(new)}
        accepted, head = self._roundtrip(request)
        if accepted and head != new:
            raise Q6DurableOutcomeUnknown("signed witness ACK head mismatch")
        return accepted
