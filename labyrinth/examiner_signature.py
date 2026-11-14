"""Verify external examiner Ed25519 attestations without treating them as custody.

A valid signature binds a caller-pinned public key to an exact precommit SHA.
It does NOT prove independent key custody, signature timing, model execution,
or that the public key was independently authenticated.
"""
from __future__ import annotations

import base64
import binascii
from hashlib import sha256
import hmac

from .live_model_probe import _canonical, _file_bytes, _load, SHA, ID

ENVELOPE_SCHEMA = "labyrinth.examiner-signed-precommit.v1"
PAYLOAD_SCHEMA = "labyrinth.examiner-signed-precommit-payload.v1"
SCOPE = "EXAMINER_AUTHORSHIP_OF_PRECOMMIT_ONLY"
MAX_ENVELOPE_BYTES = 4096


def verify_examiner_attestation(attestation_path, public_key_path, *,
                                expected_key_sha256: str,
                                expected_exam_id: str,
                                expected_precommit_sha256: str) -> dict:
    """Check exact envelope, signed payload, key digest and original precommit.

    The expected public-key hash is a caller-supplied pin, NOT a proof of
    signer independence. Never accept unrecognized 'custody' declarations.
    """
    if (type(expected_key_sha256) is not str
            or SHA.fullmatch(expected_key_sha256) is None
            or type(expected_exam_id) is not str
            or ID.fullmatch(expected_exam_id) is None
            or type(expected_precommit_sha256) is not str
            or SHA.fullmatch(expected_precommit_sha256) is None):
        raise ValueError("invalid examiner signature trust-anchor parameters")
    public = _file_bytes(public_key_path, 32)
    if len(public) != 32 or not hmac.compare_digest(
            sha256(public).hexdigest(), expected_key_sha256):
        raise ValueError("examiner public key differs from caller-pinned trust anchor")

    doc = _load(_file_bytes(attestation_path, MAX_ENVELOPE_BYTES))
    if (type(doc) is not dict
            or set(doc) != {"schema", "payload", "signature_ed25519_b64"}
            or doc.get("schema") != ENVELOPE_SCHEMA):
        raise ValueError("invalid examiner attestation envelope schema")
    payload = doc["payload"]
    if (type(payload) is not dict
            or set(payload) != {"schema", "exam_id", "precommit_sha256",
                                "examiner_key_sha256", "scope"}
            or payload.get("schema") != PAYLOAD_SCHEMA
            or payload.get("scope") != SCOPE):
        raise ValueError("invalid examiner attestation signed payload fields")
    if (payload["exam_id"] != expected_exam_id
            or payload["precommit_sha256"] != expected_precommit_sha256
            or payload["examiner_key_sha256"] != expected_key_sha256):
        raise ValueError("examiner attestation does not bind expected exam or precommit")
    encoded = doc["signature_ed25519_b64"]
    if type(encoded) is not str or len(encoded) != 88:
        raise ValueError("invalid examiner signature encoding")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != 64 or base64.b64encode(raw).decode("ascii") != encoded:
            raise ValueError("invalid examiner signature bytes")
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid examiner signature encoding") from exc
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise ValueError("Ed25519 verification requires optional cryptography dependency") from exc
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(raw, _canonical(payload))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("invalid examiner signature for signed precommit") from exc
    return {"exam_id": expected_exam_id, "examiner_key_sha256": expected_key_sha256,
            "evidence": "ED25519_VALID_CALLER_PINNED_KEY",
            "independent_custody_verified": False}
