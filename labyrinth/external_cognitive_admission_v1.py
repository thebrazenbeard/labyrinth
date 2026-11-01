"""PRIVATE external-evidence consumer for cognitive TRAIN rights and exclusions.

This module *verifies* two externally administered Ed25519 signatures, full
private corpus binding, and a separately attested protected exclusion roster.
It never issues credentials, installs trust anchors, opens protected gold, or
authorizes execution. No corpus/roster digest or evidence content may be put
on a public issue, PR, chat bus, or generic training receipt: deterministic
low-entropy source rows make even their SHA-256 values answer oracles.

An external operator must install the trust-anchor file outside the source
checkout and retain all inputs/nonce database under PRIVATE access controls.
The source repository cannot establish who actually administers that file;
the physical admission authority must check that fact independently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import time
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .cognitive_rights_candidate_v1 import (
    PrivateStewardManifest,
    prepare_private_cognitive_steward_manifest,
)
from .cognitive_train_dataset_export import CognitiveTrainDataset


_TRUST_SCHEMA = "labyrinth.external-cognitive-trust-anchors.v1"
_STEWARD_SCHEMA = "labyrinth.external-rights-attestation.v1"
_EVALUATOR_SCHEMA = "labyrinth.external-protected-exclusion-attestation.v1"
_EXCLUSION_SCHEMA = "labyrinth.protected-semantic-exclusions.v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_NONCE = re.compile(r"[A-Za-z0-9_-]{16,128}\Z")
_REF = re.compile(r"private://[A-Za-z0-9._/-]{4,200}\Z")
_EVIDENCE_NAMES = frozenset({"source_authorship", "license", "consent", "privacy_review"})
_RIGHTS_REF_NAMES = {
    "source_authorship": "source_authorship_evidence_ref",
    "license": "license_evidence_ref",
    "consent": "consent_evidence_ref",
    "privacy_review": "privacy_review_evidence_ref",
}
_SOURCE_NAMES = frozenset({"repository", "producer_commit", "producer_file_sha256",
                          "dataset_sha256", "objective_id", "contract_sha256",
                          "cycle_index"})
_SENTINEL = object()


class ExternalAdmissionHeld(RuntimeError):
    """An external steward/evaluator proof was absent or failed verification."""


class PrivateExternalAdmissionVerification:
    """Opaque PRIVATE verification evidence; deliberately never a GPU grant.

    This class is not a dataclass and cannot be recursively serialized using
    asdict. The only ordinary public representations are constant HOLD text.
    """

    __slots__ = ("_verified",)

    def __init__(self, proof: object) -> None:
        if proof is not _SENTINEL:
            raise ExternalAdmissionHeld("external admission verification HOLD")
        self._verified = True

    def private_evidence_verified(self) -> bool:
        """For an authenticated PRIVATE operator channel only."""
        return self._verified

    @property
    def authorizes_training(self) -> bool:
        return False

    @property
    def authorizes_host_execution(self) -> bool:
        return False

    @property
    def authorizes_exam(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<PrivateExternalAdmissionVerification PRIVATE; GPU_HOLD>"


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(data: bytes, *, max_bytes: int = 65536) -> dict:
    if type(data) is not bytes or not 0 < len(data) <= max_bytes:
        raise ValueError("invalid signed JSON size")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict) or _canonical(value) != data:
        raise ValueError("signed JSON must be a canonical object")
    return value


def _exact(value: object, keys: frozenset[str] | set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("unexpected attestation fields")
    return value


def _hash(value: object) -> str:
    if type(value) is not str or not _HEX64.fullmatch(value):
        raise ValueError("invalid hash")
    return value


def _claim_time(claim: dict, now: int) -> None:
    issued = claim["issued_at"]
    expires = claim["expires_at"]
    if (type(issued) is not int or type(expires) is not int or
            not issued <= now + 60 or not now < expires or
            not issued < expires <= issued + 86400):
        raise ValueError("attestation outside accepted time interval")
    if type(claim["nonce"]) is not str or not _NONCE.fullmatch(claim["nonce"]):
        raise ValueError("invalid nonce")


def signing_bytes_for_external_claim(role: str, canonical_claim_bytes: bytes) -> bytes:
    """Public protocol bytes, **not** signing authority or a key generator."""
    if role not in ("steward", "evaluator") or type(canonical_claim_bytes) is not bytes:
        raise ValueError("invalid external signing scope")
    # Role and version separation prevent cross-purpose signature substitution.
    return b"VERA-LABYRINTH-PRIVATE-COGNITIVE-ADMISSION-V1/" + role.encode() + b"\0" + canonical_claim_bytes


def _load_external_anchors(path: Path) -> tuple[dict, dict]:
    """Load only pre-installed pins, never claim-supplied keys.

    The actual independent administration and filesystem ACL of this external
    path must also be authenticated by the host. A file outside Git alone is
    insufficient proof of independent administration.
    """
    if not isinstance(path, Path):
        raise ValueError("external trust path must be a Path")
    if path.is_symlink():
        raise ValueError("symbolic trust anchor rejected")
    resolved = path.resolve(strict=True)
    if resolved.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("source-repo trust anchor rejected")
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
        raise ValueError("insecure trust-anchor permissions")
    if info.st_size > 8192:
        raise ValueError("trust-anchor file too large")
    obj = _exact(_read_json(resolved.read_bytes(), max_bytes=8192),
                 {"schema", "steward", "evaluator"})
    if obj["schema"] != _TRUST_SCHEMA:
        raise ValueError("unsupported trust-anchor schema")
    steward = _exact(obj["steward"], {"id", "ed25519_public_key_hex"})
    evaluator = _exact(obj["evaluator"], {"id", "ed25519_public_key_hex"})
    for issuer in (steward, evaluator):
        if (type(issuer["id"]) is not str or not issuer["id"] or
                len(issuer["id"]) > 128):
            raise ValueError("invalid external issuer")
        key = issuer["ed25519_public_key_hex"]
        if type(key) is not str or len(key) != 64 or not _HEX64.fullmatch(key):
            raise ValueError("invalid Ed25519 trust pin")
    if (steward["id"] == evaluator["id"] or steward["ed25519_public_key_hex"] ==
            evaluator["ed25519_public_key_hex"]):
        raise ValueError("rights and protected evaluators must be independent")
    return steward, evaluator


def _verify_signature(claim_bytes: bytes, signature_hex: str,
                      role: str, pinned: dict) -> dict:
    claim = _read_json(claim_bytes)
    if claim.get("issuer_id") != pinned["id"]:
        raise ValueError("issuer is not externally pinned")
    if type(signature_hex) is not str or len(signature_hex) != 128:
        raise ValueError("missing signature")
    sig = bytes.fromhex(signature_hex)
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(pinned["ed25519_public_key_hex"])).verify(
        sig, signing_bytes_for_external_claim(role, claim_bytes)
    )
    return claim


def _verify_rights_evidence(rights: object, evidence: Mapping[str, bytes]) -> None:
    rights = _exact(rights, {"permission", "source_authorship_evidence_ref",
                            "license_identifier", "license_evidence_ref",
                            "consent_evidence_ref", "privacy_review_evidence_ref",
                            "evidence_sha256"})
    if rights["permission"] != "GRANTED_FOR_CANDIDATE":
        raise ValueError("steward has not approved candidate rights")
    if (type(rights["license_identifier"]) is not str or
            not 4 <= len(rights["license_identifier"]) <= 200):
        raise ValueError("missing license identifier")
    hashes = _exact(rights["evidence_sha256"], _EVIDENCE_NAMES)
    if not isinstance(evidence, Mapping) or set(evidence) != _EVIDENCE_NAMES:
        raise ValueError("missing private evidence exhibits")
    for name, ref_key in _RIGHTS_REF_NAMES.items():
        ref = rights[ref_key]
        if type(ref) is not str or not _REF.fullmatch(ref):
            raise ValueError("missing private evidence reference")
        content = evidence[name]
        if type(content) is not bytes or not 0 < len(content) <= 16_777_216:
            raise ValueError("missing/oversized private evidence bytes")
        if hashlib.sha256(content).hexdigest() != _hash(hashes[name]):
            raise ValueError("rights evidence hash mismatch")


def _reserve_nonces(path: Path, steward: dict, evaluator: dict) -> None:
    """Atomically consume BOTH signed nonces in a durable SQLite registry."""
    if not isinstance(path, Path) or str(path) in (":memory:", ""):
        raise ValueError("durable nonce database required")
    if path.is_symlink():
        raise ValueError("symlink nonce registry rejected")
    resolved = path.resolve(strict=False)
    if resolved.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("nonce database cannot live in public source tree")
    if not resolved.parent.is_dir():
        raise ValueError("nonce registry parent unavailable")
    parent_stat = resolved.parent.stat()
    if parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("nonce registry parent is not private")
    if resolved.exists():
        current = resolved.stat()
        if (not stat.S_ISREG(current.st_mode) or
                current.st_mode & (stat.S_IRWXG | stat.S_IRWXO)):
            raise ValueError("nonce registry permissions are not private")
    else:
        # Protect the data BEFORE SQLite opens the file, not after the first
        # write (which could briefly leak private replay metadata via umask).
        descriptor = os.open(str(resolved), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(descriptor)
    # Nonces are opaque and must remain private with the rights/custody inputs.
    connection = sqlite3.connect(str(resolved), timeout=10)
    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE IF NOT EXISTS external_cognitive_nonce_v1 ("
                           "issuer_id TEXT NOT NULL, nonce TEXT NOT NULL, "
                           "expires_at INTEGER NOT NULL, PRIMARY KEY (issuer_id, nonce))")
        connection.execute("BEGIN IMMEDIATE")
        for claim in (steward, evaluator):
            connection.execute("INSERT INTO external_cognitive_nonce_v1 "
                               "(issuer_id, nonce, expires_at) VALUES (?, ?, ?)",
                               (claim["issuer_id"], claim["nonce"], claim["expires_at"]))
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def verify_external_cognitive_admission(
    *,
    dataset: CognitiveTrainDataset,
    private_steward_manifest: PrivateStewardManifest,
    private_rights_evidence: Mapping[str, bytes],
    exclusion_semantic_sha256: tuple[str, ...],
    trust_anchor_path: Path,
    nonce_registry_path: Path,
    steward_claim_bytes: bytes,
    steward_signature_hex: str,
    evaluator_claim_bytes: bytes,
    evaluator_signature_hex: str,
    expected_source_commit: str,
    expected_source_file_sha256: str,
    now: int | None = None,
) -> PrivateExternalAdmissionVerification:
    """Verify private, independently pinned rights and evaluator attestations.

    All inputs, log lines, exceptions, attestations, private manifest and nonce
    DB must remain under authenticated PRIVATE custody. There is NO positive
    host/GPU grant here even after both signatures verify. Caller/host must
    independently authenticate trust-anchor administration, NAS route, lease,
    protected-gold custody and physical execution authority.

    `now` is for deterministic tests; production callers must use host time.
    """
    try:
        if (not isinstance(private_steward_manifest, PrivateStewardManifest) or
                not isinstance(dataset, CognitiveTrainDataset) or
                type(exclusion_semantic_sha256) is not tuple or
                type(expected_source_commit) is not str or
                not _HEX40.fullmatch(expected_source_commit)):
            raise ValueError("invalid private producer inputs")
        _hash(expected_source_file_sha256)
        if now is None:
            now = int(time.time())
        if type(now) is not int or now < 0:
            raise ValueError("invalid trusted clock")
        if len(exclusion_semantic_sha256) < 1 or len(set(exclusion_semantic_sha256)) != len(exclusion_semantic_sha256):
            raise ValueError("missing or duplicate protected exclusions")
        exclusions = sorted(_hash(value) for value in exclusion_semantic_sha256)
        protected_digest = hashlib.sha256(_canonical({
            "schema": _EXCLUSION_SCHEMA,
            "complete": True,
            "semantic_sha256": exclusions,
        })).hexdigest()
        # This function rechecks the original strict 15-row producer contract,
        # semantic duplicates, no train/holdout collision, and every private
        # row/source binding. The old R163 HOLD module remains untouched.
        recomputed = prepare_private_cognitive_steward_manifest(
            dataset,
            source_repo="thebrazenbeard/labyrinth",
            source_commit=expected_source_commit,
            source_file_sha256=expected_source_file_sha256,
            exclusion_semantic_sha256=exclusion_semantic_sha256,
            exclusion_roster_sha256=protected_digest,
        ).private_bytes()
        if private_steward_manifest.private_bytes() != recomputed:
            raise ValueError("private manifest mismatches actual corpus/roster")
        anchors_steward, anchors_evaluator = _load_external_anchors(trust_anchor_path)
        steward = _verify_signature(steward_claim_bytes, steward_signature_hex,
                                    "steward", anchors_steward)
        evaluator = _verify_signature(evaluator_claim_bytes, evaluator_signature_hex,
                                      "evaluator", anchors_evaluator)
        _exact(steward, {"schema", "issuer_id", "nonce", "issued_at", "expires_at",
                         "source", "private_manifest_sha256", "protected_roster_sha256",
                         "rights"})
        _exact(evaluator, {"schema", "issuer_id", "nonce", "issued_at", "expires_at",
                           "dataset_sha256", "protected_roster_sha256",
                           "protected_roster_count", "roster_status",
                           "cross_system_exposure_ledger"})
        if steward["schema"] != _STEWARD_SCHEMA or evaluator["schema"] != _EVALUATOR_SCHEMA:
            raise ValueError("unknown external attestation schemas")
        _claim_time(steward, now)
        _claim_time(evaluator, now)
        manifest_source = _read_json(recomputed.rstrip(b"\n"))["source"]
        source = _exact(steward["source"], _SOURCE_NAMES)
        if source != {key: manifest_source[key] for key in _SOURCE_NAMES}:
            raise ValueError("signed source does not bind actual corpus")
        if _hash(steward["private_manifest_sha256"]) != hashlib.sha256(recomputed).hexdigest():
            raise ValueError("signed private manifest does not match corpus")
        if (_hash(steward["protected_roster_sha256"]) != protected_digest or
                _hash(evaluator["protected_roster_sha256"]) != protected_digest or
                _hash(evaluator["dataset_sha256"]) != dataset.sha256):
            raise ValueError("evaluator and steward bind different source/roster")
        if (type(evaluator["protected_roster_count"]) is not int or
                evaluator["protected_roster_count"] != len(exclusions) or
                evaluator["roster_status"] != "SEALED_COMPLETE" or
                evaluator["cross_system_exposure_ledger"] != "REVIEWED_NO_TRAIN_EXPOSURE"):
            raise ValueError("protected exclusion completeness/ledger not attested")
        _verify_rights_evidence(steward["rights"], private_rights_evidence)
        # The nonce consumption occurs only after all private verification and
        # both signatures pass, atomically, for both distinct external issuers.
        _reserve_nonces(nonce_registry_path, steward, evaluator)
        return PrivateExternalAdmissionVerification(_SENTINEL)
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error,
            InvalidSignature, UnicodeDecodeError, OverflowError) as exc:
        raise ExternalAdmissionHeld("external cognitive admission HOLD") from None
