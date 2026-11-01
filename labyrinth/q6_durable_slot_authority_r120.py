"""SOURCE-ONLY Q6 operator-side, durable per-exposure authority prototype.

This is a standalone authority *service core*, not a trainer wrapper.  Only an
operator-controlled process may construct it, hold its signing key, open its
SQLite database or map an authenticated Unix peer to ``principal``.  It never
stores/reveals prompts or gold and never invokes a model.  The separately
operated private evaluator must refuse release until a committed BEGIN receipt
has been checked against its own pin.  No such evaluator or isolated host is
installed by this module.  A SQLite database alone is NOT antirollback: a
different administrator-owned monotonic witness is required.

One global append-only signed hash chain covers every campaign and every slot,
and a global semantic index refuses cross-run re-use. WAL + synchronous FULL,
BEGIN IMMEDIATE and unique slot/nonce constraints supply local atomicity.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Callable, Mapping, Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .q6_governed_campaign_contract import (
    Q6CampaignManifest, Q6CampaignReservationReceipt, _MANIFEST_DOMAIN,
)
from .q6_slot_cas_reference import Q6SlotAddress, _SLOT_DOMAIN
from .authority_unix_service_r22 import _require_private_parent


SCHEMA = "labyrinth.q6-durable-slot-authority.source-r120.v1"
_RECEIPT_DOMAIN = b"labyrinth.q6-durable-slot-receipt.source-r120.v1\0"
_HEAD_DOMAIN = b"labyrinth.q6-durable-slot-head.source-r120.v1\0"
_GENESIS_DOMAIN = b"labyrinth.q6-durable-slot-genesis.source-r120.v1\0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_HEX = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ROLES = frozenset(("operator", "trainer", "evaluator"))


class Q6DurableError(ValueError):
    """Malformed, forbidden or conflicting source-only custody request."""


class Q6DurableConflict(Q6DurableError):
    """An immutable event, campaign, slot or ledger pin disagrees."""


class Q6DurableOutcomeUnknown(Q6DurableError):
    """After a storage/ACK fault, inspect the read-only signed journal."""


def _canonical(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=True) + "\n").encode("ascii")


def _id(value: object, field: str) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise Q6DurableError(f"invalid {field}")
    return value


def _hash(value: object, field: str) -> str:
    if type(value) is not str or not _HEX.fullmatch(value):
        raise Q6DurableError(f"invalid {field}")
    return value


@dataclass(frozen=True)
class Q6DurableHead:
    commit_index: int
    head_sha256: str
    source_only_unqualified: bool = True


@dataclass(frozen=True)
class Q6DurableReceipt:
    schema: str
    event: str
    run_id: str
    slot_id: str
    manifest_sha256: str
    principal: str
    commit_index: int
    previous_head_sha256: str
    new_head_sha256: str
    result_sha256: str | None
    execution_sha256: str | None
    checkpoint_sha256: str | None
    completion_nonce: str | None
    signature_hex: str
    source_only_unqualified: bool = True
    model_execution_authenticated: bool = False
    evaluator_prompt_custody_verified: bool = False
    host_antirollback_verified: bool = False
    graduation_authorized: bool = False

    def signed_payload(self) -> bytes:
        if self.schema != SCHEMA or self.event not in ("PROVISION", "BEGIN", "COMPLETE"):
            raise Q6DurableError("unrecognized Q6 receipt")
        for key in ("run_id", "slot_id", "manifest_sha256", "principal",
                    "previous_head_sha256", "new_head_sha256"):
            (_hash if key.endswith("sha256") or key == "slot_id" else _id)(
                getattr(self, key), key)
        if type(self.commit_index) is not int or self.commit_index < 1:
            raise Q6DurableError("invalid receipt index")
        if (self.source_only_unqualified is not True
                or any(getattr(self, flag) is not False for flag in (
                    "model_execution_authenticated", "evaluator_prompt_custody_verified",
                    "host_antirollback_verified", "graduation_authorized"))):
            raise Q6DurableError("source receipt cannot claim host qualification")
        extras = (self.result_sha256, self.execution_sha256,
                  self.checkpoint_sha256, self.completion_nonce)
        if self.event != "COMPLETE" and extras != (None,) * 4:
            raise Q6DurableError("non-completion receipt contains completion")
        if self.event == "COMPLETE":
            for key in ("result_sha256", "execution_sha256", "checkpoint_sha256"):
                _hash(getattr(self, key), key)
            _id(self.completion_nonce, "completion nonce")
        return _RECEIPT_DOMAIN + _canonical({k: v for k, v in asdict(self).items()
                                             if k != "signature_hex"})


@dataclass(frozen=True)
class Q6DurableStatus:
    slot_id: str
    state: str
    begin_receipt: Q6DurableReceipt | None
    complete_receipt: Q6DurableReceipt | None
    source_only_unqualified: bool = True


class Q6IndependentMonotonicWitness(Protocol):
    """Operator-injected external high-water interface, NOT implemented here.

    A real implementation must independently authenticate and persist this
    value outside the model/trainer/service database rollback domain. Source
    mocks and a caller-passed Python object do not prove that separation.
    """

    def read_head(self) -> Q6DurableHead: ...
    def compare_and_advance(self, expected: Q6DurableHead,
                            new: Q6DurableHead) -> bool: ...


def verify_receipt(receipt: Q6DurableReceipt, pinned_ed25519_public_key: bytes) -> None:
    """Evaluator/client verification seam; this does not authenticate a host."""
    if type(receipt) is not Q6DurableReceipt or type(pinned_ed25519_public_key) is not bytes:
        raise Q6DurableError("typed receipt and pinned operator public key required")
    try:
        Ed25519PublicKey.from_public_bytes(pinned_ed25519_public_key).verify(
            bytes.fromhex(receipt.signature_hex), receipt.signed_payload())
    except (ValueError, TypeError, KeyError) as exc:
        raise Q6DurableConflict("invalid signed receipt") from exc
    except Exception as exc:
        raise Q6DurableConflict("receipt signature rejected") from exc


class Q6DurableSlotAuthority:
    """Operator-side database core. No trainer-accessible provisioning API on wire.

    ``principal`` is trusted ONLY if injected by an authenticated operator
    transport, never from a request JSON body.  Tests call the seam directly.
    ``create`` must be explicit; a missing/replaced DB on restart fails closed.
    """

    def __init__(self, *, database_path: Path, signing_seed: bytes,
                 campaign_public_key: bytes, principal_roles: Mapping[str, str],
                 create: bool = False,
                 external_witness: Q6IndependentMonotonicWitness | None = None,
                 allow_unwitnessed_source_simulation: bool = False,
                 after_commit_hook: Callable[[str], None] | None = None):
        if (not isinstance(database_path, Path) or not database_path.is_absolute()
                or not database_path.parent.is_dir() or database_path.is_symlink()):
            raise Q6DurableError("existing absolute private database parent required")
        _require_private_parent(database_path, label="Q6 operator database")
        parent = database_path.parent.stat()
        if (parent.st_uid != os.geteuid() or parent.st_mode & 0o022):
            raise Q6DurableError("operator database directory writable by another group/UID")
        if (type(signing_seed) is not bytes or len(signing_seed) != 32
                or type(campaign_public_key) is not bytes or len(campaign_public_key) != 32):
            raise Q6DurableError("operator seed and campaign public key must be 32 bytes")
        if (not isinstance(principal_roles, Mapping) or not principal_roles
                or not any(role == "operator" for role in principal_roles.values())):
            raise Q6DurableError("operator-controlled role roster required")
        roster = dict(principal_roles)
        if any(_id(k, "principal") != k or type(v) is not str or v not in _ROLES
               for k, v in roster.items()):
            raise Q6DurableError("invalid operator role roster")
        self._db_path = database_path
        self._signer = Ed25519PrivateKey.from_private_bytes(signing_seed)
        self.public_key = self._signer.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self._campaign_key = campaign_public_key
        self._roles = roster
        if type(allow_unwitnessed_source_simulation) is not bool:
            raise Q6DurableError("source simulation flag must be explicit boolean")
        if external_witness is not None and allow_unwitnessed_source_simulation:
            raise Q6DurableError("witnessed service cannot use unwitnessed simulation bypass")
        self._witness = external_witness
        self._unwitnessed_simulation = allow_unwitnessed_source_simulation
        self._config_sha = sha256(_canonical({
            "schema": SCHEMA, "public_key": self.public_key.hex(),
            "campaign_public_key": campaign_public_key.hex(),
            "roles": roster,
            "authority_mode": ("UNWITNESSED_SOURCE_SIMULATION" if
                               allow_unwitnessed_source_simulation else
                               "EXTERNAL_WITNESS_REQUIRED"),
        })).hexdigest()
        self._genesis = sha256(_GENESIS_DOMAIN + bytes.fromhex(self._config_sha)).hexdigest()
        self._after_commit_hook = after_commit_hook
        self._initialize(create=create)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self._db_path), isolation_level=None, timeout=10.0)
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _initialize(self, *, create: bool) -> None:
        if type(create) is not bool:
            raise Q6DurableError("explicit create flag required")
        exists = self._db_path.exists()
        if exists == create:
            raise Q6DurableConflict("create requires absent DB; reopen requires existing DB")
        if create:
            try:
                fd = os.open(self._db_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY |
                             getattr(os, "O_NOFOLLOW", 0), 0o600)
                os.close(fd)
            except OSError as exc:
                raise Q6DurableOutcomeUnknown("operator DB exclusive creation failed") from exc
        info = self._db_path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077 or info.st_nlink != 1):
            raise Q6DurableConflict("operator DB file ownership/mode/link unsafe")
        try:
            with closing(self._connect()) as db:
                db.execute("PRAGMA journal_mode=WAL")
                if create:
                    db.executescript("""
                    CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK (id=1),
                      config_sha TEXT NOT NULL, commit_index INTEGER NOT NULL,
                      head_sha TEXT NOT NULL);
                    CREATE TABLE campaigns (run_id TEXT PRIMARY KEY,
                      manifest_sha TEXT NOT NULL, manifest_json TEXT NOT NULL,
                      reservation_json TEXT NOT NULL);
                    CREATE TABLE semantics (semantic_sha TEXT PRIMARY KEY,
                      run_id TEXT NOT NULL REFERENCES campaigns(run_id));
                    CREATE TABLE slots (slot_id TEXT PRIMARY KEY,
                      run_id TEXT NOT NULL REFERENCES campaigns(run_id),
                      partition TEXT NOT NULL, case_id TEXT NOT NULL,
                      arm TEXT NOT NULL, phase TEXT NOT NULL, visit INTEGER NOT NULL,
                      prompt_sha TEXT NOT NULL, semantic_sha TEXT NOT NULL,
                      state TEXT NOT NULL CHECK(state IN ('RESERVED','BEGUN','COMPLETED')),
                      begin_index INTEGER UNIQUE, complete_index INTEGER UNIQUE);
                    CREATE TABLE journal (commit_index INTEGER PRIMARY KEY,
                      slot_id TEXT NOT NULL,
                      receipt_json TEXT NOT NULL);
                    CREATE TABLE nonces (completion_nonce TEXT PRIMARY KEY,
                      slot_id TEXT NOT NULL UNIQUE REFERENCES slots(slot_id));
                    """)
                    db.execute("INSERT INTO meta VALUES(1,?,?,?)",
                               (self._config_sha, 0, self._genesis))
                self._verify(db)
        except sqlite3.Error as exc:
            raise Q6DurableOutcomeUnknown("durable service storage unavailable") from exc

    def _verify_campaign(self, manifest_json: str, reservation_json: str,
                         run_id: str, manifest_sha: str) -> dict:
        try:
            source = json.loads(manifest_json)
            receipt = Q6CampaignReservationReceipt(**json.loads(reservation_json))
            if (receipt.run_id != run_id or receipt.manifest_sha256 != manifest_sha
                    or sha256(_MANIFEST_DOMAIN + _canonical(source)).hexdigest() != manifest_sha):
                raise Q6DurableConflict("manifest/receipt changed")
            Ed25519PublicKey.from_public_bytes(self._campaign_key).verify(
                bytes.fromhex(receipt.signature_hex), receipt.signed_payload())
            return source
        except Q6DurableError:
            raise
        except Exception as exc:
            raise Q6DurableOutcomeUnknown("stored signed campaign changed") from exc

    def _verify(self, db: sqlite3.Connection) -> Q6DurableHead:
        """Verify full signed prefix and materialized slot state; refuse tamper."""
        row = db.execute("SELECT config_sha,commit_index,head_sha FROM meta WHERE id=1").fetchone()
        if row is None or row[0] != self._config_sha:
            raise Q6DurableConflict("missing or substituted operator configuration")
        idx = 0
        head = self._genesis
        replay: dict[str, tuple[str, int | None, int | None]] = {}
        replay_nonces: dict[str, str] = {}
        provisioned: set[str] = set()
        for number, slot_id, stored in db.execute(
                "SELECT commit_index,slot_id,receipt_json FROM journal ORDER BY commit_index"):
            if number != idx + 1:
                raise Q6DurableOutcomeUnknown("signed event journal has a gap")
            try:
                receipt = Q6DurableReceipt(**json.loads(stored))
                verify_receipt(receipt, self.public_key)
            except (ValueError, TypeError, KeyError) as exc:
                raise Q6DurableOutcomeUnknown("signed event journal invalid") from exc
            if (receipt.commit_index != number or receipt.slot_id != slot_id
                    or receipt.previous_head_sha256 != head):
                raise Q6DurableOutcomeUnknown("signed journal order changed")
            payload = asdict(receipt)
            payload.pop("signature_hex")
            payload.pop("new_head_sha256")
            computed = sha256(_HEAD_DOMAIN + bytes.fromhex(head) + _canonical(payload)).hexdigest()
            if computed != receipt.new_head_sha256:
                raise Q6DurableOutcomeUnknown("signed head chain mismatch")
            prior = replay.get(slot_id)
            if receipt.event == "PROVISION":
                if receipt.run_id in provisioned:
                    raise Q6DurableOutcomeUnknown("replayed campaign provisioning")
                provisioned.add(receipt.run_id)
            elif receipt.run_id not in provisioned:
                raise Q6DurableOutcomeUnknown("slot event precedes campaign")
            elif receipt.event == "BEGIN" and prior is None:
                replay[slot_id] = ("BEGUN", number, None)
            elif receipt.event == "COMPLETE" and prior is not None and prior[0] == "BEGUN":
                if receipt.completion_nonce in replay_nonces:
                    raise Q6DurableOutcomeUnknown("reused completion nonce")
                replay_nonces[receipt.completion_nonce] = slot_id
                replay[slot_id] = ("COMPLETED", prior[1], number)
            else:
                raise Q6DurableOutcomeUnknown("illegal journal transition")
            idx, head = number, computed
        if row[1:] != (idx, head):
            raise Q6DurableOutcomeUnknown("persisted ledger head mismatch")
        expected_slots: dict[str, tuple] = {}
        expected_semantics: set[tuple[str, str]] = set()
        for run_id, manifest_sha, manifest_json, reservation_json in db.execute(
                "SELECT run_id,manifest_sha,manifest_json,reservation_json FROM campaigns"):
            manifest = self._verify_campaign(manifest_json, reservation_json, run_id, manifest_sha)
            cases = {case["case_id"]: case for case in manifest["cases"]}
            for case in manifest["cases"]:
                expected_semantics.add((case["arithmetic_semantic_sha256"], run_id))
            receipt = Q6CampaignReservationReceipt(**json.loads(reservation_json))
            for visit_def in manifest["planned_exposures"]:
                case = cases[visit_def["case_id"]]
                for visit in range(visit_def["planned_visits"]):
                    addr = (manifest_sha, receipt.reservation_id,
                            visit_def["partition"], visit_def["case_id"],
                            visit_def["arm"], visit_def["phase"], visit,
                            case["prompt_sha256"], case["arithmetic_semantic_sha256"])
                    slot_id = sha256(_SLOT_DOMAIN + _canonical(addr)).hexdigest()
                    expected_slots[slot_id] = (run_id, *addr[2:])
        if provisioned != set(row[0] for row in db.execute("SELECT run_id FROM campaigns")):
            raise Q6DurableOutcomeUnknown("campaign registry differs from signed provisions")
        stored_slots = list(db.execute(
            "SELECT slot_id,run_id,partition,case_id,arm,phase,visit,prompt_sha,"
            "semantic_sha,state,begin_index,complete_index FROM slots"))
        if len(stored_slots) != len(expected_slots):
            raise Q6DurableOutcomeUnknown("missing or extra preregistered Q6 slot")
        for slot_id, *values in stored_slots:
            if (expected_slots.get(slot_id) != tuple(values[:8])
                    or tuple(values[8:]) != replay.get(slot_id, ("RESERVED", None, None))):
                raise Q6DurableOutcomeUnknown("slot table changed outside signed journal")
        if set(db.execute("SELECT semantic_sha,run_id FROM semantics")) != expected_semantics:
            raise Q6DurableOutcomeUnknown("cross-run semantic index altered")
        if set(db.execute("SELECT completion_nonce,slot_id FROM nonces")) != set(replay_nonces.items()):
            raise Q6DurableOutcomeUnknown("completion nonce registry altered")
        return Q6DurableHead(idx, head)

    def _role(self, principal: str, allowed: frozenset[str]) -> None:
        _id(principal, "authenticated principal")
        if self._roles.get(principal) not in allowed:
            raise Q6DurableConflict("authenticated peer unauthorized")

    def _read(self, operation: Callable[[sqlite3.Connection], object]):
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN")
                self._verify(db)
                value = operation(db)
                db.commit()
                return value
        except sqlite3.Error as exc:
            raise Q6DurableOutcomeUnknown("durable read unavailable") from exc

    def _write(self, operation: Callable[[sqlite3.Connection], object]):
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                head = self._verify(db)
                self._require_witness_equal(head)
                value = operation(db, head)
                post = self._verify(db)
                db.commit()
            if post != head:
                self._advance_witness(head, post)
            return value
        except sqlite3.Error as exc:
            raise Q6DurableOutcomeUnknown("durable transaction outcome unknown") from exc

    def _require_witness_equal(self, local: Q6DurableHead) -> None:
        if self._witness is None:
            if self._unwitnessed_simulation:
                return
            raise Q6DurableOutcomeUnknown(
                "no independently administered monotonic witness: no BEGIN/reveal")
        try:
            observed = self._witness.read_head()
        except Exception as exc:
            raise Q6DurableOutcomeUnknown("external high-water witness unavailable") from exc
        if type(observed) is not Q6DurableHead or observed != local:
            raise Q6DurableOutcomeUnknown("external witness differs from DB; no reveal")

    def _advance_witness(self, prior: Q6DurableHead, new: Q6DurableHead) -> None:
        if self._witness is None:
            if self._unwitnessed_simulation:
                return
            raise Q6DurableOutcomeUnknown("no external witness CAS after local commit")
        try:
            acknowledged = self._witness.compare_and_advance(prior, new)
        except Exception as exc:
            raise Q6DurableOutcomeUnknown(
                "local commit may have succeeded, external witness ACK unknown") from exc
        if acknowledged is not True:
            raise Q6DurableOutcomeUnknown(
                "external monotonic witness refused CAS; no prompt release")
        self._require_witness_equal(new)

    def reconcile_external_anchor(self, *, principal: str) -> Q6DurableHead:
        """Operator-only recovery after local commit but lost witness ACK.

        This does not return any new BEGIN capability. The witness is allowed
        to advance by exactly ONE already signed, locally verified event whose
        predecessor equals its current independently authenticated head.
        Any rollback/ahead/divergent witness halts without taking a shortcut.
        """
        self._role(principal, frozenset(("operator",)))
        if self._witness is None:
            raise Q6DurableOutcomeUnknown("independent witness missing")
        local = self.read_head()  # complete journal validation
        try:
            external = self._witness.read_head()
        except Exception as exc:
            raise Q6DurableOutcomeUnknown("external high-water unavailable") from exc
        if type(external) is not Q6DurableHead:
            raise Q6DurableOutcomeUnknown("untyped witness head")
        if external == local:
            return local
        if local.commit_index != external.commit_index + 1:
            raise Q6DurableOutcomeUnknown("witness is divergent or more than one event behind")
        def evidence(db: sqlite3.Connection) -> bool:
            receipt = self._receipt(db, local.commit_index)
            return (receipt is not None
                    and receipt.previous_head_sha256 == external.head_sha256
                    and receipt.new_head_sha256 == local.head_sha256)
        if not self._read(evidence):
            raise Q6DurableOutcomeUnknown("local receipt does not extend witness")
        self._advance_witness(external, local)
        return local

    def provision_campaign(self, *, manifest: Q6CampaignManifest,
                           receipt: Q6CampaignReservationReceipt,
                           principal: str) -> None:
        """Operator-only source provisioning after independently signed campaign."""
        self._role(principal, frozenset(("operator",)))
        if type(manifest) is not Q6CampaignManifest or type(receipt) is not Q6CampaignReservationReceipt:
            raise Q6DurableError("typed externally reserved campaign required")
        manifest_json = manifest.payload().decode("ascii")
        receipt_json = _canonical(asdict(receipt)).decode("ascii")
        receipt = Q6CampaignReservationReceipt(**json.loads(receipt_json))
        manifest_sha = sha256(_MANIFEST_DOMAIN + manifest_json.encode("ascii")).hexdigest()
        self._verify_campaign(manifest_json, receipt_json, receipt.run_id, manifest_sha)
        source = json.loads(manifest_json)

        def insert(db: sqlite3.Connection, head: Q6DurableHead) -> None:
            existing = db.execute("SELECT 1 FROM campaigns WHERE run_id=?", (receipt.run_id,)).fetchone()
            if existing:
                raise Q6DurableConflict("run already provisioned; no silent replay")
            db.execute("INSERT INTO campaigns VALUES(?,?,?,?)",
                       (receipt.run_id, manifest_sha, manifest_json, receipt_json))
            cases = {case["case_id"]: case for case in source["cases"]}
            for case in source["cases"]:
                try:
                    db.execute("INSERT INTO semantics VALUES(?,?)",
                               (case["arithmetic_semantic_sha256"], receipt.run_id))
                except sqlite3.IntegrityError as exc:
                    raise Q6DurableConflict("semantic case already exposed in another run") from exc
            for plan in source["planned_exposures"]:
                case = cases[plan["case_id"]]
                for visit in range(plan["planned_visits"]):
                    slot_tuple = (manifest_sha, receipt.reservation_id,
                                  plan["partition"], plan["case_id"], plan["arm"],
                                  plan["phase"], visit, case["prompt_sha256"],
                                  case["arithmetic_semantic_sha256"])
                    slot_id = sha256(_SLOT_DOMAIN + _canonical(slot_tuple)).hexdigest()
                    db.execute("INSERT INTO slots VALUES(?,?,?,?,?,?,?,?,?,'RESERVED',NULL,NULL)",
                               (slot_id, receipt.run_id, *slot_tuple[2:]))
            marker = sha256(b"labyrinth.q6-durable-provision.source-r120.v1\0"
                            + receipt.signed_payload()).hexdigest()
            self._append(db, head, event="PROVISION", run_id=receipt.run_id,
                         slot_id=marker, manifest_sha=manifest_sha,
                         principal=principal)
        self._write(insert)

    def read_head(self) -> Q6DurableHead:
        return self._read(lambda db: self._verify(db))

    def assert_external_head_matches(self, externally_authenticated_head: Q6DurableHead | None) -> None:
        """Fail closed against a *supplied* witness; does not own the witness."""
        if (type(externally_authenticated_head) is not Q6DurableHead
                or self.read_head() != externally_authenticated_head):
            raise Q6DurableOutcomeUnknown("independent monotonic witness missing/stale/rollback")

    def address(self, *, run_id: str, partition: str, case_id: str,
                arm: str, phase: str, visit: int) -> Q6SlotAddress:
        _id(run_id, "run")
        if type(visit) is not int or visit < 0:
            raise Q6DurableError("invalid slot visit")
        def lookup(db: sqlite3.Connection) -> Q6SlotAddress:
            row = db.execute(
                "SELECT slot_id,prompt_sha,semantic_sha FROM slots WHERE run_id=? "
                "AND partition=? AND case_id=? AND arm=? AND phase=? AND visit=?",
                (run_id, partition, case_id, arm, phase, visit)).fetchone()
            if row is None:
                raise Q6DurableConflict("slot outside sealed campaign")
            return Q6SlotAddress(partition, case_id, arm, phase, visit,
                                 row[0], row[1], row[2])
        return self._read(lookup)

    def _receipt(self, db: sqlite3.Connection, index: int | None) -> Q6DurableReceipt | None:
        if index is None:
            return None
        row = db.execute("SELECT receipt_json FROM journal WHERE commit_index=?", (index,)).fetchone()
        if row is None:
            raise Q6DurableOutcomeUnknown("missing signed event receipt")
        receipt = Q6DurableReceipt(**json.loads(row[0]))
        verify_receipt(receipt, self.public_key)
        return receipt

    def reconcile(self, slot_id: str, *, principal: str) -> Q6DurableStatus:
        """Read-only recovery; RESERVED never proves a prompt was not leaked."""
        self._role(principal, frozenset(("operator", "trainer", "evaluator")))
        _hash(slot_id, "slot ID")
        def lookup(db: sqlite3.Connection) -> Q6DurableStatus:
            row = db.execute("SELECT state,begin_index,complete_index,partition "
                             "FROM slots WHERE slot_id=?", (slot_id,)).fetchone()
            if row is None:
                raise Q6DurableConflict("unknown slot")
            required = "trainer" if row[3].startswith("train") else "evaluator"
            if self._roles[principal] not in ("operator", required):
                raise Q6DurableConflict("peer cannot inspect other role's Q6 slot")
            return Q6DurableStatus(slot_id, row[0], self._receipt(db, row[1]),
                                   self._receipt(db, row[2]))
        return self._read(lookup)

    def _after_commit(self, event: str) -> None:
        if self._after_commit_hook is not None:
            try:
                self._after_commit_hook(event)
            except Exception as exc:
                raise Q6DurableOutcomeUnknown(
                    "commit ACK lost; read-only reconcile before any exposure") from exc

    def _append(self, db: sqlite3.Connection, head: Q6DurableHead, *,
                event: str, run_id: str, slot_id: str, manifest_sha: str,
                principal: str, result_sha256: str | None = None,
                execution_sha256: str | None = None,
                checkpoint_sha256: str | None = None,
                completion_nonce: str | None = None) -> Q6DurableReceipt:
        body = {
            "schema": SCHEMA, "event": event, "run_id": run_id,
            "slot_id": slot_id, "manifest_sha256": manifest_sha,
            "principal": principal, "commit_index": head.commit_index + 1,
            "previous_head_sha256": head.head_sha256,
            "result_sha256": result_sha256,
            "execution_sha256": execution_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "completion_nonce": completion_nonce,
            "source_only_unqualified": True,
            "model_execution_authenticated": False,
            "evaluator_prompt_custody_verified": False,
            "host_antirollback_verified": False,
            "graduation_authorized": False,
        }
        digest = sha256(_HEAD_DOMAIN + bytes.fromhex(head.head_sha256)
                        + _canonical(body)).hexdigest()
        body["new_head_sha256"] = digest
        unsigned = Q6DurableReceipt(**body, signature_hex="")
        sig = self._signer.sign(unsigned.signed_payload()).hex()
        signed = Q6DurableReceipt(**body, signature_hex=sig)
        db.execute("INSERT INTO journal VALUES(?,?,?)", (
            signed.commit_index, slot_id, _canonical(asdict(signed)).decode("ascii")))
        db.execute("UPDATE meta SET commit_index=?,head_sha=? WHERE id=1",
                   (signed.commit_index, digest))
        return signed

    def _transition(self, *, slot_id: str, expected_head: Q6DurableHead,
                    principal: str, event: str,
                    result_sha256: str | None = None,
                    execution_sha256: str | None = None,
                    checkpoint_sha256: str | None = None,
                    completion_nonce: str | None = None) -> Q6DurableReceipt:
        _hash(slot_id, "slot ID")
        if type(expected_head) is not Q6DurableHead:
            raise Q6DurableError("exact current signed ledger head required")
        def transact(db: sqlite3.Connection, head: Q6DurableHead) -> Q6DurableReceipt:
            row = db.execute("SELECT run_id,partition,state,begin_index,complete_index "
                             "FROM slots WHERE slot_id=?", (slot_id,)).fetchone()
            if row is None:
                raise Q6DurableConflict("unknown slot")
            run_id, partition, state, begin_index, complete_index = row
            required = "trainer" if partition.startswith("train") else "evaluator"
            self._role(principal, frozenset((required,)))
            if event == "COMPLETE" and state == "COMPLETED":
                prior = self._receipt(db, complete_index)
                if (prior is not None and
                        (prior.result_sha256, prior.execution_sha256,
                         prior.checkpoint_sha256, prior.completion_nonce) ==
                        (result_sha256, execution_sha256,
                         checkpoint_sha256, completion_nonce)):
                    return prior  # exact readback, NO new journal write
                raise Q6DurableConflict("completion is immutable")
            if head != expected_head:
                raise Q6DurableConflict("stale global head/fence")
            if state != ("RESERVED" if event == "BEGIN" else "BEGUN"):
                raise Q6DurableConflict("slot already exposed or not begun")
            manifest_sha = db.execute("SELECT manifest_sha FROM campaigns WHERE run_id=?",
                                     (run_id,)).fetchone()[0]
            receipt = self._append(
                db, head, event=event, run_id=run_id, slot_id=slot_id,
                manifest_sha=manifest_sha, principal=principal,
                result_sha256=result_sha256,
                execution_sha256=execution_sha256,
                checkpoint_sha256=checkpoint_sha256,
                completion_nonce=completion_nonce)
            if event == "BEGIN":
                db.execute("UPDATE slots SET state='BEGUN',begin_index=? WHERE slot_id=?",
                           (receipt.commit_index, slot_id))
            else:
                db.execute("INSERT INTO nonces VALUES(?,?)", (completion_nonce, slot_id))
                db.execute("UPDATE slots SET state='COMPLETED',complete_index=? WHERE slot_id=?",
                           (receipt.commit_index, slot_id))
            return receipt
        result = self._write(transact)
        if result.commit_index == expected_head.commit_index + 1:
            self._after_commit(event)
        return result

    def begin_exposure(self, *, slot_id: str, expected_head: Q6DurableHead,
                       principal: str) -> Q6DurableReceipt:
        """Durably consume a single slot BEFORE any external prompt access."""
        return self._transition(slot_id=slot_id, expected_head=expected_head,
                                principal=principal, event="BEGIN")

    def complete_exposure(self, *, slot_id: str, expected_head: Q6DurableHead,
                          principal: str, result_sha256: str,
                          execution_sha256: str, checkpoint_sha256: str,
                          completion_nonce: str) -> Q6DurableReceipt:
        for name, value in (("result", result_sha256), ("execution", execution_sha256),
                            ("checkpoint", checkpoint_sha256)):
            _hash(value, name)
        _id(completion_nonce, "completion nonce")
        return self._transition(slot_id=slot_id, expected_head=expected_head,
                                principal=principal, event="COMPLETE",
                                result_sha256=result_sha256,
                                execution_sha256=execution_sha256,
                                checkpoint_sha256=checkpoint_sha256,
                                completion_nonce=completion_nonce)
