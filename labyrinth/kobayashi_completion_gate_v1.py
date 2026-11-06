"""Source-only host acceptance gate for one Kobayashi maze obligation.

This module does NOT make a legacy maze's raw ``completed`` flag depend on a
Minotaur verdict. A trusted host must call ``commit`` at its *acceptance*
boundary before presenting the completed run as accepted. The caller, not a
language model, owns the registry and supplies the exact bound judge.

The resulting hashes are local consistency checks, not independent judging,
durable custody, neural training, or proof that a claimed invention works.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock

from .kobayashi_encounter_v1 import KobayashiMazeLifecycle
from .kobayashi_minotaur_v1 import KobayashiError, MinotaurDialogue


_SCHEMA = "labyrinth.kobayashi-completion-gate.v1"
_EVIDENCE = "SOURCE_ONLY_UNATTESTED"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MAZE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_LOCAL = frozenset({
    "LOCAL_LOSS", "LOCAL_STABILIZATION_SOURCE_ONLY", "REVIEW_REQUIRED", "UNRESOLVED",
})


class CompletionGateError(ValueError):
    """A source-only maze completion obligation is unfulfilled or inconsistent."""


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return sha256(b"labyrinth.kobayashi-completion-gate.v1\0" + payload).hexdigest()


def _require_hash(value: str, name: str) -> None:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise CompletionGateError(f"{name} must be a lowercase SHA-256 digest")


def _require_maze_id(value: str) -> None:
    if type(value) is not str or _MAZE_ID.fullmatch(value) is None:
        raise CompletionGateError("maze_id must be a bounded ASCII identifier")


def _require_evidence_refs(evidence_refs: tuple[str, ...]) -> None:
    """Validate host evidence metadata before any model/world side effects."""
    if type(evidence_refs) is not tuple or len(evidence_refs) > 32:
        raise CompletionGateError("bounded evidence references required")
    for ref in evidence_refs:
        _require_hash(ref, "evidence reference")
    if len(set(evidence_refs)) != len(evidence_refs):
        raise CompletionGateError("duplicate evidence references")


@dataclass(frozen=True, slots=True)
class SourceOnlyCompletionReceipt:
    """Host-gate acceptance of a source episode; NEVER an external attestation."""

    maze_id: str
    scenario_instance_sha256: str
    maze_world_sha256: str
    maze_state_sha256: str
    scenario_sha256: str
    scenario_world_sha256: str
    proposal_sha256: str
    action_sha256: str
    claim_sha256: str
    corpus_sha256: str
    dialogue_sha256: str
    review_sha256: str
    verdict_sha256: str
    verdict_status: str
    argument_count: int
    evidence_refs: tuple[str, ...]
    local_encounter_status: str
    lifecycle_receipt_sha256: str
    maze_completed_source: bool = True
    maze_failed_due_to_local_loss: bool = False
    host_acceptance: str = _EVIDENCE
    schema: str = _SCHEMA
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "receipt_sha256", self._expected_digest())

    def _expected_digest(self) -> str:
        body = {item.name: getattr(self, item.name)
                for item in fields(self) if item.name != "receipt_sha256"}
        return _digest(body)

    def _validate(self) -> None:
        _require_maze_id(self.maze_id)
        for name in (
            "scenario_instance_sha256", "maze_world_sha256", "maze_state_sha256",
            "scenario_sha256", "scenario_world_sha256", "proposal_sha256",
            "action_sha256", "claim_sha256", "corpus_sha256", "dialogue_sha256",
            "review_sha256", "verdict_sha256", "lifecycle_receipt_sha256",
        ):
            _require_hash(getattr(self, name), name)
        if type(self.argument_count) is not int or not 2 <= self.argument_count <= 5:
            raise CompletionGateError("source final requires two to five bound argument rounds")
        _require_evidence_refs(self.evidence_refs)
        if type(self.verdict_status) is not str or self.verdict_status not in {
            "REFUTED", "SURVIVES", "UNKNOWN", "NOVEL_CANDIDATE",
        }:
            raise CompletionGateError("invalid source verdict status")
        if self.local_encounter_status not in _LOCAL:
            raise CompletionGateError("invalid local encounter status")
        if (self.maze_completed_source is not True
                or self.maze_failed_due_to_local_loss is not False):
            raise CompletionGateError("local encounter loss cannot fail the full maze")
        if self.host_acceptance != _EVIDENCE or self.schema != _SCHEMA:
            raise CompletionGateError("source receipt cannot claim higher authority")

    def verify(self) -> None:
        self._validate()
        if self.receipt_sha256 != self._expected_digest():
            raise CompletionGateError("completion receipt seal mismatch")


class KobayashiCompletionLedger:
    """Host-held one-instance/one-verdict source acceptance fence.

    The default is in-memory only. An opt-in SQLite file preserves *completed*
    source receipts across restart, under the trust of the supplied host path.
    It DOES NOT reserve nonces before execution, prevent filesystem rollback,
    authenticate a hostile writer, or establish independent judge authority.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._lock = Lock()
        self._closed = False
        self._db: sqlite3.Connection | None = None
        self._records: dict[tuple[str, str], SourceOnlyCompletionReceipt] = {}
        self._instance_owners: dict[str, str] = {}
        self._source_owners: dict[tuple[str, str], KobayashiMazeLifecycle] = {}
        self._lifecycle_owners: dict[int, tuple[tuple[str, str], KobayashiMazeLifecycle]] = {}
        if storage_path is not None:
            try:
                path = Path(storage_path)
                if not str(path).strip() or str(path) == ":memory:" or "\x00" in str(path):
                    raise CompletionGateError("concrete SQLite file path required")
                if path.is_symlink() or path.parent.is_symlink():
                    raise CompletionGateError("SQLite database symlink refused")
                if not path.parent.is_dir() or (path.exists() and not path.is_file()):
                    raise CompletionGateError("SQLite parent must exist and database must be a file")
                # Preflight does not close symlink races or protect against a
                # writable ancestor. Host must independently own the path.
                db = sqlite3.connect(str(path), timeout=2.0, isolation_level=None,
                                     check_same_thread=False)
                try:
                    db.execute("PRAGMA busy_timeout=2000")
                    db.execute("PRAGMA journal_mode=DELETE")
                    db.execute("PRAGMA synchronous=FULL")
                    db.execute("""CREATE TABLE IF NOT EXISTS accepted_receipts (
                        scenario_instance_sha256 TEXT PRIMARY KEY,
                        maze_id TEXT NOT NULL,
                        receipt_json TEXT NOT NULL,
                        receipt_sha256 TEXT NOT NULL
                    )""")
                except BaseException:
                    db.close()
                    raise
                self._db = db
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                if isinstance(exc, CompletionGateError):
                    raise
                raise CompletionGateError("SQLite acceptance ledger could not open") from exc

    def _require_open(self) -> None:
        if self._closed:
            raise CompletionGateError("completion ledger closed")

    def close(self) -> None:
        """Close optional persistence and reject further access (including memory)."""
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._db is not None:
                    self._db.close()

    def _read_sqlite(self, scenario_instance_sha256: str) -> SourceOnlyCompletionReceipt | None:
        assert self._db is not None
        row = self._db.execute(
            "SELECT maze_id, receipt_json, receipt_sha256 FROM accepted_receipts "
            "WHERE scenario_instance_sha256=?", (scenario_instance_sha256,)).fetchone()
        if row is None:
            return None
        try:
            # Reject ambiguous duplicate JSON keys; the stored seal then
            # checks accidental corruption, not adversarial DB rewriting.
            def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
                result: dict[str, object] = {}
                for name, value in pairs:
                    if name in result:
                        raise CompletionGateError("duplicate persisted receipt field")
                    result[name] = value
                return result

            body = json.loads(row[1], object_pairs_hook=unique_pairs)
            if type(body) is not dict:
                raise CompletionGateError("invalid persisted receipt document")
            body["evidence_refs"] = tuple(body["evidence_refs"])
            restored = SourceOnlyCompletionReceipt(**body)
            restored.verify()
            if (restored.maze_id != row[0]
                    or restored.scenario_instance_sha256 != scenario_instance_sha256
                    or restored.receipt_sha256 != row[2]):
                raise CompletionGateError("persisted completion receipt seal mismatch")
            return restored
        except (ValueError, KeyError, TypeError) as exc:
            raise CompletionGateError("invalid or tampered persisted completion receipt") from exc

    def _sqlite_commit(self) -> None:
        """Explicit commit seam permits deterministic precommit failure testing."""
        assert self._db is not None
        self._db.execute("COMMIT")

    def lookup(self, maze_id: str, scenario_instance_sha256: str) -> SourceOnlyCompletionReceipt | None:
        _require_maze_id(maze_id)
        _require_hash(scenario_instance_sha256, "scenario_instance_sha256")
        with self._lock:
            self._require_open()
            if self._db is not None:
                try:
                    stored = self._read_sqlite(scenario_instance_sha256)
                except sqlite3.Error as exc:
                    raise CompletionGateError("SQLite acceptance lookup failed") from exc
                if stored is not None and stored.maze_id != maze_id:
                    return None
                return stored
            receipt = self._records.get((maze_id, scenario_instance_sha256))
            if receipt is not None:
                receipt.verify()
            return receipt

    def require_uncommitted(self, maze_id: str, scenario_instance_sha256: str) -> None:
        """Fail early for known prior acceptance; NOT a nonce reservation.

        A competing run can still commit after this read. Hosts with external
        side effects need a separate atomic PENDING pre-execution protocol.
        """
        _require_maze_id(maze_id)
        _require_hash(scenario_instance_sha256, "scenario_instance_sha256")
        with self._lock:
            self._require_open()
            if self._db is not None:
                try:
                    prior = self._read_sqlite(scenario_instance_sha256)
                except sqlite3.Error as exc:
                    raise CompletionGateError("SQLite acceptance preflight failed") from exc
                if prior is not None:
                    raise CompletionGateError("scenario instance already committed; recover via lookup")
            elif scenario_instance_sha256 in self._instance_owners:
                raise CompletionGateError("scenario instance already committed; recover via lookup")

    def accept(self, receipt: SourceOnlyCompletionReceipt,
               lifecycle: KobayashiMazeLifecycle) -> SourceOnlyCompletionReceipt:
        """Commit in-process source acceptance; caller owns real authority."""
        with self._lock:
            self._require_open()
            return self._accept_under_lock(receipt, lifecycle)

    def _accept_under_lock(self, receipt: SourceOnlyCompletionReceipt,
                           lifecycle: KobayashiMazeLifecycle) -> SourceOnlyCompletionReceipt:
        if type(receipt) is not SourceOnlyCompletionReceipt:
            raise CompletionGateError("typed source completion receipt required")
        if type(lifecycle) is not KobayashiMazeLifecycle:
            raise CompletionGateError("exact source lifecycle required")
        receipt.verify()
        key = (receipt.maze_id, receipt.scenario_instance_sha256)
        old = self._records.get(key)
        if old is not None:
            old.verify()
            if old == receipt and self._source_owners[key] is lifecycle:
                if self._db is None:
                    return old  # exact in-memory duplicate is an inert retry
                # Persistent mode must also recheck the file, not trust a
                # cached receipt after possible external alteration.
            else:
                raise CompletionGateError("conflicting or replayed completion for committed maze instance")
        owner = self._instance_owners.get(receipt.scenario_instance_sha256)
        if owner is not None and owner != receipt.maze_id:
            raise CompletionGateError("scenario instance replayed across mazes")
        # The same deterministic response to the same exam may legitimately
        # occur in multiple runs. Verdict content is NOT a unique event ID.
        # Reject reusing one *live lifecycle object*, not its verdict hash.
        prior = self._lifecycle_owners.get(id(lifecycle))
        if prior is not None and prior[1] is lifecycle and prior[0] != key:
            raise CompletionGateError("live maze lifecycle replayed across instances")
        if self._db is not None:
            try:
                # BEGIN IMMEDIATE serializes acceptance between independent
                # SQLite connections/processes sharing this trusted path.
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    existing = self._read_sqlite(receipt.scenario_instance_sha256)
                    if existing is not None:
                        if existing.maze_id != receipt.maze_id:
                            raise CompletionGateError("scenario instance replayed across mazes")
                        if existing != receipt:
                            raise CompletionGateError("conflicting or replayed completion for committed maze instance")
                        if old is None or self._source_owners[key] is not lifecycle:
                            # On restart, only lookup may recover a historical
                            # receipt. A newly run episode (even with exactly
                            # equal deterministic bytes) is NOT a retry.
                            raise CompletionGateError("committed instance replay; recover via lookup")
                        self._sqlite_commit()
                        # Exact persisted duplicate = delivery retry, NOT a
                        # second fresh episode or independent verdict.
                        verified = self._read_sqlite(receipt.scenario_instance_sha256)
                        if verified != receipt:
                            raise CompletionGateError("persisted receipt post-commit readback mismatch")
                        return verified
                    body = {item.name: getattr(receipt, item.name)
                            for item in fields(receipt) if item.name != "receipt_sha256"}
                    payload = json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False)
                    self._db.execute("INSERT INTO accepted_receipts "
                                     "(scenario_instance_sha256, maze_id, receipt_json, receipt_sha256) "
                                     "VALUES (?, ?, ?, ?)",
                                     (receipt.scenario_instance_sha256, receipt.maze_id,
                                      payload, receipt.receipt_sha256))
                    self._sqlite_commit()
                    verified = self._read_sqlite(receipt.scenario_instance_sha256)
                    if verified != receipt:
                        raise CompletionGateError("persisted receipt post-commit readback mismatch")
                except BaseException:
                    if self._db.in_transaction:
                        self._db.execute("ROLLBACK")
                    raise
            except sqlite3.Error as exc:
                raise CompletionGateError("SQLite acceptance transaction failed") from exc
        self._records[key] = receipt
        self._instance_owners[receipt.scenario_instance_sha256] = receipt.maze_id
        self._source_owners[key] = lifecycle
        self._lifecycle_owners[id(lifecycle)] = (key, lifecycle)
        return receipt


class KobayashiCompletionGate:
    """One pre-registered, strict-wrapped maze's source completion obligation.

    The gate binds the actual lifecycle object BEFORE its proposal. It reads
    the exact host-bound Minotaur dialogue; a model's assertion of a verdict
    or a free-floating JSON receipt can never substitute for it.
    """

    def __init__(self, ledger: KobayashiCompletionLedger, maze_id: str,
                 scenario_instance_sha256: str,
                 lifecycle: KobayashiMazeLifecycle) -> None:
        if type(ledger) is not KobayashiCompletionLedger:
            raise CompletionGateError("exact host-owned completion ledger required")
        if type(lifecycle) is not KobayashiMazeLifecycle:
            raise CompletionGateError("exact Kobayashi lifecycle required")
        _require_maze_id(maze_id)
        _require_hash(scenario_instance_sha256, "scenario_instance_sha256")
        opening = lifecycle.snapshot()
        if opening["minotaur_appeal_required"] is not True:
            raise CompletionGateError("strict final Minotaur appeal is required")
        if opening["maze_completed"] or opening["mandatory_encounter_attempted"]:
            raise CompletionGateError("bind completion obligation before a proposal or completion")
        self._ledger = ledger
        self._maze_id = maze_id
        self._instance_sha256 = scenario_instance_sha256
        self._lifecycle = lifecycle
        self._maze_world_sha256 = opening["maze_world_sha256"]
        self._scenario_sha256 = lifecycle._challenge.observe().sha256
        self._scenario_world_sha256 = lifecycle._challenge.graph_sha256

    def commit(self, dialogue: MinotaurDialogue, *,
               evidence_refs: tuple[str, ...] = ()) -> SourceOnlyCompletionReceipt:
        """Accept completed source maze only after its bound final appeal.

        The caller must withhold host acceptance until this returns. Even on
        success, it certifies neither model reasoning nor external evidence.
        """
        if type(dialogue) is not MinotaurDialogue or self._lifecycle._minotaur is not dialogue:
            raise CompletionGateError("exact bound host Minotaur dialogue required")
        _require_evidence_refs(evidence_refs)

        verdict = dialogue.final
        if verdict is None or self._lifecycle._final_verdict is not verdict:
            raise CompletionGateError("bound Minotaur final verdict required")
        try:
            verdict.verify()  # snapshot status alone does not verify this seal
            claim = dialogue.claim  # checks judge's bound source context
        except KobayashiError as exc:
            raise CompletionGateError("Minotaur verdict seal or claim invalid") from exc

        snapshot = self._lifecycle.snapshot()
        if snapshot["maze_completed"] is not True:
            raise CompletionGateError("maze must be completed before host acceptance")
        if (snapshot["maze_world_sha256"] != self._maze_world_sha256
                or snapshot["mandatory_encounter_attempted"] is not True
                or snapshot["minotaur_appeal_required"] is not True
                or snapshot["appeal_final"] is not True):
            raise CompletionGateError("mandatory strict encounter/final appeal missing")
        if snapshot["maze_failed_due_to_local_loss"] is not False:
            raise CompletionGateError("local Kobayashi loss is not maze failure")
        if snapshot["training_performed"] is not False or snapshot["weights_changed"] is not False:
            raise CompletionGateError("source completion cannot assert training")
        if snapshot["encounter_phase"] != "ENCOUNTER_ATTEMPTED":
            raise CompletionGateError("encounter was not attempted")
        if (claim.scenario_sha256 != self._scenario_sha256
                or claim.world_sha256 != self._scenario_world_sha256
                or claim.claim_text != snapshot["encounter_proposal_text"]
                or claim.action_sha256 != self._lifecycle._result["action_sha256"]):
            raise CompletionGateError("proposal claim/scenario instance mismatch")
        if (verdict.scenario_sha256 != claim.scenario_sha256
                or verdict.world_sha256 != claim.world_sha256
                or verdict.action_sha256 != claim.action_sha256
                or verdict.claim_sha256 != claim.claim_sha256
                or verdict.dialogue_sha256 != dialogue.dialogue_sha256
                or snapshot["minotaur_verdict_status"] != verdict.status):
            raise CompletionGateError("final Minotaur verdict not bound to exact proposal/dialogue")

        args = dialogue.arguments
        transcript = snapshot["minotaur_arguments"]
        if not (2 <= len(args) <= 5
                and len(transcript) == len(args)
                and snapshot["minotaur_argument_rounds"] == len(args)):
            raise CompletionGateError("bounded Minotaur transcript mismatch")
        all_evidence: set[str] = set()
        for number, (argument, row) in enumerate(zip(args, transcript), 1):
            try:
                argument.verify()
            except KobayashiError as exc:
                raise CompletionGateError("invalid Minotaur argument seal") from exc
            if (row["round"] != number
                    or row["argument_text"] != argument.text
                    or row["evidence_hashes"] != argument.evidence_sha256
                    or row["minotaur_attached"] is not True
                    or row["public_minotaur_reply"] is None
                    or row["public_minotaur_reply"]["argument_sha256"] != argument.argument_sha256):
                raise CompletionGateError("host transcript does not match lifecycle evidence")
            all_evidence.update(argument.evidence_sha256)
        if not all_evidence.issubset(evidence_refs):
            raise CompletionGateError("argument evidence omitted from completion references")

        receipt = SourceOnlyCompletionReceipt(
            maze_id=self._maze_id,
            scenario_instance_sha256=self._instance_sha256,
            maze_world_sha256=self._maze_world_sha256,
            maze_state_sha256=snapshot["maze_state_sha256"],
            scenario_sha256=claim.scenario_sha256,
            scenario_world_sha256=claim.world_sha256,
            proposal_sha256=snapshot["encounter_proposal_sha256"],
            action_sha256=claim.action_sha256,
            claim_sha256=claim.claim_sha256,
            corpus_sha256=verdict.corpus_sha256,
            dialogue_sha256=verdict.dialogue_sha256,
            review_sha256=verdict.review_sha256,
            verdict_sha256=verdict.verdict_sha256,
            verdict_status=verdict.status,
            argument_count=len(args),
            evidence_refs=evidence_refs,
            local_encounter_status=snapshot["local_encounter_status"],
            lifecycle_receipt_sha256=snapshot["receipt_sha256"],
        )
        return self._ledger.accept(receipt, self._lifecycle)


__all__ = [
    "CompletionGateError", "KobayashiCompletionLedger",
    "KobayashiCompletionGate", "SourceOnlyCompletionReceipt",
]
