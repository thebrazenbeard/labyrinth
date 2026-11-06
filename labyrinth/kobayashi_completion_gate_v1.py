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
import re
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
        if type(self.evidence_refs) is not tuple or len(self.evidence_refs) > 32:
            raise CompletionGateError("bounded evidence references required")
        for evidence in self.evidence_refs:
            _require_hash(evidence, "evidence reference")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise CompletionGateError("duplicate evidence references")
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
    """Host-held *in-memory* one-instance/one-verdict replay fence.

    A fresh ledger after restart does NOT preserve prior reservations. Durable
    authoritative replay prevention requires an external host-owned store.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: dict[tuple[str, str], SourceOnlyCompletionReceipt] = {}
        self._instance_owners: dict[str, str] = {}
        self._source_owners: dict[tuple[str, str], KobayashiMazeLifecycle] = {}
        self._lifecycle_owners: dict[int, tuple[tuple[str, str], KobayashiMazeLifecycle]] = {}

    def lookup(self, maze_id: str, scenario_instance_sha256: str) -> SourceOnlyCompletionReceipt | None:
        _require_maze_id(maze_id)
        _require_hash(scenario_instance_sha256, "scenario_instance_sha256")
        with self._lock:
            receipt = self._records.get((maze_id, scenario_instance_sha256))
            if receipt is not None:
                receipt.verify()
            return receipt

    def accept(self, receipt: SourceOnlyCompletionReceipt,
               lifecycle: KobayashiMazeLifecycle) -> SourceOnlyCompletionReceipt:
        """Commit in-process source acceptance; caller owns real authority."""
        with self._lock:
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
                return old  # exact duplicate delivery is an inert retry
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
        if type(evidence_refs) is not tuple or len(evidence_refs) > 32:
            raise CompletionGateError("bounded evidence references required")
        for ref in evidence_refs:
            _require_hash(ref, "evidence reference")
        if len(set(evidence_refs)) != len(evidence_refs):
            raise CompletionGateError("duplicate evidence references")

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
