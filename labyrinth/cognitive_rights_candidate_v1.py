"""Strict private steward candidate plus data-independent public HOLD status.

Even a prompt, semantic, source-contract, or full-dataset digest can become a
label oracle when a public deterministic producer has only A/B/C choices.
Only the constant public HOLD receipt may leave private operator custody.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .approved_cognitive_empathy_train_r90 import (
    MAX_COGNITIVE_ROOMS,
    semantic_sha256_from_prompt,
)
from .cognitive_train_dataset_export import (
    CognitiveTrainDataset,
    MAX_DATASET_BYTES,
    SCHEMA_ID as TRAIN_ROW_SCHEMA,
)


SCHEMA_ID = "labyrinth.cognitive-rights-public-status.v1"
PRIVATE_STEWARD_SCHEMA_ID = "labyrinth.cognitive-rights-private-steward.v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_ROW_FIELDS = frozenset({
    "schema", "split", "provider_id", "training_rubric_id", "objective_id",
    "objective_contract_sha256", "cycle_index", "room_id", "case_id",
    "prompt", "prompt_sha256", "semantic_sha256",
    "source_binding_sha256", "target_choice",
})


class CandidateAdmissionUnavailable(RuntimeError):
    """No independently verified rights/evaluator/host issuance exists here."""


@dataclass(frozen=True, slots=True)
class CognitiveRightsCandidate:
    """Entirely public, constant HOLD: safe even under dataclasses.asdict."""

    manifest_bytes: bytes
    sha256: str
    case_count: int

    @property
    def authorizes_training(self) -> bool:
        return False

    @property
    def authorizes_host_execution(self) -> bool:
        return False

    @property
    def authorizes_exam(self) -> bool:
        return False


class PrivateStewardManifest:
    """Opaque answer-oracle material for explicit, trusted private custody.

    This is deliberately NOT a field on the public candidate and not a
    dataclass, so recursive public serialization cannot accidentally embed
    the sensitive manifest. Python cannot enforce operator channel secrecy.
    """

    __slots__ = ("_private_bytes",)

    def __init__(self, private_bytes: bytes) -> None:
        if type(private_bytes) is not bytes:
            raise TypeError("private steward manifest requires bytes")
        self._private_bytes = private_bytes

    def private_bytes(self) -> bytes:
        """Export ONLY over independently authenticated private steward route."""
        return self._private_bytes

    def __repr__(self) -> str:
        return "<PrivateStewardManifest PRIVATE>"


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("utf-8")


# This is deliberately byte-identical across all subjects, seeds, answers,
# source commits, exclusion rosters, and runtime environments. Any producer-
# dependent field here would permit public source enumeration attacks.
_PUBLIC_HOLD_BYTES = _canonical_bytes({
    "schema": SCHEMA_ID,
    "status": "UNVERIFIED_REJECTED",
    "authorizes_training": False,
    "authorizes_host_execution": False,
}) + b"\n"


def _prepare_private_steward_bytes(
    dataset: CognitiveTrainDataset,
    *,
    source_repo: str,
    source_commit: str,
    source_file_sha256: str,
    exclusion_semantic_sha256: tuple[str, ...] = (),
    exclusion_roster_sha256: str | None = None,
) -> bytes:
    """Validate full private TRAIN data and prepare private steward bytes.

    A roster commitment is merely a digest supplied by a caller. This method
    can check collisions against explicitly supplied hashes, but it cannot
    authenticate the roster's completeness or custodian. That remains a HOLD.
    """
    if not isinstance(dataset, CognitiveTrainDataset):
        raise ValueError("expected canonical cognitive TRAIN export")
    if type(source_repo) is not str or source_repo != "thebrazenbeard/labyrinth":
        raise ValueError("unsupported source repository")
    if not isinstance(source_commit, str) or not _HEX40.fullmatch(source_commit):
        raise ValueError("invalid producer commit")
    _digest(source_file_sha256, "producer source SHA-256")
    if exclusion_roster_sha256 is not None:
        _digest(exclusion_roster_sha256, "protected exclusion roster SHA-256")
    if not isinstance(exclusion_semantic_sha256, tuple):
        raise ValueError("exclusions must be immutable tuple")
    exclusions = tuple(_digest(x, "excluded semantic SHA-256")
                       for x in exclusion_semantic_sha256)
    if len(set(exclusions)) != len(exclusions):
        raise ValueError("duplicate excluded semantic digests")
    if exclusions and exclusion_roster_sha256 is None:
        raise ValueError("exclusions require a roster commitment")
    if (not isinstance(dataset.jsonl_bytes, bytes) or
            len(dataset.jsonl_bytes) > MAX_DATASET_BYTES or
            not dataset.jsonl_bytes.endswith(b"\n")):
        raise ValueError("invalid canonical TRAIN bytes")
    if hashlib.sha256(dataset.jsonl_bytes).hexdigest() != _digest(
        dataset.sha256, "dataset SHA-256"
    ):
        raise ValueError("dataset hash mismatch")
    _digest(dataset.contract_fingerprint, "objective contract SHA-256")
    if not isinstance(dataset.objective_id, str) or not dataset.objective_id:
        raise ValueError("missing objective")
    if not isinstance(dataset.cycle_index, int) or dataset.cycle_index < 0:
        raise ValueError("invalid cycle")

    source_rows = dataset.jsonl_bytes.splitlines(keepends=True)
    if (type(dataset.case_count) is not int or
            dataset.case_count != MAX_COGNITIVE_ROOMS or
            len(source_rows) != MAX_COGNITIVE_ROOMS):
        raise ValueError("canonical TRAIN requires the full 15-row roster")
    summaries = []
    room_ids: set[str] = set()
    case_ids: set[str] = set()
    prompts: set[str] = set()
    semantics: set[str] = set()
    bindings: set[str] = set()
    for line in source_rows:
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("invalid TRAIN JSONL") from exc
        if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
            raise ValueError("unexpected TRAIN row fields")
        if _canonical_bytes(row) + b"\n" != line:
            raise ValueError("noncanonical TRAIN JSONL")
        if (row["schema"] != TRAIN_ROW_SCHEMA or row["split"] != "train" or
                row["objective_id"] != dataset.objective_id or
                row["objective_contract_sha256"] != dataset.contract_fingerprint or
                row["cycle_index"] != dataset.cycle_index or
                row["provider_id"] != dataset.provider_id):
            raise ValueError("TRAIN origin mismatch")
        for name in ("room_id", "case_id", "prompt", "provider_id",
                     "training_rubric_id"):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError("missing or invalid TRAIN identity")
        if row["target_choice"] not in ("A", "B", "C"):
            raise ValueError("invalid TRAIN target")
        if row["prompt_sha256"] != hashlib.sha256(row["prompt"].encode()).hexdigest():
            raise ValueError("TRAIN prompt hash mismatch")
        semantic = _digest(row["semantic_sha256"], "TRAIN semantic SHA-256")
        if semantic_sha256_from_prompt(row["prompt"]) != semantic:
            raise ValueError("TRAIN semantic digest does not bind public prompt")
        binding = _digest(row["source_binding_sha256"], "TRAIN source binding SHA-256")
        if semantic in exclusions:
            raise ValueError("TRAIN semantic overlaps supplied protected exclusions")
        if (row["room_id"] in room_ids or row["case_id"] in case_ids or
                row["prompt_sha256"] in prompts or semantic in semantics or
                binding in bindings):
            raise ValueError("duplicate TRAIN room, identity, prompt, binding, or semantic case")
        room_ids.add(row["room_id"])
        case_ids.add(row["case_id"])
        prompts.add(row["prompt_sha256"])
        semantics.add(semantic)
        bindings.add(binding)
        summaries.append({
            # ALL row/dataset/source commitments remain PRIVATE. Even hashes
            # constructed solely from prompts can be seed-enumeration oracles.
            "case_id_sha256": hashlib.sha256(row["case_id"].encode()).hexdigest(),
            "prompt_sha256": row["prompt_sha256"],
            "semantic_sha256": semantic,
            "source_binding_sha256": binding,
            "row_sha256": hashlib.sha256(line).hexdigest(),
        })

    # This full hash inventory is an answer oracle and belongs solely in the
    # authenticated PRIVATE steward channel. Public data must be constant.
    private_manifest = {
        "schema": PRIVATE_STEWARD_SCHEMA_ID,
        "source": {
            "repository": source_repo,
            "producer_commit": source_commit,
            "producer_file_sha256": source_file_sha256,
            "authorship_verified": False,
            "dataset_sha256": dataset.sha256,
            "objective_id": dataset.objective_id,
            "contract_sha256": dataset.contract_fingerprint,
            "cycle_index": dataset.cycle_index,
        },
        "splits": {"train": {"count": len(summaries), "rows": summaries}, "dev": None},
        "rights": {
            "permission": "NOT_ESTABLISHED",
            "source_authorship": "UNVERIFIED",
            "license_review": "NOT_ADMITTED",
            "consent_review": "NOT_ASSESSED",
            "privacy_review": "NOT_ASSESSED",
            "independent_steward_signature": None,
        },
        "protected_evaluation": {
            "roster_sha256": exclusion_roster_sha256,
            "supplied_exclusion_count": len(exclusions),
            "custodian_authenticated": False,
            "cross_system_exposure_ledger_verified": False,
        },
        "admission": {"status": "UNVERIFIED_REJECTED", "authorizes_training": False,
                      "authorizes_host_execution": False},
    }
    return _canonical_bytes(private_manifest) + b"\n"


def prepare_cognitive_rights_candidate(
    dataset: CognitiveTrainDataset,
    *,
    source_repo: str,
    source_commit: str,
    source_file_sha256: str,
    exclusion_semantic_sha256: tuple[str, ...] = (),
    exclusion_roster_sha256: str | None = None,
) -> CognitiveRightsCandidate:
    """Verify the full 15-row private data, then return public status ONLY.

    The returned object has zero private fields. Even recursive dataclass
    serialization exposes only constant, data-independent status values.
    """
    _prepare_private_steward_bytes(
        dataset, source_repo=source_repo, source_commit=source_commit,
        source_file_sha256=source_file_sha256,
        exclusion_semantic_sha256=exclusion_semantic_sha256,
        exclusion_roster_sha256=exclusion_roster_sha256,
    )
    return CognitiveRightsCandidate(
        manifest_bytes=_PUBLIC_HOLD_BYTES,
        sha256=hashlib.sha256(_PUBLIC_HOLD_BYTES).hexdigest(),
        case_count=MAX_COGNITIVE_ROOMS,
    )


def prepare_private_cognitive_steward_manifest(
    dataset: CognitiveTrainDataset,
    *,
    source_repo: str,
    source_commit: str,
    source_file_sha256: str,
    exclusion_semantic_sha256: tuple[str, ...] = (),
    exclusion_roster_sha256: str | None = None,
) -> PrivateStewardManifest:
    """Opt-in private output, entirely separate from the public candidate.

    This method returns a different type whose bytes must be kept private;
    it does NOT acquire a steward signature or authorize training.
    """
    return PrivateStewardManifest(_prepare_private_steward_bytes(
        dataset, source_repo=source_repo, source_commit=source_commit,
        source_file_sha256=source_file_sha256,
        exclusion_semantic_sha256=exclusion_semantic_sha256,
        exclusion_roster_sha256=exclusion_roster_sha256,
    ))


def require_cognitive_rights_admission(
    candidate: CognitiveRightsCandidate,
    *,
    steward_signature: str | None = None,
    evaluator_signature: str | None = None,
) -> None:
    """Reject locally supplied claims pending authenticated external issuers.

    The signature-shaped strings are intentionally ignored: this library has
    no trusted steward key or independent evaluator roster verification route.
    A positive decision belongs to a separately installed host/custodian.
    """
    if not isinstance(candidate, CognitiveRightsCandidate):
        raise ValueError("unknown rights candidate")
    raise CandidateAdmissionUnavailable(
        "rights steward, protected evaluation roster, and host-issued "
        "admission cannot be authenticated by this source-only module"
    )
