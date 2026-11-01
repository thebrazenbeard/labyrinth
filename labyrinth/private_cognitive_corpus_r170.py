"""Private, model-neutral Labyrinth cognitive corpus candidate assembly.

Builds TRAIN-only material from already-approved operator-side cycle exports,
plus an optional separately supplied *candidate* DEV set. It never generates
protected examiner rows, issues rights, or implies neural weight training.
All labeled bytes and derived hashes remain in an explicit PRIVATE object;
the public non-grant status is constant across corpus contents and labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata

from .approved_cognitive_empathy_train_r90 import (
    CognitiveTrainingRoom, PROVIDER_ID, RUBRIC_ID, _FOCUS,
    semantic_sha256_from_prompt,
)
from .cognitive_rights_candidate_v1 import _prepare_private_steward_bytes
from .cognitive_train_dataset_export import CognitiveTrainDataset
from .journey_challenges import PreparedTrainingRoom
from .maze import Challenge


SCHEMA_ID = "labyrinth.private-cognitive-corpus-candidate.r170"
DEV_ROW_SCHEMA = "labyrinth.cognitive-independent-dev-candidate.v1"
PUBLIC_SCHEMA_ID = "labyrinth.cognitive-corpus-public-hold.r170"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z][a-z0-9-]{2,100}\Z")
_PUBLIC_HOLD = (json.dumps({
    "schema": PUBLIC_SCHEMA_ID,
    "status": "QUARANTINED_UNISSUED",
    "authorizes_training": False,
    "authorizes_model_qualification": False,
    "authorizes_host_execution": False,
}, sort_keys=True, separators=(",", ":")) + "\n").encode()
_DEV_FIELDS = frozenset({"schema", "split", "case_id", "prompt",
                         "target_choice", "provenance_id"})
_MAX_PRIVATE_BYTES = 2_000_000
_MAX_SCENARIO_CHARS = 1024


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("utf-8")


def _valid_digest(value: str, name: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"invalid {name} SHA-256")
    return value


def _norm(value: str) -> str:
    if type(value) is not str or not 1 <= len(value) <= 4096:
        raise ValueError("invalid bounded cognitive scenario text")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _scenario_text(prompt: str) -> str:
    lines = prompt.splitlines()
    if len(lines) != 8 or not lines[0].startswith("Case: "):
        raise ValueError("invalid cognitive choice prompt")
    scenario = _norm(lines[1])
    if len(scenario) > _MAX_SCENARIO_CHARS:
        raise ValueError("cognitive scenario exceeds lexical comparison bound")
    return scenario


@dataclass(frozen=True, slots=True)
class _LexicalSignature:
    text: str
    words: frozenset[str]
    trigrams: Counter[str]
    trigram_count: int


def _lexical_signature(scenario: str) -> _LexicalSignature:
    """Linear-time features computed once per private scenario."""
    if type(scenario) is not str or not scenario:
        raise ValueError("missing lexical scenario")
    grams = Counter(scenario[index:index + 3]
                    for index in range(max(0, len(scenario) - 2)))
    return _LexicalSignature(scenario, frozenset(scenario.split()),
                             grams, sum(grams.values()))


def _near_duplicate_signatures(left: _LexicalSignature,
                               right: _LexicalSignature) -> bool:
    """Bounded linear multiset-trigram and word overlap, NOT semantic proof.

    A semantic steward must still review paraphrases that evade this filter.
    Here one-character variants and common minor inflections cannot become a
    fresh DEV case solely by changing its case ID or answer option ordering.
    No unbounded, quadratic SequenceMatcher path is used.
    """
    if left.text == right.text:
        return True
    longer = max(len(left.text), len(right.text))
    if abs(len(left.text) - len(right.text)) / longer > 0.08:
        return False
    overlap = len(left.words & right.words) / max(1, len(left.words | right.words))
    if overlap < 0.77 and max(len(left.words), len(right.words)) > 1:
        return False
    if not left.trigram_count or not right.trigram_count:
        return False
    smaller, other = ((left, right) if len(left.trigrams) <= len(right.trigrams)
                      else (right, left))
    shared = sum(min(count, other.trigrams.get(gram, 0))
                 for gram, count in smaller.trigrams.items())
    return 2 * shared / (left.trigram_count + right.trigram_count) >= 0.89


def _near_duplicate(left: str, right: str) -> bool:
    """Standalone lexical probe; corpus assembly uses cached signatures."""
    return _near_duplicate_signatures(_lexical_signature(left),
                                      _lexical_signature(right))


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """Source identity only. This cannot claim a license or consent decision."""

    cycle_index: int
    repository: str
    source_commit: str
    source_file_sha256: str

    def __post_init__(self) -> None:
        if (type(self.cycle_index) is not int or self.cycle_index < 0
                or self.cycle_index > 7 or self.repository != "thebrazenbeard/labyrinth"
                or type(self.source_commit) is not str
                or _HEX40.fullmatch(self.source_commit) is None):
            raise ValueError("invalid supported cognitive source declaration")
        _valid_digest(self.source_file_sha256, "producer file")


class PrivateCognitiveCorpusCandidate:
    """Opaque private labeled content with a constant public HOLD receipt."""

    __slots__ = ("_train", "_dev", "_manifest")

    def __init__(self, train: bytes, dev: bytes | None, manifest: bytes) -> None:
        if hasattr(self, "_train"):
            raise AttributeError("private corpus candidate is already initialized")
        if type(train) is not bytes or (dev is not None and type(dev) is not bytes) \
                or type(manifest) is not bytes:
            raise TypeError("private corpus candidate requires immutable bytes")
        object.__setattr__(self, "_train", train)
        object.__setattr__(self, "_dev", dev)
        object.__setattr__(self, "_manifest", manifest)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("private corpus candidate is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("private corpus candidate is immutable")

    def __repr__(self) -> str:
        return "<PrivateCognitiveCorpusCandidate PRIVATE HOLD>"

    @property
    def authorizes_training(self) -> bool:
        return False

    @property
    def authorizes_host_execution(self) -> bool:
        return False

    @property
    def authorizes_model_qualification(self) -> bool:
        return False

    def public_status_bytes(self) -> bytes:
        """Safe to publish; no dataset count, hash, seed or label oracle."""
        return _PUBLIC_HOLD

    def private_train_jsonl(self) -> bytes:
        """Operator-side only; contains answer labels."""
        return self._train

    def private_dev_jsonl(self) -> bytes | None:
        """Unadmitted DEV candidate only; NOT protected evaluation data."""
        return self._dev

    def private_manifest_json(self) -> bytes:
        """Operator-side only; contains low-entropy source commitments."""
        return self._manifest


def _parse_dev(dev: bytes) -> tuple[list[dict], list[tuple[str, _LexicalSignature]]]:
    if type(dev) is not bytes or not dev or len(dev) > _MAX_PRIVATE_BYTES:
        raise ValueError("invalid bounded private DEV candidate bytes")
    if not dev.endswith(b"\n"):
        raise ValueError("private DEV candidate is not canonical JSONL")
    rows = []
    similarities = []
    for line in dev.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("invalid private DEV JSONL") from exc
        if type(row) is not dict or set(row) != _DEV_FIELDS:
            raise ValueError("invalid DEV row fields")
        if _canonical(row) + b"\n" != line:
            raise ValueError("noncanonical private DEV JSONL")
        if (row["schema"] != DEV_ROW_SCHEMA or row["split"] != "dev"
                or type(row["case_id"]) is not str
                or not row["case_id"].startswith("dev-cognitive-")
                or _ID.fullmatch(row["case_id"]) is None
                or type(row["provenance_id"]) is not str
                or _ID.fullmatch(row["provenance_id"]) is None
                or row["target_choice"] not in ("A", "B", "C")
                or type(row["prompt"]) is not str):
            raise ValueError("invalid or unverified DEV source declaration")
        if not row["prompt"].startswith(f"Case: {row['case_id']}\n"):
            raise ValueError("DEV case ID does not bind its prompt")
        # Reuse the TRAIN parser to independently check exact choice format.
        canonical_train_prompt = row["prompt"].replace(
            f"Case: {row['case_id']}\n", f"Case: train-cognitive-{row['case_id']}\n", 1,
        )
        semantic = semantic_sha256_from_prompt(canonical_train_prompt)
        similarities.append((semantic, _lexical_signature(_scenario_text(row["prompt"]))))
        rows.append(row)
    if not rows or len(rows) > 512:
        raise ValueError("invalid DEV row count")
    return rows, similarities


def _validate_approved_train_gold_and_binding(row: dict) -> None:
    """Reconstruct the approved provider's bounded source record and gold.

    The upstream steward parser checks syntax and caller-recomputed digests,
    but the dataset is a plain NamedTuple, not an authenticated export. Its
    SHA can be recomputed after an answer edit. Cross-check the row against
    the independent approved 30-scenario gold table and original complete
    room-source commitment. The producer did not serialize its focus tag;
    boundedly enumerate the provider's five allowed tags without inventing
    a gold answer. This is source consistency, NOT a trusted host signature.
    """
    if (row["provider_id"] != PROVIDER_ID
            or row["training_rubric_id"] != RUBRIC_ID
            or row["case_id"] != f"train-cognitive-{row['room_id']}"):
        raise ValueError("TRAIN gold/source binding origin mismatch")
    for focus_tag in sorted(_FOCUS):
        challenge = Challenge(
            id=row["case_id"], kind="synthetic_cognitive_perspective",
            tier=1, split="train", prompt=row["prompt"],
            answer=row["target_choice"],
            metadata=(
                ("provider_id", PROVIDER_ID),
                ("training_rubric", RUBRIC_ID),
                ("focus_tag", focus_tag),
                ("objective_contract_sha256", row["objective_contract_sha256"]),
                ("semantic_scenario_sha256", row["semantic_sha256"]),
            ),
        )
        record = CognitiveTrainingRoom(
            prepared=PreparedTrainingRoom(row["room_id"], "cognitive_empathy", challenge),
            semantic_sha256=row["semantic_sha256"],
        )
        object.__setattr__(record, "_source_binding_sha256", row["source_binding_sha256"])
        try:
            if record.score_train_choice(row["target_choice"]) == 100:
                return
        except ValueError:
            continue
    raise ValueError("TRAIN gold/source binding does not match approved provider")


def build_private_cognitive_corpus(
    train_exports: tuple[CognitiveTrainDataset, ...],
    sources: tuple[CorpusSource, ...],
    *,
    dev_candidate_jsonl: bytes | None = None,
    protected_semantic_sha256: tuple[str, ...] = (),
    protected_roster_sha256: str | None = None,
) -> PrivateCognitiveCorpusCandidate:
    """Assemble a PRIVATE model-neutral SFT candidate, always QUARANTINED.

    Existing approved TRAIN cycles are never silently reclassified as DEV.
    The protected examiner bank is not an input and is never exported. Exact
    supplied exclusion digests are checked, but roster completeness, corpus
    rights, consent, DEV authorship and near-semantic dedup remain unverified.
    """
    if (type(train_exports) is not tuple or not 1 <= len(train_exports) <= 8
            or type(sources) is not tuple or len(sources) != len(train_exports)):
        raise ValueError("expected one bounded declaration for each TRAIN cycle")
    if type(protected_semantic_sha256) is not tuple or len(protected_semantic_sha256) > 4096:
        raise ValueError("protected exclusions must be bounded immutable tuple")
    exclusions = tuple(_valid_digest(item, "excluded semantic")
                       for item in protected_semantic_sha256)
    if len(set(exclusions)) != len(exclusions):
        raise ValueError("duplicate protected exclusions")
    if exclusions and protected_roster_sha256 is None:
        raise ValueError("protected exclusions require roster commitment")
    if protected_roster_sha256 is not None:
        _valid_digest(protected_roster_sha256, "protected roster")
    entries = list(zip(train_exports, sources))
    if any(type(dataset) is not CognitiveTrainDataset or
           type(source) is not CorpusSource or
           source.cycle_index != dataset.cycle_index for dataset, source in entries):
        raise ValueError("TRAIN cycle/source declaration mismatch")
    entries.sort(key=lambda item: item[0].cycle_index)
    if [dataset.cycle_index for dataset, _ in entries] != list(range(len(entries))):
        raise ValueError("noncontiguous or replayed TRAIN cycles")
    if len({dataset.contract_fingerprint for dataset, _ in entries}) != 1:
        raise ValueError("multiple unrelated objective contracts")
    train_lines: list[bytes] = []
    train_details = []
    seen_ids: set[str] = set()
    seen_semantic: set[str] = set()
    scenario_texts: list[_LexicalSignature] = []
    for dataset, source in entries:
        # This existing validator enforces exactly 15 canonical labeled rows,
        # genuine source bindings and no supplied protected exact overlap.
        _prepare_private_steward_bytes(
            dataset, source_repo=source.repository,
            source_commit=source.source_commit,
            source_file_sha256=source.source_file_sha256,
            exclusion_semantic_sha256=exclusions,
            exclusion_roster_sha256=protected_roster_sha256,
        )
        for line in dataset.jsonl_bytes.splitlines(keepends=True):
            row = json.loads(line)
            _validate_approved_train_gold_and_binding(row)
            semantic = row["semantic_sha256"]
            if row["case_id"] in seen_ids or semantic in seen_semantic:
                raise ValueError("duplicate TRAIN identity or semantic case")
            scenario = _lexical_signature(_scenario_text(row["prompt"]))
            if any(_near_duplicate_signatures(scenario, prior) for prior in scenario_texts):
                raise ValueError("TRAIN near-duplicate case")
            seen_ids.add(row["case_id"])
            seen_semantic.add(semantic)
            scenario_texts.append(scenario)
            train_lines.append(line)
            train_details.append({
                "case_id_sha256": hashlib.sha256(row["case_id"].encode()).hexdigest(),
                "row_sha256": hashlib.sha256(line).hexdigest(),
                "semantic_sha256": semantic,
                "source_binding_sha256": row["source_binding_sha256"],
                "source_cycle": dataset.cycle_index,
                "source_repository": source.repository,
                "source_commit": source.source_commit,
                "source_file_sha256": source.source_file_sha256,
                "license_status": "UNVERIFIED",
                "consent_status": "UNVERIFIED",
            })
    train_payload = b"".join(train_lines)
    if len(train_payload) > _MAX_PRIVATE_BYTES:
        raise ValueError("private TRAIN corpus exceeds limit")
    dev_rows = []
    if dev_candidate_jsonl is not None:
        dev_rows, dev_semantics = _parse_dev(dev_candidate_jsonl)
        seen_dev_ids = set()
        for row, (semantic, scenario) in zip(dev_rows, dev_semantics):
            if row["case_id"] in seen_dev_ids or semantic in seen_semantic:
                raise ValueError("TRAIN/DEV duplicate semantic case")
            if any(_near_duplicate_signatures(scenario, previous)
                   for previous in scenario_texts):
                raise ValueError("TRAIN/DEV near-duplicate case")
            if semantic in exclusions:
                raise ValueError("DEV overlaps supplied protected exclusion")
            seen_dev_ids.add(row["case_id"])
            seen_semantic.add(semantic)
            scenario_texts.append(scenario)
    private_manifest = {
        "schema": SCHEMA_ID,
        "model_neutral": True,
        "target_family": "cognitive_perspective_choice_A_B_C",
        "sources": [{"cycle_index": source.cycle_index,
                     "repository": source.repository,
                     "source_commit": source.source_commit,
                     "source_file_sha256": source.source_file_sha256,
                     "external_source_verification": "NOT_ESTABLISHED",
                     "dataset_sha256": dataset.sha256}
                    for dataset, source in entries],
        "splits": {
            "train": {"count": len(train_details), "sha256": hashlib.sha256(train_payload).hexdigest(),
                      "rows": train_details},
            "dev": None if dev_candidate_jsonl is None else {
                "count": len(dev_rows),
                "sha256": hashlib.sha256(dev_candidate_jsonl).hexdigest(),
                "provenance_status": "OPERATOR_SUPPLIED_UNVERIFIED",
                "semantic_equivalence_review": "REQUIRED_UNVERIFIED",
                "rows": [{"case_id_sha256": hashlib.sha256(row["case_id"].encode()).hexdigest(),
                          "row_sha256": hashlib.sha256(_canonical(row) + b"\n").hexdigest(),
                          "source_provenance_id": row["provenance_id"],
                          "license_status": "UNVERIFIED",
                          "consent_status": "UNVERIFIED"}
                         for row in dev_rows],
            },
        },
        "protected_eval": {"rows_exported": 0, "roster_sha256": protected_roster_sha256,
                           "supplied_exclusion_count": len(exclusions),
                           "custodian_authenticated": False,
                           "semantic_overlap_review": "REQUIRED_UNVERIFIED",
                           "cross_system_exposure_ledger_verified": False},
        "rights": {"status": "UNISSUED", "authorship_verified": False,
                   "consent_review": "NOT_ASSESSED", "license_review": "NOT_ADMITTED",
                   "independent_steward_signed": False},
        "qualification": {"status": "QUARANTINED", "authorizes_training": False,
                          "authorizes_model_qualification": False,
                          "authorizes_host_execution": False},
    }
    return PrivateCognitiveCorpusCandidate(
        train_payload, dev_candidate_jsonl, _canonical(private_manifest) + b"\n",
    )
