"""Offline, strictly unadmitted Dolly source-sample review for Vera TRAIN/DEV.

No downloader is provided. The sample and optional protected-exclusion digests
must arrive via a separately controlled PRIVATE operator route. Publisher
metadata does not prove row membership, legal adequacy, contributor consent,
privacy clearance, independent DEV, or a complete protected evaluation roster.
Neither building candidates nor a source license claim authorizes training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import threading
import unicodedata
import weakref


SCHEMA_ID = "labyrinth.external-licensed-corpus-private-review.r170"
PUBLIC_SCHEMA_ID = "labyrinth.external-licensed-corpus-public-hold.r170"
DOLLY_REPOSITORY = "databricks/databricks-dolly-15k"
DOLLY_DATASET_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_LFS_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
DOLLY_FILENAME = "databricks-dolly-15k.jsonl"
DOLLY_FILE_BYTES = 13_085_339
DOLLY_CARD = "https://huggingface.co/datasets/databricks/databricks-dolly-15k"
DOLLY_DECLARED_LICENSE = "CC-BY-SA-3.0"
MAX_SAMPLES = 256
MAX_SAMPLE_LINE_BYTES = 8192
MAX_SAMPLE_BYTES = 2_000_000
_DOLLY_KNOWN_ROWS = 15_011
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CATEGORY = re.compile(r"[a-z][a-z0-9_]{1,40}\Z")
_EMAIL_OR_LINK = re.compile(r"(?i)(?:[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}|https?://|www\.)")
_PHONE_LIKE = re.compile(
    r"(?<![A-Za-z0-9])(?:\+?\d{1,3}[\s.\-]?)?"
    r"(?:\(\d{2,4}\)|\d{2,4})[\s.\-]+\d{3,4}[\s.\-]+\d{3,4}"
    r"(?![A-Za-z0-9])"
)
_CREDENTIAL = re.compile(r"(?i)(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:api[_ -]?key|secret|password|access[_ -]?token)\s*(?:=|:))")
# Untrusted instruction/answer text may otherwise masquerade as chat framing.
_STRUCTURAL_CHAT_DELIMITER = re.compile(
    r"(?i)(?:<\|(?:im_start|im_end|start_header_id|end_header_id|eot_id|fim_prefix|fim_middle)\|>|\[/?INST\])"
)
_RESERVED_CONTROLS = (chr(0), chr(27), chr(127))
_DOLLY_FIELDS = frozenset({"instruction", "context", "response", "category"})
_PUBLIC_HOLD = (json.dumps({
    "schema": PUBLIC_SCHEMA_ID,
    "status": "QUARANTINED_NOT_ADMITTED",
    "authorizes_training": False,
    "authorizes_host_execution": False,
    "authorizes_model_qualification": False,
}, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
_PRIVATE_CANDIDATE_PAYLOADS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_PRIVATE_CANDIDATE_LOCK = threading.Lock()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _pairs_unique(pairs: list[tuple[str, object]]) -> dict:
    obj: dict = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate source JSON key")
        obj[key] = value
    return obj


def _reject_reserved_controls(value: str) -> None:
    if any(control in value for control in _RESERVED_CONTROLS):
        raise ValueError("reserved control character in source text")


def _normalized(value: str) -> str:
    if type(value) is not str or not 1 <= len(value) <= 4096:
        raise ValueError("invalid bounded prompt")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("invalid Unicode in source text") from exc
    _reject_reserved_controls(value)
    cleaned = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not cleaned:
        raise ValueError("empty normalized prompt")
    return cleaned


def private_prompt_sha256(instruction: str, context: str) -> str:
    """PRIVATE cross-split/exclusion fingerprint; not a semantic proof.

    Protected evaluator must independently apply the same algorithm to its
    private complete roster, if appropriate, before claiming exclusions.
    """
    if type(context) is not str or len(context) > 4096:
        raise ValueError("invalid prompt context")
    try:
        context.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("invalid Unicode in private prompt context") from exc
    _reject_reserved_controls(context)
    normalized_context = " ".join(unicodedata.normalize("NFKC", context).casefold().split())
    return hashlib.sha256((_normalized(instruction) + "\0" + normalized_context).encode()).hexdigest()


def _bounded_lexical_shingles(value: str) -> frozenset[str]:
    """O(1) bounded-memory feature size, independent of 4K input length.

    This deliberately does NOT establish semantic similarity. Truncating to
    the front/back allows false positives/negatives that the independent
    steward must review. Unlike unrestricted SequenceMatcher, even repeated
    four-thousand-character adversarial prompts cannot force quadratic work.
    """
    bounded = value[:192] + "\x1f" + value[-192:]
    return frozenset(bounded[i:i + 4] for i in range(max(1, len(bounded) - 3)))


def _near_duplicate(a: str, a_grams: frozenset[str],
                    b: str, b_grams: frozenset[str]) -> bool:
    if min(len(a), len(b)) < max(len(a), len(b)) * 0.88:
        return False
    # Bounded exact slicing catches one-char suffix changes even in low-
    # entropy repeated text where distinct n-gram sets are too small to
    # produce a reliable similarity score. This intentionally quarantines
    # some false positives; no external data is approved automatically.
    if (abs(len(a) - len(b)) <= 4 and a[:160] == b[:160]
            and (a[-160:-1] == b[-160:-1] or a[-160:] == b[-160:])):
        return True
    common = len(a_grams & b_grams)
    return 2 * common >= 0.85 * (len(a_grams) + len(b_grams))


@dataclass(frozen=True, slots=True)
class ExternalCorpusSample:
    """Caller-asserted line identity; no proof that it belongs to upstream."""

    source_line_number: int
    intended_split: str
    raw_jsonl_line: bytes


class ExternalCorpusCandidates:
    """Opaque PRIVATE artifacts, with one constant public non-grant receipt."""

    # No bytes live in assignable instance slots, even when a caller invokes
    # object.__setattr__ to bypass the conventional immutability override.
    # Weak keys release payloads with the corresponding private object.
    __slots__ = ("__weakref__",)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("external private candidate cannot be subclassed")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("external private candidate is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("external private candidate is immutable")

    def __init__(self, train: bytes | None, dev: bytes | None,
                 quarantine: bytes, manifest: bytes) -> None:
        if type(self) is not ExternalCorpusCandidates:
            raise TypeError("private candidate requires exact expected class")
        if ((train is not None and type(train) is not bytes)
                or (dev is not None and type(dev) is not bytes)
                or type(quarantine) is not bytes or type(manifest) is not bytes):
            raise ValueError("private candidate payload requires exact byte types")
        # __init__ is publicly callable on an already constructed object.
        # Check-and-insert under one lock: direct reinitialization and racing
        # calls must fail closed instead of replacing the private artifact and
        # invalidating its manifest. WeakKeyDictionary holds no strong key.
        with _PRIVATE_CANDIDATE_LOCK:
            if self in _PRIVATE_CANDIDATE_PAYLOADS:
                raise ValueError("private candidate already initialized")
            _PRIVATE_CANDIDATE_PAYLOADS[self] = (train, dev, quarantine, manifest)

    def __repr__(self) -> str:
        return "<ExternalCorpusCandidates PRIVATE HOLD>"

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
        return _PUBLIC_HOLD

    def private_train_candidate_jsonl(self) -> bytes | None:
        return _PRIVATE_CANDIDATE_PAYLOADS[self][0]

    def private_dev_candidate_jsonl(self) -> bytes | None:
        return _PRIVATE_CANDIDATE_PAYLOADS[self][1]

    def private_quarantine_json(self) -> bytes:
        return _PRIVATE_CANDIDATE_PAYLOADS[self][2]

    def private_manifest_json(self) -> bytes:
        return _PRIVATE_CANDIDATE_PAYLOADS[self][3]


def _parse_source_line(line: bytes) -> dict:
    if type(line) is not bytes or not 0 < len(line) <= MAX_SAMPLE_LINE_BYTES:
        raise ValueError("invalid bounded source line size")
    if not line.endswith(b"\n") or line.count(b"\n") != 1 or b"\r" in line:
        raise ValueError("source must be one LF-terminated JSON object")
    try:
        row = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs_unique)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid source JSON line") from exc
    if type(row) is not dict or set(row) != _DOLLY_FIELDS:
        raise ValueError("unrecognized Dolly source fields")
    if any(type(row[field]) is not str for field in _DOLLY_FIELDS):
        raise ValueError("non-string Dolly field")
    if (not 1 <= len(row["instruction"]) <= 4096
            or not 1 <= len(row["response"]) <= 4096
            or len(row["context"]) > 4096
            or _CATEGORY.fullmatch(row["category"]) is None):
        raise ValueError("oversized or invalid Dolly source value")
    _normalized(row["instruction"])
    _normalized(row["response"])
    # A JSON \\ud800 escape decodes to a lone surrogate, which cannot be
    # encoded as UTF-8. Reject before hashing, preserving a controlled error
    # instead of leaking a codec exception (including in unused contexts).
    if row["context"]:
        try:
            row["context"].encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("invalid Unicode in source context") from exc
        _reject_reserved_controls(row["context"])
    return row


def _quarantine_reason(row: dict) -> str | None:
    # Conservative, intentionally imperfect screening: review is mandatory
    # even if no pattern matches. Sources include Wikipedia reference text.
    all_text = "\n".join(row[field] for field in ("instruction", "context", "response"))
    # Fingerprints use NFKC, so quarantine cannot inspect only raw glyphs:
    # fullwidth compatibility spellings otherwise evade PII/framing patterns.
    screened = (all_text, unicodedata.normalize("NFKC", all_text))
    if any(_STRUCTURAL_CHAT_DELIMITER.search(text) for text in screened):
        return "STRUCTURAL_CHAT_DELIMITER"
    if row["context"].strip():
        return "THIRD_PARTY_CONTEXT"
    if any(_CREDENTIAL.search(text) for text in screened):
        return "POSSIBLE_CREDENTIAL"
    if any(_EMAIL_OR_LINK.search(text) or _PHONE_LIKE.search(text)
           for text in screened):
        return "POTENTIAL_PII_OR_LINK"
    return None


def build_external_corpus_candidates(
    samples: tuple[ExternalCorpusSample, ...],
    *,
    source_repository: str = DOLLY_REPOSITORY,
    source_revision: str = DOLLY_DATASET_REVISION,
    full_file_sha256: str = DOLLY_LFS_SHA256,
    protected_prompt_sha256: tuple[str, ...] = (),
) -> ExternalCorpusCandidates:
    """Create unadmitted PRIVATE TRAIN/DEV candidates from an offline sample.

    Source membership, original employee consent, CC BY-SA compliance, heldout
    isolation, and complete protected exclusions are ALWAYS unverified. The
    256-row bound intentionally prevents silently ingesting the full corpus.
    """
    if (type(samples) is not tuple or not 1 <= len(samples) <= MAX_SAMPLES
            or source_repository != DOLLY_REPOSITORY
            or source_revision != DOLLY_DATASET_REVISION
            or full_file_sha256 != DOLLY_LFS_SHA256):
        raise ValueError("untrusted or unbounded publisher source identity")
    if (type(protected_prompt_sha256) is not tuple
            or len(protected_prompt_sha256) > 4096
            or any(type(x) is not str or _HEX64.fullmatch(x) is None
                   for x in protected_prompt_sha256)
            or len(set(protected_prompt_sha256)) != len(protected_prompt_sha256)):
        raise ValueError("invalid private protected exclusion inventory")
    if any(type(s) is not ExternalCorpusSample or
           type(s.source_line_number) is not int or
           not 1 <= s.source_line_number <= _DOLLY_KNOWN_ROWS or
           s.intended_split not in ("train", "dev") or
           type(s.raw_jsonl_line) is not bytes for s in samples):
        raise ValueError("invalid declared sample identity/split")
    if len({s.source_line_number for s in samples}) != len(samples):
        raise ValueError("duplicate claimed upstream source line")
    if sum(len(s.raw_jsonl_line) for s in samples) > MAX_SAMPLE_BYTES:
        raise ValueError("sample bytes exceed private bound")

    ordered = sorted(samples, key=lambda s: s.source_line_number)
    # Fail closed for malformed source before producing any artifacts.
    parsed = [(sample, _parse_source_line(sample.raw_jsonl_line)) for sample in ordered]
    exclusion_set = set(protected_prompt_sha256)
    prior: list[tuple[str, str, frozenset[str]]] = []
    accepted: dict[str, list[dict]] = {"train": [], "dev": []}
    quarantined: list[dict] = []
    private_details: list[dict] = []
    for sample, row in parsed:
        fingerprint = private_prompt_sha256(row["instruction"], row["context"])
        normalized = _normalized(row["instruction"])
        if row["context"].strip():
            normalized += "\0" + " ".join(
                unicodedata.normalize("NFKC", row["context"]).casefold().split()
            )
        lexical = _bounded_lexical_shingles(normalized)
        reason = _quarantine_reason(row)
        if fingerprint in exclusion_set:
            reason = "PROTECTED_PROMPT_EXCLUSION"
        elif any(fingerprint == previous_fp for previous_fp, _, _ in prior):
            reason = "DUPLICATE_PROMPT"
        elif any(_near_duplicate(normalized, lexical, previous_text, previous_grams)
                 for _, previous_text, previous_grams in prior):
            reason = "NEAR_DUPLICATE_PROMPT"
        prior.append((fingerprint, normalized, lexical))
        detail = {
            "source_line_number": sample.source_line_number,
            "claimed_split": sample.intended_split,
            "raw_line_sha256": hashlib.sha256(sample.raw_jsonl_line).hexdigest(),
            "prompt_sha256": fingerprint,
            "review_status": "PENDING_INDEPENDENT_STEWARD",
        }
        private_details.append(detail)
        if reason is not None:
            quarantined.append({**detail, "reason": reason})
            continue
        accepted[sample.intended_split].append({
            "schema": "labyrinth.external-dolly-sft-unadmitted.r170",
            "split": sample.intended_split,
            "source_line_number_claimed": sample.source_line_number,
            "instruction": row["instruction"],
            "context": row["context"],
            "response": row["response"],
            "category": row["category"],
            "rights": "NOT_ADMITTED",
        })
    def _jsonl(rows: list[dict]) -> bytes | None:
        return b"".join(_canonical(r) + b"\n" for r in rows) if rows else None

    train_bytes = _jsonl(accepted["train"])
    dev_bytes = _jsonl(accepted["dev"])
    quarantine_bytes = _canonical({"schema": SCHEMA_ID + ".quarantine",
                                   "entries": quarantined}) + b"\n"
    manifest = {
        "schema": SCHEMA_ID,
        "source": {
            "repository": DOLLY_REPOSITORY,
            "revision": DOLLY_DATASET_REVISION,
            "filename": DOLLY_FILENAME,
            "full_lfs_sha256": DOLLY_LFS_SHA256,
            "full_file_size_bytes": DOLLY_FILE_BYTES,
            "owner_card": DOLLY_CARD,
            "sample_membership_verified": False,
            "observed_offline_sample_count": len(samples),
            "private_rows": private_details,
        },
        "splits": {
            "train_candidate_count": len(accepted["train"]),
            "dev_candidate_count": len(accepted["dev"]),
            "quarantined_count": len(quarantined),
            "independent_dev_verified": False,
            "train_candidate_sha256": hashlib.sha256(train_bytes).hexdigest() if train_bytes else None,
            "dev_candidate_sha256": hashlib.sha256(dev_bytes).hexdigest() if dev_bytes else None,
            "quarantine_sha256": hashlib.sha256(quarantine_bytes).hexdigest(),
        },
        "rights": {
            "license_claim": DOLLY_DECLARED_LICENSE,
            "independent_license_review": "NOT_ESTABLISHED",
            "contributor_consent": "NOT_ESTABLISHED",
            "privacy_clearance": "NOT_ESTABLISHED",
            "wikipedia_context_reuse_review": "NOT_ESTABLISHED",
            "steward_signature": None,
        },
        "evaluation": {
            "supplied_exclusion_count": len(exclusion_set),
            "complete_exclusion_roster_verified": False,
            "protected_evaluator_signature": None,
            "protected_rows_exported": 0,
        },
        "admission": {
            "status": "QUARANTINED_NOT_ADMITTED",
            "authorizes_training": False,
            "authorizes_host_execution": False,
            "authorizes_model_qualification": False,
        },
    }

    return ExternalCorpusCandidates(
        train_bytes, dev_bytes, quarantine_bytes,
        _canonical(manifest) + b"\n",
    )
