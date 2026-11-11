"""Fail-closed source-level prescreen for *mechanical* transfer evidence.

This is deliberately NOT the model graduation gate. All mechanic identities,
checkpoint SHA-256 values, optimizer-step counts and evaluator labels are
declarations that require independent physical custody to become evidence.
Even a successful HMAC proves only the bytes signed by its key holder. This
module therefore has no disposition named GRADUATED or VERIFIED_LEARNING.

The public Minotaur exam can be solved by a zero-update lookup across wording
variants. Such paraphrases count as one mechanic, not multiple independent
transfer observations. An unfamiliar structural family must be precommitted
outside training, evaluated with matched controls, and externally verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import re


SCHEMA = "labyrinth.structural-transfer-screen.v1"
_DOMAIN = b"labyrinth.structural-transfer-screen.v1\x00"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,95}\Z")
_SPLITS = ("train", "heldout", "transfer")
_CONTROLS = ("base", "zero", "corrupted", "memorizer")
_EVAL_SPLITS = ("heldout", "transfer")
_MIN_FAMILIES = 8
_MAX_CASES = 512


@dataclass(frozen=True)
class CaseCommitment:
    case_id: str
    split: str
    mechanic_sha256: str
    prompt_sha256: str


@dataclass(frozen=True)
class CheckpointLineage:
    base_sha256: str
    candidate_before_sha256: str
    candidate_after_sha256: str
    candidate_reload_sha256: str
    zero_update_sha256: str
    corrupted_label_sha256: str
    optimizer_steps: int


@dataclass(frozen=True)
class StructuralManifest:
    cases: tuple[CaseCommitment, ...]
    lineage: CheckpointLineage
    evaluator_sha256: str
    dataset_sha256: str
    source_public: bool
    sha256: str


@dataclass(frozen=True)
class TrialOutcome:
    case_id: str
    candidate_correct: bool
    candidate_valid: bool
    base_correct: bool
    zero_correct: bool
    corrupted_correct: bool
    memorizer_correct: bool


@dataclass(frozen=True)
class TransferReport:
    manifest_sha256: str
    rows: tuple[TrialOutcome, ...]


@dataclass(frozen=True)
class ScreenDecision:
    disposition: str
    reasons: tuple[str, ...]
    independent_families: tuple[int, int] = (0, 0)
    neural_learning_verified: bool = False
    private_exam_qualified: bool = False


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _valid_sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _valid_id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _copy_cases(cases: object) -> tuple[CaseCommitment, ...]:
    if type(cases) is not tuple or not 3 <= len(cases) <= _MAX_CASES:
        raise ValueError("bounded case roster is required")
    seen_cases = set()
    seen_prompts = set()
    by_split = {name: set() for name in _SPLITS}
    frozen = []
    for row in cases:
        if type(row) is not CaseCommitment:
            raise ValueError("invalid, duplicate or ambiguous committed case")
        case_id, split, mechanic, prompt = (row.case_id, row.split,
                                             row.mechanic_sha256, row.prompt_sha256)
        if (not _valid_id(case_id) or type(split) is not str or split not in _SPLITS
                or not _valid_sha(mechanic) or not _valid_sha(prompt)
                or case_id in seen_cases or prompt in seen_prompts):
            raise ValueError("invalid, duplicate or ambiguous committed case")
        seen_cases.add(case_id)
        seen_prompts.add(prompt)
        by_split[split].add(mechanic)
        frozen.append(CaseCommitment(case_id, split, mechanic, prompt))
    if any(not by_split[name] for name in _SPLITS):
        raise ValueError("train, heldout and transfer splits are required")
    if any(by_split[a] & by_split[b] for i, a in enumerate(_SPLITS)
           for b in _SPLITS[i + 1:]):
        raise ValueError("mechanic split overlap contaminates transfer")
    if any(len(by_split[name]) < _MIN_FAMILIES for name in _EVAL_SPLITS):
        raise ValueError("insufficient independent mechanics in evaluation split")
    return tuple(frozen)


def _copy_lineage(value: object) -> CheckpointLineage:
    if type(value) is not CheckpointLineage:
        raise ValueError("checkpoint lineage must be exact")
    fields = (value.base_sha256, value.candidate_before_sha256,
              value.candidate_after_sha256, value.candidate_reload_sha256,
              value.zero_update_sha256, value.corrupted_label_sha256)
    if (not all(_valid_sha(x) for x in fields)
            or type(value.optimizer_steps) is not int
            or not 0 <= value.optimizer_steps <= 1_000_000):
        raise ValueError("invalid checkpoint lineage")
    return CheckpointLineage(*fields, value.optimizer_steps)


def _manifest_payload(cases, lineage, evaluator_sha256, dataset_sha256, source_public):
    return {"schema": SCHEMA, "cases": [asdict(c) for c in cases],
            "lineage": asdict(lineage), "evaluator_sha256": evaluator_sha256,
            "dataset_sha256": dataset_sha256, "source_public": source_public}


def precommit(cases: tuple[CaseCommitment, ...], lineage: CheckpointLineage,
              *, evaluator_sha256: str, dataset_sha256: str,
              source_public: bool) -> StructuralManifest:
    """Snapshot exact declared identities before revealing evaluation outcomes.

    This is a **source digest**, not an independent timestamp or custody proof.
    """
    frozen_cases = _copy_cases(cases)
    frozen_lineage = _copy_lineage(lineage)
    if (not _valid_sha(evaluator_sha256) or not _valid_sha(dataset_sha256)
            or type(source_public) is not bool):
        raise ValueError("invalid source provenance or public-exam classification")
    digest = _digest(_manifest_payload(frozen_cases, frozen_lineage,
                                       evaluator_sha256, dataset_sha256,
                                       source_public))
    return StructuralManifest(frozen_cases, frozen_lineage, evaluator_sha256,
                              dataset_sha256, source_public, digest)


def _copy_report(report: object) -> TransferReport:
    if (type(report) is not TransferReport or not _valid_sha(report.manifest_sha256)
            or type(report.rows) is not tuple or not 1 <= len(report.rows) <= _MAX_CASES):
        raise ValueError("malformed result report")
    frozen_rows = []
    for row in report.rows:
        if type(row) is not TrialOutcome:
            raise ValueError("malformed trial outcome")
        values = (row.case_id, row.candidate_correct, row.candidate_valid,
                  row.base_correct, row.zero_correct, row.corrupted_correct,
                  row.memorizer_correct)
        if not _valid_id(values[0]) or any(type(value) is not bool for value in values[1:]):
            raise ValueError("malformed trial outcome")
        frozen_rows.append(TrialOutcome(*values))
    return TransferReport(report.manifest_sha256, tuple(frozen_rows))


def canonical_report(report: TransferReport) -> bytes:
    """Canonical bytes to sign by a genuinely independent evaluator, if any."""
    frozen = _copy_report(report)
    return json.dumps({"schema": SCHEMA, "manifest_sha256": frozen.manifest_sha256,
                       "rows": [asdict(row) for row in frozen.rows]},
                      sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _blocked(reason: str, counts: tuple[int, int] = (0, 0)) -> ScreenDecision:
    return ScreenDecision("BLOCKED", (reason,), counts)


def assess(manifest: StructuralManifest, report: TransferReport,
           *, signature_hex: str, independent_key: bytes) -> ScreenDecision:
    """Only screen declared evidence; never graduate or certify neural learning.

    A self-controlled key or fabricated semantic-family labels can make source
    tests pass. A successful screen is PENDING independent physical readback.
    """
    try:
        if type(manifest) is not StructuralManifest:
            return _blocked("invalid manifest")
        cases, lineage, evaluator_sha, dataset_sha, source_public, original_sha = (
            manifest.cases, manifest.lineage, manifest.evaluator_sha256,
            manifest.dataset_sha256, manifest.source_public, manifest.sha256)
        if not _valid_sha(original_sha):
            return _blocked("invalid manifest")
        frozen_plan = precommit(cases, lineage,
                                evaluator_sha256=evaluator_sha,
                                dataset_sha256=dataset_sha,
                                source_public=source_public)
        if original_sha != frozen_plan.sha256:
            return _blocked("manifest digest disagrees with committed fields")
        frozen_report = _copy_report(report)
        if frozen_report.manifest_sha256 != frozen_plan.sha256:
            return _blocked("receipt refers to another manifest")
        if (type(independent_key) is not bytes or len(independent_key) < 32
                or type(signature_hex) is not str or not _valid_sha(signature_hex)):
            return _blocked("missing authenticated evaluator signature")
        canonical = canonical_report(frozen_report)
        expected = hmac.new(independent_key, _DOMAIN + canonical, "sha256").hexdigest()
        if not hmac.compare_digest(expected, signature_hex):
            return _blocked("invalid evaluator signature")
    except Exception:
        return _blocked("malformed source-level transfer receipt")

    planned = {c.case_id: c for c in frozen_plan.cases if c.split in _EVAL_SPLITS}
    indexed: dict[str, TrialOutcome] = {}
    for row in frozen_report.rows:
        if row.case_id in indexed or row.case_id not in planned:
            return _blocked("evaluation roster has duplicate or uncommitted case")
        indexed[row.case_id] = row
    if indexed.keys() != planned.keys():
        return _blocked("evaluation roster was changed or cherry-picked")

    by_split: dict[str, dict[str, list[TrialOutcome]]] = {
        split: {} for split in _EVAL_SPLITS}
    for case_id, row in indexed.items():
        commitment = planned[case_id]
        by_split[commitment.split].setdefault(commitment.mechanic_sha256, []).append(row)
    counts = tuple(len(by_split[name]) for name in _EVAL_SPLITS)

    lineage = frozen_plan.lineage
    if (lineage.optimizer_steps == 0
            or lineage.candidate_after_sha256 == lineage.candidate_before_sha256):
        return _blocked("no declared candidate parameter update", counts)
    if lineage.candidate_after_sha256 != lineage.candidate_reload_sha256:
        return _blocked("candidate checkpoint reload hash mismatch", counts)
    for control_name, control_sha in (
        ("base", lineage.base_sha256),
        ("zero-update", lineage.zero_update_sha256),
        ("corrupted-label", lineage.corrupted_label_sha256),
    ):
        if lineage.candidate_after_sha256 == control_sha:
            return _blocked(f"candidate checkpoint identical to {control_name} control", counts)
    if frozen_plan.source_public:
        return _blocked("source-public examination cannot qualify blinded transfer", counts)

    for split in _EVAL_SPLITS:
        family_rows = tuple(by_split[split].values())
        size = len(family_rows)
        successes = sum(all(row.candidate_correct and row.candidate_valid for row in rows)
                        for rows in family_rows)
        if successes < size * .75:
            return _blocked(f"candidate failed independent mechanics on {split}", counts)
        for control in _CONTROLS:
            control_successes = sum(all(getattr(row, control + "_correct") for row in rows)
                                    for rows in family_rows)
            if (successes - control_successes) / size < .10:
                return _blocked(f"no measured gain against {control} on {split}", counts)

    return ScreenDecision(
        "PENDING_INDEPENDENT_CUSTODY",
        ("source declarations and HMAC cannot authenticate private evaluation, "
         "parameter update execution or mechanics labels",),
        counts,
    )
