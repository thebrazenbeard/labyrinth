"""Evaluator-side, source-level hostile checks for regenerative neural learning.

This module must be run by a distinct evaluator custodian, with private gold,
precommitted cases, real fresh-process model outputs and externally verified
training/weight receipts. It NEVER opens a model, signs an exam, issues a
graduation decision or authenticates the evidence supplied to it. In-process
use by the trainer is only a protocol dry run.

The public case commitment intentionally excludes gold: publishing a hash of
small answer keys would allow a dictionary attack. An examiner-private HMAC
seal also freezes gold while the key stays outside trainer custody.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest, new as hmac_new
from math import comb
import json
from pathlib import Path
import re
import sqlite3
from typing import Mapping

from .evaluation import score_response
from .maze import Challenge


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ANSWER = re.compile(r"(?:0|-?[1-9][0-9]{0,11})\Z", re.ASCII)
_SPLITS = ("heldout", "transfer", "retention")
_ARMS = ("baseline", "zero_update", "correct_labels", "corrupted_labels")
MAX_CASES = 512
MIN_HELDOUT = 16
MIN_TRANSFER = 16
MIN_RETENTION = 8
MIN_MASTERY = 0.75
MIN_GAIN = 0.10
MAX_RETENTION_DROP = 0.10
# Three simultaneous comparator tests: conservative Bonferroni family-wise bound.
MAX_P_VALUE = 0.05 / 3


def _identifier(value: object, label: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


def _digest(value: object, label: str) -> None:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


@dataclass(frozen=True)
class ExamCase:
    case_id: str
    split: str
    skill: str
    prompt: str
    answer: str  # private; NEVER serialize this object for the learner
    structure_key: str

    def validate(self) -> None:
        _identifier(self.case_id, "case ID")
        _identifier(self.skill, "skill")
        _identifier(self.structure_key, "structure key")
        if self.split not in _SPLITS:
            raise ValueError("unknown evaluator split")
        if type(self.prompt) is not str or not 1 <= len(self.prompt) <= 8192:
            raise ValueError("invalid private exam prompt")
        if type(self.answer) is not str or _ANSWER.fullmatch(self.answer) is None:
            raise ValueError("invalid private oracle answer")


@dataclass(frozen=True)
class TrainingExposure:
    """Evaluator-trusted per-case training exposures, in matching tuple order.

    The three tuples must have identical lengths. A structure key is mandatory
    for every exposed TRAIN case; omitting the list cannot establish transfer
    novelty. These declarations still require external provenance attestation.
    """

    case_ids: tuple[str, ...]
    prompt_sha256: tuple[str, ...]
    structure_keys: tuple[str, ...]

    def validate(self) -> None:
        for field, values, check in (
            ("training case", self.case_ids, _identifier),
            ("training prompt", self.prompt_sha256, _digest),
            ("training structure", self.structure_keys, _identifier),
        ):
            if type(values) is not tuple or len(values) > 4096:
                raise ValueError(f"invalid or duplicate {field} exposure")
            for value in values:
                check(value, field)
            if field != "training structure" and len(values) != len(set(values)):
                raise ValueError(f"invalid or duplicate {field} exposure")
        if (not self.case_ids or len(self.case_ids) != len(self.prompt_sha256)
                or len(self.case_ids) != len(self.structure_keys)):
            raise ValueError("incomplete per-case training exposure")


@dataclass(frozen=True)
class ArmRun:
    """Untrusted execution metadata plus full responses for custodian scoring."""

    name: str
    base_sha256: str
    initial_adapter_sha256: str
    final_adapter_sha256: str
    optimizer_updates: int
    compute_steps: int
    visited_case_ids: tuple[str, ...]
    responses: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        if self.name not in _ARMS:
            raise ValueError("unknown evaluation arm")
        for field in ("base_sha256", "initial_adapter_sha256", "final_adapter_sha256"):
            _digest(getattr(self, field), field)
        if (type(self.optimizer_updates) is not int or type(self.compute_steps) is not int
                or not 0 <= self.optimizer_updates <= self.compute_steps <= 100000):
            raise ValueError("invalid reported optimizer/compute steps")
        if type(self.visited_case_ids) is not tuple or len(self.visited_case_ids) > 100000:
            raise ValueError("invalid training visit sequence")
        for case_id in self.visited_case_ids:
            _identifier(case_id, "training visit ID")
        if type(self.responses) is not tuple or len(self.responses) > MAX_CASES:
            raise ValueError("invalid predictions")
        for trial in self.responses:
            if (type(trial) is not tuple or len(trial) != 2
                    or type(trial[1]) is not str or len(trial[1]) > 8192):
                raise ValueError("invalid predictions")
            _identifier(trial[0], "prediction case ID")


def predictions_commitment(responses: tuple[tuple[str, str], ...]) -> str:
    """Bind an inference receipt to exact ordered case IDs and output bytes.

    This is an integrity digest, NOT evidence the named model generated them.
    """
    if type(responses) is not tuple or len(responses) > MAX_CASES:
        raise ValueError("invalid prediction claim")
    for trial in responses:
        if (type(trial) is not tuple or len(trial) != 2
                or type(trial[1]) is not str or len(trial[1]) > 8192):
            raise ValueError("invalid prediction claim")
        _identifier(trial[0], "prediction case ID")
    payload = json.dumps(responses, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(b"labyrinth.exam-predictions.v1\0" + payload).hexdigest()


@dataclass(frozen=True)
class InferenceClaim:
    """Untrusted metadata for an examiner to compare with EXTERNAL receipts.

    ``process_id`` and ``inference_report_sha256`` are declarations, not proof
    of fresh execution or authentic report bytes. Never treat passing source
    validation as verification of independent inference provenance.
    """

    arm_name: str
    process_id: str
    adapter_sha256: str
    predictions_sha256: str
    inference_report_sha256: str

    def validate(self) -> None:
        if self.arm_name not in _ARMS:
            raise ValueError("unknown inference claim arm")
        _identifier(self.process_id, "fresh process claim")
        for field in ("adapter_sha256", "predictions_sha256", "inference_report_sha256"):
            _digest(getattr(self, field), field)


def _custodian_db_file(value: str | Path) -> str:
    """Reject ephemeral SQLite names, aliases and noncanonical file targets.

    This check is not a substitute for an examiner-controlled, persistent,
    access-restricted directory: path swaps between checking and SQLite open
    remain possible if an attacker can modify that directory (TOCTOU).
    """
    if not isinstance(value, (str, Path)):
        raise ValueError("custodian requires a stable file-backed SQLite path")
    raw = str(value)
    if (not raw or "\x00" in raw or raw.lower().startswith("file:")
            or raw == ":memory:"):
        raise ValueError("custodian requires a stable file-backed SQLite path")
    path = Path(raw)
    if not path.is_absolute() or not path.name:
        raise ValueError("custodian requires a stable file-backed SQLite path")
    try:
        canonical = path.resolve(strict=False)
    except (RuntimeError, OSError) as exc:
        raise ValueError("custodian requires a canonical SQLite file path") from exc
    components = (path, *path.parents)
    if (path != canonical or any(
            component.is_symlink()
            or getattr(component, "is_junction", lambda: False)()
            for component in components)):
        raise ValueError("custodian requires a canonical SQLite file path")
    if path.exists() and not path.is_file():
        raise ValueError("custodian requires a regular SQLite file path")
    return str(path)


@dataclass(frozen=True)
class HoldoutVerdict:
    status: str  # PROVISIONAL, FAIL or INCONCLUSIVE; never formal graduation
    reasons: tuple[str, ...]
    candidate_accuracy: float
    paired_p_values: dict[str, float]
    retention_drop: float | None
    graduation_authorized: bool = False
    evidence_authenticated: bool = False


def case_commitment(cases: tuple[ExamCase, ...]) -> str:
    """Commit to *public question identity/content*, not guessable secret gold."""
    if type(cases) is not tuple or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError("invalid exam case set")
    for case in cases:
        if type(case) is not ExamCase:
            raise ValueError("invalid exam case")
        case.validate()
    public = [(case.case_id, case.split, case.skill,
               sha256(case.prompt.encode("utf-8")).hexdigest(), case.structure_key)
              for case in sorted(cases, key=lambda item: item.case_id)]
    payload = json.dumps(public, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(b"labyrinth.evaluator-public-manifest.v1\0" + payload).hexdigest()


def private_gold_seal(cases: tuple[ExamCase, ...], custodian_key: bytes) -> str:
    """Freeze answers without publishing a dictionary-attackable answer hash.

    The key must be examiner-held *before* training and never written to model
    artifacts, reports, prompts, logs or source control.
    """
    case_commitment(cases)
    if type(custodian_key) is not bytes or len(custodian_key) < 32:
        raise ValueError("an examiner-held 32-byte or longer HMAC key is required")
    private = [(case.case_id, case.answer)
               for case in sorted(cases, key=lambda item: item.case_id)]
    payload = json.dumps(private, separators=(",", ":")).encode("utf-8")
    return hmac_new(custodian_key, b"labyrinth.evaluator-gold.v1\0" + payload, sha256).hexdigest()


def _exact_pair_p(candidate: list[bool], comparator: list[bool]) -> float:
    """Exact one-sided paired sign test; ties do not add evidence."""
    wins = sum(a and not b for a, b in zip(candidate, comparator))
    losses = sum(b and not a for a, b in zip(candidate, comparator))
    discordant = wins + losses
    if not discordant:
        return 1.0
    return sum(comb(discordant, i) for i in range(wins, discordant + 1)) / (2 ** discordant)


def evaluate_holdout(
    cases: tuple[ExamCase, ...],
    evaluator_precommitment: str,
    exposure: TrainingExposure,
    arms: tuple[ArmRun, ...],
    retention_before: Mapping[str, str],
    *,
    examiner_gold_seal: str,
    examiner_key: bytes,
) -> HoldoutVerdict:
    """Score independently held inputs and reject unbalanced negative controls.

    Both commitments, the HMAC key and ``retention_before`` MUST originate
    from evaluator-controlled state predating submission. Trainer-provided
    values could be fabricated. No value permits a Minotaur graduation.
    """
    if type(cases) is not tuple or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError("invalid exam cases")
    case_commitment(cases)  # validate cases before accessing any answer
    ids = [case.case_id for case in cases]
    hashes = [sha256(case.prompt.encode("utf-8")).hexdigest() for case in cases]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("duplicate exam case ID or prompt")
    _digest(evaluator_precommitment, "evaluator precommitment")
    if case_commitment(cases) != evaluator_precommitment:
        raise ValueError("exam changed since evaluator precommit")
    _digest(examiner_gold_seal, "examiner gold seal")
    if not compare_digest(private_gold_seal(cases, examiner_key), examiner_gold_seal):
        raise ValueError("private oracle gold changed since examiner seal")
    if type(exposure) is not TrainingExposure:
        raise ValueError("invalid training exposure")
    exposure.validate()
    if (set(ids) & set(exposure.case_ids)
            or set(hashes) & set(exposure.prompt_sha256)
            or {case.structure_key for case in cases if case.split == "transfer"}
            & set(exposure.structure_keys)):
        raise ValueError("exam/training leakage by case, prompt or transfer structure")

    if (type(arms) is not tuple or len(arms) != len(_ARMS)
            or any(type(arm) is not ArmRun for arm in arms)):
        raise ValueError("all four control arms required")
    for arm in arms:
        arm.validate()
    by_arm = {arm.name: arm for arm in arms}
    if set(by_arm) != set(_ARMS) or len(by_arm) != len(arms):
        raise ValueError("duplicate or missing control arm")
    bases = {arm.base_sha256 for arm in arms}
    starts = {arm.initial_adapter_sha256 for arm in arms}
    if len(bases) != 1 or len(starts) != 1:
        raise ValueError("control arms must share the identical base and initial adapter")
    baseline, sham, candidate, corrupt = (by_arm[name] for name in _ARMS)
    if (baseline.optimizer_updates != 0 or baseline.compute_steps != 0
            or baseline.visited_case_ids or baseline.final_adapter_sha256 != baseline.initial_adapter_sha256):
        raise ValueError("baseline must be frozen without training")
    if (sham.optimizer_updates != 0 or sham.final_adapter_sha256 != sham.initial_adapter_sha256
            or candidate.optimizer_updates < 1 or corrupt.optimizer_updates < 1
            or candidate.optimizer_updates != corrupt.optimizer_updates
            or sham.compute_steps != candidate.compute_steps
            or corrupt.compute_steps != candidate.compute_steps
            or not candidate.visited_case_ids
            or candidate.visited_case_ids != sham.visited_case_ids
            or candidate.visited_case_ids != corrupt.visited_case_ids
            or len(candidate.visited_case_ids) != candidate.compute_steps
            or any(case_id not in exposure.case_ids for case_id in candidate.visited_case_ids)):
        raise ValueError("control arms lack matched training exposure and compute")
    if (candidate.final_adapter_sha256 == candidate.initial_adapter_sha256
            or corrupt.final_adapter_sha256 == corrupt.initial_adapter_sha256):
        raise ValueError("trained candidate/corrupted adapter was not updated")
    if candidate.final_adapter_sha256 == corrupt.final_adapter_sha256:
        raise ValueError("correct/corrupted label arms need distinct final parameters")

    case_ids = set(ids)
    score_by_arm: dict[str, dict[str, bool]] = {}
    for name, arm in by_arm.items():
        submitted_ids = [case_id for case_id, _ in arm.responses]
        if len(submitted_ids) != len(set(submitted_ids)) or set(submitted_ids) != case_ids:
            raise ValueError(f"{name} predictions missing, duplicate or foreign exam IDs")
        predictions = dict(arm.responses)
        score_by_arm[name] = {
            case.case_id: score_response(
                Challenge(case.case_id, "independent", 0, case.split, case.prompt, case.answer),
                predictions[case.case_id],
            ).correct
            for case in cases
        }

    sections = {split: [case for case in cases if case.split == split] for split in _SPLITS}
    # Retention may contain earlier mastered skills outside the current target.
    skills = {case.skill for case in sections["heldout"] + sections["transfer"]}
    reasons: list[str] = []
    missing_coverage = False
    for split, required in (("heldout", MIN_HELDOUT), ("transfer", MIN_TRANSFER),
                            ("retention", MIN_RETENTION)):
        if len(sections[split]) < required:
            missing_coverage = True
            reasons.append(f"insufficient {split} case count")
    for skill in skills:
        if any(sum(case.skill == skill for case in sections[split]) < 4
               for split in ("heldout", "transfer")):
            missing_coverage = True
            reasons.append(f"insufficient heldout/transfer coverage for {skill}")
        skill_transfer = [case for case in sections["transfer"] if case.skill == skill]
        if len({case.structure_key for case in skill_transfer}) < 4:
            missing_coverage = True
            reasons.append(f"insufficient novel transfer structures for {skill}")

    def accuracy(name: str, sample: list[ExamCase]) -> float:
        return sum(score_by_arm[name][case.case_id] for case in sample) / len(sample) if sample else 0.0

    candidate_accuracy = accuracy("correct_labels", sections["heldout"] + sections["transfer"])
    pairs: dict[str, float] = {}
    main_cases = sections["heldout"] + sections["transfer"]
    for control in ("baseline", "zero_update", "corrupted_labels"):
        pairs[control] = _exact_pair_p(
            [score_by_arm["correct_labels"][case.case_id] for case in main_cases],
            [score_by_arm[control][case.case_id] for case in main_cases],
        ) if main_cases else 1.0
        if pairs[control] > MAX_P_VALUE:
            reasons.append(f"paired benefit over {control} is not established")
        for split in ("heldout", "transfer"):
            if (accuracy("correct_labels", sections[split])
                    - accuracy(control, sections[split]) < MIN_GAIN):
                reasons.append(f"{split} gain over {control} is below threshold")
    for split in ("heldout", "transfer"):
        if accuracy("correct_labels", sections[split]) < MIN_MASTERY:
            reasons.append(f"{split} candidate mastery is below threshold")
        for skill in skills:
            skill_cases = [case for case in sections[split] if case.skill == skill]
            if not skill_cases:
                continue  # Inconclusive missing-skill coverage is recorded above.
            if accuracy("correct_labels", skill_cases) < MIN_MASTERY:
                reasons.append(f"{split} mastery below threshold for {skill}")
            for control in ("baseline", "zero_update", "corrupted_labels"):
                if (accuracy("correct_labels", skill_cases)
                        - accuracy(control, skill_cases) < MIN_GAIN):
                    reasons.append(f"{split} gain over {control} below threshold for {skill}")

    retention_drop: float | None = None
    required_retention_ids = {case.case_id for case in sections["retention"]}
    if (not isinstance(retention_before, Mapping)
            or set(retention_before) != required_retention_ids
            or any(type(response) is not str for response in retention_before.values())):
        missing_coverage = True
        reasons.append("independently captured predecessor retention responses absent")
    elif sections["retention"]:
        before_scores = {
            case.case_id: score_response(
                Challenge(case.case_id, "independent", 0, case.split, case.prompt, case.answer),
                retention_before[case.case_id],
            ).correct for case in sections["retention"]
        }
        before = sum(before_scores.values()) / len(sections["retention"])
        after = accuracy("correct_labels", sections["retention"])
        retention_drop = before - after
        if before < MIN_MASTERY:
            reasons.append("predecessor did not demonstrate retained-skill mastery")
        if after < MIN_MASTERY or retention_drop > MAX_RETENTION_DROP:
            reasons.append("regenerated learner forgot previously mastered behavior")
        for skill in {case.skill for case in sections["retention"]}:
            prior_skill_cases = [case for case in sections["retention"] if case.skill == skill]
            if len(prior_skill_cases) < 4:
                missing_coverage = True
                reasons.append(f"insufficient retention coverage for {skill}")
                continue
            skill_before = sum(before_scores[case.case_id] for case in prior_skill_cases) / len(prior_skill_cases)
            skill_after = accuracy("correct_labels", prior_skill_cases)
            if skill_before < MIN_MASTERY:
                reasons.append(f"previously mastered {skill} not established")
            if skill_after < MIN_MASTERY or skill_before - skill_after > MAX_RETENTION_DROP:
                reasons.append(f"retention loss for previously mastered {skill}")

    status = "INCONCLUSIVE" if missing_coverage else "FAIL" if reasons else "PROVISIONAL"
    return HoldoutVerdict(status, tuple(reasons), candidate_accuracy, pairs, retention_drop)


def evaluate_one_shot_holdout(
    cases: tuple[ExamCase, ...],
    evaluator_precommitment: str,
    exposure: TrainingExposure,
    arms: tuple[ArmRun, ...],
    retention_before: Mapping[str, str],
    *,
    examiner_gold_seal: str,
    examiner_key: bytes,
    attempt_id: str,
    inference_claims: tuple[InferenceClaim, ...],
    custodian_db_path: str | Path,
) -> HoldoutVerdict:
    """Reserve one attempt *atomically*, then evaluate, even if scoring fails.

    The DB MUST be owned by an independent examiner, protected from trainer
    writes, persistent across process restarts and shared by all exam workers.
    The unique public-exam key prevents a second submission under a new ID.
    An unavailable or permission-denied DB raises; no in-memory fallback.

    The model identity/process/report fields are still merely untrusted claims.
    Independent inference observation, private-bank custody and source-to-model
    traceability remain EXTERNAL requirements. This path cannot grant graduation.
    Canonical path checking does not defeat path-swap races in an attacker-
    writable parent directory; custody of filesystem permissions is mandatory.
    """
    _identifier(attempt_id, "exam attempt ID")
    _digest(evaluator_precommitment, "evaluator precommitment")
    if case_commitment(cases) != evaluator_precommitment:
        raise ValueError("exam changed since evaluator precommit")
    _digest(examiner_gold_seal, "examiner gold seal")
    if not compare_digest(private_gold_seal(cases, examiner_key), examiner_gold_seal):
        raise ValueError("private oracle gold changed since examiner seal")
    if type(exposure) is not TrainingExposure:
        raise ValueError("invalid training exposure")
    exposure.validate()
    if (type(arms) is not tuple or len(arms) != len(_ARMS)
            or any(type(arm) is not ArmRun for arm in arms)):
        raise ValueError("all four control arms required")
    if (type(inference_claims) is not tuple or len(inference_claims) != len(_ARMS)
            or any(type(claim) is not InferenceClaim for claim in inference_claims)):
        raise ValueError("all four inference claims required")
    for arm in arms:
        arm.validate()
    for claim in inference_claims:
        claim.validate()
    by_arm = {arm.name: arm for arm in arms}
    by_claim = {claim.arm_name: claim for claim in inference_claims}
    if len(by_arm) != len(_ARMS) or len(by_claim) != len(_ARMS):
        raise ValueError("duplicate control arm or inference claim")
    if len({claim.process_id for claim in inference_claims}) != len(_ARMS):
        raise ValueError("fresh process claims must be distinct")
    for name in _ARMS:
        arm, claim = by_arm[name], by_claim[name]
        if claim.adapter_sha256 != arm.final_adapter_sha256:
            raise ValueError(f"{name} adapter claim does not match submitted run")
        if claim.predictions_sha256 != predictions_commitment(arm.responses):
            raise ValueError(f"{name} prediction claim does not match submitted run")
    if not isinstance(retention_before, Mapping):
        # A malformed predecessor observation must not be serialized or scored.
        raise ValueError("invalid predecessor retention responses")
    if any(type(key) is not str or type(value) is not str
           for key, value in retention_before.items()):
        raise ValueError("invalid predecessor retention responses")

    # Only public exam identity and submitted bytes enter the reservation DB.
    # In particular, private gold and the examiner HMAC key do not.
    submission = {
        "exam": evaluator_precommitment,
        "attempt": attempt_id,
        "exposure": exposure.__dict__,
        "arms": [arm.__dict__ for arm in arms],
        "retention_before": sorted(retention_before.items()),
        "inference_claims": [claim.__dict__ for claim in inference_claims],
    }
    payload = json.dumps(submission, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    submission_sha256 = sha256(b"labyrinth.exam-submission.v1\0" + payload).hexdigest()

    stable_db_path = _custodian_db_file(custodian_db_path)

    # A custodian-owned SQLite file gives a real unique-key reservation across
    # threads/processes. If a later test fails, the exam remains consumed to
    # prevent repeated private-score probing. Never auto-clear this record.
    with closing(sqlite3.connect(stable_db_path, timeout=10.0)) as connection:
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS exam_attempts ("
                    "exam_commitment TEXT PRIMARY KEY, "
                    "attempt_id TEXT NOT NULL UNIQUE, "
                    "submission_sha256 TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO exam_attempts "
                    "(exam_commitment, attempt_id, submission_sha256) VALUES (?, ?, ?)",
                    (evaluator_precommitment, attempt_id, submission_sha256),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("exam already consumed by a prior submission") from exc

    return evaluate_holdout(
        cases, evaluator_precommitment, exposure, arms, retention_before,
        examiner_gold_seal=examiner_gold_seal, examiner_key=examiner_key,
    )
