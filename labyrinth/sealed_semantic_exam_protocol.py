"""Evaluator-only *candidate* for a sealed semantic transfer/retention exam.

The 20-family/400-situation shape and five-arm controls are a PROPOSAL, not
Patrick's approved graduation thresholds. A hosting evaluator must own the
private cases, signing keys, clock, model inference, semantic judgments and
durable one-shot ledger. In-process HMAC checks and a local replay guard
cannot establish that such a host or examination exists.

No private prompt, gold label, case ID, or semantic key is included in the
receipts. Evaluator-side scores include keyed, pseudonymous family aggregates
to detect concentrated forgetting; these MUST NOT be returned to the trainee
as tuning feedback. A retention situation has two parallel before/after
prompts; transfer has only an after prompt. This means 400 *situations*, not
necessarily 400 raw prompt strings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import hmac
import json
import re
import threading
from typing import Any

from .retention_family_registry import (
    canonical_family_id, canonical_prompt_alias_sha256,
)
from .arithmetic_semantic_case_key import (
    UnsupportedArithmeticCase, arithmetic_semantic_case_key,
)


SCHEMA = "labyrinth.sealed-semantic-exam-proposal.v1"
ARMS = ("correct", "zero_update", "inverted", "zero_lr", "randomized_labels")
_BANK_DOMAIN = b"labyrinth.sealed-semantic-bank.v1\0"
_PLAN_DOMAIN = b"labyrinth.sealed-semantic-plan.v1\0"
_SCORE_DOMAIN = b"labyrinth.sealed-semantic-score.v1\0"
_SUBMISSION_DOMAIN = b"labyrinth.sealed-semantic-submission.v1\0"
_TRAINER_KEY_DOMAIN = b"labyrinth.sealed-semantic-trainer-verifier.v1\0"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class SealedExamError(ValueError):
    """An examination prerequisite or sealed evidence check failed."""


@dataclass(frozen=True)
class SealedSituation:
    """Keep these private on the evaluator host; never send to the trainer."""

    case_id: str
    family_id: str
    split: str  # transfer or retention
    after_prompt: str
    gold_label: int
    semantic_key: str
    before_prompt: str | None = None


@dataclass(frozen=True)
class TrainingOverlapInventory:
    """Externally frozen TRAIN inventories, not a model's self-report."""

    family_ids: tuple[str, ...]
    prompt_alias_sha256s: tuple[str, ...]
    semantic_keys: tuple[str, ...]
    # Complete, externally validated TRAIN roster for the narrow, parsed
    # single-addition grammar. An empty roster is NOT proof of no arithmetic
    # in training; evaluator must separately certify completeness.
    arithmetic_semantic_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SignedSealedPlan:
    public_bytes: bytes
    signature_hex: str
    graduation_authorized: bool = False
    independent_evaluator_host_verified: bool = False


@dataclass(frozen=True)
class PredictionSubmission:
    arm: str
    seed: int
    steps_per_cycle: int
    run_id: str
    initial_trainable_sha256: str
    training_visits_sha256: str
    candidate_artifact_sha256: str
    predictions: tuple[tuple[str, int], ...]
    retention_before_predictions: tuple[tuple[str, int], ...]
    prior_skill_checkpoint_sha256: str
    signature_hex: str = ""


@dataclass(frozen=True)
class SignedAggregateScore:
    receipt_bytes: bytes
    signature_hex: str
    graduation_authorized: bool = False
    physical_execution_authenticated: bool = False
    independent_evaluator_host_verified: bool = False


class LocalOneShotLedger:
    """Thread-safe in-process veto; NEW INSTANCE/SNAPSHOT CAN REPLAY."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._used: set[str] = set()

    def consume(self, event_id: str) -> None:
        _identifier(event_id, "exam event")
        with self._lock:
            if event_id in self._used:
                raise SealedExamError("exam event already consumed locally")
            self._used.add(event_id)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise SealedExamError(f"invalid {name} digest")
    return value


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or _IDENT.fullmatch(value) is None:
        raise SealedExamError(f"invalid {name}")
    return value


def _key(value: Any, name: str) -> bytes:
    if type(value) is not bytes or len(value) < 32:
        raise SealedExamError(f"independently held {name} key required")
    return value


def _keys(*keys: bytes) -> None:
    for index, key in enumerate(keys):
        _key(key, f"key-{index}")
    if any(hmac.compare_digest(first, second)
           for index, first in enumerate(keys)
           for second in keys[index + 1:]):
        raise SealedExamError("distinct evaluator/bank/trainer keys required")


def training_inventory_sha256(inventory: TrainingOverlapInventory) -> str:
    """A digest only; external precommit must prevent changing this inventory."""
    if type(inventory) is not TrainingOverlapInventory:
        raise SealedExamError("typed training overlap inventory required")
    for name, values in (("families", inventory.family_ids),
                         ("prompts", inventory.prompt_alias_sha256s),
                         ("semantics", inventory.semantic_keys)):
        if type(values) is not tuple or not values:
            raise SealedExamError(f"nonempty TRAIN {name} inventory required")
        if len(set(values)) != len(values):
            raise SealedExamError(f"duplicate TRAIN {name} entry")
    if any(canonical_family_id(value) != value for value in inventory.family_ids):
        raise SealedExamError("TRAIN family not canonical")
    for value in inventory.prompt_alias_sha256s:
        _digest(value, "TRAIN prompt")
    for value in inventory.semantic_keys:
        _identifier(value, "TRAIN semantic key")
    if (type(inventory.arithmetic_semantic_keys) is not tuple
            or len(set(inventory.arithmetic_semantic_keys))
            != len(inventory.arithmetic_semantic_keys)):
        raise SealedExamError("malformed TRAIN arithmetic semantic inventory")
    for value in inventory.arithmetic_semantic_keys:
        if (type(value) is not str
                or re.fullmatch(r"add:(0|[1-9][0-9]?):(0|[1-9][0-9]?)", value)
                is None):
            raise SealedExamError("invalid TRAIN arithmetic semantic identity")
        _, first, second = value.split(":")
        if int(first) > int(second):
            raise SealedExamError("TRAIN arithmetic semantic identity not canonical")
    return sha256(_PLAN_DOMAIN + _canonical({
        "families": sorted(inventory.family_ids),
        "prompts": sorted(inventory.prompt_alias_sha256s),
        "semantics": sorted(inventory.semantic_keys),
        "narrow_arithmetic_semantics": sorted(inventory.arithmetic_semantic_keys),
    })).hexdigest()


def _validate_bank(cases: tuple[SealedSituation, ...],
                   inventory: TrainingOverlapInventory) -> tuple[SealedSituation, ...]:
    if (type(cases) is not tuple or len(cases) != 400
            or any(type(case) is not SealedSituation for case in cases)):
        raise SealedExamError("require exactly 400 privately held situations")
    seen_ids: set[str] = set()
    seen_semantics: set[str] = set(inventory.semantic_keys)
    seen_prompts: set[str] = set(inventory.prompt_alias_sha256s)
    seen_narrow_arithmetic: dict[str, str] = {}
    family_split: dict[str, str] = {}
    families: dict[str, list[SealedSituation]] = {}
    for case in cases:
        _identifier(case.case_id, "private case")
        _identifier(case.semantic_key, "semantic key")
        try:
            family = canonical_family_id(case.family_id)
        except ValueError as exc:
            raise SealedExamError("invalid semantic family") from exc
        if (family != case.family_id or family in inventory.family_ids
                or case.case_id in seen_ids or case.semantic_key in seen_semantics
                or type(case.gold_label) is not int
                or case.gold_label not in (0, 1)
                or case.split not in ("transfer", "retention")):
            raise SealedExamError("duplicate, overlapping or malformed private case")
        seen_ids.add(case.case_id)
        seen_semantics.add(case.semantic_key)
        if (type(case.after_prompt) is not str
                or not 1 <= len(case.after_prompt) <= 4096):
            raise SealedExamError("invalid after prompt")
        texts = (case.after_prompt,)
        if case.split == "retention":
            if (type(case.before_prompt) is not str
                    or not 1 <= len(case.before_prompt) <= 4096):
                raise SealedExamError("parallel prior-skill retention form required")
            texts += (case.before_prompt,)
        elif case.before_prompt is not None:
            raise SealedExamError("transfer cannot expose a prior-skill form")
        case_narrow_keys: list[str | None] = []
        for text in texts:
            try:
                commitment = canonical_prompt_alias_sha256(text)
            except ValueError as exc:
                raise SealedExamError("invalid private prompt") from exc
            if commitment in seen_prompts:
                raise SealedExamError("private prompt overlaps TRAIN or another form")
            seen_prompts.add(commitment)
            # This only covers the existing strictly parsed single-addition
            # grammar; every other semantic family still requires an external
            # evaluator's human/independent equivalence judgment.
            try:
                narrow_key = arithmetic_semantic_case_key(text)
            except UnsupportedArithmeticCase:
                narrow_key = None
            case_narrow_keys.append(narrow_key)
            if narrow_key is not None:
                if narrow_key in inventory.arithmetic_semantic_keys:
                    raise SealedExamError(
                        "private arithmetic semantics overlap frozen TRAIN roster")
                if (narrow_key in seen_narrow_arithmetic
                        and seen_narrow_arithmetic[narrow_key] != case.case_id):
                    raise SealedExamError("private arithmetic semantic alias reused")
                # Retention before/after forms are deliberately two
                # expressions of ONE situation. Their equivalence is
                # expected; cross-situation reuse is not.
                seen_narrow_arithmetic[narrow_key] = case.case_id
        if (case.split == "retention" and any(case_narrow_keys)
                and (None in case_narrow_keys
                     or len(set(case_narrow_keys)) != 1)):
            raise SealedExamError(
                "prior-skill before/after arithmetic semantics differ or unverified")
        if family in family_split and family_split[family] != case.split:
            raise SealedExamError("semantic family crossed private partitions")
        family_split[family] = case.split
        families.setdefault(family, []).append(case)
    if len(families) != 20 or sorted(len(group) for group in families.values()) != [20] * 20:
        raise SealedExamError("20 semantic families with 20 cases each required")
    if (sum(split == "transfer" for split in family_split.values()) != 10
            or sum(split == "retention" for split in family_split.values()) != 10):
        raise SealedExamError("ten novel-transfer and ten prior-skill families required")
    for group in families.values():
        if sum(case.gold_label for case in group) != 10:
            raise SealedExamError("each semantic family must be binary balanced")
    return tuple(sorted(cases, key=lambda row: row.case_id))


def _bank_commitment(cases: tuple[SealedSituation, ...], bank_key: bytes) -> str:
    private = [{"id": case.case_id, "family": case.family_id,
                "split": case.split, "after": case.after_prompt,
                "before": case.before_prompt, "gold": case.gold_label,
                "semantic_key": case.semantic_key} for case in cases]
    return hmac.new(bank_key, _BANK_DOMAIN + _canonical(private), sha256).hexdigest()


def build_sealed_plan(
    *, cases: tuple[SealedSituation, ...],
    training_inventory: TrainingOverlapInventory,
    externally_pinned_training_inventory_sha256: str,
    exam_id: str, event_id: str, objective_id: str,
    scope_sha256: str, pretrained_model_sha256: str, tokenizer_sha256: str,
    optimizer_config_sha256: str, source_sha256: str,
    prior_skill_checkpoint_sha256: str,
    seeds: tuple[int, ...], step_budgets: tuple[int, ...],
    evaluator_key: bytes, bank_key: bytes, trainer_key: bytes,
) -> SignedSealedPlan:
    """Plan commitment has no private data or independent timestamp claim."""
    _keys(evaluator_key, bank_key, trainer_key)
    for value, name in ((exam_id, "exam"), (event_id, "event"),
                        (objective_id, "objective")):
        _identifier(value, name)
    for value, name in ((scope_sha256, "scope"),
                        (pretrained_model_sha256, "base model"),
                        (tokenizer_sha256, "tokenizer"),
                        (optimizer_config_sha256, "optimizer"),
                        (source_sha256, "source"),
                        (prior_skill_checkpoint_sha256, "prior-skill checkpoint"),
                        (externally_pinned_training_inventory_sha256, "TRAIN pin")):
        _digest(value, name)
    if (training_inventory_sha256(training_inventory)
            != externally_pinned_training_inventory_sha256):
        raise SealedExamError("TRAIN overlap inventory differs from evaluator precommit")
    if (type(seeds) is not tuple or not 5 <= len(seeds) <= 10
            or any(type(seed) is not int or not 0 <= seed < 2**31
                   for seed in seeds) or len(set(seeds)) != len(seeds)
            or type(step_budgets) is not tuple
            or not 1 <= len(step_budgets) <= 2
            or any(type(step) is not int or not 1 <= step <= 64
                   for step in step_budgets)
            or len(set(step_budgets)) != len(step_budgets)):
        raise SealedExamError("five to ten fixed seeds and one/two bounded budgets required")
    frozen = _validate_bank(cases, training_inventory)
    body = {
        "schema": SCHEMA, "exam_id": exam_id, "event_id": event_id,
        "objective_id": objective_id, "scope_sha256": scope_sha256,
        "pretrained_model_sha256": pretrained_model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "optimizer_config_sha256": optimizer_config_sha256,
        "source_sha256": source_sha256,
        "prior_skill_checkpoint_sha256": prior_skill_checkpoint_sha256,
        "training_inventory_sha256": externally_pinned_training_inventory_sha256,
        "trainer_key_commitment": hmac.new(
            evaluator_key, _TRAINER_KEY_DOMAIN + trainer_key,
            sha256).hexdigest(),
        "private_bank_commitment": _bank_commitment(frozen, bank_key),
        "family_count": 20, "situations_per_family": 20,
        "transfer_situations": 200, "retention_situations": 200,
        "case_count": 400, "seeds": list(seeds),
        "step_budgets": list(step_budgets), "arms": list(ARMS),
        "decision_policy_status": "PROPOSED_NOT_APPROVED",
        "proposal_primary_margin_pp": 10,
        "proposal_retention_aggregate_loss_pp": 5,
        "proposal_retention_family_loss_pp": 10,
        "proposal_family_stratified_method":
            "PAIRED_FAMILY_BOOTSTRAP_MULTIPLE_COMPARISON_PENDING_IMPLEMENTATION",
        "decision_policy_qualified": False,
        "status": "EVALUATOR_SOURCE_PRECOMMIT_ONLY",
        "external_timestamp_verified": False,
        "host_custody_verified": False,
        "independent_model_execution_verified": False,
        "graduation_authorized": False,
    }
    public = _canonical(body)
    signature = hmac.new(evaluator_key, _PLAN_DOMAIN + public, sha256).hexdigest()
    return SignedSealedPlan(public, signature)


def _verified_plan(plan: SignedSealedPlan, *, evaluator_key: bytes) -> dict[str, Any]:
    _key(evaluator_key, "evaluator")
    if (type(plan) is not SignedSealedPlan
            or plan.graduation_authorized is not False
            or plan.independent_evaluator_host_verified is not False
            or type(plan.public_bytes) is not bytes or len(plan.public_bytes) > 16_384):
        raise SealedExamError("untrusted sealed plan")
    try:
        body = json.loads(plan.public_bytes)
        if _canonical(body) != plan.public_bytes:
            raise ValueError("noncanonical")
    except (UnicodeError, ValueError, TypeError) as exc:
        raise SealedExamError("invalid sealed plan encoding") from exc
    expected = hmac.new(evaluator_key, _PLAN_DOMAIN + plan.public_bytes,
                        sha256).hexdigest()
    if (type(plan.signature_hex) is not str
            or not hmac.compare_digest(expected, plan.signature_hex)):
        raise SealedExamError("sealed plan signature did not authenticate")
    if (type(body) is not dict or body.get("schema") != SCHEMA
            or body.get("arms") != list(ARMS)
            or body.get("case_count") != 400
            or body.get("family_count") != 20
            or body.get("status") != "EVALUATOR_SOURCE_PRECOMMIT_ONLY"
            or body.get("graduation_authorized") is not False):
        raise SealedExamError("sealed plan shape altered")
    return body


def _submission_payload(row: PredictionSubmission) -> bytes:
    if type(row) is not PredictionSubmission:
        raise SealedExamError("typed trainer submission required")
    return _canonical({
        "schema": SCHEMA + ".trainer-submission.v1",
        "arm": row.arm, "seed": row.seed,
        "steps_per_cycle": row.steps_per_cycle,
        "run_id": row.run_id,
        "initial_trainable_sha256": row.initial_trainable_sha256,
        "training_visits_sha256": row.training_visits_sha256,
        "candidate_artifact_sha256": row.candidate_artifact_sha256,
        "prior_skill_checkpoint_sha256": row.prior_skill_checkpoint_sha256,
        "predictions": row.predictions,
        "retention_before_predictions": row.retention_before_predictions,
    })


def sign_prediction_submission(row: PredictionSubmission, *,
                               trainer_key: bytes) -> PredictionSubmission:
    """Authenticate claimed response bytes, NOT actual model execution."""
    _key(trainer_key, "trainer")
    if row.signature_hex:
        raise SealedExamError("trainer submission already carries a signature")
    digest = hmac.new(trainer_key, _SUBMISSION_DOMAIN + _submission_payload(row),
                      sha256).hexdigest()
    return replace(row, signature_hex=digest)


def score_sealed_exam(
    *, plan: SignedSealedPlan, cases: tuple[SealedSituation, ...],
    submissions: tuple[PredictionSubmission, ...],
    training_inventory: TrainingOverlapInventory,
    evaluator_key: bytes, bank_key: bytes, trainer_key: bytes,
    local_ledger: LocalOneShotLedger,
) -> SignedAggregateScore:
    """One in-process sealed score; aggregate only, never exit authority.

    Every attempt consumes the local event AFTER source preflight. A malicious
    operator could instantiate another ledger or restore a snapshot, so this
    is not a protected final exam or graduate gate.
    """
    _keys(evaluator_key, bank_key, trainer_key)
    body = _verified_plan(plan, evaluator_key=evaluator_key)
    trainer_commitment = hmac.new(
        evaluator_key, _TRAINER_KEY_DOMAIN + trainer_key,
        sha256).hexdigest()
    if not hmac.compare_digest(trainer_commitment,
                               body.get("trainer_key_commitment", "")):
        raise SealedExamError("trainer verifier identity differs from sealed plan")
    if (type(local_ledger) is not LocalOneShotLedger
            or training_inventory_sha256(training_inventory)
            != body["training_inventory_sha256"]):
        raise SealedExamError("untrusted local ledger or TRAIN inventory")
    frozen = _validate_bank(cases, training_inventory)
    if not hmac.compare_digest(_bank_commitment(frozen, bank_key),
                               body["private_bank_commitment"]):
        raise SealedExamError("private bank changed since source precommit")
    if (type(submissions) is not tuple or not submissions
            or any(type(row) is not PredictionSubmission for row in submissions)):
        raise SealedExamError("complete typed blind submissions required")
    expected_jobs = {(seed, steps, arm)
                     for seed in body["seeds"]
                     for steps in body["step_budgets"]
                     for arm in ARMS}
    observed_jobs = {(row.seed, row.steps_per_cycle, row.arm) for row in submissions}
    if (len(submissions) != len(expected_jobs) or observed_jobs != expected_jobs):
        raise SealedExamError("missing, duplicate or posthoc model control arm")
    case_ids = {case.case_id for case in frozen}
    retention_ids = {case.case_id for case in frozen if case.split == "retention"}
    expected_splits = {case.case_id: case for case in frozen}
    per_job: dict[tuple[int, int, str], dict[str, Any]] = {}
    matching: dict[tuple[int, int], set[tuple[str, str]]] = {}
    prior_predictions_by_match: dict[tuple[int, int], tuple[tuple[str, int], ...]] = {}
    seen_run_ids: set[str] = set()
    for row in submissions:
        expected_signature = hmac.new(
            trainer_key, _SUBMISSION_DOMAIN + _submission_payload(row),
            sha256).hexdigest()
        if (type(row.signature_hex) is not str
                or not hmac.compare_digest(expected_signature, row.signature_hex)):
            raise SealedExamError("trainer submission signature not authenticated")
        _identifier(row.run_id, "candidate run")
        if row.run_id in seen_run_ids:
            raise SealedExamError("repeated neural candidate run ID")
        seen_run_ids.add(row.run_id)
        for value, name in ((row.initial_trainable_sha256, "initial weights"),
                            (row.training_visits_sha256, "TRAIN visits"),
                            (row.candidate_artifact_sha256, "candidate artifact"),
                            (row.prior_skill_checkpoint_sha256, "prior-skill checkpoint")):
            _digest(value, name)
        if row.prior_skill_checkpoint_sha256 != body["prior_skill_checkpoint_sha256"]:
            raise SealedExamError("retention before-state is not independently pinned")
        if (type(row.predictions) is not tuple
                or len(row.predictions) != 400
                or any(type(pair) is not tuple or len(pair) != 2
                       or type(pair[0]) is not str
                       or type(pair[1]) is not int or pair[1] not in (0, 1)
                       for pair in row.predictions)
                or {case_id for case_id, _ in row.predictions} != case_ids):
            raise SealedExamError("missing/duplicate or malformed blinded prediction")
        if (type(row.retention_before_predictions) is not tuple
                or len(row.retention_before_predictions) != 200
                or any(type(pair) is not tuple or len(pair) != 2
                       or type(pair[0]) is not str or type(pair[1]) is not int
                       or pair[1] not in (0, 1)
                       for pair in row.retention_before_predictions)
                or {case_id for case_id, _ in row.retention_before_predictions}
                != retention_ids):
            raise SealedExamError("parallel prior-skill before predictions incomplete")
        matching.setdefault((row.seed, row.steps_per_cycle), set()).add(
            (row.initial_trainable_sha256, row.training_visits_sha256))
        match_key = (row.seed, row.steps_per_cycle)
        frozen_before = tuple(sorted(row.retention_before_predictions))
        previous_before = prior_predictions_by_match.setdefault(
            match_key, frozen_before)
        if previous_before != frozen_before:
            raise SealedExamError(
                "matched control arms disagree on prior-skill before predictions")
        counts = {"transfer": [0, 0], "retention": [0, 0]}
        by_family: dict[str, dict[str, int]] = {}
        prior_correct = 0
        for case_id, predicted in row.retention_before_predictions:
            case = expected_splits[case_id]
            correct = int(predicted == case.gold_label)
            prior_correct += correct
            family = by_family.setdefault(case.family_id,
                                          {"before": 0, "after": 0})
            family["before"] += correct
        for case_id, predicted in row.predictions:
            case = expected_splits[case_id]
            counts[case.split][1] += 1
            correct = int(predicted == case.gold_label)
            counts[case.split][0] += correct
            if case.split == "retention":
                by_family[case.family_id]["after"] += correct
        blinded_families = tuple(sorted((
            {"family_token": hmac.new(
                 bank_key, _BANK_DOMAIN + b"family\0" + family_id.encode("ascii"),
                 sha256).hexdigest(),
             "before_correct": values["before"],
             "after_correct": values["after"],
             "total": 20}
            for family_id, values in by_family.items()),
            key=lambda value: value["family_token"]))
        worst_family_drop = max(
            entry["before_correct"] - entry["after_correct"]
            for entry in blinded_families)
        per_job[(row.seed, row.steps_per_cycle, row.arm)] = {
            "seed": row.seed, "steps_per_cycle": row.steps_per_cycle,
            "arm": row.arm, "correct_transfer": counts["transfer"][0],
            "transfer_total": counts["transfer"][1],
            "correct_retention_after": counts["retention"][0],
            "correct_retention_before": prior_correct,
            "retention_total": counts["retention"][1],
            "blinded_retention_families": blinded_families,
            "worst_family_retention_loss_count": worst_family_drop,
            "worst_family_retention_loss_pp": worst_family_drop * 5,
        }
    if any(len(pair_set) != 1 for pair_set in matching.values()):
        raise SealedExamError("controls used different initial weights or TRAIN visits")
    # A scorer-side local event-consumption veto. An independent durable
    # examiner must replace this before any protected one-shot assessment.
    local_ledger.consume(body["event_id"])
    report = {
        "schema": SCHEMA + ".aggregate.v1",
        "exam_id": body["exam_id"], "event_id": body["event_id"],
        "plan_sha256": sha256(plan.public_bytes).hexdigest(),
        "public_aggregate": [per_job[key] for key in sorted(per_job)],
        "private_case_count": 400,
        "proposed_only": True,
        "decision_policy_qualified": False,
        "physical_execution_authenticated": False,
        "host_custody_verified": False,
        "graduation_authorized": False,
        "private_exam_authorizes_graduation": False,
    }
    raw = _canonical(report)
    return SignedAggregateScore(
        raw, hmac.new(evaluator_key, _SCORE_DOMAIN + raw, sha256).hexdigest())
