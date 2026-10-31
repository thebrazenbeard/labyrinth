"""Authenticated *development-only* score -> HF FAIL transition bridge.

The blind evaluator's HMAC evidence and the HF successor's HMAC evidence
deliberately have different schemas and domains. An untrusted caller may not
copy one MAC, relabel it, or declare weaknesses. This bridge authenticates a
scored development receipt, a separately signed candidate file/LoRA lineage,
the independently precommitted public exam roster, and a fixed score policy.
It derives remediation skills from signed aggregate scores and *only then*
signs the already-existing HF development-FAIL contract.

This is an opt-in SOURCE-level adapter. The distinct secrets must be
externally retained by evaluator/lineage custodian/transition authority. A
lineage HMAC authenticates a custodian claim; it does not attest model
inference, honest human custody, power-loss-safe storage, or neural learning.
The scorer explicitly reports model_execution_authenticated=False. The
result grants NO graduation authority, cannot consume an event by itself,
and must still pass run_journey_bound_successor's complete predecessor,
Journey, and one-shot reservation checks before any training side effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import re
from typing import Any

from .blind_evaluation_custody import (
    BlindCustodyError, _PUBLIC_DOMAIN, _canonical, verify_assessment,
)
from .hf_development_transition import (
    SCHEMA as TRANSITION_SCHEMA, _DOMAIN as TRANSITION_DOMAIN,
    DevelopmentCaseRef, development_case_commitment, verify_development_transition,
)
from .hf_resume_bridge import ResumeContext
from .training_scope import TrainingScopeSpec


CANDIDATE_LINEAGE_SCHEMA = "labyrinth.blind-candidate-lineage.v1"
CANDIDATE_LINEAGE_DOMAIN = b"labyrinth.blind-candidate-lineage.v1\0"
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_DIGEST = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
_FIELDS = frozenset({
    "schema", "run_id", "exam_id", "objective_id", "scope_fingerprint", "cycle_index",
    "checkpoint_artifact_sha256", "predecessor_report_sha256",
    "predecessor_lora_value_sha256", "public_exam_sha256",
})


@dataclass(frozen=True)
class PublicAssessmentCase:
    """Evaluator-public metadata. Never contains private answers or prompts."""

    case_id: str
    skill: str
    split: str
    prompt_sha256: str
    structure_key: str


@dataclass(frozen=True)
class SignedRegenerationTransition:
    """Authenticated training-remediation hint, never a graduation token."""

    receipt_bytes: bytes
    signature_hex: str
    development_public_roster: tuple[DevelopmentCaseRef, ...]
    scored_receipt_sha256: str
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False


def _identifier(value: Any, name: str) -> None:
    if type(value) is not str or not _IDENT.fullmatch(value):
        raise ValueError(f"invalid {name}")


def _digest(value: Any, name: str) -> None:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"invalid {name}")


def _key(value: Any, name: str) -> None:
    if type(value) is not bytes or len(value) < 32:
        raise ValueError(f"{name} requires a separate 32+ byte secret")


def _distinct_keys(*keys: bytes) -> None:
    for i, key in enumerate(keys):
        _key(key, "trusted bridge authority")
        if any(hmac.compare_digest(key, other) for other in keys[:i]):
            raise ValueError("bridge authorities must hold distinct secrets")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, value in pairs:
        if field in result:
            raise ValueError("duplicate candidate lineage field")
        result[field] = value
    return result


def _verify_lineage(
    raw: bytes, signature_hex: str, key: bytes, *,
    scope: TrainingScopeSpec, context: ResumeContext, run_id: str, exam_id: str,
    artifact_sha256: str, public_exam_sha256: str,
) -> None:
    if (type(raw) is not bytes or not 0 < len(raw) <= 8192
            or type(signature_hex) is not str or not _DIGEST.fullmatch(signature_hex)):
        raise ValueError("malformed candidate lineage evidence")
    expected = hmac.new(key, CANDIDATE_LINEAGE_DOMAIN + raw, sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_hex):
        raise ValueError("candidate lineage signature not authenticated")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=lambda _x: (_ for _ in ()).throw(
                               ValueError("nonfinite candidate lineage number")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed candidate lineage evidence") from exc
    if type(value) is not dict or value.keys() != _FIELDS or _canonical(value) != raw:
        raise ValueError("noncanonical or incomplete candidate lineage")
    for field in ("run_id", "exam_id", "objective_id"):
        _identifier(value[field], field)
    for field in ("scope_fingerprint", "checkpoint_artifact_sha256",
                  "predecessor_report_sha256", "predecessor_lora_value_sha256",
                  "public_exam_sha256"):
        _digest(value[field], field)
    expected_fields = {
        "schema": CANDIDATE_LINEAGE_SCHEMA,
        "run_id": run_id,
        "exam_id": exam_id,
        "objective_id": scope.objective_id,
        "scope_fingerprint": scope.fingerprint,
        "cycle_index": context.cycle_index - 1,
        "checkpoint_artifact_sha256": artifact_sha256,
        "predecessor_report_sha256": context.predecessor_report_sha256,
        "predecessor_lora_value_sha256": context.predecessor_lora_value_sha256,
        "public_exam_sha256": public_exam_sha256,
    }
    if value != expected_fields or type(value["cycle_index"]) is not int:
        raise ValueError("candidate lineage differs from authenticated scored learner")


def _bridge_roster(
    cases: tuple[PublicAssessmentCase, ...], *, expected_digest: str,
    scope: TrainingScopeSpec,
) -> tuple[DevelopmentCaseRef, ...]:
    if type(cases) is not tuple or not 4 <= len(cases) <= 512:
        raise ValueError("development bridge requires 4..512 public cases")
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    families: dict[tuple[str, str], str] = {}
    groups: dict[tuple[str, str], int] = {}
    for case in cases:
        if type(case) is not PublicAssessmentCase:
            raise ValueError("invalid public assessment case")
        _identifier(case.case_id, "case ID")
        _identifier(case.skill, "case skill")
        _identifier(case.structure_key, "case structure")
        _digest(case.prompt_sha256, "case prompt commitment")
        if (case.skill not in scope.skills or case.split not in ("heldout", "transfer")
                or case.case_id in seen_ids or case.prompt_sha256 in seen_prompts):
            raise ValueError("duplicate or unauthorized development public case")
        seen_ids.add(case.case_id)
        seen_prompts.add(case.prompt_sha256)
        family = (case.skill, case.structure_key)
        if family in families and families[family] != case.split:
            raise ValueError("structure family cannot cross development exam splits")
        families[family] = case.split
        groups[(case.split, case.skill)] = groups.get((case.split, case.skill), 0) + 1
    # The scored roster must cover *every declared objective skill*, not only
    # the subset the presenter chose to include. Otherwise a valid signature
    # over a partial exam could drive regeneration for a whole objective.
    if (any(groups.get((split, skill), 0) < 2
            for skill in scope.skills
            for split in ("heldout", "transfer"))):
        raise ValueError("missing or undersized declared scope skills/splits in public exam")
    public = [
        (case.case_id, case.skill, case.split,
         case.prompt_sha256, case.structure_key)
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    actual = sha256(_PUBLIC_DOMAIN + _canonical(public)).hexdigest()
    if not hmac.compare_digest(expected_digest, actual):
        raise ValueError("scored exam and independently precommitted public roster differ")
    return tuple(DevelopmentCaseRef(
        c.case_id, c.skill, c.prompt_sha256, c.structure_key,
    ) for c in cases)


def authorize_signed_regeneration(
    *,
    scope: TrainingScopeSpec,
    context: ResumeContext,
    scored_receipt_bytes: bytes,
    scored_signature_hex: str,
    candidate_run_id: str,
    candidate_artifact_sha256: str,
    public_cases: tuple[PublicAssessmentCase, ...],
    evaluator_precommitment_sha256: str,
    development_roster_precommitment_sha256: str,
    candidate_lineage_bytes: bytes,
    candidate_lineage_signature_hex: str,
    exam_id: str,
    event_id: str,
    failure_threshold_percent: int,
    independent_evaluator_key: bytes,
    independent_lineage_key: bytes,
    independent_transition_key: bytes,
) -> SignedRegenerationTransition:
    """Sign a failed *development* exam into the existing HF successor schema.

    A weakness is any declared skill whose authenticated heldout OR transfer
    aggregate is strictly below the fixed whole-percent threshold. A complete
    pass, invalid lineage, uncommitted roster or absent independent key fails
    closed. This returns signed bytes only; it does NOT launch a trainer.
    """
    if type(scope) is not TrainingScopeSpec or type(context) is not ResumeContext:
        raise ValueError("immutable training scope and resume context required")
    TrainingScopeSpec.__post_init__(scope)
    ResumeContext.__post_init__(context)
    _distinct_keys(independent_evaluator_key, independent_lineage_key,
                   independent_transition_key)
    _identifier(candidate_run_id, "candidate run")
    _identifier(exam_id, "development exam ID")
    _identifier(event_id, "development event ID")
    _digest(candidate_artifact_sha256, "candidate artifact")
    _digest(evaluator_precommitment_sha256, "evaluator public precommitment")
    _digest(development_roster_precommitment_sha256, "successor roster precommitment")
    if (type(failure_threshold_percent) is not int
            or not 1 <= failure_threshold_percent <= 100):
        raise ValueError("development failure threshold must be 1..100 percent")
    if context.allow_legacy_unscoped or context.scope_fingerprint != scope.fingerprint:
        raise ValueError("only a same-objective scoped successor may be authorized")
    if context.cycle_index >= scope.max_cycles:
        raise ValueError("declared objective has exhausted its successor cycles")
    try:
        scored = verify_assessment(
            scored_receipt_bytes, scored_signature_hex,
            evaluator_key=independent_evaluator_key,
            run_id=candidate_run_id,
            checkpoint_sha256=candidate_artifact_sha256,
        )
    except BlindCustodyError as exc:
        raise ValueError("independent scored development receipt not authenticated") from exc
    if scored["objective_id"] != scope.objective_id:
        raise ValueError("scored objective differs from declared training objective")
    if scored["public_exam_sha256"] != evaluator_precommitment_sha256:
        raise ValueError("scored public cases differ from evaluator precommitment")
    roster = _bridge_roster(
        public_cases, expected_digest=evaluator_precommitment_sha256, scope=scope,
    )
    # Different development and blind-scoring roster domains must both have
    # been independently pinned BEFORE the exam. Deriving the successor
    # precommitment for the first time after scoring is not sufficient custody.
    observed_development_roster = development_case_commitment(roster)
    if not hmac.compare_digest(
        observed_development_roster, development_roster_precommitment_sha256,
    ):
        raise ValueError("successor public roster was not independently precommitted")
    _verify_lineage(
        candidate_lineage_bytes, candidate_lineage_signature_hex,
        independent_lineage_key,
        scope=scope, context=context, run_id=candidate_run_id, exam_id=exam_id,
        artifact_sha256=candidate_artifact_sha256,
        public_exam_sha256=evaluator_precommitment_sha256,
    )
    groups = {(g["split"], g["skill"]): g for g in scored["groups"]}
    observed = {(case.split, case.skill): 0 for case in public_cases}
    for case in public_cases:
        observed[(case.split, case.skill)] += 1
    if (set(groups) != set(observed)
            or any(groups[key]["total"] != count for key, count in observed.items())):
        raise ValueError("signed aggregate does not correspond to committed case groups")
    weaknesses = tuple(skill for skill in scope.skills if any(
        groups[(split, skill)]["correct"] * 100
        < groups[(split, skill)]["total"] * failure_threshold_percent
        for split in ("heldout", "transfer") if (split, skill) in groups
    ))
    if not weaknesses:
        raise ValueError("no authenticated weakness; regeneration not authorized")
    body = {
        "schema": TRANSITION_SCHEMA,
        "event_id": event_id,
        "exam_id": exam_id,
        "exam_kind": "development",
        "exam_status": "FAIL",
        "objective_id": scope.objective_id,
        "scope_fingerprint": scope.fingerprint,
        "from_cycle": context.cycle_index - 1,
        "to_cycle": context.cycle_index,
        "predecessor_report_sha256": context.predecessor_report_sha256,
        "predecessor_lora_value_sha256": context.predecessor_lora_value_sha256,
        "resume_authority_ref": context.authority_ref,
        "weak_skills": list(weaknesses),
        "exam_case_commitment_sha256": observed_development_roster,
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(independent_transition_key,
                         TRANSITION_DOMAIN + raw, sha256).hexdigest()
    # Treat the real successor verifier as a contract test at production time;
    # never return newly signed bytes that its own lineage policy rejects.
    verify_development_transition(
        scope, context, raw, signature,
        independent_development_key=independent_transition_key,
    )
    return SignedRegenerationTransition(
        receipt_bytes=raw, signature_hex=signature,
        development_public_roster=roster,
        scored_receipt_sha256=sha256(scored_receipt_bytes).hexdigest(),
    )
