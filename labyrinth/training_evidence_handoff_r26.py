"""Versioned, explicitly non-authorizing Labyrinth evidence export (R26).

This is a **source declaration** intended for a separately governed Vera Mono
consumer. It neither verifies an examiner's independence nor confers a right
to execute, train, graduate, deploy, change plastic state or transfer consent.
Even a cryptographically matching SHA-256 proves only these transported bytes.

The consumer must acquire expected subject/adapter/evaluator bindings from its
own trusted state. Caller-supplied expectations are mere equality checks;
they do not constitute a trusted host verifier or an effect capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any

from .graduation_gate import GateState, GraduationPolicy, _policy_fingerprint
from .trainingday_graduation_bridge import (
    AdapterTrainingReceipt, BridgeDecision, CurriculumAdmission, LearningIntent,
    assess_graduation_handoff,
)


SCHEMA = "labyrinth.training-evidence-handoff.v1"
_DOMAIN = b"labyrinth.training-evidence-handoff.v1\x00"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_MAX_PACKET = 32 * 1024
_GATE_STATUS_TO_DISPOSITION = {
    "EXAM_REQUIRED": "NOT_CALLED",
    "SIGNED_GATE_REPORTED_GRADUATED": "GRADUATED",
    "PRIOR_TRUSTED_GATE_GRADUATION_REPORTED": "GRADUATED",
    "GATE_BLOCKED": "BLOCKED",
    "GATE_STAY": "STAY",
    "GATE_EXHAUSTED": "EXHAUSTED",
}
_NON_AUTHORITATIVE = {
    "requested_effect": "NONE",
    "source_evidence_status": "DECLARED_UNATTESTED",
    "host_authority_verified": False,
    "physical_learning_verified": False,
    "graduation_authorized": False,
    "plastic_write_authorized": False,
    "deployment_authorized": False,
    "identity_consent_transferred": False,
}


@dataclass(frozen=True)
class HandoffSourceProvenance:
    """Required provenance DIGEST CLAIMS; none independently authenticated."""

    labyrinth_commit: str
    training_report_sha256: str
    optimizer_receipt_sha256: str
    examiner_receipt_sha256: str
    examiner_public_key_sha256: str
    minotaur_policy_sha256: str
    training_run_id: str


@dataclass(frozen=True)
class ExpectedConsumerBinding:
    """Caller must get these constraints outside the transported packet."""

    objective_id: str
    objective_kind: str
    intent_sha256: str
    curriculum_manifest_sha256: str
    admitted_corpus_sha256: str
    training_source_sha256: str
    tokenizer_sha256: str
    base_model_sha256: str
    initial_adapter_sha256: str
    trained_adapter_sha256: str
    completed_steps: int
    evaluator_source_sha256: str
    evaluator_conditions_sha256: str
    minotaur_policy_sha256: str
    examiner_public_key_sha256: str
    examiner_receipt_sha256: str
    optimizer_receipt_sha256: str
    training_report_sha256: str
    training_run_id: str
    labyrinth_commit: str
    reported_gate_status: str


@dataclass(frozen=True)
class HandoffInspection:
    schema: str
    digest_sha256: str
    objective_id: str
    trained_adapter_sha256: str
    reported_gate_status: str
    disposition: str = "SOURCE_ONLY_NONAUTHORIZING"
    evaluator_independence_verified: bool = False
    host_authority_verified: bool = False
    physical_learning_verified: bool = False
    graduation_authorized: bool = False
    plastic_write_authorized: bool = False
    deployment_authorized: bool = False
    identity_consent_transferred: bool = False


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _hash(value: Any) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _id(value: Any) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _object(value: Any, fields: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"invalid {label} fields")
    return value


def _require_digest(value: Any, label: str) -> None:
    if not _hash(value):
        raise ValueError(f"invalid {label} SHA-256")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate handoff JSON field")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError("non-finite handoff JSON number")


def _source_record(
    intent: LearningIntent,
    curriculum: CurriculumAdmission,
    training: AdapterTrainingReceipt,
    policy: GraduationPolicy,
    decision: BridgeDecision,
    provenance: HandoffSourceProvenance,
) -> dict[str, Any]:
    if (type(intent) is not LearningIntent
            or type(curriculum) is not CurriculumAdmission
            or type(training) is not AdapterTrainingReceipt
            or type(policy) is not GraduationPolicy
            or type(decision) is not BridgeDecision
            or type(provenance) is not HandoffSourceProvenance):
        raise ValueError("exact typed Labyrinth evidence and provenance are required")
    # Reuse the existing public source-level binding invariant without passing
    # fake examiner signature/host claims through the graduation gate.
    checked = assess_graduation_handoff(intent, curriculum, training, policy, GateState())
    if checked.status != "EXAM_REQUIRED":
        raise ValueError("training source does not satisfy Labyrinth pre-exam handoff")
    if (type(decision.status) is not str
            or type(decision.minotaur_gate_disposition) is not str
            or decision.status not in _GATE_STATUS_TO_DISPOSITION
            or decision.promotion_allowed is not False
            or decision.qualified_neural_improvement is not False
            or decision.independently_verified_host_receipts is not False):
        raise ValueError("source decision cannot assert trusted host or promotion effects")
    if decision.minotaur_gate_disposition != _GATE_STATUS_TO_DISPOSITION[decision.status]:
        raise ValueError("reported Minotaur disposition does not match status")

    if (type(provenance.labyrinth_commit) is not str
            or _COMMIT.fullmatch(provenance.labyrinth_commit) is None):
        raise ValueError("exact Labyrinth source commit is required")
    for name in ("training_report_sha256", "optimizer_receipt_sha256",
                 "examiner_receipt_sha256", "examiner_public_key_sha256",
                 "minotaur_policy_sha256"):
        _require_digest(getattr(provenance, name), name)
    if not _id(provenance.training_run_id):
        raise ValueError("invalid exact training run identity")
    if provenance.minotaur_policy_sha256 != _policy_fingerprint(policy):
        raise ValueError("Minotaur policy digest does not bind passed policy")

    return {
        "provenance": {
            "labyrinth_repo": "thebrazenbeard/labyrinth",
            "labyrinth_commit": provenance.labyrinth_commit,
            "training_report_sha256": provenance.training_report_sha256,
            "optimizer_receipt_sha256": provenance.optimizer_receipt_sha256,
            "training_run_id": provenance.training_run_id,
        },
        "subject": {
            "objective_id": intent.objective_id,
            "objective_kind": intent.objective_kind,
            "intent_sha256": intent.intent_sha256,
            "curriculum_manifest_sha256": curriculum.manifest_sha256,
        },
        "training": {
            "source_sha256": training.source_sha256,
            "admitted_corpus_sha256": training.admitted_corpus_sha256,
            "base_model_sha256": training.base_sha256,
            "initial_adapter_sha256": training.initial_adapter_sha256,
            "trained_adapter_sha256": training.trained_adapter_sha256,
            "tokenizer_sha256": training.tokenizer_sha256,
            "completed_steps": training.completed_steps,
            "time_integrity_status": training.time_integrity_status,
        },
        "evaluator": {
            "evaluator_source_sha256": policy.expected_evaluator_sha256,
            "conditions_sha256": policy.expected_conditions_sha256,
            "examiner_receipt_sha256": provenance.examiner_receipt_sha256,
            "examiner_public_key_sha256": provenance.examiner_public_key_sha256,
            "minotaur_policy_sha256": provenance.minotaur_policy_sha256,
            "independence": "DECLARED_UNVERIFIED",
            "receipt_signature_verified": False,
        },
        "minotaur": {
            "reported_gate_status": decision.status,
            "reported_gate_disposition": decision.minotaur_gate_disposition,
            "effect_status": "REPORT_ONLY",
        },
        "authority": dict(_NON_AUTHORITATIVE),
    }


def export_handoff(
    intent: LearningIntent,
    curriculum: CurriculumAdmission,
    training: AdapterTrainingReceipt,
    policy: GraduationPolicy,
    decision: BridgeDecision,
    provenance: HandoffSourceProvenance,
) -> bytes:
    """Serialize a bounded deterministic evidence envelope, never a capability."""
    record = _source_record(intent, curriculum, training, policy, decision, provenance)
    sha = sha256(_DOMAIN + _canonical(record)).hexdigest()
    payload = _canonical({"schema": SCHEMA, "record": record, "sha256": sha})
    if len(payload) > _MAX_PACKET:
        raise ValueError("handoff evidence exceeds transport budget")
    return payload


def inspect_handoff(payload: bytes, expected: ExpectedConsumerBinding) -> HandoffInspection:
    """Verify canonical transport + consumer equality only; no authority gain."""
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_PACKET:
        raise ValueError("invalid bounded handoff payload")
    if type(expected) is not ExpectedConsumerBinding:
        raise ValueError("expected independently pinned consumer binding is required")
    if (not _id(expected.objective_id)
            or type(expected.objective_kind) is not str
            or expected.objective_kind not in ("training_effect", "skill_mastery")
            or not _id(expected.training_run_id)
            or type(expected.labyrinth_commit) is not str
            or _COMMIT.fullmatch(expected.labyrinth_commit) is None
            or type(expected.reported_gate_status) is not str
            or expected.reported_gate_status not in _GATE_STATUS_TO_DISPOSITION
            or type(expected.completed_steps) is not int
            or not 1 <= expected.completed_steps <= 100_000
            or not all(_hash(v) for v in (
                expected.intent_sha256, expected.curriculum_manifest_sha256,
                expected.admitted_corpus_sha256, expected.training_source_sha256,
                expected.tokenizer_sha256,
                expected.base_model_sha256, expected.trained_adapter_sha256,
                expected.initial_adapter_sha256,
                expected.evaluator_source_sha256, expected.minotaur_policy_sha256,
                expected.examiner_public_key_sha256, expected.evaluator_conditions_sha256,
                expected.examiner_receipt_sha256, expected.optimizer_receipt_sha256,
                expected.training_report_sha256,
            ))):
        raise ValueError("malformed expected consumer binding")
    try:
        value = json.loads(payload.decode("utf-8"),
                           object_pairs_hook=_unique, parse_constant=_nonfinite)
    except (UnicodeError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError("malformed handoff envelope") from exc
    outer = _object(value, {"schema", "record", "sha256"}, "envelope")
    if outer["schema"] != SCHEMA or _canonical(outer) != payload:
        raise ValueError("noncanonical or unsupported handoff schema")
    record = _object(outer["record"],
                     {"provenance", "subject", "training", "evaluator", "minotaur", "authority"},
                     "record")
    if not _hash(outer["sha256"]) or sha256(_DOMAIN + _canonical(record)).hexdigest() != outer["sha256"]:
        raise ValueError("handoff digest mismatch")

    provenance = _object(record["provenance"],
                         {"labyrinth_repo", "labyrinth_commit", "training_report_sha256",
                          "optimizer_receipt_sha256", "training_run_id"}, "provenance")
    if (provenance["labyrinth_repo"] != "thebrazenbeard/labyrinth"
            or type(provenance["labyrinth_commit"]) is not str
            or _COMMIT.fullmatch(provenance["labyrinth_commit"]) is None
            or not _id(provenance["training_run_id"])):
        raise ValueError("invalid provenance source and run identity")
    for label in ("training_report_sha256", "optimizer_receipt_sha256"):
        _require_digest(provenance[label], label)

    subject = _object(record["subject"],
                      {"objective_id", "objective_kind", "intent_sha256",
                       "curriculum_manifest_sha256"}, "subject")
    if (not _id(subject["objective_id"])
            or subject["objective_kind"] not in ("training_effect", "skill_mastery")
            or not all(_hash(subject[k]) for k in ("intent_sha256", "curriculum_manifest_sha256"))):
        raise ValueError("invalid learning subject")
    training = _object(record["training"],
                       {"source_sha256", "admitted_corpus_sha256", "base_model_sha256",
                        "initial_adapter_sha256", "trained_adapter_sha256", "tokenizer_sha256",
                        "completed_steps", "time_integrity_status"}, "training")
    if (not all(_hash(training[k]) for k in (
            "source_sha256", "admitted_corpus_sha256", "base_model_sha256",
            "initial_adapter_sha256", "trained_adapter_sha256", "tokenizer_sha256"))
            or type(training["completed_steps"]) is not int
            or not 1 <= training["completed_steps"] <= 100_000
            or training["time_integrity_status"] != "VALID_TIME_BUDGET"
            or len({training["base_model_sha256"], training["initial_adapter_sha256"],
                    training["trained_adapter_sha256"]}) != 3):
        raise ValueError("invalid or self-contradictory adapter report")

    evaluator = _object(record["evaluator"],
                        {"evaluator_source_sha256", "conditions_sha256",
                         "examiner_receipt_sha256", "examiner_public_key_sha256",
                         "minotaur_policy_sha256", "independence",
                         "receipt_signature_verified"}, "evaluator")
    if (not all(_hash(evaluator[k]) for k in (
            "evaluator_source_sha256", "conditions_sha256", "examiner_receipt_sha256",
            "examiner_public_key_sha256", "minotaur_policy_sha256"))
            or evaluator["independence"] != "DECLARED_UNVERIFIED"
            or evaluator["receipt_signature_verified"] is not False):
        raise ValueError("source handoff must not impersonate an independent evaluator")
    gate = _object(record["minotaur"],
                   {"reported_gate_status", "reported_gate_disposition", "effect_status"},
                   "Minotaur report")
    if (type(gate["reported_gate_status"]) is not str
            or type(gate["reported_gate_disposition"]) is not str
            or gate["reported_gate_status"] not in _GATE_STATUS_TO_DISPOSITION
            or gate["effect_status"] != "REPORT_ONLY"):
        raise ValueError("invalid source-only gate report")
    if gate["reported_gate_disposition"] != _GATE_STATUS_TO_DISPOSITION[gate["reported_gate_status"]]:
        raise ValueError("inconsistent gate source status")
    authority = _object(record["authority"], _NON_AUTHORITATIVE.keys(), "authority")
    if any(type(authority[key]) is not type(expected_value)
           or authority[key] != expected_value
           for key, expected_value in _NON_AUTHORITATIVE.items()):
        raise ValueError("source evidence cannot grant effects or transfer consent")
    if (subject["objective_id"] != expected.objective_id
            or subject["objective_kind"] != expected.objective_kind
            or subject["intent_sha256"] != expected.intent_sha256
            or subject["curriculum_manifest_sha256"] != expected.curriculum_manifest_sha256
            or training["admitted_corpus_sha256"] != expected.admitted_corpus_sha256
            or training["source_sha256"] != expected.training_source_sha256
            or training["tokenizer_sha256"] != expected.tokenizer_sha256
            or training["base_model_sha256"] != expected.base_model_sha256
            or training["initial_adapter_sha256"] != expected.initial_adapter_sha256
            or training["trained_adapter_sha256"] != expected.trained_adapter_sha256
            or training["completed_steps"] != expected.completed_steps
            or evaluator["evaluator_source_sha256"] != expected.evaluator_source_sha256
            or evaluator["conditions_sha256"] != expected.evaluator_conditions_sha256
            or evaluator["minotaur_policy_sha256"] != expected.minotaur_policy_sha256
            or evaluator["examiner_public_key_sha256"] != expected.examiner_public_key_sha256
            or evaluator["examiner_receipt_sha256"] != expected.examiner_receipt_sha256
            or provenance["optimizer_receipt_sha256"] != expected.optimizer_receipt_sha256
            or provenance["training_report_sha256"] != expected.training_report_sha256
            or provenance["training_run_id"] != expected.training_run_id
            or provenance["labyrinth_commit"] != expected.labyrinth_commit
            or gate["reported_gate_status"] != expected.reported_gate_status):
        raise ValueError("handoff subject/model/evaluator does not match consumer binding")
    return HandoffInspection(
        schema=SCHEMA, digest_sha256=outer["sha256"],
        objective_id=subject["objective_id"],
        trained_adapter_sha256=training["trained_adapter_sha256"],
        reported_gate_status=gate["reported_gate_status"],
    )
