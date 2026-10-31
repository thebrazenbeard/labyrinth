"""Labyrinth/Minotaur source-only cross-project training provenance boundary.

This intentionally does not execute a trainer or reinterpret control arms.
The frozen R18 comparison consumes the exact Labyrinth PR59 R16 corpus; the
regenerative PR62 MiniLM experiment is a *different* learner/task/plan.
Task/control-set IDs here are new bridge-local taxonomy identifiers, not
identifiers asserted to exist in PR59/PR62/PR143 payloads.
Passing this validator never grants host authority, model promotion or a
private examination. Exact historical heads below are V1 pins, not a claim
that those branches remain current or that a physical run occurred.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA = "labyrinth.cross-training-source-provenance.v1"
PR59 = "labyrinth-pr59-corpus-r16"
PR62 = "labyrinth-pr62-regenerative-minilm"
R18 = "model-training-pr143-r18"
SOURCE_ONLY = "SOURCE_METADATA_ONLY_NOT_EXECUTION_ATTESTATION"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_STRING_FIELDS = (
    "schema", "campaign", "source_repo", "source_head", "corpus_repo",
    "corpus_head", "task_id", "objective_family", "method_id",
    "control_set_id", "eval_source", "evidence_level",
)
_OPTIONAL_STRING_FIELDS = (
    "corpus_manifest_sha256", "model_id", "model_revision", "control_id",
    "training_plan_sha256", "baseline_checkpoint_sha256",
    "result_checkpoint_sha256", "eval_custody_sha256",
)
_BOOL_FIELDS = (
    "host_authority_attested", "independent_exam_pass",
    "graduation_authorized",
)
_DIGEST_FIELDS = (
    "corpus_manifest_sha256", "training_plan_sha256",
    "baseline_checkpoint_sha256", "result_checkpoint_sha256",
    "eval_custody_sha256",
)


class IncompatibleTrainingEvidence(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    source_repo: str
    source_head: str
    corpus_repo: str
    corpus_head: str
    corpus_manifest_sha256: str | None
    task_id: str
    objective_family: str
    model_id: str | None
    model_revision: str | None
    methods: tuple[str, ...]
    control_set_id: str
    control_set: tuple[str, ...]
    eval_source: str


PROFILES = MappingProxyType({
    PR59: Profile(
        "thebrazenbeard/labyrinth", "1d38e4f25b5663bf8a84ff5e139875d4152e3680",
        "thebrazenbeard/labyrinth", "1d38e4f25b5663bf8a84ff5e139875d4152e3680",
        "90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836",
        "r16_integer_chain_v2", "verified_integer_operation_chains", None, None,
        ("source_only_corpus",), "r16_source_only_no_controls", (),
        "EVAL_SHARDS_HASH_ONLY_NOT_OPENED",
    ),
    PR62: Profile(
        "thebrazenbeard/labyrinth", "b698ede24d9a69e551e1cad83dee83a825feb9a6",
        "thebrazenbeard/labyrinth", "b698ede24d9a69e551e1cad83dee83a825feb9a6",
        None, "pr62_regenerative_sentiment_v1", "binary_sentiment_polarity",
        "sentence-transformers/all-MiniLM-L6-v2", None,
        ("adamw_lora",),
        "pr62_minilm_preregistered_five_arms",
        ("correct", "zero_update", "inverted", "zero_lr", "randomized_labels"),
        "PUBLIC_UNSEALED_DEVELOPMENT",
    ),
    R18: Profile(
        "thebrazenbeard/model-training", "a14118e87dbba9fd436ef2ae6ad04293966d4d8b",
        "thebrazenbeard/labyrinth", "1d38e4f25b5663bf8a84ff5e139875d4152e3680",
        "90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836",
        "r16_integer_chain_v2", "verified_integer_operation_chains",
        "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        ("mezo_lora_zo", "qlora_nf4"),
        "r18_qwen_frozen_zero_correct_corrupt",
        ("frozen", "zero", "correct", "corrupt"),
        "R18_NEW_SEALED_NOT_R16_FRESH64",
    ),
})


@dataclass(frozen=True)
class ProvenanceEnvelope:
    schema: str
    campaign: str
    source_repo: str
    source_head: str
    corpus_repo: str
    corpus_head: str
    corpus_manifest_sha256: str | None
    task_id: str
    objective_family: str
    model_id: str | None
    model_revision: str | None
    method_id: str
    control_set_id: str
    control_set: tuple[str, ...]
    control_id: str | None
    training_plan_sha256: str | None
    baseline_checkpoint_sha256: str | None
    result_checkpoint_sha256: str | None
    eval_source: str
    eval_custody_sha256: str | None
    evidence_level: str
    host_authority_attested: bool
    independent_exam_pass: bool
    graduation_authorized: bool

    def to_json(self) -> dict[str, Any]:
        self.validate()
        return {field.name: getattr(self, field.name) for field in fields(self)} | {
            "control_set": list(self.control_set)
        }

    @property
    def digest(self) -> str:
        self.validate()
        return sha256((json.dumps(self.to_json(), sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=True,
                                  allow_nan=False) + "\n").encode("ascii")).hexdigest()

    def validate(self) -> None:
        # Reject Python objects and str subclasses *before* any equality,
        # membership, or hash operation can invoke attacker-defined methods.
        for key in _REQUIRED_STRING_FIELDS:
            if type(getattr(self, key)) is not str:
                raise IncompatibleTrainingEvidence(f"{key} must be an exact string")
        for key in _OPTIONAL_STRING_FIELDS:
            value = getattr(self, key)
            if value is not None and type(value) is not str:
                raise IncompatibleTrainingEvidence(f"{key} must be an exact string or null")
        for key in _BOOL_FIELDS:
            if type(getattr(self, key)) is not bool:
                raise IncompatibleTrainingEvidence(f"{key} must be an exact bool")
        if type(self.control_set) is not tuple or any(
                type(item) is not str for item in self.control_set):
            raise IncompatibleTrainingEvidence("control set must be an exact string tuple")
        for key in _DIGEST_FIELDS:
            value = getattr(self, key)
            if value is not None and not _valid_digest(value):
                raise IncompatibleTrainingEvidence(f"{key} must be lowercase SHA-256")
        try:
            spec = PROFILES[self.campaign]
        except (KeyError, TypeError) as exc:
            raise IncompatibleTrainingEvidence("unregistered training campaign") from exc
        for key in ("source_repo", "source_head", "corpus_repo", "corpus_head",
                    "corpus_manifest_sha256", "task_id", "objective_family",
                    "model_id", "model_revision", "control_set_id", "eval_source"):
            if getattr(self, key) != getattr(spec, key):
                raise IncompatibleTrainingEvidence(f"{key} differs from exact campaign pin")
        if self.schema != SCHEMA or self.evidence_level != SOURCE_ONLY:
            raise IncompatibleTrainingEvidence("unrecognized source evidence schema/class")
        if (type(self.host_authority_attested) is not bool
                or type(self.independent_exam_pass) is not bool
                or type(self.graduation_authorized) is not bool
                or any((self.host_authority_attested, self.independent_exam_pass,
                        self.graduation_authorized))):
            raise IncompatibleTrainingEvidence("source metadata cannot confer authority")
        if (type(self.control_set) is not tuple or self.control_set != spec.control_set
                or type(self.method_id) is not str or self.method_id not in spec.methods):
            raise IncompatibleTrainingEvidence("method or control taxonomy was transferred")
        if self.campaign == PR59:
            if (self.control_id is not None or self.training_plan_sha256 is not None
                    or self.baseline_checkpoint_sha256 is not None
                    or self.result_checkpoint_sha256 is not None
                    or self.eval_custody_sha256 is not None):
                raise IncompatibleTrainingEvidence("a source-only corpus is not a trained model")
        else:
            if (type(self.control_id) is not str or self.control_id not in spec.control_set
                    or not _valid_digest(self.training_plan_sha256)
                    or not _valid_digest(self.baseline_checkpoint_sha256)):
                raise IncompatibleTrainingEvidence("training plan/control/initial learner unbound")
            if (self.result_checkpoint_sha256 is not None
                    and not _valid_digest(self.result_checkpoint_sha256)):
                raise IncompatibleTrainingEvidence("malformed result checkpoint digest")
            if self.campaign == R18:
                if not _valid_digest(self.eval_custody_sha256):
                    raise IncompatibleTrainingEvidence("R18 sealed evaluation custody unbound")
            elif self.eval_custody_sha256 is not None:
                raise IncompatibleTrainingEvidence("PR62 public development is not sealed")


def _valid_digest(value: object) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def make_envelope(campaign: str, *, method_id: str,
                  control_id: str | None = None,
                  training_plan_sha256: str | None = None,
                  baseline_checkpoint_sha256: str | None = None,
                  result_checkpoint_sha256: str | None = None,
                  eval_custody_sha256: str | None = None) -> ProvenanceEnvelope:
    if type(campaign) is not str or campaign not in PROFILES:
        raise IncompatibleTrainingEvidence("unregistered training campaign")
    p = PROFILES[campaign]
    value = ProvenanceEnvelope(
        SCHEMA, campaign, p.source_repo, p.source_head, p.corpus_repo,
        p.corpus_head, p.corpus_manifest_sha256, p.task_id, p.objective_family,
        p.model_id, p.model_revision, method_id, p.control_set_id,
        p.control_set, control_id,
        training_plan_sha256, baseline_checkpoint_sha256,
        result_checkpoint_sha256, p.eval_source, eval_custody_sha256,
        SOURCE_ONLY, False, False, False,
    )
    value.validate()
    return value


def verify_json(value: Mapping[str, Any]) -> ProvenanceEnvelope:
    if (type(value) is not dict or any(type(key) is not str for key in value)
            or set(value) != {field.name for field in fields(ProvenanceEnvelope)}):
        raise IncompatibleTrainingEvidence("missing or extra envelope fields")
    if type(value.get("control_set")) is not list or any(
            type(item) is not str for item in value["control_set"]):
        raise IncompatibleTrainingEvidence("control set must be a string array")
    normalized = dict(value, control_set=tuple(value["control_set"]))
    result = ProvenanceEnvelope(**normalized)
    result.validate()
    return result


def assert_relation(left: ProvenanceEnvelope, right: ProvenanceEnvelope,
                    relation: str) -> str:
    """Validate metadata relationship, never runtime training or authority."""
    if (type(left) is not ProvenanceEnvelope or type(right) is not ProvenanceEnvelope
            or type(relation) is not str):
        raise IncompatibleTrainingEvidence("relation requires exact envelopes and type")
    left.validate()
    right.validate()
    if relation == "R16_CORPUS_INPUT":
        if not (left.campaign == PR59 and right.campaign == R18
                and left.source_repo == right.corpus_repo
                and left.source_head == right.corpus_head
                and left.corpus_manifest_sha256 == right.corpus_manifest_sha256):
            raise IncompatibleTrainingEvidence("R18 is not bound to the frozen PR59 corpus")
        return "SOURCE_INPUT_PROVENANCE_ONLY"
    if relation == "MATCHED_CONTROL_METADATA":
        if (left.campaign == PR59 or right.campaign != left.campaign
                or left.method_id != right.method_id
                or left.training_plan_sha256 != right.training_plan_sha256
                or left.baseline_checkpoint_sha256 != right.baseline_checkpoint_sha256
                or left.eval_custody_sha256 != right.eval_custody_sha256
                or left.control_id == right.control_id):
            raise IncompatibleTrainingEvidence("control candidates are not metadata-matched")
        return "SOURCE_CONTROL_METADATA_ONLY"
    raise IncompatibleTrainingEvidence("no implicit learner/exam/identity migration")
