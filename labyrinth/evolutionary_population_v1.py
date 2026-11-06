"""Development-only evolutionary population policy over physical HF genome receipts.

This module does not train, execute generated Python, evaluate a neural model,
hold private exam cases, or authorize deployment/graduation. Its local examiner
records are untrusted *fixtures* until separately authenticated in a real host.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re

from .campaign_full_state_v1 import verify_campaign_state, verify_generation_ancestry


_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_CONTROL_IDS = ("parent", "control")


class PopulationError(ValueError):
    """A population, its provenance, or its exam evidence is inadmissible."""


class MutationKind(str, Enum):
    ADAPTER_DELTA = "lora_adapter"
    FULL_MODEL_EDIT = "full_weight_edit"
    PROMPT_MEMORY = "prompt_memory"  # Not an HF neural mutation; unsupported here.


def _sha(value: str, label: str) -> None:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise PopulationError(f"{label} must be an exact lowercase SHA-256")


def _identity(value: str, label: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise PopulationError(f"{label} must have a bounded stable identifier")


def deterministic_seed(parent_state_sha256: str, development_generation: int,
                       candidate_id: str) -> int:
    """Public *training* seed, never a secret evaluation-case seed."""
    _sha(parent_state_sha256, "parent_state_sha256")
    _identity(candidate_id, "candidate_id")
    if type(development_generation) is not int or development_generation < 1:
        raise PopulationError("development generation must be a positive integer")
    payload = ("labyrinth.population.training-seed.v1\0"
               f"{parent_state_sha256}\0{development_generation}\0{candidate_id}")
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    manifest_path: Path
    mutation_kind: MutationKind
    trainer_id: str
    trainer_receipt_sha256: str
    preupdate_parent_state_sha256: str
    seed: int


@dataclass(frozen=True)
class EvaluationPlan:
    examiner_id: str
    suite_sha256: str
    sealed_case_commitment: str
    preregistration_sha256: str
    private_cohort_commitment: str
    cohort_policy: str = "MATCHED_RANDOMIZED_PRIVATE_EXAMINER_ONLY"


@dataclass(frozen=True)
class Metrics:
    holdout: float
    transfer: float
    identity: float
    retention: float
    regression_rate: float


@dataclass(frozen=True)
class Examination:
    subject_id: str
    state_sha256: str
    examiner_id: str
    suite_sha256: str
    sealed_case_commitment: str
    preregistration_sha256: str
    private_cohort_commitment: str
    examiner_run_sha256: str
    metrics: Metrics
    producer_role: str = "INDEPENDENT_DEV_EXAMINER"


@dataclass(frozen=True)
class Population:
    development_generation: int
    origin_manifest_path: Path
    parent_manifest_path: Path
    control_manifest_path: Path
    candidates: tuple[Candidate, ...]
    plan: EvaluationPlan
    origin_state_sha256: str
    parent_state_sha256: str
    parent_manifest_sha256: str
    control_manifest_sha256: str
    candidate_states: tuple[tuple[str, str, str], ...]  # id, state, manifest
    original_plan_fields: tuple[str, ...]
    qualification: str = "SOURCE_ONLY_UNAUTHENTICATED_EXAMINER_FIXTURE"


@dataclass(frozen=True)
class Decision:
    development_generation: int
    next_development_generation: int
    winner_id: str | None
    selected_manifest_path: Path
    selected_state_sha256: str
    parent_state_sha256: str
    rejected_candidate_ids: tuple[str, ...]
    qualification: str = "SOURCE_ONLY_SELECTION_POLICY_NOT_NEURAL_EVIDENCE"


def _manifest(path: Path) -> dict:
    if not isinstance(path, Path) or not path.is_absolute():
        raise PopulationError("full-state manifest needs an absolute Path")
    try:
        attestation = verify_campaign_state(path, fresh_process=True)
        manifest = json.loads(path.read_bytes())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise PopulationError("physical full-state manifest readback failed") from exc
    if (manifest["state_sha256"] != attestation["state_sha256"]
            or manifest["manifest_sha256"] != attestation["manifest_sha256"]):
        raise PopulationError("manifest changed during physical readback")
    return manifest


def _check_plan(plan: EvaluationPlan, trainers: set[str]) -> None:
    if type(plan) is not EvaluationPlan:
        raise PopulationError("predeclared independent evaluation plan required")
    _identity(plan.examiner_id, "examiner_id")
    if plan.examiner_id in trainers:
        raise PopulationError("trainer cannot be its own examiner")
    for field in ("suite_sha256", "sealed_case_commitment",
                  "preregistration_sha256", "private_cohort_commitment"):
        _sha(getattr(plan, field), field)
    if plan.cohort_policy != "MATCHED_RANDOMIZED_PRIVATE_EXAMINER_ONLY":
        raise PopulationError("private matched randomized examiner cohorts are mandatory")


def _plan_fields(plan: EvaluationPlan) -> tuple[str, ...]:
    if type(plan) is not EvaluationPlan:
        raise PopulationError("expected exact source-owned evaluation plan")
    return (plan.examiner_id, plan.suite_sha256,
            plan.sealed_case_commitment, plan.preregistration_sha256,
            plan.private_cohort_commitment, plan.cohort_policy)


def _check_child(parent: dict, child: dict) -> None:
    if (child["family"] != parent["family"]
            or child["mutation_kind"] != parent["mutation_kind"]
            or child["generation_index"] != parent["generation_index"] + 1
            or child["parent_state_sha256"] != parent["state_sha256"]
            or child["origin_state_sha256"] != parent["origin_state_sha256"]
            or child["state_sha256"] == parent["state_sha256"]):
        raise PopulationError("candidate not an exact claimed parent/origin descendant")
    if parent["mutation_kind"] == MutationKind.ADAPTER_DELTA.value:
        def adapter_weights(manifest: dict) -> dict:
            return {name: record for name, record in manifest["adapter"]["files"].items()
                    if name.endswith(".safetensors")}

        if (child["base"]["files"] != parent["base"]["files"]
                or not adapter_weights(parent)
                or not adapter_weights(child)
                or adapter_weights(child) == adapter_weights(parent)):
            raise PopulationError("LoRA child must preserve whole base and mutate adapter tensor weights")
    elif parent["mutation_kind"] == MutationKind.FULL_MODEL_EDIT.value:
        def metadata(manifest: dict) -> dict:
            return {key: value for key, value in manifest["base"]["files"].items()
                    if not key.endswith(".safetensors")
                    and key != "model.safetensors.index.json"}

        def weights(manifest: dict) -> dict:
            return {key: value for key, value in manifest["base"]["files"].items()
                    if key.endswith(".safetensors")}

        if metadata(parent) != metadata(child) or weights(parent) == weights(child):
            raise PopulationError("full-weight child must preserve tokenizer/config and mutate weights")
    else:
        raise PopulationError("unsupported mutation family; prompt memory is not neural weights")


def prepare_population(*, development_generation: int, origin_manifest_path: Path,
                       parent_manifest_path: Path, control_manifest_path: Path,
                       candidates: tuple[Candidate, ...], plan: EvaluationPlan) -> Population:
    """Bind 2-8 sibling byte manifests to the same physical parent and control.

    A child's *claimed* pre-update parent and trainer receipt are checked for
    consistency, but are NOT authenticated proof of an actual optimizer step.
    """
    if type(development_generation) is not int or development_generation < 1:
        raise PopulationError("invalid development generation")
    if (type(candidates) is not tuple or not 2 <= len(candidates) <= 8
            or any(type(candidate) is not Candidate for candidate in candidates)):
        raise PopulationError("each generation needs two to eight sibling variants")
    paths = (parent_manifest_path, control_manifest_path,
             *(candidate.manifest_path for candidate in candidates))
    if (len(set(paths)) != len(paths)
            or (origin_manifest_path != parent_manifest_path
                and origin_manifest_path in paths)):
        raise PopulationError("parent, control and sibling manifests need distinct paths")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids) or set(ids) & set(_CONTROL_IDS):
        raise PopulationError("duplicate or reserved candidate id")
    origin = _manifest(origin_manifest_path)
    parent = _manifest(parent_manifest_path)
    control = _manifest(control_manifest_path)
    if (origin["generation_index"] != 0
            or origin["state_sha256"] != origin["origin_state_sha256"]
            or parent["origin_state_sha256"] != origin["state_sha256"]
            or control["state_sha256"] != parent["state_sha256"]
            or control["family"] != parent["family"]
            or control["mutation_kind"] != parent["mutation_kind"]):
        raise PopulationError("origin/unchanged control/parent provenance mismatch")
    _check_plan(plan, {candidate.trainer_id for candidate in candidates} | set(ids))
    if len({candidate.trainer_receipt_sha256 for candidate in candidates}) != len(candidates):
        raise PopulationError("siblings need distinct claimed trainer receipts")
    states = []
    for candidate in candidates:
        _identity(candidate.candidate_id, "candidate_id")
        _identity(candidate.trainer_id, "trainer_id")
        _sha(candidate.trainer_receipt_sha256, "trainer_receipt_sha256")
        _sha(candidate.preupdate_parent_state_sha256, "preupdate_parent_state_sha256")
        if (type(candidate.mutation_kind) is not MutationKind
                or candidate.mutation_kind is MutationKind.PROMPT_MEMORY
                or candidate.mutation_kind.value != parent["mutation_kind"]
                or candidate.preupdate_parent_state_sha256 != parent["state_sha256"]
                or type(candidate.seed) is not int
                or candidate.seed != deterministic_seed(parent["state_sha256"],
                                                         development_generation,
                                                         candidate.candidate_id)):
            raise PopulationError("candidate trainer origin, typed mutation or seed mismatch")
        child = _manifest(candidate.manifest_path)
        _check_child(parent, child)
        states.append((candidate.candidate_id, child["state_sha256"],
                       child["manifest_sha256"]))
    if len(set(state for _, state, _ in states)) != len(states):
        raise PopulationError("duplicate physical sibling full state")
    return Population(development_generation, origin_manifest_path,
                      parent_manifest_path, control_manifest_path, candidates,
                      plan, origin["state_sha256"], parent["state_sha256"],
                      parent["manifest_sha256"], control["manifest_sha256"],
                      tuple(states), _plan_fields(plan))


def _check_metric(metrics: Metrics) -> None:
    if (type(metrics) is not Metrics or any(
        type(value) not in (int, float) or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in (metrics.holdout, metrics.transfer, metrics.identity,
                      metrics.retention, metrics.regression_rate)
    )):
        raise PopulationError("examiner metrics must be finite [0,1] values")


def select_generation(population: Population, examinations: tuple[Examination, ...], *,
                      min_holdout_gain: float = .01, min_transfer_gain: float = .01,
                      identity_floor: float = .95, retention_floor: float = .95,
                      max_regression_rate: float = .05,
                      max_identity_drop: float = .01,
                      max_retention_drop: float = .01) -> Decision:
    """Compare siblings with separately examined unchanged parent and control.

    Selection is deterministic given purported examiner metrics. Neither these
    records nor a successful local policy decision authenticate neural fitness.
    """
    if type(population) is not Population:
        raise PopulationError("exact prepared population required")
    if _plan_fields(population.plan) != population.original_plan_fields:
        raise PopulationError("examiner preregistration integrity changed after preparation")
    gates = (min_holdout_gain, min_transfer_gain, identity_floor, retention_floor,
             max_regression_rate, max_identity_drop, max_retention_drop)
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in gates):
        raise PopulationError("selection gates must be bounded finite fractions")
    if min_holdout_gain <= 0 or min_transfer_gain <= 0:
        raise PopulationError("development requires positive holdout and transfer gains")
    # Reopen all bytes: an in-memory population is not proof against mutation.
    fresh = prepare_population(
        development_generation=population.development_generation,
        origin_manifest_path=population.origin_manifest_path,
        parent_manifest_path=population.parent_manifest_path,
        control_manifest_path=population.control_manifest_path,
        candidates=population.candidates, plan=population.plan)
    if fresh != population:
        raise PopulationError("prepared population provenance changed")
    subjects = {"parent": (population.parent_state_sha256,
                           population.parent_manifest_sha256),
                "control": (population.parent_state_sha256,
                            population.control_manifest_sha256),
                **{name: (state, manifest)
                   for name, state, manifest in population.candidate_states}}
    if (type(examinations) is not tuple or len(examinations) != len(subjects)
            or any(type(item) is not Examination for item in examinations)
            or {item.subject_id for item in examinations} != set(subjects)):
        raise PopulationError("independent parent, control and every sibling exam required")
    if len({item.examiner_run_sha256 for item in examinations}) != len(examinations):
        raise PopulationError("examiner runs must have distinct receipts")
    by_id = {}
    for examination in examinations:
        _sha(examination.examiner_run_sha256, "examiner_run_sha256")
        _check_metric(examination.metrics)
        if (examination.state_sha256 != subjects[examination.subject_id][0]
                or examination.examiner_id != population.plan.examiner_id
                or examination.suite_sha256 != population.plan.suite_sha256
                or examination.sealed_case_commitment != population.plan.sealed_case_commitment
                or examination.preregistration_sha256 != population.plan.preregistration_sha256
                or examination.private_cohort_commitment != population.plan.private_cohort_commitment
                or examination.producer_role != "INDEPENDENT_DEV_EXAMINER"):
            raise PopulationError("forged/self-graded or mismatched independent examiner result")
        by_id[examination.subject_id] = examination.metrics
    parent, control = by_id["parent"], by_id["control"]
    minimum_holdout = max(parent.holdout, control.holdout) + min_holdout_gain
    minimum_transfer = max(parent.transfer, control.transfer) + min_transfer_gain
    minimum_identity = max(identity_floor,
                           max(parent.identity, control.identity) - max_identity_drop)
    minimum_retention = max(retention_floor,
                            max(parent.retention, control.retention) - max_retention_drop)
    accepted = []
    for candidate in population.candidates:
        m = by_id[candidate.candidate_id]
        if (m.holdout >= minimum_holdout and m.transfer >= minimum_transfer
                and m.identity >= minimum_identity and m.retention >= minimum_retention
                and m.regression_rate <= max_regression_rate
                and m.regression_rate <= min(parent.regression_rate, control.regression_rate)):
            score = (min(m.holdout - minimum_holdout,
                         m.transfer - minimum_transfer),
                     m.holdout + m.transfer, m.identity + m.retention,
                     -m.regression_rate)
            accepted.append((score, candidate.candidate_id))
    winner_id = (sorted(accepted, key=lambda pair: (tuple(-v for v in pair[0]),
                                                     pair[1]))[0][1]
                 if accepted else None)
    selected = (next(candidate.manifest_path for candidate in population.candidates
                     if candidate.candidate_id == winner_id)
                if winner_id else population.parent_manifest_path)
    selected_state = (next(state for name, state, _ in population.candidate_states
                           if name == winner_id)
                      if winner_id else population.parent_state_sha256)
    return Decision(population.development_generation,
                    population.development_generation + 1, winner_id, selected,
                    selected_state, population.parent_state_sha256,
                    tuple(sorted(set(subjects) - {"parent", "control"}
                                 - ({winner_id} if winner_id else set()))))


def verify_selected_entry(population: Population, decision: Decision,
                          next_entry_path: Path) -> dict:
    """Verify exact champion -> next entry full-state content, no weight merge.

    No-winner generations *retain* the original parent manifest directly.
    """
    if (type(population) is not Population or type(decision) is not Decision
            or decision.winner_id is None
            or decision.development_generation != population.development_generation
            or decision.parent_state_sha256 != population.parent_state_sha256
            or decision.winner_id not in {c.candidate_id for c in population.candidates}):
        raise PopulationError("selected physical child required for ancestry verification")
    winner = next(candidate for candidate in population.candidates
                  if candidate.candidate_id == decision.winner_id)
    if decision.selected_manifest_path != winner.manifest_path:
        raise PopulationError("decision champion was swapped")
    try:
        result = verify_generation_ancestry(
            population.parent_manifest_path,
            tuple(candidate.manifest_path for candidate in population.candidates),
            winner.manifest_path, next_entry_path)
    except (ValueError, OSError) as exc:
        raise PopulationError("selected full-state next entry is not exact") from exc
    if result["champion_state_sha256"] != decision.selected_state_sha256:
        raise PopulationError("selected state digest disagrees with champion")
    return result
