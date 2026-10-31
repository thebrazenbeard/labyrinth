"""Source-only preregistration and audit for a five-arm regenerative MiniLM run.

This module deliberately cannot launch a model or authorize an examination.
The operator, optimizer, evaluator, and signing keys are not independent merely
because their identifiers or SHA-256 claims match.  Its useful guarantee is
that a reported experiment has a *consistent, frozen source contract*.

This v1 profile covers the canonical (unmeasured) Maze batch selector. A later
measured Minotaur schedule needs its own independently pinned contract; it
must not silently reuse this source-only manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any

from labyrinth.training_scope import TrainingScopeSpec
from scripts.qualify_maze_conditioned_minilm_cpu import (
    NEGATIVES, OBJECTIVE, POSITIVES, PREREG_ARMS, SKILLS, TEMPLATES,
    build_training_cycles, optimizer_learning_rate,
    preregistered_randomized_labels, select_training_batch,
    training_batch_labels, validate_scope,
)

SCHEMA = "labyrinth.minilm-five-arm-independent-evidence-source.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_HEAD = re.compile(r"[0-9a-f]{40}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_TRAIN_FAMILIES = tuple(f"sentiment.generated.template.{i}" for i in range(len(TEMPLATES)))
_APPROVED_ADJECTIVES = frozenset(POSITIVES + NEGATIVES)


class ExperimentContractError(ValueError):
    """A declared source-only experiment contradicts its precommitment."""


def _require_digest(value: Any, name: str) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ExperimentContractError(f"{name} must be a SHA-256 hex digest")


def _require_id(value: Any, name: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ExperimentContractError(f"{name} must be a bounded identifier")


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _digest(obj: Any) -> str:
    return sha256(_canonical(obj)).hexdigest()


def _normalized(text: str) -> str:
    if type(text) is not str or not text or len(text) > 8192:
        raise ExperimentContractError("bounded text required")
    return " ".join(text.casefold().split())


def _approved_words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+", _normalized(text)))


def _canonical_training_payload(
    scope: TrainingScopeSpec, steps_per_cycle: int,
    weak_cycle_one: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[int, ...]], ...], tuple[str, ...]]:
    """Re-derive TRAIN visits/labels from the actual source runner, not claims."""
    cycles = build_training_cycles(scope, weak_cycle_one=weak_cycle_one)
    visits: list[str] = []
    labels: dict[str, list[int]] = {arm: [] for arm in PREREG_ARMS}
    hashes: list[str] = []
    for cycle in cycles:
        hashes.extend(sha256(_normalized(row.prompt).encode("utf-8")).hexdigest()
                      for row in cycle.examples)
        randomized = preregistered_randomized_labels(cycle, scope.seed)
        for step in range(steps_per_cycle):
            batch = select_training_batch(cycle, step)
            visits.extend(row.exposure_id for row in batch)
            for arm in PREREG_ARMS:
                labels[arm].extend(training_batch_labels(
                    batch, arm, randomized if arm == "randomized_labels" else None
                ))
    return (
        tuple(visits),
        tuple((arm, tuple(labels[arm])) for arm in PREREG_ARMS),
        tuple(sorted(hashes)),
    )


@dataclass(frozen=True)
class FrozenFiveArmPlan:
    schema: str
    run_id: str
    source_head: str
    source_tree: str
    scope_fingerprint: str
    scope_seed: int
    scope_specificity: int
    scope_scale: int
    weak_cycle_one: tuple[str, ...]
    steps_per_cycle: int
    nominal_learning_rate: float
    maximum_wall_milliseconds_per_arm: int
    base_model_sha256: str
    initial_trainable_sha256: str
    initial_optimizer_state_sha256: str
    train_visits: tuple[str, ...]
    expected_labels: tuple[tuple[str, tuple[int, ...]], ...]
    train_prompt_hashes: tuple[str, ...]
    generated_family_ids: tuple[str, ...]
    private_transfer_roster_commitment: str
    private_final_bank_commitment: str
    retention_case_ids: tuple[str, ...]
    maximum_prior_correct_regressions: int
    source_contract_sha256: str

    def validate(self) -> None:
        if type(self) is not FrozenFiveArmPlan or type(self.schema) is not str or self.schema != SCHEMA:
            raise ExperimentContractError("unrecognized exact source contract")
        _require_id(self.run_id, "run")
        if (type(self.source_head) is not str
                or type(self.source_tree) is not str
                or not _HEAD.fullmatch(self.source_head)
                or not _HEAD.fullmatch(self.source_tree)):
            raise ExperimentContractError("exact Git source/tree is required")
        _require_digest(self.scope_fingerprint, "scope fingerprint")
        for name in ("base_model_sha256", "initial_trainable_sha256",
                     "initial_optimizer_state_sha256",
                     "private_transfer_roster_commitment",
                     "private_final_bank_commitment", "source_contract_sha256"):
            _require_digest(getattr(self, name), name)
        if self.private_transfer_roster_commitment == self.private_final_bank_commitment:
            raise ExperimentContractError("private final and transfer bank commitments must differ")
        if (type(self.scope_seed) is not int or self.scope_seed < 0
                or type(self.scope_specificity) is not int
                or not 1 <= self.scope_specificity <= 5
                or type(self.scope_scale) is not int
                or not 1 <= self.scope_scale <= 5
                or type(self.weak_cycle_one) is not tuple
                or any(type(x) is not str or x not in SKILLS
                       for x in self.weak_cycle_one)
                or len(set(self.weak_cycle_one)) != len(self.weak_cycle_one)
                or type(self.steps_per_cycle) is not int
                or not 4 <= self.steps_per_cycle <= 16
                or type(self.nominal_learning_rate) is not float
                or not math.isfinite(self.nominal_learning_rate)
                or not 0 < self.nominal_learning_rate <= .01
                or type(self.maximum_wall_milliseconds_per_arm) is not int
                or not 1 <= self.maximum_wall_milliseconds_per_arm <= 24 * 3600 * 1000):
            raise ExperimentContractError("invalid bounded training budget")
        if (type(self.train_visits) is not tuple
                or len(self.train_visits) != 8 * self.steps_per_cycle
                or any(type(x) is not str or not _ID.fullmatch(x)
                       for x in self.train_visits)):
            raise ExperimentContractError("ordered two-cycle 4-wide visits required")
        if (type(self.expected_labels) is not tuple
                or len(self.expected_labels) != len(PREREG_ARMS)):
            raise ExperimentContractError("all five expected label arms required")
        for item in self.expected_labels:
            if (type(item) is not tuple or len(item) != 2
                    or type(item[0]) is not str or type(item[1]) is not tuple
                    or len(item[1]) != len(self.train_visits)
                    or any(type(y) is not int or y not in (0, 1) for y in item[1])):
                raise ExperimentContractError("malformed arm label assignments")
        if tuple(item[0] for item in self.expected_labels) != PREREG_ARMS:
            raise ExperimentContractError("all five expected label arms required")
        if (type(self.train_prompt_hashes) is not tuple
                or not self.train_prompt_hashes
                or any(type(x) is not str for x in self.train_prompt_hashes)
                or len(set(self.train_prompt_hashes)) != len(self.train_prompt_hashes)):
            raise ExperimentContractError("unique TRAIN prompt inventory required")
        for token in self.train_prompt_hashes:
            _require_digest(token, "TRAIN prompt")
        if (type(self.generated_family_ids) is not tuple
                or any(type(x) is not str for x in self.generated_family_ids)
                or self.generated_family_ids != _TRAIN_FAMILIES):
            raise ExperimentContractError("complete generator family inventory required")
        if (type(self.retention_case_ids) is not tuple
                or not self.retention_case_ids
                or any(type(x) is not str or not _ID.fullmatch(x)
                       for x in self.retention_case_ids)
                or len(set(self.retention_case_ids)) != len(self.retention_case_ids)
                or type(self.maximum_prior_correct_regressions) is not int
                or not 0 <= self.maximum_prior_correct_regressions <= len(self.retention_case_ids) // 20):
            raise ExperimentContractError("strict, precommitted prior-skill cases and deterioration cap required")
        scope = TrainingScopeSpec(
            objective_id=OBJECTIVE, domain="sentiment", skills=SKILLS,
            specificity=self.scope_specificity, scale=self.scope_scale,
            max_cycles=2, seed=self.scope_seed,
        )
        validate_scope(scope)
        if scope.fingerprint != self.scope_fingerprint:
            raise ExperimentContractError("source scope fingerprint does not match declared Maze")
        expected_visits, expected_labels, expected_hashes = _canonical_training_payload(
            scope, self.steps_per_cycle, self.weak_cycle_one
        )
        if (self.train_visits != expected_visits
                or self.expected_labels != expected_labels
                or self.train_prompt_hashes != expected_hashes):
            raise ExperimentContractError("TRAIN visits/labels do not match canonical runner")
        data = asdict(self)
        expected = data.pop("source_contract_sha256")
        if _digest(data) != expected:
            raise ExperimentContractError("five-arm plan changed after precommitment")


@dataclass(frozen=True)
class ArmClaim:
    arm: str
    plan_sha256: str
    initial_trainable_sha256: str
    initial_optimizer_state_sha256: str
    executed_visits: tuple[str, ...]
    executed_labels: tuple[int, ...]
    effective_learning_rate: float
    forward_calls: int
    backward_calls: int
    optimizer_step_calls: int
    elapsed_wall_milliseconds: int
    cycle_zero_trainable_sha256: str
    cycle_one_start_trainable_sha256: str
    cycle_zero_optimizer_state_sha256: str
    cycle_one_start_optimizer_state_sha256: str
    final_trainable_sha256: str
    final_optimizer_state_sha256: str


@dataclass(frozen=True)
class TransferCase:
    case_id: str
    structure_family_id: str
    prompt: str
    expected_label: int


@dataclass(frozen=True)
class PriorCaseObservation:
    case_id: str
    before_correct: bool
    after_correct: bool


def build_frozen_plan(
    scope: TrainingScopeSpec, *, run_id: str, source_head: str,
    source_tree: str, base_model_sha256: str,
    initial_trainable_sha256: str,
    initial_optimizer_state_sha256: str,
    private_transfer_roster_commitment: str,
    private_final_bank_commitment: str,
    retention_case_ids: tuple[str, ...],
    maximum_prior_correct_regressions: int,
    steps_per_cycle: int = 8, nominal_learning_rate: float = .001,
    maximum_wall_milliseconds_per_arm: int = 3600000,
    weak_cycle_one: tuple[str, ...] = (),
) -> FrozenFiveArmPlan:
    """Freeze actual unmeasured runner batches; no optimizer is executed."""
    validate_scope(scope)
    if type(steps_per_cycle) is not int or not 4 <= steps_per_cycle <= 16:
        raise ExperimentContractError("source protocol requires 4..16 steps per cycle")
    if (type(nominal_learning_rate) not in (float, int)
            or not 0 < nominal_learning_rate <= .01
            or (type(nominal_learning_rate) is float
                and not math.isfinite(nominal_learning_rate))):
        raise ExperimentContractError("nominal optimizer rate must be finite and in (0, .01]")
    if type(weak_cycle_one) is not tuple or any(type(x) is not str for x in weak_cycle_one):
        raise ExperimentContractError("weakness selection must be an exact tuple")
    visits, expected_labels, prompt_hashes = _canonical_training_payload(
        scope, steps_per_cycle, weak_cycle_one
    )
    fields = dict(
        schema=SCHEMA, run_id=run_id, source_head=source_head,
        source_tree=source_tree, scope_fingerprint=scope.fingerprint,
        scope_seed=scope.seed, scope_specificity=scope.specificity,
        scope_scale=scope.scale, weak_cycle_one=weak_cycle_one,
        steps_per_cycle=steps_per_cycle,
        nominal_learning_rate=float(nominal_learning_rate),
        maximum_wall_milliseconds_per_arm=maximum_wall_milliseconds_per_arm,
        base_model_sha256=base_model_sha256,
        initial_trainable_sha256=initial_trainable_sha256,
        initial_optimizer_state_sha256=initial_optimizer_state_sha256,
        train_visits=visits,
        expected_labels=expected_labels,
        train_prompt_hashes=prompt_hashes,
        generated_family_ids=_TRAIN_FAMILIES,
        private_transfer_roster_commitment=private_transfer_roster_commitment,
        private_final_bank_commitment=private_final_bank_commitment,
        retention_case_ids=retention_case_ids,
        maximum_prior_correct_regressions=maximum_prior_correct_regressions,
    )
    plan = FrozenFiveArmPlan(**fields, source_contract_sha256=_digest(fields))
    plan.validate()
    return plan


def audit_five_arm_claims(plan: FrozenFiveArmPlan, claims: tuple[ArmClaim, ...]) -> dict[str, Any]:
    """Reject mismatched arms/exposures/controls; never attest actual weights."""
    if type(plan) is not FrozenFiveArmPlan:
        raise ExperimentContractError("a frozen source plan is mandatory")
    plan.validate()
    if (type(claims) is not tuple or len(claims) != 5
            or any(type(c) is not ArmClaim for c in claims)):
        raise ExperimentContractError("exactly five typed arm claims required")
    if any(type(c.arm) is not str for c in claims):
        raise ExperimentContractError("exact primitive arm names required")
    if tuple(c.arm for c in claims) != PREREG_ARMS:
        raise ExperimentContractError("control arms omitted or reordered")
    expected_labels = dict(plan.expected_labels)
    steps = plan.steps_per_cycle * 2
    for c in claims:
        for name in ("plan_sha256", "initial_trainable_sha256",
                     "initial_optimizer_state_sha256",
                     "cycle_zero_trainable_sha256",
                     "cycle_one_start_trainable_sha256",
                     "cycle_zero_optimizer_state_sha256",
                     "cycle_one_start_optimizer_state_sha256",
                     "final_trainable_sha256", "final_optimizer_state_sha256"):
            _require_digest(getattr(c, name), name)
        if (type(c.executed_visits) is not tuple
                or any(type(x) is not str for x in c.executed_visits)
                or type(c.executed_labels) is not tuple
                or any(type(x) is not int for x in c.executed_labels)):
            raise ExperimentContractError("exact primitive executed TRAIN visits and labels required")
        if (type(c.arm) is not str or c.plan_sha256 != plan.source_contract_sha256
                or c.initial_trainable_sha256 != plan.initial_trainable_sha256
                or c.initial_optimizer_state_sha256 != plan.initial_optimizer_state_sha256
                or c.executed_visits != plan.train_visits
                or c.executed_labels != expected_labels[c.arm]):
            raise ExperimentContractError("arm claim deviates from frozen TRAIN cases/labels")
        if (type(c.effective_learning_rate) is not float
                or not math.isfinite(c.effective_learning_rate)
                or c.effective_learning_rate != optimizer_learning_rate(
                    c.arm, plan.nominal_learning_rate)):
            raise ExperimentContractError("control learning rate differs from declared optimizer")
        if (any(type(x) is not int for x in (
                c.forward_calls, c.backward_calls, c.optimizer_step_calls,
                c.elapsed_wall_milliseconds))
                or c.forward_calls != steps or c.backward_calls != steps
                or c.optimizer_step_calls != (0 if c.arm == "zero_update" else steps)
                or not 0 <= c.elapsed_wall_milliseconds <= plan.maximum_wall_milliseconds_per_arm):
            raise ExperimentContractError("arm exposure, backprop, optimizer or time budget mismatch")
        if (c.arm in ("zero_update", "zero_lr")
                and c.final_trainable_sha256 != plan.initial_trainable_sha256):
            raise ExperimentContractError("zero-weight-change control claims updated parameters")
        if (c.arm == "zero_update"
                and (c.cycle_zero_optimizer_state_sha256 != plan.initial_optimizer_state_sha256
                     or c.final_optimizer_state_sha256 != plan.initial_optimizer_state_sha256)):
            raise ExperimentContractError("zero-update control altered optimizer moments")
        if (c.arm != "zero_update"
                and (c.cycle_zero_optimizer_state_sha256 == plan.initial_optimizer_state_sha256
                     or c.final_optimizer_state_sha256 == plan.initial_optimizer_state_sha256)):
            raise ExperimentContractError("stepped optimizer shows no state evolution")
        if (c.cycle_zero_trainable_sha256 != c.cycle_one_start_trainable_sha256
                or c.cycle_zero_optimizer_state_sha256 != c.cycle_one_start_optimizer_state_sha256):
            raise ExperimentContractError("learner tensor/optimizer state reset with Maze regeneration")
        if (c.arm in ("zero_update", "zero_lr")
                and c.cycle_zero_trainable_sha256 != plan.initial_trainable_sha256):
            raise ExperimentContractError("zero-weight-change control moved before regeneration")
        if (c.arm in ("correct", "inverted", "randomized_labels")
                and c.cycle_zero_trainable_sha256 == plan.initial_trainable_sha256):
            raise ExperimentContractError("first training cycle shows no changed learned parameters")
        if (c.arm in ("correct", "inverted", "randomized_labels")
                and c.final_trainable_sha256 == plan.initial_trainable_sha256):
            raise ExperimentContractError("trained arm reports no parameter changes")
    return {
        "status": "SOURCE_ONLY_UNATTESTED",
        "plan_sha256": plan.source_contract_sha256,
        "five_arm_claims_present": True,
        "exposures_and_budgets_source_consistent": True,
        "physical_optimizer_execution_verified": False,
        "checkpoint_bytes_independently_read": False,
        "independent_evaluation_verified": False,
        "can_authorize_graduation": False,
    }


def transfer_roster_commitment(
    cases: tuple[TransferCase, ...], private_salt: str,
) -> str:
    if (type(cases) is not tuple or not cases
            or any(type(c) is not TransferCase for c in cases)
            or type(private_salt) is not str or not re.fullmatch(r"[0-9a-f]{64}", private_salt)):
        raise ExperimentContractError("typed private transfer roster and random salt required")
    for case in cases:
        _require_id(case.case_id, "transfer case")
        _require_id(case.structure_family_id, "transfer family")
        if type(case.expected_label) is not int or case.expected_label not in (0, 1):
            raise ExperimentContractError("binary evaluator reference label required")
        _normalized(case.prompt)
    return _digest({"domain": "labyrinth.private-transfer-roster.v1",
                    "salt": private_salt, "cases": [asdict(c) for c in cases]})


def audit_transfer_roster(
    plan: FrozenFiveArmPlan, cases: tuple[TransferCase, ...], private_salt: str,
) -> dict[str, Any]:
    """Source overlap veto; semantic independence requires external human QA."""
    if type(plan) is not FrozenFiveArmPlan:
        raise ExperimentContractError("typed plan required")
    plan.validate()
    if transfer_roster_commitment(cases, private_salt) != plan.private_transfer_roster_commitment:
        raise ExperimentContractError("uncommitted transfer roster or secret salt")
    if len(set(c.case_id for c in cases)) != len(cases):
        raise ExperimentContractError("reused private transfer case IDs")
    if len(set(_normalized(c.prompt) for c in cases)) != len(cases):
        raise ExperimentContractError("duplicate private transfer prompts")
    for case in cases:
        normalized = _normalized(case.prompt)
        if (sha256(normalized.encode("utf-8")).hexdigest() in plan.train_prompt_hashes
                or case.structure_family_id in plan.generated_family_ids
                or _approved_words(normalized) & _APPROVED_ADJECTIVES):
            raise ExperimentContractError("transfer case overlaps TRAIN text/family/approved cue")
    return {
        "status": "SOURCE_OVERLAP_VETO_ONLY",
        "source_lexical_overlap_detected": False,
        "independent_semantic_transfer_verified": False,
        "private_bank_custody_verified": False,
        "can_authorize_graduation": False,
    }


def audit_prior_skill_retention(
    plan: FrozenFiveArmPlan, observations: tuple[PriorCaseObservation, ...],
) -> dict[str, Any]:
    """Count previously mastered cases lost, without net-accuracy masking."""
    if type(plan) is not FrozenFiveArmPlan:
        raise ExperimentContractError("typed plan required")
    plan.validate()
    if (type(observations) is not tuple
            or len(observations) != len(plan.retention_case_ids)
            or any(type(x) is not PriorCaseObservation for x in observations)):
        raise ExperimentContractError("complete paired prior-skill observations required")
    if any(type(x.case_id) is not str or type(x.before_correct) is not bool
           or type(x.after_correct) is not bool for x in observations):
        raise ExperimentContractError("typed paired outcomes required")
    if tuple(sorted(x.case_id for x in observations)) != tuple(sorted(plan.retention_case_ids)):
        raise ExperimentContractError("retention case IDs drifted from plan")
    prior_correct = sum(x.before_correct for x in observations)
    regressed = sum(x.before_correct and not x.after_correct for x in observations)
    if prior_correct == 0:
        raise ExperimentContractError("no established baseline cases")
    if prior_correct * 4 < len(observations) * 3:
        raise ExperimentContractError("prior skill baseline accuracy below 75%")
    if regressed > plan.maximum_prior_correct_regressions:
        raise ExperimentContractError("previously mastered cases regressed beyond precommitment")
    return {
        "status": "SOURCE_RETENTION_ACCOUNTING_ONLY",
        "previously_correct_count": prior_correct,
        "previously_correct_regressed": regressed,
        "independent_observation_verified": False,
        "can_authorize_graduation": False,
    }


def audit_distinct_eval_seeds(plans: tuple[FrozenFiveArmPlan, ...]) -> None:
    """Block reused source commitments; private semantic reuse remains unknown."""
    if (type(plans) is not tuple or len(plans) < 2
            or any(type(p) is not FrozenFiveArmPlan for p in plans)):
        raise ExperimentContractError("at least two typed seed plans required")
    for plan in plans:
        plan.validate()
    for name in ("run_id", "scope_seed", "private_transfer_roster_commitment",
                 "private_final_bank_commitment"):
        values = [getattr(p, name) for p in plans]
        if len(set(values)) != len(values):
            raise ExperimentContractError(f"independent seed plans reuse {name}")
