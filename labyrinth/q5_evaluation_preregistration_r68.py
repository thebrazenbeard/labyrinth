"""Q5 source-only pre-outcome roster binding for five-arm MiniLM evaluation.

The statistical screen checks a digest supplied next to the results. Such a
digest is self-consistent even when an operator changes the case roster after
looking at the results. This separate gate requires a different, externally
recorded pre-outcome digest and witness for an *exact* seed/split/case roster.

Neither the witness callback nor any hashes establish trusted time, independent
case custody, a pretrained neural model, semantic holdout, learning, retention,
or graduation. The host must supply and govern those authorities separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import re
from typing import Callable

from labyrinth.five_arm_statistical_screen import (
    FAMILYWISE_ALPHA, MAXIMUM_CASES_PER_SPLIT, MAXIMUM_SEEDS,
    MINIMUM_CASES_PER_SPLIT, MINIMUM_MEAN_GAIN, MINIMUM_SEEDS, SCHEMA as SCREEN_SCHEMA,
    SPLITS, SeedOutcome, StatisticalScreenError, screen_five_arm_seed_outcomes,
)
from labyrinth.maze_conditioned_minilm_cpu_experiment import PREREG_ARMS


SCHEMA = "labyrinth.q5-evaluation-preoutcome-source-roster.v1"
METHOD = "one_sided_seed_level_exact_sign_holm_8_comparisons"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)


class EvaluationPrecommitError(ValueError):
    """An untrusted Q5 evaluation roster is not source-precommitted."""


def _digest(fields: dict[str, object]) -> str:
    data = (json.dumps(fields, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return sha256(data).hexdigest()


def _is_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


@dataclass(frozen=True)
class PredeclaredSeedRoster:
    run_id: str
    scope_seed: int
    plan_sha256: str
    development_cases: tuple[str, ...]
    transfer_cases: tuple[str, ...]


@dataclass(frozen=True)
class FrozenEvaluationPrecommit:
    schema: str
    screen_schema: str
    method: str
    arms: tuple[str, ...]
    splits: tuple[str, ...]
    minimum_seeds: int
    maximum_seeds: int
    minimum_cases_per_split: int
    maximum_cases_per_split: int
    minimum_mean_gain: float
    familywise_alpha: float
    seed_rosters: tuple[PredeclaredSeedRoster, ...]
    precommit_sha256: str

    def validate(self) -> None:
        """Check full policy, exact roster, and canonical source commitment."""
        if (type(self) is not FrozenEvaluationPrecommit
                or type(self.schema) is not str or self.schema != SCHEMA
                or type(self.screen_schema) is not str or self.screen_schema != SCREEN_SCHEMA
                or type(self.method) is not str or self.method != METHOD
                or type(self.arms) is not tuple or self.arms != PREREG_ARMS
                or type(self.splits) is not tuple or self.splits != SPLITS
                or type(self.minimum_seeds) is not int or self.minimum_seeds != MINIMUM_SEEDS
                or type(self.maximum_seeds) is not int or self.maximum_seeds != MAXIMUM_SEEDS
                or type(self.minimum_cases_per_split) is not int
                or self.minimum_cases_per_split != MINIMUM_CASES_PER_SPLIT
                or type(self.maximum_cases_per_split) is not int
                or self.maximum_cases_per_split != MAXIMUM_CASES_PER_SPLIT
                or type(self.minimum_mean_gain) is not float
                or self.minimum_mean_gain != MINIMUM_MEAN_GAIN
                or type(self.familywise_alpha) is not float
                or self.familywise_alpha != FAMILYWISE_ALPHA):
            raise EvaluationPrecommitError("fixed statistical policy changed")
        if (type(self.seed_rosters) is not tuple
                or not MINIMUM_SEEDS <= len(self.seed_rosters) <= MAXIMUM_SEEDS
                or any(type(seed) is not PredeclaredSeedRoster for seed in self.seed_rosters)):
            raise EvaluationPrecommitError("predeclared seed count or type is invalid")

        run_ids: set[str] = set()
        scope_seeds: set[int] = set()
        plans: set[str] = set()
        global_cases: set[str] = set()
        for seed in self.seed_rosters:
            if (type(seed.run_id) is not str or _RUN_ID.fullmatch(seed.run_id) is None
                    or type(seed.scope_seed) is not int or seed.scope_seed < 0):
                raise EvaluationPrecommitError("invalid run or scope seed identifier")
            if not _is_digest(seed.plan_sha256):
                raise EvaluationPrecommitError("invalid plan SHA256")
            if (seed.run_id in run_ids or seed.scope_seed in scope_seeds
                    or seed.plan_sha256 in plans):
                raise EvaluationPrecommitError("duplicate predeclared seed or plan identity")
            run_ids.add(seed.run_id)
            scope_seeds.add(seed.scope_seed)
            plans.add(seed.plan_sha256)
            for split in SPLITS:
                cases = getattr(seed, f"{split}_cases")
                if (type(cases) is not tuple
                        or not MINIMUM_CASES_PER_SPLIT <= len(cases) <= MAXIMUM_CASES_PER_SPLIT
                        or any(not _is_digest(case) for case in cases)):
                    raise EvaluationPrecommitError("invalid predeclared case count or hash")
                for case in cases:
                    if case in global_cases:
                        raise EvaluationPrecommitError("duplicate case across seed or split")
                    global_cases.add(case)
        if not _is_digest(self.precommit_sha256):
            raise EvaluationPrecommitError("invalid precommit digest")
        payload = asdict(self)
        recorded = payload.pop("precommit_sha256")
        if _digest(payload) != recorded:
            raise EvaluationPrecommitError("changed predeclared policy or roster digest")


def build_frozen_evaluation_precommit(
    seed_rosters: tuple[PredeclaredSeedRoster, ...],
) -> FrozenEvaluationPrecommit:
    """Build the roster **before** outcomes; this cannot attest when it was built."""
    fields = dict(
        schema=SCHEMA, screen_schema=SCREEN_SCHEMA, method=METHOD,
        arms=PREREG_ARMS, splits=SPLITS,
        minimum_seeds=MINIMUM_SEEDS, maximum_seeds=MAXIMUM_SEEDS,
        minimum_cases_per_split=MINIMUM_CASES_PER_SPLIT,
        maximum_cases_per_split=MAXIMUM_CASES_PER_SPLIT,
        minimum_mean_gain=MINIMUM_MEAN_GAIN, familywise_alpha=FAMILYWISE_ALPHA,
        seed_rosters=seed_rosters,
    )
    if (type(seed_rosters) is not tuple
            or not MINIMUM_SEEDS <= len(seed_rosters) <= MAXIMUM_SEEDS
            or any(type(item) is not PredeclaredSeedRoster for item in seed_rosters)):
        raise EvaluationPrecommitError("typed bounded predeclared seed tuple required")
    # Bounds before dataclass serialization prevent unbounded self-supplied input.
    for seed in seed_rosters:
        for split in SPLITS:
            cases = getattr(seed, f"{split}_cases")
            if (type(cases) is not tuple
                    or not MINIMUM_CASES_PER_SPLIT <= len(cases) <= MAXIMUM_CASES_PER_SPLIT
                    or any(not _is_digest(value) for value in cases)):
                raise EvaluationPrecommitError("invalid predeclared case count or hash")
    provisional = FrozenEvaluationPrecommit(**fields, precommit_sha256="0" * 64)
    payload = asdict(provisional)
    payload.pop("precommit_sha256")
    # The dataclass validator enforces every identity, duplicate, and policy bound.
    frozen = FrozenEvaluationPrecommit(**fields, precommit_sha256=_digest(payload))
    frozen.validate()
    return frozen


def screen_predeclared_evaluation(
    *, precommit: FrozenEvaluationPrecommit,
    external_expected_precommit_sha256: str,
    preoutcome_witness: bytes,
    verify_external_preoutcome_witness: Callable[[bytes, str], bool],
    outcomes: tuple[SeedOutcome, ...],
) -> dict[str, object]:
    """Reject self-recomputed rosters unless an EXTERNAL exact digest is pinned.

    The caller MUST supply the expected digest from an independently governed
    pre-outcome record and its own verifier. This Python process cannot prove
    record origin, timestamp, verifier identity, or non-replay. Even a passing
    source comparison conveys NO learning, retention or graduation authority.
    """
    if type(precommit) is not FrozenEvaluationPrecommit:
        raise EvaluationPrecommitError("typed evaluation precommit required")
    FrozenEvaluationPrecommit.validate(precommit)
    # A frozen dataclass does not protect against object.__setattr__. In
    # particular, a caller-controlled witness callback executes between the
    # first digest check and roster comparison. Use a separate typed snapshot
    # (including *new* nested roster instances), never the callback-visible
    # original, for the remainder of the audit.
    sealed = replace(precommit, seed_rosters=tuple(
        replace(seed) for seed in precommit.seed_rosters))
    FrozenEvaluationPrecommit.validate(sealed)
    if (not _is_digest(external_expected_precommit_sha256)
            or sealed.precommit_sha256 != external_expected_precommit_sha256):
        raise EvaluationPrecommitError("external expected precommit digest differs")
    if (type(preoutcome_witness) is not bytes
            or not 16 <= len(preoutcome_witness) <= 4096
            or not callable(verify_external_preoutcome_witness)):
        raise EvaluationPrecommitError("external witness is missing or malformed")
    try:
        witness_accepted = verify_external_preoutcome_witness(
            preoutcome_witness, external_expected_precommit_sha256)
    except Exception as exc:
        raise EvaluationPrecommitError("external witness verifier failed closed") from exc
    if witness_accepted is not True:
        raise EvaluationPrecommitError("external witness was not accepted")
    if precommit != sealed:
        raise EvaluationPrecommitError("precommit mutated during witness verification")
    # Same-interpreter callbacks can inspect their caller frame and corrupt the
    # otherwise private `sealed` local as well as the original. Equality then
    # passes. Re-derive the canonical commitment AFTER callback execution.
    try:
        FrozenEvaluationPrecommit.validate(sealed)
    except EvaluationPrecommitError as exc:
        raise EvaluationPrecommitError("post-witness sealed digest changed") from exc
    if sealed.precommit_sha256 != external_expected_precommit_sha256:
        raise EvaluationPrecommitError("post-witness external digest pin changed")

    if (type(outcomes) is not tuple
            or len(outcomes) != len(sealed.seed_rosters)
            or any(type(item) is not SeedOutcome for item in outcomes)):
        raise EvaluationPrecommitError("outcome seed order/count differs from preregistration")
    for expected, observed in zip(sealed.seed_rosters, outcomes, strict=True):
        plan = observed.plan
        if (getattr(plan, "run_id", None) != expected.run_id
                or getattr(plan, "scope_seed", None) != expected.scope_seed
                or getattr(plan, "source_contract_sha256", None) != expected.plan_sha256):
            raise EvaluationPrecommitError("outcome seed order/plan differs from preregistration")
        for split in SPLITS:
            observed_cases = getattr(observed, split)
            if (type(observed_cases) is not tuple
                    or tuple(getattr(case, "case_sha256", None) for case in observed_cases)
                    != getattr(expected, f"{split}_cases")):
                raise EvaluationPrecommitError("observed evaluation case roster drifted from precommit")
    try:
        screen = screen_five_arm_seed_outcomes(outcomes)
    except (StatisticalScreenError, ValueError, TypeError, AttributeError) as exc:
        raise EvaluationPrecommitError("unqualified five-arm source statistical screen") from exc
    # Preserve the exact external pin through the final result construction.
    try:
        FrozenEvaluationPrecommit.validate(sealed)
    except EvaluationPrecommitError as exc:
        raise EvaluationPrecommitError("post-screen sealed digest changed") from exc
    if sealed.precommit_sha256 != external_expected_precommit_sha256:
        raise EvaluationPrecommitError("post-screen external digest pin changed")
    return {
        "schema": SCHEMA,
        "status": "EXTERNALLY_PINNED_SOURCE_ROSTER_ONLY_UNATTESTED",
        "precommit_sha256": sealed.precommit_sha256,
        "source_roster_matches_precommit": True,
        "source_screen_pass": screen["source_screen_pass"],
        "screen": screen,
        "independent_preoutcome_witness_custody_verified": False,
        "independent_evaluator_custody_verified": False,
        "physical_pretrained_model_verified": False,
        "semantic_holdout_verified": False,
        "retention_authorized": False,
        "learning_authorized": False,
        "graduation_authorized": False,
    }
