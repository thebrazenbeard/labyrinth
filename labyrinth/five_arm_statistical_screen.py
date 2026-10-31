"""Prespecified source-only statistical screen for five-arm MiniLM results.

Each independent seed is ONE statistical unit. Reusing the same evaluation
cases across seeds is rejected rather than counting repeated prompts as
independent samples. No model, evaluator, signer or private case bank is
accessed here. The caller supplies correctness bits; even a passing screen
is never evidence that a physical model was trained or independently graded.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import comb
import re

from labyrinth.five_arm_experiment_contract import (
    FrozenFiveArmPlan, PriorCaseObservation, audit_distinct_eval_seeds,
    audit_prior_skill_retention,
)
from scripts.qualify_maze_conditioned_minilm_cpu import PREREG_ARMS

SCHEMA = "labyrinth.five-arm-seed-paired-source-screen.v1"
CONTROLS = tuple(arm for arm in PREREG_ARMS if arm != "correct")
SPLITS = ("development", "transfer")
MINIMUM_SEEDS = 8
MAXIMUM_SEEDS = 64
MINIMUM_CASES_PER_SPLIT = 20
MAXIMUM_CASES_PER_SPLIT = 512
MINIMUM_MEAN_GAIN = 0.10
FAMILYWISE_ALPHA = 0.05
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class StatisticalScreenError(ValueError):
    """The proposed outcome data contradict the frozen statistical screen."""


def _is_digest(value: object) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _case_digest(cases: tuple["PairedCase", ...]) -> str:
    # This binds the supplied pseudonyms only; it is NOT an external precommit.
    return sha256(("\n".join(c.case_sha256 for c in cases) + "\n").encode()).hexdigest()


@dataclass(frozen=True)
class PairedCase:
    case_sha256: str
    # Same case evaluated by all five arms, in the declared exact order.
    arm_correctness: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class SeedOutcome:
    plan: FrozenFiveArmPlan
    development_case_digest: str
    transfer_case_digest: str
    development: tuple[PairedCase, ...]
    transfer: tuple[PairedCase, ...]
    prior_retention: tuple[PriorCaseObservation, ...]


def _exact_sign_tail(wins: int, losses: int) -> float:
    """One-sided exact sign test; ties are not independent wins/trials."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(wins, n + 1)) / (2**n)


def screen_five_arm_seed_outcomes(
    outcomes: tuple[SeedOutcome, ...],
) -> dict[str, object]:
    """Return an explicitly untrusted source screen; NEVER permit graduation.

    Fixed preregistered method: eight seed-level paired sign tests (correct vs
    each of four controls in both splits), Holm familywise 0.05 correction,
    minimum mean correct-label advantage of 10 percentage points, eight
    distinct seed plans, twenty pseudonymous cases in EACH split per seed,
    no repeated case hashes across ANY seed or split, and mandatory prior
    retention accounting for each seed. Inputs remain self-reported.
    """
    if (type(outcomes) is not tuple
            or not MINIMUM_SEEDS <= len(outcomes) <= MAXIMUM_SEEDS
            or any(type(item) is not SeedOutcome for item in outcomes)):
        raise StatisticalScreenError("eight to 64 exact seed outcomes required")
    plans = tuple(item.plan for item in outcomes)
    if any(type(plan) is not FrozenFiveArmPlan for plan in plans):
        raise StatisticalScreenError("typed five-arm plan required for each seed")
    try:
        audit_distinct_eval_seeds(plans)
    except ValueError as exc:
        raise StatisticalScreenError("seed plans reuse preregistered identities") from exc

    common = ("source_head", "source_tree", "base_model_sha256",
              "initial_trainable_sha256", "initial_optimizer_state_sha256",
              "scope_specificity", "scope_scale", "steps_per_cycle",
              "nominal_learning_rate", "maximum_wall_milliseconds_per_arm",
              "weak_cycle_one")
    if any(getattr(plan, field) != getattr(plans[0], field)
           for plan in plans[1:] for field in common):
        raise StatisticalScreenError("seeds have nonmatched initial model/optimizer or budgets")

    all_cases: set[str] = set()
    correct_by_split: dict[str, list[float]] = {split: [] for split in SPLITS}
    advantage: dict[tuple[str, str], list[float]] = {
        (split, control): [] for split in SPLITS for control in CONTROLS
    }
    retention_counts: list[tuple[int, int]] = []
    for seed in outcomes:
        try:
            retention = audit_prior_skill_retention(seed.plan, seed.prior_retention)
        except ValueError as exc:
            raise StatisticalScreenError("unqualified source prior-skill retention") from exc
        retention_counts.append((retention["previously_correct_count"],
                                 retention["previously_correct_regressed"]))
        for split in SPLITS:
            cases = getattr(seed, split)
            commitment = getattr(seed, f"{split}_case_digest")
            if (type(cases) is not tuple
                    or not MINIMUM_CASES_PER_SPLIT <= len(cases) <= MAXIMUM_CASES_PER_SPLIT
                    or any(type(case) is not PairedCase for case in cases)
                    or not _is_digest(commitment)):
                raise StatisticalScreenError("complete typed case roster commitment required")
            current: set[str] = set()
            count_by_arm = dict.fromkeys(PREREG_ARMS, 0)
            for case in cases:
                if (not _is_digest(case.case_sha256)
                        or case.case_sha256 in current
                        or case.case_sha256 in all_cases
                        or type(case.arm_correctness) is not tuple
                        or len(case.arm_correctness) != len(PREREG_ARMS)):
                    raise StatisticalScreenError("duplicate or malformed independent evaluation case")
                current.add(case.case_sha256)
                for i, value in enumerate(case.arm_correctness):
                    if (type(value) is not tuple or len(value) != 2
                            or type(value[0]) is not str
                            or value[0] != PREREG_ARMS[i]
                            or type(value[1]) is not bool):
                        raise StatisticalScreenError("five ordered exact binary arm outcomes required")
                    count_by_arm[value[0]] += int(value[1])
            if commitment != _case_digest(cases):
                raise StatisticalScreenError("evaluated case pseudonyms changed after declared digest")
            all_cases.update(current)
            correct_accuracy = count_by_arm["correct"] / len(cases)
            correct_by_split[split].append(correct_accuracy)
            for control in CONTROLS:
                advantage[(split, control)].append(
                    correct_accuracy - count_by_arm[control] / len(cases))

    results: list[dict[str, object]] = []
    for split in SPLITS:
        for control in CONTROLS:
            deltas = advantage[(split, control)]
            wins = sum(delta > 0 for delta in deltas)
            losses = sum(delta < 0 for delta in deltas)
            p = _exact_sign_tail(wins, losses)
            mean_gain = sum(deltas) / len(deltas)
            results.append({"split": split, "control": control,
                            "seed_wins": wins, "seed_losses": losses,
                            "seed_ties": len(deltas) - wins - losses,
                            "mean_gain": mean_gain, "unadjusted_p": p})
    # Holm's step-down correction is fixed BEFORE inspecting the claims.
    order = sorted(range(len(results)), key=lambda i: results[i]["unadjusted_p"])
    familywise_pass = True
    for rank, index in enumerate(order):
        threshold = FAMILYWISE_ALPHA / (len(results) - rank)
        eligible = (familywise_pass
                    and results[index]["unadjusted_p"] <= threshold
                    and results[index]["mean_gain"] >= MINIMUM_MEAN_GAIN)
        results[index]["holm_pass"] = eligible
        familywise_pass = eligible
    return {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_UNATTESTED_STATISTICAL_SCREEN",
        "source_screen_pass": all(result["holm_pass"] for result in results),
        "seed_count": len(outcomes),
        "minimum_cases_per_split": MINIMUM_CASES_PER_SPLIT,
        "minimum_mean_gain": MINIMUM_MEAN_GAIN,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "method": "one_sided_seed_level_exact_sign_holm_8_comparisons",
        "comparisons": tuple(results),
        "retention_source_counts": tuple(retention_counts),
        "actual_model_updates_verified": False,
        "independent_case_custody_verified": False,
        "semantic_transfer_verified": False,
        "physical_retention_verified": False,
        "can_authorize_training": False,
        "can_authorize_graduation": False,
    }
