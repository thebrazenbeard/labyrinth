"""Deterministic, matched-exposure schedule for a future neural pilot.

The plan has *no* answer keys, inference results, optimizer, or model runtime.
It prevents declared development/final case IDs from entering the generated
training schedule, but cannot authenticate the caller's split assignments,
prove structural disjointness or verify independent evaluator custody.
Public deterministic IDs and catalog hashes are not a secrecy boundary.
No independent/final-set membership commitment is released by this planner;
final-set attestation belongs to an external evaluator with private custody.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Sequence


_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z", re.ASCII)
_ARMS = ("trained", "sham", "corrupted")
_MAX_CASES = 100_000
_MAX_STEPS = 100_000


def _ids(values: Sequence[str], split: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{split} IDs must be a sequence")
    if not 1 <= len(values) <= _MAX_CASES:
        raise ValueError(f"{split} IDs must contain 1 to {_MAX_CASES} cases")
    if any(type(value) is not str or _ID_RE.fullmatch(value) is None for value in values):
        raise ValueError(f"{split} IDs must be bounded printable ASCII identifiers")
    if len(set(values)) != len(values):
        raise ValueError(f"{split} contains duplicate case IDs")
    return tuple(sorted(values))


def _digest_ids(split: str, values: tuple[str, ...]) -> str:
    # Length-delimited canonical JSON prevents concatenation ambiguities.
    message = json.dumps(["labyrinth-trial-split-v1", split, values], separators=(",", ":"))
    return hashlib.sha256(message.encode("ascii")).hexdigest()


def _permute(train_ids: tuple[str, ...], seed: int, cycle: int) -> list[str]:
    def rank(identifier: str) -> tuple[bytes, str]:
        message = f"labyrinth-trial-exposure-v1\x00{seed}\x00{cycle}\x00{identifier}".encode("ascii")
        return hashlib.sha256(message).digest(), identifier

    return sorted(train_ids, key=rank)


@dataclass(frozen=True)
class TrialPlan:
    """Immutable rehearsal schedule; exposure parity alone is not training parity."""

    seed: int
    steps: int
    available_training_cases: int
    development_case_count: int
    independent_case_count: int
    train_ids_commitment_sha256: str
    development_ids_commitment_sha256: str
    scheduled_case_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        scheduled = list(self.scheduled_case_ids)
        body = {
            "schema_version": 1,
            "qualification": "control_schedule_only",
            "sampler": "sha256_seeded_epoch_permutation_v1",
            "seed": self.seed,
            "available_training_cases": self.available_training_cases,
            "development_case_count": self.development_case_count,
            "independent_case_count": self.independent_case_count,
            "train_ids_commitment_sha256": self.train_ids_commitment_sha256,
            "development_ids_commitment_sha256": self.development_ids_commitment_sha256,
            "scheduled_training_visits": self.steps,
            "unique_cases_scheduled": len(set(scheduled)),
            "scheduled_case_ids": scheduled,
            "arms": list(_ARMS),
            "schedule_by_arm": {arm: list(scheduled) for arm in _ARMS},
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        body["manifest_sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
        return body


def build_trial_plan(
    *,
    train_case_ids: Sequence[str],
    development_case_ids: Sequence[str],
    independent_case_ids: Sequence[str],
    steps: int,
    seed: int,
) -> TrialPlan:
    """Plan identical case exposures for all three intervention arms.

    The sampler covers every available training ID before repeating an ID.
    Inputs are identifiers only: no gold labels or response payloads are read.
    A future trainer must separately attest *completed* updates, token budgets,
    checkpoint custody and fresh-process inference for each matched arm.
    Independent IDs are checked only for collisions and counts. Their public
    commitment would enable dictionary guessing of predictable hidden IDs.
    This receipt is not pre-registration or proof of sealed final-set custody;
    any private evaluator commitment must be maintained outside this module.
    """
    if type(steps) is not int or not 1 <= steps <= _MAX_STEPS:
        raise ValueError(f"steps must be an integer from 1 to {_MAX_STEPS}")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    train = _ids(train_case_ids, "train")
    development = _ids(development_case_ids, "development")
    independent = _ids(independent_case_ids, "independent")
    if len(set(train) | set(development) | set(independent)) != (
        len(train) + len(development) + len(independent)
    ):
        raise ValueError("case IDs overlap across splits")

    visits: list[str] = []
    cycle = 0
    while len(visits) < steps:
        visits.extend(_permute(train, seed, cycle)[: steps - len(visits)])
        cycle += 1
    return TrialPlan(
        seed=seed,
        steps=steps,
        available_training_cases=len(train),
        development_case_count=len(development),
        independent_case_count=len(independent),
        train_ids_commitment_sha256=_digest_ids("train", train),
        development_ids_commitment_sha256=_digest_ids("development", development),
        scheduled_case_ids=tuple(visits),
    )
