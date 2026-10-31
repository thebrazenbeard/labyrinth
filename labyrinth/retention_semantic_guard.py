"""Optional, arithmetic-only semantic-identity preflight for retention.

This companion checks a *restricted, syntactically recognized* family of
single-addition prompts. It rejects repeated arithmetic instances hidden by
distinct situation IDs, family labels, forms, or commuted operands. Unknown
natural language fails closed; this is not a general semantic judge.

The caller must separately establish a genuinely independent examiner, pinned
policy and training inventory, physical model inference, and protected bank.
This module does not sign evidence, evaluate neural behavior, authorize
graduation, or silently change any existing retention entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .arithmetic_semantic_case_key import (
    UnsupportedArithmeticCase, arithmetic_semantic_case_key,
)
from .retention_qualification import (
    RetentionError, RetentionPair, RetentionPolicy, _campaign,
    retention_roster_sha256,
)


SCHEMA = "labyrinth.retention-arithmetic-semantic-preflight.v1"
_DOMAIN = b"labyrinth.retention-arithmetic-semantic-preflight.v1\0"


class StrictArithmeticRetentionError(RetentionError):
    """Arithmetic semantic identity cannot establish a diverse private roster."""


@dataclass(frozen=True)
class StrictArithmeticPreflight:
    objective_id: str
    roster_sha256: str
    semantic_inventory_sha256: str
    semantic_case_count: int
    schema: str = SCHEMA
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False
    independent_semantics_authenticated: bool = False


def require_distinct_arithmetic_retention(
    policy: RetentionPolicy, roster: tuple[RetentionPair, ...],
) -> StrictArithmeticPreflight:
    """Check a committed retention roster has distinct, supported ADD facts.

    Every before/after pair must paraphrase the *same* single addition,
    each situation must have a different commutative operand key, and no
    retention key may be present in the complete declared TRAIN inventory.
    At least 64 such facts are required, not merely 64 different labels.

    This only checks the supplied bytes; it cannot authenticate whether the
    operator's training inventory is complete or independently held out.
    """
    if type(policy) is not RetentionPolicy or type(roster) is not tuple:
        raise StrictArithmeticRetentionError("exact retention policy and tuple roster required")
    if policy.prior_skill_ids != ("arithmetic",):
        raise StrictArithmeticRetentionError("strict arithmetic-only prior skill required")
    try:
        # Frozen Python dataclasses can still be mutated with object.__setattr__.
        # Revalidate at this entry point rather than relying on construction.
        RetentionPolicy.__post_init__(policy)
        _campaign(policy, roster)
        train_keys = tuple(
            arithmetic_semantic_case_key(prompt)
            for prompt in policy.excluded_training_prompts
        )
    except (RetentionError, ValueError, TypeError) as exc:
        raise StrictArithmeticRetentionError(
            "invalid policy, source roster, or unsupported training arithmetic"
        ) from exc

    seen: set[str] = set()
    train_set = set(train_keys)
    witnessed: list[tuple[str, str]] = []
    for item in roster:
        try:
            before = arithmetic_semantic_case_key(item.before_prompt_text)
            after = arithmetic_semantic_case_key(item.after_prompt_text)
        except UnsupportedArithmeticCase as exc:
            raise StrictArithmeticRetentionError(
                "unsupported arithmetic retention form; cannot establish semantic identity"
            ) from exc
        if before != after:
            raise StrictArithmeticRetentionError(
                "paired forms do not express the same arithmetic computation"
            )
        if before in train_set:
            raise StrictArithmeticRetentionError(
                "retention arithmetic computation overlaps training"
            )
        if before in seen:
            raise StrictArithmeticRetentionError(
                "duplicate arithmetic computation under distinct retention identities"
            )
        seen.add(before)
        witnessed.append((item.situation_id, before))

    if len(seen) < 64:
        raise StrictArithmeticRetentionError(
            "at least 64 distinct arithmetic computations required"
        )
    payload = json.dumps(
        {"schema": SCHEMA, "objective_id": policy.objective_id,
         "roster_sha256": retention_roster_sha256(roster),
         "semantic_cases": witnessed},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return StrictArithmeticPreflight(
        objective_id=policy.objective_id,
        roster_sha256=retention_roster_sha256(roster),
        semantic_inventory_sha256=sha256(_DOMAIN + payload).hexdigest(),
        semantic_case_count=len(seen),
    )
