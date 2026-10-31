"""Strict, opt-in arithmetic semantic veto before retention development admission.

This narrow source companion closes a known *single-addition* overlap route:
different spelling and structure-family IDs cannot make the same arithmetic
instance independent. It does not recognize general-language paraphrases,
establish the completeness of the caller's TRAIN inventory, authenticate
model execution, hold a private exam, or confer graduation. The two trusted
digest expectations MUST be frozen by an external evaluator before seeing
candidate results. This module does not install such a custodian.

Both forms of one retention pair must identify the same semantic situation;
different pairs must identify different situations. Unknown grammar blocks
instead of being silently accepted as a new situation.
"""

from __future__ import annotations

from hashlib import sha256
import hmac
import json
import re
from typing import Any

from .arithmetic_semantic_case_key import (
    ArithmeticCaseOverlap, UnsupportedArithmeticCase,
    arithmetic_semantic_case_key, require_disjoint_arithmetic_cases,
)
from .retention_entrypoint_guard import (
    RetentionAdmission, evaluate_retention_admission,
)
from .retention_qualification import (
    RetentionPair, RetentionPolicy, retention_roster_sha256,
)


_DOMAIN = b"labyrinth.arithmetic-retention-training-inventory.v1\0"
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def arithmetic_training_inventory_sha256(prompts: tuple[str, ...]) -> str:
    """Versioned raw-TRAIN inventory commitment, for external precommit.

    This is a hash only; generating it from trainee-owned inventory does not
    independently establish that all training exposures were inventoried.
    """
    if (type(prompts) is not tuple or not 1 <= len(prompts) <= 100_000
            or any(type(prompt) is not str for prompt in prompts)):
        raise ValueError("bounded, typed TRAIN prompt inventory required")
    raw = (json.dumps(list(prompts), ensure_ascii=True,
                      separators=(",", ":")) + "\n").encode("ascii")
    return sha256(_DOMAIN + raw).hexdigest()


def evaluate_arithmetic_retention_admission(
    request: Any,
    policy: RetentionPolicy,
    roster: tuple[RetentionPair, ...],
    evidence: Any,
    signed_taxonomy: Any,
    *,
    trusted_training_inventory_sha256: str,
    trusted_retention_roster_sha256: str,
    **custody_inputs: Any,
) -> RetentionAdmission:
    """Veto known semantic leakage, then enforce the signed retention gate.

    The evaluator must pin *both* expectations separately from the candidate
    and pass the existing evaluator/taxonomy custody inputs. This function
    never returns a graduation credential; even PASS means only development
    retention review survived this narrow veto.
    """
    blocked = lambda: RetentionAdmission(
        "BLOCKED", "arithmetic semantic overlap, unsupported expression, or unpinned roster", None)
    if (type(policy) is not RetentionPolicy or type(roster) is not tuple
            or type(trusted_training_inventory_sha256) is not str
            or type(trusted_retention_roster_sha256) is not str
            or _HEX.fullmatch(trusted_training_inventory_sha256) is None
            or _HEX.fullmatch(trusted_retention_roster_sha256) is None):
        return blocked()

    try:
        policy.__post_init__()
        expected_training = arithmetic_training_inventory_sha256(
            policy.excluded_training_prompts)
        if (not hmac.compare_digest(expected_training,
                                   trusted_training_inventory_sha256)
                or not hmac.compare_digest(retention_roster_sha256(roster),
                                           trusted_retention_roster_sha256)
                or not hmac.compare_digest(policy.expected_roster_sha256,
                                           trusted_retention_roster_sha256)):
            return blocked()
        if any(type(pair) is not RetentionPair for pair in roster):
            return blocked()
        training = policy.excluded_training_prompts
        # Reuse the narrow existing grammar rather than inventing a second
        # semantic judgment; all forms must be representable.
        forms = tuple(
            text for pair in roster
            for text in (pair.before_prompt_text, pair.after_prompt_text)
        )
        train_keys, _ = require_disjoint_arithmetic_cases(training, forms)
        if len(set(train_keys)) != len(train_keys):
            return blocked()
        pair_keys: set[str] = set()
        for pair in roster:
            before = arithmetic_semantic_case_key(pair.before_prompt_text)
            after = arithmetic_semantic_case_key(pair.after_prompt_text)
            if before != after or before in pair_keys:
                return blocked()
            pair_keys.add(before)
    except (ArithmeticCaseOverlap, UnsupportedArithmeticCase,
            TypeError, ValueError, AttributeError, OverflowError):
        return blocked()
    return evaluate_retention_admission(
        request, policy, roster, evidence, signed_taxonomy, **custody_inputs)
