"""Conservative, separately governed prior-skill retention *development* gate.

This is an opt-in veto substrate: it does not modify the existing graduation
policy, call a neural model, issue a graduation credential, or install an
independent evaluator. A trusted custodian must PRECOMMIT the policy, paired
independent semantic situations/forms, training overlap inventory and HMAC key;
then actually examine both pinned learner checkpoints. HMAC authenticates a
custodian's claim, not the physical inference, case independence, or host
isolation. A passing receipt alone MUST NOT graduate a learner.

For each prior skill the exact one-sided Clopper-Pearson upper confidence
bound is computed on the event: correct BEFORE, incorrect AFTER. Bounding the
deterioration event (ignoring offsetting new gains) is deliberately more
conservative than merely comparing mean pre/post scores. The statistical
interpretation requires independently sampled semantic situations; the
declared structure-family IDs cannot prove this property by themselves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import math
import re
import unicodedata
from typing import Any


SCHEMA = "labyrinth.prior-skill-retention-development.v2"
PROMPT_COMMITMENT_VERSION = "labyrinth.prompt-nfkc-casefold-whitespace.v2"
_PROMPT_DOMAIN = b"labyrinth.prompt-nfkc-casefold-whitespace.v2\0"
_DOMAIN = b"labyrinth.prior-skill-retention-development.v2\0"
_POLICY_DOMAIN = b"labyrinth.prior-skill-retention-policy.v2\0"
_ROSTER_DOMAIN = b"labyrinth.prior-skill-retention-roster.v2\0"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_RECEIPT_BYTES = 1_048_576
_MAX_ROSTER = 6_144


class RetentionError(ValueError):
    """The precommitted retention record is inadequate or untrusted."""


def _digest(value: Any, name: str) -> None:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"invalid {name} SHA-256")


def _identifier(value: Any, name: str) -> None:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {name} identifier")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")


def normalized_prompt_sha256(prompt_text: str) -> str:
    """Versioned commitment to NFKC/casefold/whitespace-normalized text.

    The raw prompt is required in the evaluator's private custody. A caller
    supplying only an old SHA-256 cannot prove the prescribed normalization.
    No raw prompt is put in a signed observation/result receipt.

    Unicode format/invisible characters (category Cf) are prohibited because
    NFKC/casefold/whitespace normalization does not remove all of them, and
    visually identical strings can otherwise receive distinct commitments.
    This does not detect all semantic confusables (such as U+2010 HYPHEN);
    externally governed structure-family separation remains mandatory.
    """
    if type(prompt_text) is not str or not 1 <= len(prompt_text) <= 8192:
        raise ValueError("bounded raw prompt text required for normalized commitment")
    if any(unicodedata.category(char) == "Cf" for char in prompt_text):
        raise ValueError("invisible Unicode format characters forbidden in prompts")
    normalized = unicodedata.normalize(
        "NFKC", unicodedata.normalize("NFKC", prompt_text).casefold()
    )
    if any(unicodedata.category(char) == "Cf" for char in normalized):
        raise ValueError("invisible Unicode format characters forbidden in prompts")
    normalized = " ".join(normalized.split())
    if not normalized or len(normalized.encode("utf-8")) > 8192:
        raise ValueError("empty or oversized normalized prompt")
    return sha256(_PROMPT_DOMAIN + normalized.encode("utf-8")).hexdigest()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RetentionError("duplicate signed evidence key")
        result[key] = value
    return result


@dataclass(frozen=True)
class RetentionPair:
    """Two different private forms for one predeclared semantic situation."""

    skill_id: str
    situation_id: str
    structure_family_id: str
    before_case_id: str
    after_case_id: str
    before_prompt_text: str
    after_prompt_text: str
    before_prompt_sha256: str
    after_prompt_sha256: str


@dataclass(frozen=True)
class RetentionObservation:
    situation_id: str
    before_correct: bool
    after_correct: bool


@dataclass(frozen=True)
class RetentionPolicy:
    """Independent custodian must freeze before any final learner examination."""

    objective_id: str
    scope_sha256: str
    before_checkpoint_sha256: str
    after_checkpoint_sha256: str
    evaluator_sha256: str
    conditions_sha256: str
    session_nonce: str
    prior_skill_ids: tuple[str, ...]
    expected_roster_sha256: str
    prompt_commitment_version: str
    excluded_training_prompts: tuple[str, ...]
    excluded_training_prompt_sha256s: tuple[str, ...]
    excluded_training_structure_family_ids: tuple[str, ...]
    min_units_per_skill: int = 64
    min_before_accuracy: float = 0.75
    max_deterioration_rate: float = 0.05
    significance_alpha: float = 0.05

    def __post_init__(self) -> None:
        _identifier(self.objective_id, "objective")
        for name in ("scope_sha256", "before_checkpoint_sha256",
                     "after_checkpoint_sha256", "evaluator_sha256",
                     "conditions_sha256", "expected_roster_sha256"):
            _digest(getattr(self, name), name)
        if self.before_checkpoint_sha256 == self.after_checkpoint_sha256:
            raise ValueError("retention requires distinct pre/post learner checkpoints")
        if type(self.session_nonce) is not str or not re.fullmatch(
                r"[0-9a-f]{32,128}", self.session_nonce):
            raise ValueError("fresh precommitted retention nonce is required")
        skills = self.prior_skill_ids
        if (type(skills) is not tuple or not 1 <= len(skills) <= 12
                or any(type(skill) is not str or not _IDENTIFIER.fullmatch(skill)
                       for skill in skills) or len(set(skills)) != len(skills)):
            raise ValueError("at least one unique prior skill is mandatory")
        if self.prompt_commitment_version != PROMPT_COMMITMENT_VERSION:
            raise ValueError("explicit current normalized prompt commitment version required")
        prompts = self.excluded_training_prompt_sha256s
        texts = self.excluded_training_prompts
        if (type(prompts) is not tuple or type(texts) is not tuple
                or len(prompts) != len(texts) or not 1 <= len(prompts) <= 100000
                or any(type(d) is not str or not _DIGEST.fullmatch(d) for d in prompts)
                or len(set(prompts)) != len(prompts)):
            raise ValueError("complete independently declared training overlap inventory required")
        if any(normalized_prompt_sha256(text) != digest
               for text, digest in zip(texts, prompts)):
            raise ValueError("training prompt inventory lacks normalized versioned commitments")
        families = self.excluded_training_structure_family_ids
        if (type(families) is not tuple or not 1 <= len(families) <= 100000
                or any(type(item) is not str or not _IDENTIFIER.fullmatch(item)
                       for item in families) or len(set(families)) != len(families)):
            raise ValueError("explicit training structure-family exclusion inventory required")
        if type(self.min_units_per_skill) is not int or not 64 <= self.min_units_per_skill <= 512:
            raise ValueError("minimum independent units must be in [64, 512]")
        for value, limit, name in (
            (self.max_deterioration_rate, 0.05, "deterioration margin"),
            (self.significance_alpha, 0.05, "significance alpha"),
        ):
            if (type(value) not in (float, int) or not math.isfinite(value)
                    or not 0 < value <= limit):
                raise ValueError(f"invalid {name}; cannot weaken default retention gate")
        if (type(self.min_before_accuracy) not in (float, int)
                or not math.isfinite(self.min_before_accuracy)
                or not 0.75 <= self.min_before_accuracy <= 1):
            raise ValueError("prior skill needs established baseline proficiency")


@dataclass(frozen=True)
class SignedRetentionEvidence:
    receipt_bytes: bytes
    signature_hex: str
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False


@dataclass(frozen=True)
class SkillRetentionResult:
    skill_id: str
    total: int
    baseline_correct: int
    baseline_accuracy: float
    deteriorations: int
    upper_deterioration_rate: float
    passed: bool


@dataclass(frozen=True)
class RetentionDecision:
    disposition: str  # PASS or HOLD; neither is a graduation decision.
    groups: tuple[SkillRetentionResult, ...]
    graduation_authorized: bool = False
    model_execution_authenticated: bool = False


def retention_roster_sha256(pairs: tuple[RetentionPair, ...]) -> str:
    if type(pairs) is not tuple or not 1 <= len(pairs) <= _MAX_ROSTER:
        raise RetentionError("invalid bounded retention roster")
    if any(type(pair) is not RetentionPair for pair in pairs):
        raise RetentionError("invalid retention roster member")
    return sha256(_ROSTER_DOMAIN + _canonical([asdict(pair) for pair in pairs])).hexdigest()


def _campaign(policy: RetentionPolicy, roster: tuple[RetentionPair, ...]) -> None:
    if type(policy) is not RetentionPolicy:
        raise RetentionError("precommitted retention policy required")
    # A frozen dataclass prevents normal assignment, not object.__setattr__ or
    # changes made through a compromised in-process caller. The constructor's
    # safety limits must still hold at BOTH signing and decision time. Call
    # the class method explicitly so an injected instance attribute cannot
    # shadow the validator.
    try:
        RetentionPolicy.__post_init__(policy)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise RetentionError("invalid retention policy at custody boundary") from exc
    if not hmac.compare_digest(retention_roster_sha256(roster),
                               policy.expected_roster_sha256):
        raise RetentionError("retention roster commitment does not match policy")
    seen_ids: set[str] = set()
    seen_situations: set[str] = set()
    seen_families: set[str] = set()
    seen_prompts = set(policy.excluded_training_prompt_sha256s)
    count: dict[str, int] = {skill: 0 for skill in policy.prior_skill_ids}
    for pair in roster:
        for name in ("skill_id", "situation_id", "structure_family_id",
                     "before_case_id", "after_case_id"):
            _identifier(getattr(pair, name), name)
        for name in ("before_prompt_sha256", "after_prompt_sha256"):
            _digest(getattr(pair, name), name)
        if (normalized_prompt_sha256(pair.before_prompt_text) != pair.before_prompt_sha256
                or normalized_prompt_sha256(pair.after_prompt_text) != pair.after_prompt_sha256):
            raise RetentionError("retention form lacks normalized prompt commitment")
        if pair.skill_id not in count:
            raise RetentionError("undeclared prior skill in retention roster")
        if pair.structure_family_id in policy.excluded_training_structure_family_ids:
            raise RetentionError("retention form overlaps training structure family")
        if (pair.situation_id in seen_situations
                or pair.structure_family_id in seen_families):
            raise RetentionError("duplicate semantic situation or structure family")
        seen_situations.add(pair.situation_id)
        seen_families.add(pair.structure_family_id)
        for case_id in (pair.before_case_id, pair.after_case_id):
            if case_id in seen_ids:
                raise RetentionError("duplicate retention case ID")
            seen_ids.add(case_id)
        for prompt in (pair.before_prompt_sha256, pair.after_prompt_sha256):
            if prompt in seen_prompts:
                raise RetentionError("repeated or contaminated prompt commitment")
            seen_prompts.add(prompt)
        count[pair.skill_id] += 1
    missing = tuple(skill for skill, n in count.items() if n == 0)
    if missing:
        raise RetentionError("missing prior-skill coverage: " + ", ".join(missing))
    if any(n < policy.min_units_per_skill for n in count.values()):
        raise RetentionError("insufficient independent retention units")


def _observations(roster: tuple[RetentionPair, ...],
                  observed: tuple[RetentionObservation, ...]) -> list[dict[str, Any]]:
    if type(observed) is not tuple or len(observed) != len(roster):
        raise RetentionError("observation identity does not cover exact roster")
    expected = {pair.situation_id for pair in roster}
    result: dict[str, RetentionObservation] = {}
    for item in observed:
        if (type(item) is not RetentionObservation
                or type(item.before_correct) is not bool
                or type(item.after_correct) is not bool
                or item.situation_id not in expected
                or item.situation_id in result):
            raise RetentionError("invalid or reused observation identity")
        result[item.situation_id] = item
    if set(result) != expected:
        raise RetentionError("observation identity does not cover exact roster")
    return [{"situation_id": pair.situation_id,
             "before_correct": result[pair.situation_id].before_correct,
             "after_correct": result[pair.situation_id].after_correct}
            for pair in roster]


def _policy_sha256(policy: RetentionPolicy) -> str:
    return sha256(_POLICY_DOMAIN + _canonical(asdict(policy))).hexdigest()


def _receipt_body(policy: RetentionPolicy, roster: tuple[RetentionPair, ...],
                  observed: tuple[RetentionObservation, ...]) -> dict[str, Any]:
    _campaign(policy, roster)
    return {
        "schema": SCHEMA,
        "policy_sha256": _policy_sha256(policy),
        "objective_id": policy.objective_id,
        "scope_sha256": policy.scope_sha256,
        "before_checkpoint_sha256": policy.before_checkpoint_sha256,
        "after_checkpoint_sha256": policy.after_checkpoint_sha256,
        "evaluator_sha256": policy.evaluator_sha256,
        "conditions_sha256": policy.conditions_sha256,
        "session_nonce": policy.session_nonce,
        "roster_sha256": policy.expected_roster_sha256,
        "observations": _observations(roster, observed),
        "graduation_authorized": False,
        "model_execution_authenticated": False,
    }


def _key(value: Any) -> None:
    if type(value) is not bytes or not 32 <= len(value) <= 4096:
        raise RetentionError("independent evaluator HMAC key required")


def sign_retention_development_evidence(
    policy: RetentionPolicy,
    roster: tuple[RetentionPair, ...],
    observed: tuple[RetentionObservation, ...], *,
    independent_evaluator_key: bytes,
) -> SignedRetentionEvidence:
    """Evaluator-side signing; this does not run the declared model."""
    _key(independent_evaluator_key)
    raw = _canonical(_receipt_body(policy, roster, observed))
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise RetentionError("retention evidence exceeds bounded bytes")
    signature = hmac.new(independent_evaluator_key, _DOMAIN + raw, sha256).hexdigest()
    return SignedRetentionEvidence(raw, signature)


def _binomial_cdf(n: int, k: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k == n else 0.0
    lp = math.log(p)
    lq = math.log1p(-p)
    gam_n = math.lgamma(n + 1)
    terms = [gam_n - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * lp + (n - i) * lq for i in range(k + 1)]
    maximum = max(terms)
    return math.exp(maximum) * sum(math.exp(item - maximum) for item in terms)


def one_sided_exact_deterioration_upper(n: int, losses: int,
                                        alpha: float = 0.05) -> float:
    """Clopper-Pearson upper bound for independent deterioration indicators."""
    if (type(n) is not int or not 1 <= n <= _MAX_ROSTER
            or type(losses) is not int or not 0 <= losses <= n
            or type(alpha) not in (float, int) or not math.isfinite(alpha)
            or not 0 < alpha <= 0.05):
        raise ValueError("invalid exact binomial confidence-bound inputs")
    if losses == n:
        return 1.0
    if losses == 0:
        return -math.expm1(math.log(alpha) / n)
    low, high = 0.0, 1.0
    for _ in range(64):
        mid = (low + high) / 2
        if _binomial_cdf(n, losses, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def decide_retention(
    policy: RetentionPolicy,
    roster: tuple[RetentionPair, ...],
    evidence: SignedRetentionEvidence, *,
    independent_evaluator_key: bytes,
) -> RetentionDecision:
    """Return an independent prior-skill retention veto, never graduation."""
    _key(independent_evaluator_key)
    _campaign(policy, roster)
    if (type(evidence) is not SignedRetentionEvidence
            or evidence.graduation_authorized is not False
            or evidence.model_execution_authenticated is not False
            or type(evidence.receipt_bytes) is not bytes
            or not 0 < len(evidence.receipt_bytes) <= _MAX_RECEIPT_BYTES
            or type(evidence.signature_hex) is not str
            or not _DIGEST.fullmatch(evidence.signature_hex)):
        raise RetentionError("invalid retention evidence or claimed authority")
    expected = hmac.new(independent_evaluator_key,
                        _DOMAIN + evidence.receipt_bytes, sha256).hexdigest()
    if not hmac.compare_digest(expected, evidence.signature_hex):
        raise RetentionError("retention signature is not authentic")
    try:
        body = json.loads(evidence.receipt_bytes, object_pairs_hook=_unique_keys)
    except (ValueError, UnicodeError) as exc:
        raise RetentionError("invalid signed retention receipt JSON") from exc
    if (type(body) is not dict or _canonical(body) != evidence.receipt_bytes
            or set(body) != {"schema", "policy_sha256", "objective_id", "scope_sha256",
                             "before_checkpoint_sha256", "after_checkpoint_sha256",
                             "evaluator_sha256", "conditions_sha256", "session_nonce",
                             "roster_sha256", "observations", "graduation_authorized",
                             "model_execution_authenticated"}
            or body["schema"] != SCHEMA
            or body["graduation_authorized"] is not False
            or body["model_execution_authenticated"] is not False):
        raise RetentionError("invalid retention receipt schema or authority")
    for name, expected_value in (
        ("policy_sha256", _policy_sha256(policy)),
        ("objective_id", policy.objective_id),
        ("scope_sha256", policy.scope_sha256),
        ("before_checkpoint_sha256", policy.before_checkpoint_sha256),
        ("after_checkpoint_sha256", policy.after_checkpoint_sha256),
        ("evaluator_sha256", policy.evaluator_sha256),
        ("conditions_sha256", policy.conditions_sha256),
        ("session_nonce", policy.session_nonce),
        ("roster_sha256", policy.expected_roster_sha256),
    ):
        if body[name] != expected_value:
            raise RetentionError("retention policy or model identity mismatch")
    raw = body["observations"]
    if type(raw) is not list or len(raw) != len(roster):
        raise RetentionError("invalid signed observations")
    observations = []
    for item in raw:
        if (type(item) is not dict
                or item.keys() != {"situation_id", "before_correct", "after_correct"}
                or type(item.get("situation_id")) is not str
                or type(item.get("before_correct")) is not bool
                or type(item.get("after_correct")) is not bool):
            raise RetentionError("malformed signed observation")
        observations.append(RetentionObservation(
            item["situation_id"], item["before_correct"], item["after_correct"]
        ))
    _observations(roster, tuple(observations))
    by_skill: dict[str, list[RetentionObservation]] = {
        skill: [] for skill in policy.prior_skill_ids
    }
    for pair, observation in zip(roster, observations):
        if pair.situation_id != observation.situation_id:
            raise RetentionError("signed observations not in committed roster order")
        by_skill[pair.skill_id].append(observation)
    groups = []
    for skill in policy.prior_skill_ids:
        items = by_skill[skill]
        baseline_correct = sum(int(item.before_correct) for item in items)
        baseline_accuracy = baseline_correct / len(items)
        losses = sum(int(item.before_correct and not item.after_correct)
                     for item in items)
        upper = one_sided_exact_deterioration_upper(
            len(items), losses, policy.significance_alpha / len(by_skill)
        )
        groups.append(SkillRetentionResult(
            skill, len(items), baseline_correct, baseline_accuracy,
            losses, upper,
            baseline_accuracy >= policy.min_before_accuracy
            and upper <= policy.max_deterioration_rate,
        ))
    return RetentionDecision(
        "PASS" if all(group.passed for group in groups) else "HOLD",
        tuple(groups),
    )
