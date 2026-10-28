"""Minotaur's fail-closed qualification gate for models leaving Labyrinth.

This module makes *no* inference call, trains no weights, and performs no
deployment.  It accepts only an attestation from a separately trusted evaluator
that actually ran the pinned candidate, frozen baseline, zero-update control,
and corrupted-label control on common held-out cases with a private oracle.

The evaluator, **not** Minotaur or the trainee, must own the HMAC key.  HMAC
authenticates the signed receipt's origin, not whether the evaluator told the
truth, had genuinely unseen questions, or ran the declared checkpoints.
The trusted owner must persist GateState across processes and issue a fresh
session nonce for each qualification campaign; neither is automatic here.
An unsigned Labyrinth report, even one that passes its consistency auditor,
cannot satisfy this gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import hmac
import json
import math
import re
from typing import Any


SCHEMA = "labyrinth.minotaur-evaluator-attestation.v1"
_DOMAIN = b"labyrinth.minotaur.graduation.v1\x00"
_PRIVATE_EXAM_SCORE_PROVENANCE = "independent_private_exam.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_NONCE = re.compile(r"[0-9a-f]{32,128}\Z", re.ASCII)
_MAX_RECEIPT_BYTES = 1_048_576
_MAX_TRIALS = 2_000


@dataclass(frozen=True)
class GraduationPolicy:
    """Declared criterion; the trusted orchestrator must freeze it before eval."""

    objective_id: str
    trainee_sha256: str
    base_sha256: str
    session_nonce: str
    expected_evaluator_sha256: str
    expected_conditions_sha256: str
    required_slices: tuple[str, ...] = ("heldout", "transfer")
    min_cases_per_slice: int = 16
    min_accuracy: float = 0.75
    min_format_rate: float = 0.90
    min_gain_over_controls: float = 0.10
    max_attempts: int = 3
    objective_kind: str = "training_effect"  # alternatively skill_mastery
    significance_alpha: float = 0.05  # family-wise one-sided paired-test limit

    def __post_init__(self) -> None:
        for field in (
            "objective_id", "trainee_sha256", "base_sha256", "session_nonce",
            "expected_evaluator_sha256", "expected_conditions_sha256",
        ):
            if type(getattr(self, field)) is not str:
                raise ValueError(f"{field} must be a string")
        if type(self.required_slices) is not tuple:
            raise ValueError("required_slices must be an immutable tuple")
        if not _ID.fullmatch(self.objective_id):
            raise ValueError("invalid objective ID")
        if not _SHA256.fullmatch(self.trainee_sha256):
            raise ValueError("trainee checkpoint must have a SHA-256 identity")
        if not _SHA256.fullmatch(self.base_sha256):
            raise ValueError("pretrained base checkpoint must have a SHA-256 identity")
        if not _NONCE.fullmatch(self.session_nonce):
            raise ValueError("a fresh random hexadecimal session nonce is required")
        if not _SHA256.fullmatch(self.expected_evaluator_sha256):
            raise ValueError("expected evaluator source SHA-256 must be pinned")
        if not _SHA256.fullmatch(self.expected_conditions_sha256):
            raise ValueError("expected evaluation conditions SHA-256 must be pinned")
        if (not self.required_slices or len(self.required_slices) > 12
                or any(type(s) is not str or not _ID.fullmatch(s) for s in self.required_slices)
                or len(set(self.required_slices)) != len(self.required_slices)):
            raise ValueError("invalid or duplicate required slices")
        if (not {"heldout", "transfer"}.issubset(self.required_slices)
                or any(s.lower().startswith("train") for s in self.required_slices)):
            raise ValueError("graduation requires heldout and transfer, never a training slice")
        if type(self.min_cases_per_slice) is not int or not 16 <= self.min_cases_per_slice <= 500:
            raise ValueError("min_cases_per_slice must be in [16, 500]")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 100:
            raise ValueError("max_attempts must be in [1, 100]")
        for name in ("min_accuracy", "min_format_rate", "min_gain_over_controls"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        # These are independent *absolute* minimums; the caller must not turn
        # the admission gate into a tautology by supplying zero thresholds.
        # Lower task-specific requirements need a reviewed protocol/version
        # change, not a runtime switch in a receipt policy.
        if self.min_accuracy < 0.75:
            raise ValueError("a graduation objective requires at least 75% semantic accuracy")
        if self.min_format_rate < 0.90:
            raise ValueError("a graduation objective requires at least 90% valid output format")
        if self.min_gain_over_controls < 0.10:
            raise ValueError("training-effect objectives require a nontrivial declared gain")
        if self.objective_kind not in ("training_effect", "skill_mastery"):
            raise ValueError("objective_kind must distinguish mastery from training effect")
        if self.objective_kind == "skill_mastery" and self.min_accuracy == 1:
            raise ValueError("finite samples cannot establish perfect population mastery")
        if (type(self.significance_alpha) not in (float, int)
                or not math.isfinite(self.significance_alpha)
                or not 0 < self.significance_alpha <= 0.05):
            raise ValueError("significance_alpha must be finite and in (0, 0.05]")


@dataclass(frozen=True)
class GateState:
    """Trusted durable state, maintained *outside* trainee-controlled storage."""

    next_attempt: int = 1
    consumed_run_ids: frozenset[str] = frozenset()
    consumed_suite_sha256s: frozenset[str] = frozenset()
    consumed_eval_case_ids: frozenset[str] = frozenset()
    consumed_eval_prompt_commitments: frozenset[str] = frozenset()
    graduated: bool = False
    policy_fingerprint: str | None = None


@dataclass(frozen=True)
class GateDecision:
    disposition: str  # GRADUATED, STAY, EXHAUSTED, or BLOCKED
    reasons: tuple[str, ...]
    state: GateState


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _exact_keys(item: Any, expected: frozenset[str]) -> bool:
    return type(item) is dict and item.keys() == expected


def _policy_fingerprint(policy: GraduationPolicy) -> str:
    """Bind durable admission state to every criterion and the subject."""
    canonical = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    return hashlib.sha256(b"labyrinth.minotaur.policy.v1\x00" + canonical).hexdigest()


_ROOT_KEYS = frozenset({
    "schema", "objective_id", "trainee_sha256", "base_sha256", "session_nonce", "run_id",
    "attempt", "evaluator_sha256", "conditions_sha256", "fresh_process_reload",
    "independent_private_oracle", "score_provenance", "training_case_ids", "training_prompt_commitments",
    "slices", "suite_sha256", "arm_checkpoint_sha256s", "arm_training_steps",
})
_TRIAL_KEYS = frozenset({
    "case_id", "candidate_correct", "candidate_format", "baseline_correct",
    "zero_update_correct", "corrupted_correct", "prompt_commitment",
    "baseline_format", "zero_update_format", "corrupted_format",
    "candidate_semantic_correct", "baseline_semantic_correct",
    "zero_update_semantic_correct", "corrupted_semantic_correct",
    "candidate_response_sha256", "baseline_response_sha256",
    "zero_update_response_sha256", "corrupted_response_sha256",
})
_ARMS = frozenset({"candidate", "baseline", "zero_update", "corrupted"})


def _one_sided_exact_paired_p(wins: int, losses: int) -> float:
    """Exact one-sided McNemar/binomial tail for discordant paired cases.

    The null is no greater correctness probability for candidate than control.
    It assumes the individual held-out problems, not their paraphrases or
    correlated variants, are exchangeable independent evaluation units.
    """
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / (1 << discordant)


def _one_sided_mastery_tail_p(successes: int, total: int, minimum_accuracy: float) -> float:
    """Exact binomial survival probability under the specified mastery floor.

    This only interprets genuinely independent evaluation problems. Repeating
    paraphrases of one public structure invalidates a confidence claim.
    """
    if successes == 0:
        return 1.0
    return sum(math.comb(total, k) * minimum_accuracy**k
               * (1 - minimum_accuracy)**(total - k)
               for k in range(successes, total + 1))


def decide_graduation(
    policy: GraduationPolicy,
    state: GateState,
    receipt_bytes: bytes | None,
    signature_hex: str | None,
    *,
    independent_evaluator_key: bytes | None,
) -> GateDecision:
    """Return a decision, never a deployment or production-inference cutover.

    The key must come from a trusted evaluator not accessible to trainee or
    neural Minotaur.  A valid receipt consumes one budgeted attempt; malformed
    or untrusted receipts never graduate or consume an authenticated attempt.
    """

    if not isinstance(policy, GraduationPolicy) or not isinstance(state, GateState):
        raise TypeError("expected GraduationPolicy and trusted GateState")
    if (type(state.next_attempt) is not int or state.next_attempt < 1
            or type(state.consumed_run_ids) is not frozenset
            or any(type(r) is not str or not _ID.fullmatch(r) for r in state.consumed_run_ids)
            or type(state.consumed_suite_sha256s) is not frozenset
            or any(type(r) is not str or not _SHA256.fullmatch(r)
                   for r in state.consumed_suite_sha256s)
            or type(state.consumed_eval_case_ids) is not frozenset
            or any(type(r) is not str or not _ID.fullmatch(r)
                   for r in state.consumed_eval_case_ids)
            or type(state.consumed_eval_prompt_commitments) is not frozenset
            or any(type(r) is not str or not _SHA256.fullmatch(r)
                   for r in state.consumed_eval_prompt_commitments)
            or type(state.graduated) is not bool
            or (state.policy_fingerprint is not None
                and (type(state.policy_fingerprint) is not str
                     or not _SHA256.fullmatch(state.policy_fingerprint)))):
        return GateDecision("BLOCKED", ("invalid trusted state",), state)
    binding = _policy_fingerprint(policy)
    if state.policy_fingerprint not in (None, binding):
        return GateDecision("BLOCKED", ("trusted state belongs to another policy or model",), state)
    if (state.policy_fingerprint is None
            and (state.next_attempt != 1 or state.consumed_run_ids
                 or state.consumed_suite_sha256s or state.consumed_eval_case_ids
                 or state.consumed_eval_prompt_commitments or state.graduated)):
        return GateDecision("BLOCKED", ("unbound trusted state cannot authorize graduation",), state)
    if state.graduated:
        return GateDecision("GRADUATED", ("previous trusted graduation retained",), state)
    if state.next_attempt > policy.max_attempts:
        return GateDecision("EXHAUSTED", ("training attempt budget exhausted",), state)

    if (type(independent_evaluator_key) is not bytes or len(independent_evaluator_key) < 32
            or type(receipt_bytes) is not bytes
            or not 1 <= len(receipt_bytes) <= _MAX_RECEIPT_BYTES
            or type(signature_hex) is not str or not _SHA256.fullmatch(signature_hex)):
        return GateDecision("BLOCKED", ("missing authenticated independent evidence",), state)
    expected = hmac.new(independent_evaluator_key, _DOMAIN + receipt_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature_hex, expected):
        return GateDecision("BLOCKED", ("independent evaluator attestation invalid",), state)
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"), object_pairs_hook=_no_duplicate_keys,
                             parse_constant=_reject_nonfinite)
    except (ValueError, UnicodeError, RecursionError):
        return GateDecision("BLOCKED", ("malformed signed evaluation evidence",), state)
    if not _exact_keys(receipt, _ROOT_KEYS):
        return GateDecision("BLOCKED", ("unexpected evaluation receipt schema",), state)
    if (receipt["schema"] != SCHEMA or receipt["objective_id"] != policy.objective_id
            or receipt["trainee_sha256"] != policy.trainee_sha256
            or receipt["base_sha256"] != policy.base_sha256
            or receipt["session_nonce"] != policy.session_nonce
            or type(receipt["attempt"]) is not int or receipt["attempt"] != state.next_attempt
            or type(receipt["run_id"]) is not str or not _ID.fullmatch(receipt["run_id"])
            or receipt["run_id"] in state.consumed_run_ids
            or type(receipt["suite_sha256"]) is not str
            or not _SHA256.fullmatch(receipt["suite_sha256"])
            or receipt["suite_sha256"] in state.consumed_suite_sha256s
            or receipt["evaluator_sha256"] != policy.expected_evaluator_sha256
            or receipt["conditions_sha256"] != policy.expected_conditions_sha256
            or receipt["fresh_process_reload"] is not True
            or receipt["independent_private_oracle"] is not True
            # A correct score from Minotaur's bounded/public advisor remains
            # advisory evidence. Only the independently custodied examiner
            # may attest private-exam provenance; callers cannot silently
            # recast a domain-limited score as final qualification.
            or receipt["score_provenance"] != _PRIVATE_EXAM_SCORE_PROVENANCE):
        return GateDecision("BLOCKED", ("untrusted, stale, or wrong-subject evaluation",), state)

    checkpoints = receipt["arm_checkpoint_sha256s"]
    steps = receipt["arm_training_steps"]
    if (not _exact_keys(checkpoints, _ARMS) or not _exact_keys(steps, _ARMS)
            or any(type(item) is not str or not _SHA256.fullmatch(item)
                   for item in checkpoints.values())
            or checkpoints["candidate"] != policy.trainee_sha256
            or checkpoints["baseline"] != policy.base_sha256
            or checkpoints["candidate"] == checkpoints["baseline"]
            or checkpoints["candidate"] == checkpoints["zero_update"]
            or checkpoints["candidate"] == checkpoints["corrupted"]
            or any(type(item) is not int or item < 0 or item > 100_000
                   for item in steps.values())
            or steps["baseline"] != 0 or steps["candidate"] == 0
            or steps["candidate"] != steps["zero_update"]
            or steps["candidate"] != steps["corrupted"]):
        return GateDecision("BLOCKED", ("unbound or unequal-budget trial arms",), state)

    training_ids = receipt["training_case_ids"]
    training_prompts = receipt["training_prompt_commitments"]
    groups = receipt["slices"]
    if (type(training_ids) is not list or not 1 <= len(training_ids) <= _MAX_TRIALS
            or any(type(x) is not str or not _ID.fullmatch(x) for x in training_ids)
            or len(set(training_ids)) != len(training_ids)
            or type(training_prompts) is not list or len(training_prompts) != len(training_ids)
            or any(type(x) is not str or not _SHA256.fullmatch(x)
                   for x in training_prompts)
            or len(set(training_prompts)) != len(training_prompts)
            or type(groups) is not dict or set(groups) != set(policy.required_slices)):
        return GateDecision("BLOCKED", ("invalid or incomplete evaluation slice custody",), state)

    if (set(training_ids) & state.consumed_eval_case_ids
            or set(training_prompts) & state.consumed_eval_prompt_commitments):
        return GateDecision("BLOCKED", ("prior final evaluation leaked into training",), state)

    encountered = set(training_ids) | state.consumed_eval_case_ids
    seen_prompts = set(training_prompts) | state.consumed_eval_prompt_commitments
    new_case_ids: set[str] = set()
    new_prompt_commitments: set[str] = set()
    satisfied = True
    count_total = 0
    for slice_name in policy.required_slices:
        trials = groups[slice_name]
        if type(trials) is not list or not policy.min_cases_per_slice <= len(trials) <= 500:
            return GateDecision("BLOCKED", (f"missing or undersized slice: {slice_name}",), state)
        count_total += len(trials)
        if count_total > _MAX_TRIALS:
            return GateDecision("BLOCKED", ("too many evaluation trials",), state)
        tally = {name: 0 for name in (
            "candidate_correct", "candidate_format", "baseline_correct",
            "zero_update_correct", "corrupted_correct", "baseline_format",
            "zero_update_format", "corrupted_format", "candidate_semantic_correct",
            "baseline_semantic_correct", "zero_update_semantic_correct",
            "corrupted_semantic_correct")}
        paired = {arm: [0, 0] for arm in ("baseline", "zero_update", "corrupted")}
        for trial in trials:
            if not _exact_keys(trial, _TRIAL_KEYS):
                return GateDecision("BLOCKED", ("malformed paired trial",), state)
            case_id = trial["case_id"]
            if type(case_id) is not str or not _ID.fullmatch(case_id) or case_id in encountered:
                return GateDecision("BLOCKED", ("training/evaluation overlap or repeated case",), state)
            encountered.add(case_id)
            new_case_ids.add(case_id)
            prompt_commitment = trial["prompt_commitment"]
            if (type(prompt_commitment) is not str or not _SHA256.fullmatch(prompt_commitment)
                    or prompt_commitment in seen_prompts):
                return GateDecision("BLOCKED", ("training/evaluation prompt overlap",), state)
            seen_prompts.add(prompt_commitment)
            new_prompt_commitments.add(prompt_commitment)
            for name in tally:
                if type(trial[name]) is not bool:
                    return GateDecision("BLOCKED", ("nonboolean scoring evidence",), state)
                tally[name] += int(trial[name])
            for arm, (wins, losses) in paired.items():
                candidate = trial["candidate_semantic_correct"]
                control = trial[f"{arm}_semantic_correct"]
                paired[arm] = [wins + int(candidate and not control),
                               losses + int(control and not candidate)]
            for name in ("candidate_response_sha256", "baseline_response_sha256",
                         "zero_update_response_sha256", "corrupted_response_sha256"):
                if type(trial[name]) is not str or not _SHA256.fullmatch(trial[name]):
                    return GateDecision("BLOCKED", ("missing bound arm predictions",), state)
            for arm in _ARMS:
                # Strict score = substantive correctness AND valid final answer
                # form; substantive correctness is separately oracle-checked.
                if trial[f"{arm}_correct"] is not (
                    trial[f"{arm}_semantic_correct"] and trial[f"{arm}_format"]):
                    return GateDecision("BLOCKED", ("format and substantive scores disagree",), state)
            # Identical response bytes on one question cannot receive
            # contradictory correctness labels from one deterministic oracle.
            arms = ("candidate", "baseline", "zero_update", "corrupted")
            for index, left in enumerate(arms):
                for right in arms[index + 1:]:
                    if (trial[f"{left}_response_sha256"] == trial[f"{right}_response_sha256"]
                            and (trial[f"{left}_correct"] != trial[f"{right}_correct"]
                                 or trial[f"{left}_semantic_correct"]
                                 != trial[f"{right}_semantic_correct"]
                                 or trial[f"{left}_format"] != trial[f"{right}_format"])):
                        return GateDecision("BLOCKED", ("contradictory paired scoring evidence",), state)

        n = len(trials)
        # Reasoning qualification uses answer correctness *irrespective of*
        # serialization, so a format-only fix cannot fake a reasoning gain.
        accuracy = tally["candidate_semantic_correct"] / n
        format_rate = tally["candidate_format"] / n
        strongest_control = max(
            tally["baseline_semantic_correct"], tally["zero_update_semantic_correct"],
            tally["corrupted_semantic_correct"]) / n
        gain = accuracy - strongest_control
        if accuracy < policy.min_accuracy:
            satisfied = False
        if format_rate < policy.min_format_rate:
            satisfied = False
        if policy.objective_kind == "skill_mastery":
            # H0: true success rate <= the frozen competency floor. Observing
            # its empirical threshold is not the same as confidently meeting
            # it. Adjust for both held-out slices and multiple attempts.
            if (_one_sided_mastery_tail_p(tally["candidate_semantic_correct"], n,
                                          policy.min_accuracy)
                    * len(policy.required_slices) * policy.max_attempts
                    >= policy.significance_alpha):
                satisfied = False
        else:  # training_effect
            # A nonzero sample gain alone is weak evidence. Check *paired*
            # discordant wins against all three controls in each slice.
            # Bonferroni controls the family-wise false-positive risk across
            # arms, slices, and all budgeted attempts, even when arms overlap.
            # This only applies if individual cases are genuinely independent
            # within their slices, an evaluator-side custody responsibility.
            comparisons = 3 * len(policy.required_slices) * policy.max_attempts
            if gain <= 0 or gain < policy.min_gain_over_controls:
                satisfied = False
            for wins, losses in paired.values():
                if _one_sided_exact_paired_p(wins, losses) * comparisons >= policy.significance_alpha:
                    satisfied = False

    advanced = replace(state, next_attempt=state.next_attempt + 1,
                       policy_fingerprint=binding,
                       consumed_run_ids=state.consumed_run_ids | {receipt["run_id"]},
                       consumed_suite_sha256s=state.consumed_suite_sha256s | {receipt["suite_sha256"]},
                       consumed_eval_case_ids=state.consumed_eval_case_ids | new_case_ids,
                       consumed_eval_prompt_commitments=(
                           state.consumed_eval_prompt_commitments | new_prompt_commitments))
    if satisfied:
        return GateDecision("GRADUATED", ("objective demonstrated under signed conditions",),
                            replace(advanced, graduated=True))
    if advanced.next_attempt > policy.max_attempts:
        return GateDecision("EXHAUSTED", ("objective not demonstrated; training budget exhausted",),
                            advanced)
    # Never leak per-slice final-set scores or hints through the public gate.
    return GateDecision("STAY", ("objective not demonstrated",), advanced)
