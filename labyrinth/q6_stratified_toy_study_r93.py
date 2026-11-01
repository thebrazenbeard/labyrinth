"""Source-only Q6 toy generalization and retention study with adverse controls.

This deliberately measures extrapolation *within the existing canonical
arithmetic grammar*. The learner is the four-parameter ToyLinearAdapter, not a
pretrained language model. Fixed public seeds, caller-owned labels, and local
storage cannot qualify independent holdout custody or a Q6 release gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory

from .evaluation import score_response
from .maze import Challenge
from .toy_backend import ToyLinearAdapter


SCHEMA = "labyrinth.q6-stratified-toy-study-r93.v1"
_EXPOSURE_DOMAIN = b"labyrinth.q6.public-evaluation-exposures.v1\0"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_MAX_EXPOSURES = 100_000
_PROMPT = re.compile(
    r"Calculate ([0-9]{1,3}) ([+-]) ([0-9]{1,3})\.\n"
    r"Respond with exactly one line: ANSWER: <integer>\.", re.ASCII,
)
_PROFILES = ("add_carry", "add_no_carry", "subtract_borrow", "subtract_no_borrow")
_SPLITS = {"train_a": "train", "train_b": "train",
           "heldout": "q6_heldout", "transfer": "q6_transfer"}


class Q6StudyError(ValueError):
    """Invalid source-only study declarations, case partition or comparison."""


class Q6ExposureReuse(Q6StudyError):
    """An evaluation problem is semantically present in prior public history."""


class Q6ExposureExhaustion(Q6StudyError):
    """A bounded source cohort/history could not be safely admitted."""


@dataclass(frozen=True)
class Q6Cohorts:
    train_a: tuple[Challenge, ...]
    train_b: tuple[Challenge, ...]
    heldout: tuple[Challenge, ...]
    transfer: tuple[Challenge, ...]


def _parse(prompt: str) -> tuple[int, str, int]:
    if type(prompt) is not str:
        raise Q6StudyError("canonical arithmetic prompt required")
    match = _PROMPT.fullmatch(prompt)
    if match is None:
        raise Q6StudyError("canonical arithmetic prompt required")
    return int(match[1]), match[2], int(match[3])


def semantic_case_key(case: Challenge) -> tuple[str, int, int, int]:
    """Canonicalize addition and all equivalent inverse subtraction facts."""
    if type(case) is not Challenge:
        raise Q6StudyError("exact Challenge required")
    return _semantic_prompt_key(case.prompt)


def _semantic_prompt_key(prompt: str) -> tuple[str, int, int, int]:
    a, op, b = _parse(prompt)
    # Normalize a+b=c, c-a=b, c-b=a (including negative subtractive
    # rearrangements) to the same positive-addends equation. Distinct text,
    # case IDs, and operator tokens are not independent semantic problems.
    if op == "+":
        addends, total = (a, b), a + b
    elif a >= b:
        addends, total = (b, a - b), a
    else:
        addends, total = (a, b - a), b
    return ("positive-sum", *sorted(addends), total)


def _profile(prompt: str) -> str:
    a, op, b = _parse(prompt)
    if op == "+":
        return "add_carry" if a % 10 + b % 10 >= 10 else "add_no_carry"
    return "subtract_borrow" if a % 10 < b % 10 else "subtract_no_borrow"


def _freeze_exposures(value: set[str] | frozenset[str]) -> frozenset[str]:
    """Snapshot a caller's mutable public history before other code executes.

    This guards TOCTOU inside one process, not completeness, authenticity,
    concurrent host-atomic reads, or rollback resistance of an outside ledger.
    """
    if type(value) not in (set, frozenset):
        raise Q6StudyError("explicit exposed prompt set required")
    if len(value) > _MAX_EXPOSURES:
        raise Q6ExposureExhaustion("bounded exposure history exhausted")
    try:
        frozen = frozenset(value)
    except RuntimeError as exc:
        raise Q6StudyError("exposure history changed during snapshot") from exc
    if any(type(p) is not str for p in frozen):
        raise Q6StudyError("explicit exposed prompt set required")
    for prompt in frozen:
        _parse(prompt)
    return frozen


def q6_exposure_history_sha256(value: set[str] | frozenset[str]) -> str:
    """Canonical content fingerprint; NEVER a signature or owner attestation."""
    frozen = _freeze_exposures(value)
    payload = json.dumps(sorted(frozen), ensure_ascii=True,
                         separators=(",", ":")).encode("ascii")
    return sha256(_EXPOSURE_DOMAIN + payload).hexdigest()


def _seed(seed: int, label: str, profile: str) -> int:
    payload = f"labyrinth.q6.r93\x00{seed}\x00{label}\x00{profile}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _source_cases(
    *, seed: int, label: str, split: str, low: int, high: int,
    per_profile: int, forbidden_semantics: set[tuple[str, int, int, int]],
) -> tuple[Challenge, ...]:
    result: list[Challenge] = []
    # The fixed quotas, ranges, and seed are selected without model evaluation.
    for profile in _PROFILES:
        rng = random.Random(_seed(seed, label, profile))
        for ordinal in range(per_profile):
            for attempt in range(4096):
                a, b = rng.randint(low, high), rng.randint(low, high)
                op = "+" if profile.startswith("add_") else "-"
                prompt = (f"Calculate {a} {op} {b}.\n"
                          "Respond with exactly one line: ANSWER: <integer>.")
                if _profile(prompt) != profile:
                    continue
                key = _semantic_prompt_key(prompt)
                if key in forbidden_semantics:
                    continue
                forbidden_semantics.add(key)
                answer = a + b if op == "+" else a - b
                digest = sha256(
                    f"{seed}\x1f{label}\x1f{profile}\x1f{ordinal}\x1f{prompt}".encode("ascii")
                ).hexdigest()[:24]
                result.append(Challenge(
                    id=f"q6-r93-{label}-{profile}-{ordinal}-{digest}",
                    kind="arithmetic", tier=1, split=split, prompt=prompt,
                    answer=str(answer), metadata=(("generator", "q6-r93-public"),),
                ))
                break
            else:
                raise Q6ExposureExhaustion("bounded cohort generation exhausted")
    return tuple(result)


def construct_q6_cohorts(*, seed: int, per_profile: int = 6) -> Q6Cohorts:
    """Preregister public low-range TRAIN/heldout and high-range extrapolation.

    All four carry/borrow profiles get equal quota. ``transfer`` demands both
    operands 800..999 whereas both TRAIN stages and heldout use 11..499.
    Lexical grammar is the same; this is NOT unseen-language transfer.
    """
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise Q6StudyError("seed must be a bounded exact integer")
    if type(per_profile) is not int or not 2 <= per_profile <= 24:
        raise Q6StudyError("per_profile must be 2..24")
    seen: set[tuple[str, int, int, int]] = set()
    groups = {}
    for label in _SPLITS:
        low, high = (800, 999) if label == "transfer" else (11, 499)
        groups[label] = _source_cases(
            seed=seed, label=label, split=_SPLITS[label], low=low, high=high,
            per_profile=per_profile, forbidden_semantics=seen,
        )
    cohorts = Q6Cohorts(**groups)
    validate_q6_cohorts(cohorts, per_profile=per_profile)
    return cohorts


def validate_q6_cohorts(
    cohorts: Q6Cohorts, *, per_profile: int,
    previously_exposed_eval_prompts: set[str] | frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Reject exact and algebraic leakage across all four partitions.

    An externally maintained exposure register is not implemented. Supplying
    an empty history again does NOT make a replay independently fresh.
    """
    if type(cohorts) is not Q6Cohorts:
        raise Q6StudyError("exact cohort structure required")
    if type(per_profile) is not int or not 2 <= per_profile <= 24:
        raise Q6StudyError("invalid declared profile quota")
    # Ensure an external caller cannot mutate history after the initial check
    # while the cohort validator walks the remaining (potentially large) bank.
    exposure_snapshot = _freeze_exposures(previously_exposed_eval_prompts)
    all_ids: set[str] = set()
    all_prompt_sha256: set[str] = set()
    all_semantics: set[tuple[str, int, int, int]] = set()
    by_group: dict[str, dict[str, int]] = {}
    for label, expected_split in _SPLITS.items():
        cases = getattr(cohorts, label)
        if type(cases) is not tuple or len(cases) != 4 * per_profile:
            raise Q6StudyError("wrong preregistered cohort cardinality")
        counts: Counter[str] = Counter()
        for case in cases:
            if (type(case) is not Challenge or type(case.id) is not str or not case.id
                    or case.kind != "arithmetic" or case.tier != 1
                    or case.split != expected_split
                    or case.metadata != (("generator", "q6-r93-public"),)):
                raise Q6StudyError("invalid cohort case identity or scope")
            a, op, b = _parse(case.prompt)
            minimum, maximum = (800, 999) if label == "transfer" else (11, 499)
            if not minimum <= a <= maximum or not minimum <= b <= maximum:
                raise Q6StudyError("cohort range shift or training support violated")
            answer = a + b if op == "+" else a - b
            if case.answer != str(answer):
                raise Q6StudyError("case gold is inconsistent with arithmetic")
            fingerprint = sha256(case.prompt.encode("ascii")).hexdigest()
            semantic = semantic_case_key(case)
            if semantic in all_semantics:
                raise Q6StudyError("semantic overlap across study partitions")
            if case.id in all_ids or fingerprint in all_prompt_sha256:
                raise Q6StudyError("duplicate case identity or prompt across partitions")
            all_ids.add(case.id)
            all_prompt_sha256.add(fingerprint)
            all_semantics.add(semantic)
            counts[_profile(case.prompt)] += 1
        if set(counts) != set(_PROFILES) or any(n != per_profile for n in counts.values()):
            raise Q6StudyError("case strata are not balanced")
        by_group[label] = dict(counts)
    previously_exposed_semantics = {
        _semantic_prompt_key(prompt) for prompt in exposure_snapshot
    }
    # A previous public evaluation problem must never become a later TRAIN
    # problem either. Checking only the current evaluation splits permits
    # cross-seed exposure to contaminate the optimizer before it is scored.
    for case in cohorts.train_a + cohorts.train_b + cohorts.heldout + cohorts.transfer:
        if (case.prompt in exposure_snapshot
                or semantic_case_key(case) in previously_exposed_semantics):
            raise Q6ExposureReuse("evaluation prompt previously exposed; replay refused")
    return {"total": len(all_ids), "by_group": by_group,
            "all_prompt_sha256": tuple(sorted(all_prompt_sha256))}


def _score(model: ToyLinearAdapter, cases: tuple[Challenge, ...]) -> int:
    # Only prompt TEXT crosses the learner boundary; the local evaluator holds
    # Challenge.answer and all partition/rubric information.
    return sum(score_response(case, model.generate(case.prompt)).points for case in cases)


def _snapshot(model: ToyLinearAdapter, cohorts: Q6Cohorts) -> dict[str, int]:
    return {"heldout": _score(model, cohorts.heldout),
            "transfer": _score(model, cohorts.transfer)}


def run_q6_toy_study(
    *, seed: int = 7, per_profile: int = 6, epochs_stage_a: int = 80,
    epochs_stage_b: int = 80, learning_rate: float = 0.5,
    previously_exposed_eval_prompts: set[str] | frozenset[str] = frozenset(),
    expected_exposure_history_sha256: str | None = None,
) -> dict[str, object]:
    """Execute both real optimizer stages and matched wrong/no-update arms.

    The schedule is validated/frozen before scoring; there is no test-driven
    seed choice or threshold promotion. Training uses TRAIN gold, while the
    heldout/transfer scores are read-only at baseline, after A, and after B.
    """
    for name, count in (("epochs_stage_a", epochs_stage_a), ("epochs_stage_b", epochs_stage_b)):
        if type(count) is not int or not 1 <= count <= 1000:
            raise Q6StudyError(f"{name} must be 1..1000")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise Q6StudyError("learning_rate must be finite in (0,1]")
    # Validate and snapshot at entry, ahead of generation and optimizer use.
    # A caller-supplied pin binds source bytes only; it does not prove that
    # the caller owns a monotonic independent exposure register.
    exposure_snapshot = _freeze_exposures(previously_exposed_eval_prompts)
    exposure_sha256 = q6_exposure_history_sha256(exposure_snapshot)
    if expected_exposure_history_sha256 is not None:
        if (type(expected_exposure_history_sha256) is not str
                or _HEX64.fullmatch(expected_exposure_history_sha256) is None):
            raise Q6StudyError("malformed exposure history pin")
        if expected_exposure_history_sha256 != exposure_sha256:
            raise Q6StudyError("exposure history pin mismatch; run refused")
    cohorts = construct_q6_cohorts(seed=seed, per_profile=per_profile)
    # The caller controls this exposure history, so this remains source-only.
    # Nevertheless it must be checked *inside the executable runner* before
    # any optimizer writes, not merely by an optional standalone validator.
    verified = validate_q6_cohorts(
        cohorts, per_profile=per_profile,
        previously_exposed_eval_prompts=exposure_snapshot,
    )
    arms = {label: ToyLinearAdapter(seed=seed) for label in
            ("trained", "zero_update", "inverted_labels")}
    scores = {label: {"initial": _snapshot(model, cohorts)}
              for label, model in arms.items()}
    initial = arms["trained"].model_hash
    zero_initial = arms["zero_update"].model_hash
    optimizer: dict[str, dict[str, dict]] = {}
    trained_hash_a = ""
    with TemporaryDirectory(prefix="labyrinth-q6-r93-") as scratch:
        for phase, batch, epochs in (("after_a", cohorts.train_a, epochs_stage_a),
                                     ("after_b", cohorts.train_b, epochs_stage_b)):
            stages: dict[str, dict] = {}
            for label, model in arms.items():
                receipt = model.train(
                    batch, epochs=epochs, learning_rate=float(learning_rate),
                    update_enabled=label != "zero_update",
                    shuffled_labels=label == "inverted_labels",
                )
                stages[label] = receipt
                # A fully converged learner may legitimately make no further
                # update in stage B; require genuine updates across the whole
                # fixed campaign, not fabricated per-stage movement.
                if (phase == "after_a" and label != "zero_update"
                        and receipt["before_model_hash"] == receipt["after_model_hash"]):
                    raise Q6StudyError("initial optimizer made no real toy update")
                if label == "zero_update" and (receipt["parameter_updates"] != 0
                                               or model.model_hash != zero_initial):
                    raise Q6StudyError("zero-update control mutated")
            optimizer[phase] = stages
            for label, model in arms.items():
                scores[label][phase] = _snapshot(model, cohorts)
            checkpoint = Path(scratch) / f"trained-{phase}.json"
            saved_digest = arms["trained"].save_checkpoint(checkpoint)
            if saved_digest != arms["trained"].model_hash:
                raise Q6StudyError("checkpoint publication/readback hash mismatch")
            arms["trained"] = ToyLinearAdapter.load_checkpoint(checkpoint)
            if arms["trained"].model_hash != saved_digest:
                raise Q6StudyError("fresh-load lost trained toy parameters")
            if phase == "after_a":
                trained_hash_a = saved_digest
    expected_visits = len(cohorts.train_a) * epochs_stage_a + len(cohorts.train_b) * epochs_stage_b
    counts = {label: sum(optimizer[phase][label]["case_visits"] for phase in optimizer)
              for label in arms}
    if any(v != expected_visits for v in counts.values()):
        raise Q6StudyError("matched-exposure control visit count diverged")
    train_updates = sum(optimizer[phase]["trained"]["parameter_updates"] for phase in optimizer)
    inverted_updates = sum(optimizer[phase]["inverted_labels"]["parameter_updates"] for phase in optimizer)
    if train_updates <= 0 or inverted_updates <= 0:
        raise Q6StudyError("no real toy parameter updates")
    return {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_TOY_Q6_UNQUALIFIED",
        "case_seed": seed,
        "cohort_counts": {label: len(getattr(cohorts, label)) for label in _SPLITS},
        "profiles": list(_PROFILES),
        "semantic_partition_validated": verified["total"] == 16 * per_profile,
        "evaluated_prompts": [c.prompt for c in cohorts.heldout + cohorts.transfer],
        "evaluation_prompt_sha256": [sha256(c.prompt.encode("ascii")).hexdigest()
                                      for c in cohorts.heldout + cohorts.transfer],
        "supplied_exposure_prompt_count": len(exposure_snapshot),
        "supplied_exposure_history_sha256": exposure_sha256,
        "caller_supplied_history_pin_checked": expected_exposure_history_sha256 is not None,
        "scores": scores,
        "retention_delta_after_stage_b": (
            scores["trained"]["after_b"]["heldout"]
            - scores["trained"]["after_a"]["heldout"]
        ),
        "high_range_transfer_gain_vs_zero": (
            scores["trained"]["after_b"]["transfer"]
            - scores["zero_update"]["after_b"]["transfer"]
        ),
        "trained_hash_initial": initial,
        "trained_hash_after_stage_a": trained_hash_a,
        "trained_hash_after_stage_b": arms["trained"].model_hash,
        "fresh_reload_sha256": arms["trained"].model_hash,
        "zero_hash_initial": zero_initial,
        "zero_hash_after_stage_b": arms["zero_update"].model_hash,
        "matched_visits": expected_visits,
        "zero_parameter_updates": 0,
        "trained_parameter_updates": train_updates,
        "inverted_parameter_updates": inverted_updates,
        "q6_independently_qualified": False,
        "pretrained_learning_established": False,
        "graduation_authorized": False,
        "limitations": [
            "ToyLinearAdapter has only four SGD weights and a hand-coded arithmetic parser.",
            "This is low-to-high magnitude same-grammar extrapolation, not new-language transfer.",
            "The heldout and transfer generators are public deterministic fixtures, not sealed exams.",
            "Retained-score delta covers one immediate later toy training stage, not long-term forgetting.",
            "Caller-supplied previous-exposure claims cannot replace an independent monotonic ledger.",
            "A matching caller-supplied history digest authenticates no owner or previous-run custody.",
            "The returned public evaluation prompts can seed a later caller-owned exposure check, not host custody.",
            "No pretrained model, independent Q5/Q6 evaluation, or host-controlled graduation is established.",
        ],
    }
