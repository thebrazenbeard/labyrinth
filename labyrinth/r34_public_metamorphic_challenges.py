"""Deterministic PUBLIC TOY metamorphic practice pairs, never an exam gate.

Public text variants preserve the R31 toy oracle while changing surface form.
No live model, hidden bank, training effect, or qualification is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from . import r31_public_challenge_synth as r31
from .r29_objective_regeneration import PublicRegenerationReceipt
from .trainingday_objective_metamorphic import FormalObjectiveContract


_MAX_LOOKUP_ATTEMPTS = 128  # Bound mirrors R31's semantic-collision retry domain.
_PAIR_STR_FIELDS = (
    "case_id", "skill_id", "split", "original_prompt",
    "original_prompt_sha256", "variant_prompt", "variant_prompt_sha256",
    "semantic_sha256", "pair_sha256",
)
_AUTH_FLAGS = (
    "training_authorized", "graduation_authorized",
    "independent_exam_verified", "promotion_allowed",
)


def _exact_instance_fields(value: object, expected: tuple[str, ...]) -> dict[str, object]:
    """Copy instance-owned fields, never falling back to dataclass defaults.

    Deleting a frozen dataclass field with a class-level default can make
    `getattr` silently return that default. Require the complete exact instance
    schema before accepting any status or authority flag.
    """
    raw = object.__getattribute__(value, "__dict__")
    if (type(raw) is not dict or len(raw) != len(expected)
            or any(type(key) is not str for key in raw)
            or set(raw) != set(expected)):
        raise ValueError("missing or unexpected public metamorphic instance field")
    return raw.copy()


@dataclass(frozen=True)
class PublicMetamorphicPair:
    case_id: str
    skill_id: str
    split: str
    original_prompt: str
    original_prompt_sha256: str
    variant_prompt: str
    variant_prompt_sha256: str
    semantic_sha256: str
    pair_sha256: str
    expected_answer: str | None


@dataclass(frozen=True)
class PublicMetamorphicBundle:
    receipt_commitment_sha256: str
    development: tuple[PublicMetamorphicPair, ...]
    challenge: tuple[PublicMetamorphicPair, ...]
    commitment_sha256: str
    training_authorized: bool = False
    graduation_authorized: bool = False
    independent_exam_verified: bool = False
    promotion_allowed: bool = False


@dataclass(frozen=True)
class PublicMetamorphicVerdict:
    status: str
    reasons: tuple[str, ...]
    training_authorized: bool = False
    graduation_authorized: bool = False
    independent_exam_verified: bool = False
    promotion_allowed: bool = False


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def _hash(domain: str, value: object) -> str:
    return sha256(domain.encode("ascii") + b"\0" + _canonical(value)).hexdigest()


def _variant_question(skill_id: str, params: tuple[int, int, int, int],
                      style: str, case_id: str) -> tuple[str, str]:
    """An independently worded toy problem with the R31 exact numeric oracle."""
    a, b, c, multiplier = params
    if style not in ("hint_free", "fresh_examples", "structure_variation"):
        raise ValueError("unknown public toy style")
    if skill_id == "arithmetic":
        if style == "structure_variation":
            question = f"Begin with {a}; multiply it by {multiplier}; increase by {c}."
            answer = str(a * multiplier + c)
        elif style == "fresh_examples":
            question = (f"First add {a} and {b}. Multiply that sum by {multiplier}. "
                        f"Subtract {c}.")
            answer = str((a + b) * multiplier - c)
        else:
            question = f"Take {a}, add {b}, and finally subtract {c}."
            answer = str(a + b - c)
        prompt = f"Public toy arithmetic case {case_id}. {question} Reply with the integer."
    elif skill_id == "source_priority":
        # Deliberately invert which source is mentioned first. A response
        # must follow the declared priority, not recency or first occurrence.
        higher = r31._LABELS[a % len(r31._LABELS)]
        lower = r31._LABELS[(a + 1 + b % (len(r31._LABELS) - 1)) % len(r31._LABELS)]
        if higher == lower:
            lower = r31._LABELS[(a + 1) % len(r31._LABELS)]
        if style == "hint_free":
            question = (f"The lower-priority recap reports {lower}. "
                        f"The higher-priority versioned toy record reports {higher}.")
        else:
            question = (f"The higher-priority versioned toy record reports {higher}. "
                        f"The lower-priority recap reports {lower}.")
        prompt = (f"Public toy source-priority case {case_id}. {question} "
                  "Reply with the higher-priority label only.")
        answer = higher
    elif skill_id == "program_reasoning":
        if style == "structure_variation":
            question = (f"Let x = {a}; then x *= {multiplier}; "
                        f"then x -= {b}; then x += {c}.")
            answer = str(a * multiplier - b + c)
        elif style == "fresh_examples":
            question = (f"Let x = {a}; then x += {b}; "
                        f"then x -= {c}; then x *= {multiplier}.")
            answer = str((a + b - c) * multiplier)
        else:
            question = (f"Let x = {a}; then x += {b}; "
                        f"then x *= {multiplier}; then x -= {c}.")
            answer = str((a + b) * multiplier - c)
        prompt = (f"Public toy program-operation case {case_id}. {question} "
                  "Reply with the final integer value of x.")
    else:
        raise ValueError("unsupported public toy skill")
    return prompt, answer


def _pair_from_item(item: r31.PublicToyExercise,
                    receipt: PublicRegenerationReceipt,
                    template_digest: str) -> PublicMetamorphicPair:
    style = r31._style(receipt)
    for attempt in range(_MAX_LOOKUP_ATTEMPTS):
        params = r31._parameters(receipt, item.split, item.skill_id,
                                 item.case_id, template_digest, attempt)
        expected_prompt, answer, semantic = r31._question(
            item.skill_id, params, style, item.case_id)
        if (expected_prompt != item.prompt or semantic != item.semantic_sha256
                or (item.expected_answer is not None and item.expected_answer != answer)):
            continue
        variant, variant_answer = _variant_question(
            item.skill_id, params, style, item.case_id)
        if variant == item.prompt or variant_answer != answer:
            raise ValueError("metamorphic public toy variant is not equivalent")
        variant_digest = sha256(variant.encode("utf-8")).hexdigest()
        pair_digest = _hash("labyrinth.r34.public-toy-metamorphic-pair.v1", {
            "case": item.case_id, "skill": item.skill_id, "split": item.split,
            "original": item.prompt_sha256, "variant": variant_digest,
            "semantic": item.semantic_sha256,
        })
        return PublicMetamorphicPair(
            item.case_id, item.skill_id, item.split, item.prompt,
            item.prompt_sha256, variant, variant_digest,
            item.semantic_sha256, pair_digest, item.expected_answer)
    raise ValueError("cannot independently recover R31 public content parameters")


def _public_payload(pair: PublicMetamorphicPair, *, include_answer: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_id": pair.case_id, "skill_id": pair.skill_id,
        "split": pair.split, "original_prompt": pair.original_prompt,
        "original_prompt_sha256": pair.original_prompt_sha256,
        "variant_prompt": pair.variant_prompt,
        "variant_prompt_sha256": pair.variant_prompt_sha256,
        "semantic_sha256": pair.semantic_sha256, "pair_sha256": pair.pair_sha256,
    }
    if include_answer:
        payload["expected_answer"] = pair.expected_answer
    return payload


def materialize_public_metamorphic_challenges(
    contract: FormalObjectiveContract, receipt: PublicRegenerationReceipt,
    *, predecessor: PublicRegenerationReceipt | None = None,
    predecessor_ancestors: tuple[PublicRegenerationReceipt, ...] = (),
) -> PublicMetamorphicBundle:
    """Bind paraphrases to an R29 receipt; public synthetic source only."""
    base = r31.materialize_public_toy_exercises(
        contract, receipt, predecessor=predecessor,
        predecessor_ancestors=predecessor_ancestors)
    templates = {item.skill_id: item for item in receipt.template_families}
    dev = tuple(_pair_from_item(item, receipt, templates[item.skill_id].development_sha256)
                for item in base.development)
    challenge = tuple(_pair_from_item(item, receipt, templates[item.skill_id].challenge_sha256)
                      for item in base.challenge)
    # Each underlying R31 semantic fingerprint is unique across both splits.
    if len({pair.semantic_sha256 for pair in dev + challenge}) != len(dev) + len(challenge):
        raise ValueError("public metamorphic split semantic collision")
    if len({pair.variant_prompt_sha256 for pair in dev + challenge}) != len(dev) + len(challenge):
        raise ValueError("public metamorphic prompt collision")
    commitment = _hash("labyrinth.r34.public-toy-metamorphic-bundle.v1", {
        "receipt": receipt.commitment_sha256,
        "development": [_public_payload(item, include_answer=True) for item in dev],
        "challenge": [_public_payload(item, include_answer=False) for item in challenge],
    })
    return PublicMetamorphicBundle(receipt.commitment_sha256, dev, challenge, commitment)


def _checked_public_bundle_snapshot(candidate: object) -> tuple[object, ...]:
    """Copy exact, bounded builtins before any comparison with public oracle.

    `frozen=True` is not a trust boundary: callers can delete or replace fields
    via object.__delattr__/__setattr__. A normalized primitive snapshot avoids
    invoking generated dataclass equality on those untrusted live objects.
    """
    if type(candidate) is not PublicMetamorphicBundle:
        raise ValueError("not an exact public metamorphic bundle")
    fields = _exact_instance_fields(candidate, (
        "receipt_commitment_sha256", "development", "challenge",
        "commitment_sha256") + _AUTH_FLAGS)
    for name in _AUTH_FLAGS:
        flag = fields[name]
        if type(flag) is not bool or flag:
            raise ValueError("unauthorized metamorphic bundle flag")
    receipt_sha = fields["receipt_commitment_sha256"]
    commitment_sha = fields["commitment_sha256"]
    if (type(receipt_sha) is not str or type(commitment_sha) is not str
            or len(receipt_sha) > 2048 or len(commitment_sha) > 2048):
        raise ValueError("malformed public metamorphic commitment")
    development = fields["development"]
    challenge = fields["challenge"]
    if (type(development) is not tuple or type(challenge) is not tuple
            or not 1 <= len(development) <= 512
            or not 1 <= len(challenge) <= 512):
        raise ValueError("malformed public metamorphic splits")

    def snapshot_pairs(pairs: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
        result = []
        for pair in pairs:
            if type(pair) is not PublicMetamorphicPair:
                raise ValueError("malformed public metamorphic pair type")
            pair_fields = _exact_instance_fields(
                pair, _PAIR_STR_FIELDS + ("expected_answer",))
            values = tuple(pair_fields[name] for name in _PAIR_STR_FIELDS)
            if any(type(value) is not str or len(value) > 2048 for value in values):
                raise ValueError("malformed public metamorphic pair field")
            answer = pair_fields["expected_answer"]
            if answer is not None and (type(answer) is not str or len(answer) > 2048):
                raise ValueError("malformed public metamorphic answer")
            result.append(values + (answer,))
        return tuple(result)

    return (receipt_sha, commitment_sha,
            snapshot_pairs(development), snapshot_pairs(challenge))


def verify_public_metamorphic_challenges(
    contract: FormalObjectiveContract, receipt: PublicRegenerationReceipt,
    candidate: object, *, predecessor: PublicRegenerationReceipt | None = None,
    predecessor_ancestors: tuple[PublicRegenerationReceipt, ...] = (),
) -> PublicMetamorphicVerdict:
    denied = PublicMetamorphicVerdict(
        "PUBLIC_TOY_METAMORPHIC_INVALID", ("MALFORMED_OR_UNATTESTED_PUBLIC_TOY",))
    try:
        candidate_snapshot = _checked_public_bundle_snapshot(candidate)
        actual = materialize_public_metamorphic_challenges(
            contract, receipt, predecessor=predecessor,
            predecessor_ancestors=predecessor_ancestors)
        actual_snapshot = _checked_public_bundle_snapshot(actual)
    except (TypeError, ValueError, AttributeError, KeyError):
        return denied
    if candidate_snapshot != actual_snapshot:
        return denied
    return PublicMetamorphicVerdict(
        "PUBLIC_TOY_METAMORPHIC_CONSISTENT_ONLY",
        ("PUBLIC_SYNTHETIC_PRACTICE_NO_INDEPENDENT_ORACLE",))


def serialize_public_metamorphic_challenge_jsonl(
    contract: FormalObjectiveContract, receipt: PublicRegenerationReceipt,
    candidate: PublicMetamorphicBundle, *, predecessor: PublicRegenerationReceipt | None = None,
    predecessor_ancestors: tuple[PublicRegenerationReceipt, ...] = (),
) -> str:
    """Only public challenge prompts; no answer fields or private bank access."""
    if verify_public_metamorphic_challenges(
        contract, receipt, candidate, predecessor=predecessor,
        predecessor_ancestors=predecessor_ancestors).status != "PUBLIC_TOY_METAMORPHIC_CONSISTENT_ONLY":
        raise ValueError("unverified public metamorphic toy bundle")
    # Reconstruct independently after validation; avoid mutable caller alias.
    private = materialize_public_metamorphic_challenges(
        contract, receipt, predecessor=predecessor,
        predecessor_ancestors=predecessor_ancestors)
    return "".join(json.dumps(_public_payload(pair, include_answer=False),
                              sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
                   for pair in private.challenge)
