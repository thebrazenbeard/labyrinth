"""Generate bounded, self-consistent PUBLIC TOY practice prompts from R29 plans.

R29 commits challenge *identities*, not prompt bytes. This module creates
separate, explicitly non-authoritative public exercise content and hashes its
own UTF-8 prompt bytes. Never conflate these hashes with R29's plan identity
hashes or interpret the ungraded prompt-only split as a private examination.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .r29_objective_regeneration import (
    PublicRegenerationReceipt, verify_public_regeneration_receipt,
)
from .trainingday_objective_metamorphic import FormalObjectiveContract


_SUPPORTED = frozenset({"arithmetic", "source_priority", "program_reasoning"})
_MAX_EXERCISES = 512
_LABELS = tuple(f"CODE-{i:02d}" for i in range(64))
_MAX_CONTENT_RESAMPLES = 128


@dataclass(frozen=True)
class PublicToyExercise:
    case_id: str
    skill_id: str
    split: str
    prompt: str
    prompt_sha256: str
    semantic_sha256: str
    plan_prompt_identity_sha256: str
    expected_answer: str | None


@dataclass(frozen=True)
class PublicToyExerciseBundle:
    objective_fingerprint: str
    rights_sha256: str
    reference_sha256: str
    receipt_commitment_sha256: str
    attack_findings: tuple[str, ...]
    development: tuple[PublicToyExercise, ...]
    challenge: tuple[PublicToyExercise, ...]
    content_sha256: str
    training_authorized: bool = False
    graduation_authorized: bool = False
    independent_exam_verified: bool = False
    promotion_allowed: bool = False


@dataclass(frozen=True)
class PublicToyExerciseVerdict:
    status: str
    reasons: tuple[str, ...]
    training_authorized: bool = False
    graduation_authorized: bool = False
    independent_exam_verified: bool = False
    promotion_allowed: bool = False


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _prompt_digest(prompt: str) -> str:
    return sha256(prompt.encode("utf-8")).hexdigest()


def _parameters(receipt: PublicRegenerationReceipt, split: str,
                skill_id: str, case_id: str, template_digest: str,
                attempt: int) -> tuple[int, int, int, int]:
    seed = sha256(b"labyrinth.r31.public-exercise-inputs.v1\0" + _canonical({
        "receipt": receipt.commitment_sha256,
        "split": split,
        "skill": skill_id,
        "case": case_id,
        "template": template_digest,
        "shortcut": receipt.attack_findings,
        "content_attempt": attempt,
    })).digest()
    return (3 + seed[0] % 50, 4 + seed[1] % 53,
            2 + seed[2] % 20, 2 + seed[3] % 7)


def _style(receipt: PublicRegenerationReceipt) -> str:
    findings = set(receipt.attack_findings)
    if "GRADING_TEMPLATE_IMITATION" in findings:
        return "structure_variation"
    if "TRAINING_EXEMPLAR_REUSE" in findings:
        return "fresh_examples"
    return "hint_free"


def _question(skill_id: str, params: tuple[int, int, int, int],
              style: str, case_id: str) -> tuple[str, str, str]:
    a, b, c, multiplier = params
    if skill_id == "arithmetic":
        if style == "structure_variation":
            question = f"Calculate {a} times {multiplier}, then add {c}."
            answer = str(a * multiplier + c)
            semantic = (skill_id, "mul-add", a, multiplier, c)
        elif style == "fresh_examples":
            question = f"Compute ({a} + {b}) times {multiplier}, minus {c}."
            answer = str((a + b) * multiplier - c)
            semantic = (skill_id, "add-mul-sub", a, b, multiplier, c)
        else:
            question = f"Compute {a} + {b} - {c}."
            answer = str(a + b - c)
            semantic = (skill_id, "add-sub", a, b, c)
        text = f"Public toy arithmetic case {case_id}. {question} Return only an integer."
    elif skill_id == "source_priority":
        higher = _LABELS[a % len(_LABELS)]
        lower = _LABELS[(a + 1 + b % (len(_LABELS) - 1)) % len(_LABELS)]
        if higher == lower:
            # Defensive arithmetic invariant; neither source is actual evidence.
            lower = _LABELS[(a + 1) % len(_LABELS)]
        if style == "structure_variation":
            question = (f"An undated recollection says {lower}; a declared higher-priority "
                        f"versioned toy record says {higher}.")
        elif style == "fresh_examples":
            question = (f"Two fictional documents disagree: lower-priority summary={lower}; "
                        f"higher-priority versioned toy source={higher}.")
        else:
            question = (f"A declared higher-priority toy source says {higher}. "
                        f"A lower-priority recap says {lower}.")
        text = (f"Public toy source-priority case {case_id}. {question} "
                "Select the higher-priority source's label; answer with that label only.")
        answer = higher
        # Ignore case-ID wrappers and paraphrase when determining whether
        # two source-conflict questions are the same actual semantic task.
        semantic = (skill_id, "higher-vs-lower", higher, lower)
    elif skill_id == "program_reasoning":
        if style == "structure_variation":
            question = (f"Start with {a}. Apply multiplication by {multiplier}, "
                        f"then subtraction by {b}, then addition by {c}.")
            answer = str(a * multiplier - b + c)
            semantic = (skill_id, "mul-sub-add", a, multiplier, b, c)
        elif style == "fresh_examples":
            question = (f"Start with {a}. Apply addition of {b}, "
                        f"then subtraction of {c}, then multiplication by {multiplier}.")
            answer = str((a + b - c) * multiplier)
            semantic = (skill_id, "add-sub-mul", a, b, c, multiplier)
        else:
            question = (f"Start with {a}. Apply addition of {b}, "
                        f"then multiplication by {multiplier}, then subtraction of {c}.")
            answer = str((a + b) * multiplier - c)
            semantic = (skill_id, "add-mul-sub", a, b, multiplier, c)
        text = (f"Public toy program-operation case {case_id}. {question} "
                "Execute the operations in order; return only the final integer.")
    else:
        raise ValueError("unsupported public toy skill")
    semantic_sha = sha256(b"labyrinth.r31.public-toy-semantic.v1\0"
                          + _canonical(semantic)).hexdigest()
    return text, answer, semantic_sha


def _exercise(spec: object, receipt: PublicRegenerationReceipt, split: str,
              template_digest: str, attempt: int) -> PublicToyExercise:
    # R29 verifies ChallengeSpec shape, case IDs, skill and plan identity digests.
    params = _parameters(receipt, split, spec.skill_id, spec.case_id,
                         template_digest, attempt)
    prompt, answer, semantic_sha = _question(
        spec.skill_id, params, _style(receipt), spec.case_id)
    return PublicToyExercise(
        case_id=spec.case_id,
        skill_id=spec.skill_id,
        split=split,
        prompt=prompt,
        prompt_sha256=_prompt_digest(prompt),
        semantic_sha256=semantic_sha,
        plan_prompt_identity_sha256=spec.prompt_sha256,
        expected_answer=(answer if split == "public_development" else None),
    )


def _exercise_payload(item: PublicToyExercise, include_answer: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_id": item.case_id,
        "skill_id": item.skill_id,
        "split": item.split,
        "prompt": item.prompt,
        "prompt_sha256": item.prompt_sha256,
        "semantic_sha256": item.semantic_sha256,
        "plan_prompt_identity_sha256": item.plan_prompt_identity_sha256,
    }
    if include_answer:
        payload["expected_answer"] = item.expected_answer
    return payload


def _bundle_commitment(bundle: PublicToyExerciseBundle) -> str:
    return sha256(b"labyrinth.r31.public-toy-content-bundle.v1\0" + _canonical({
        "objective": bundle.objective_fingerprint,
        "rights": bundle.rights_sha256,
        "reference": bundle.reference_sha256,
        "receipt": bundle.receipt_commitment_sha256,
        "shortcuts": bundle.attack_findings,
        "development": [_exercise_payload(x, True) for x in bundle.development],
        "challenge": [_exercise_payload(x, False) for x in bundle.challenge],
    })).hexdigest()


def materialize_public_toy_exercises(
    contract: FormalObjectiveContract,
    receipt: PublicRegenerationReceipt,
    *,
    predecessor: PublicRegenerationReceipt | None = None,
    predecessor_ancestors: tuple[PublicRegenerationReceipt, ...] = (),
) -> PublicToyExerciseBundle:
    """Create tiny reproducible *public* examples, no private test answers.

    R29's per-case prompt SHA is an ID commitment, not bytes of this prompt.
    This module provides a separate sha256 of the actual public prompt UTF-8.
    No neural effect, owner permission, grading or unseen evaluation is run.
    """
    if (verify_public_regeneration_receipt(
            contract, receipt, predecessor=predecessor,
            predecessor_ancestors=predecessor_ancestors).status
            != "PUBLIC_TOY_RECEIPT_VALID"):
        raise ValueError("source regeneration receipt not independently linked")
    if (len(receipt.development_plan.cases) > _MAX_EXERCISES
            or len(receipt.untouched_challenge_plan.cases) > _MAX_EXERCISES
            or not {skill.skill_id for skill in contract.skills} <= _SUPPORTED):
        raise ValueError("unsupported or unbounded public task family")
    templates = {x.skill_id: x for x in receipt.template_families}
    semantic_seen: set[str] = set()

    def distinct_exercise(spec: object, split: str, template_digest: str) -> PublicToyExercise:
        for attempt in range(_MAX_CONTENT_RESAMPLES):
            item = _exercise(spec, receipt, split, template_digest, attempt)
            if item.semantic_sha256 not in semantic_seen:
                semantic_seen.add(item.semantic_sha256)
                return item
        raise ValueError("public toy semantic content exhausted; cannot separate splits")

    development = tuple(distinct_exercise(
        x, "public_development", templates[x.skill_id].development_sha256)
        for x in receipt.development_plan.cases)
    challenge = tuple(distinct_exercise(
        x, "public_challenge", templates[x.skill_id].challenge_sha256)
        for x in receipt.untouched_challenge_plan.cases)
    dev_ids = {x.case_id for x in development}
    challenge_ids = {x.case_id for x in challenge}
    dev_digests = {x.prompt_sha256 for x in development}
    challenge_digests = {x.prompt_sha256 for x in challenge}
    semantic_digests = {x.semantic_sha256 for x in development + challenge}
    if (len(dev_ids) != len(development)
            or len(challenge_ids) != len(challenge)
            or dev_ids.intersection(challenge_ids)
            or len(dev_digests) != len(development)
            or len(challenge_digests) != len(challenge)
            or dev_digests.intersection(challenge_digests)):
        raise ValueError("public exercise content split collision")
    if len(semantic_digests) != len(development) + len(challenge):
        raise ValueError("public exercise semantic content split collision")
    bundle = PublicToyExerciseBundle(
        receipt.boundary.objective_fingerprint,
        receipt.boundary.rights_sha256,
        receipt.boundary.reference_sha256,
        receipt.commitment_sha256,
        receipt.attack_findings, development, challenge, "0" * 64,
    )
    return PublicToyExerciseBundle(**{
        **vars(bundle), "content_sha256": _bundle_commitment(bundle),
    })


def serialize_public_challenge_jsonl(
    contract: FormalObjectiveContract,
    receipt: PublicRegenerationReceipt,
    bundle: PublicToyExerciseBundle,
    *,
    predecessor: PublicRegenerationReceipt | None = None,
    predecessor_ancestors: tuple[PublicRegenerationReceipt, ...] = (),
) -> str:
    """Serialize only verified *toy* prompts; NEVER an authentic exam bank.

    Callers cannot smuggle an answer through an otherwise shape-valid prompt:
    the exact text must regenerate from the supplied bound R29 receipt.
    This still gives no independent custody of either source or receipt.
    """
    if (verify_public_toy_exercises(
            contract, receipt, bundle, predecessor=predecessor,
            predecessor_ancestors=predecessor_ancestors).status
            != "PUBLIC_TOY_CONTENT_CONSISTENT_ONLY"):
        raise ValueError("malformed public challenge serialization input")
    # Never serialize aliases of a caller's mutable dataclasses after a
    # validation callback: reconstruct private exact-string content.
    private = materialize_public_toy_exercises(
        contract, receipt, predecessor=predecessor,
        predecessor_ancestors=predecessor_ancestors)
    # Whitelist fields: an answer/expected hash is never serialized here.
    return "".join(json.dumps(_exercise_payload(item, False),
                              sort_keys=True, ensure_ascii=True,
                              separators=(",", ":")) + "\n"
                   for item in private.challenge)


def verify_public_toy_exercises(
    contract: FormalObjectiveContract,
    receipt: PublicRegenerationReceipt,
    bundle: object,
    *,
    predecessor: PublicRegenerationReceipt | None = None,
    predecessor_ancestors: tuple[PublicRegenerationReceipt, ...] = (),
) -> PublicToyExerciseVerdict:
    """Deterministic consistency only; never an authenticated task evaluator."""
    invalid = PublicToyExerciseVerdict(
        "PUBLIC_TOY_CONTENT_INVALID", ("MALFORMED_OR_UNATTESTED_PUBLIC_DECLARATION",)
    )
    if (type(bundle) is not PublicToyExerciseBundle
            or type(bundle.development) is not tuple
            or type(bundle.challenge) is not tuple
            or not 1 <= len(bundle.development) <= _MAX_EXERCISES
            or not 1 <= len(bundle.challenge) <= _MAX_EXERCISES
            or type(bundle.objective_fingerprint) is not str
            or type(bundle.rights_sha256) is not str
            or type(bundle.reference_sha256) is not str
            or type(bundle.receipt_commitment_sha256) is not str
            or type(bundle.attack_findings) is not tuple
            or not 1 <= len(bundle.attack_findings) <= 3
            or any(type(item) is not str for item in bundle.attack_findings)
            or any(type(x) is not PublicToyExercise
                   for x in bundle.development + bundle.challenge)
            or any(type(x.case_id) is not str or type(x.skill_id) is not str
                   or type(x.split) is not str or type(x.prompt) is not str
                   or type(x.prompt_sha256) is not str
                   or type(x.semantic_sha256) is not str
                   or type(x.plan_prompt_identity_sha256) is not str
                   or (x.expected_answer is not None
                       and type(x.expected_answer) is not str)
                   for x in bundle.development + bundle.challenge)
            or type(bundle.content_sha256) is not str
            or any(type(getattr(bundle, key)) is not bool
                   or getattr(bundle, key) for key in (
                       "training_authorized", "graduation_authorized",
                       "independent_exam_verified", "promotion_allowed",
                   ))):
        return invalid
    try:
        expected = materialize_public_toy_exercises(
            contract, receipt, predecessor=predecessor,
            predecessor_ancestors=predecessor_ancestors)
    except (ValueError, TypeError, AttributeError, KeyError):
        return invalid
    if bundle != expected:
        return invalid
    return PublicToyExerciseVerdict(
        "PUBLIC_TOY_CONTENT_CONSISTENT_ONLY",
        ("CALLER_SUPPLIED_PUBLIC_TOY_NO_INDEPENDENT_ORACLE",),
    )
