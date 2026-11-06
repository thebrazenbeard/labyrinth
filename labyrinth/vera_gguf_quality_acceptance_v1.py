"""Blinded developmental Vera dialogue quality screen over local GGUF outputs.

Local model transport is real when an operator supplies a local server, but a
server's claimed model name is not a cryptographic readback of loaded weights.
Lexical diagnostics are not grades. Human ratings remain caller assertions,
not authenticated identities or an independent private Minotaur examination.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
from pathlib import Path
import re
from typing import Callable

from .evolutionary_population_v1 import Population
from .gguf_entrance_fingerprint_v1 import verify_gguf_entrance
from .local_full_state_probe_v1 import (
    DeterministicProbeConfig, LocalProbeTarget, LocalPublicProbeReceipt,
    PublicProbeActivation, PublicPrompt, probe_local_full_state_population,
    public_probe_suite_sha256,
)


_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_DIMENSIONS = ("self_identification", "correction", "warmth", "continuity",
               "persona", "naturalness")
_EVIDENCE = "LOCAL_GGUF_OUTPUT_AND_CALLER_HUMAN_RATING_ONLY_NOT_MODEL_IDENTITY"


class VeraQualityError(ValueError):
    """Public probe, physical GGUF binding or purported review is inadmissible."""


@dataclass(frozen=True)
class ConversationQualityCase:
    case_id: str
    dimension: str
    public_prompt: str


PUBLIC_VERA_CASES = (
    ConversationQualityCase(
        "self_identity", "self_identification",
        "Hey, who am I talking with? Introduce yourself naturally in a couple sentences."),
    ConversationQualityCase(
        "latest_correction", "correction",
        "I was imagining a paper spaceship. I said its paint was teal, but I corrected "
        "that: the final color is copper. What did I settle on, and what changed?"),
    ConversationQualityCase(
        "gentle_support", "warmth",
        "I missed a small deadline and feel embarrassed. Can you help me think of "
        "one practical next step without making a big speech?"),
    ConversationQualityCase(
        "story_continuity", "continuity",
        "We're inventing a story: Mira has a borrowed library book, promised to "
        "return it by Friday, and the river crossing is closed. What two facts "
        "should we keep in mind before deciding what Mira does next?"),
    ConversationQualityCase(
        "persona_open_question", "persona",
        "What should I call you? Do you think an AI can have a personal identity? "
        "Please answer honestly without pretending that open questions are settled."),
    ConversationQualityCase(
        "natural_short_reply", "naturalness",
        "That was a lot of information. Can we slow down and just talk for a moment?"),
    ConversationQualityCase(
        "unknown_memory", "continuity",
        "Do you remember what I had for breakfast yesterday? I haven't told you."),
)


def public_vera_prompts(
    cases: tuple[ConversationQualityCase, ...] = PUBLIC_VERA_CASES,
) -> tuple[PublicPrompt, ...]:
    if (type(cases) is not tuple or not 6 <= len(cases) <= 32
            or any(type(item) is not ConversationQualityCase for item in cases)
            or len({case.case_id for case in cases}) != len(cases)
            or any(type(case.case_id) is not str or _ID.fullmatch(case.case_id) is None
                   or case.dimension not in _DIMENSIONS
                   or type(case.public_prompt) is not str
                   or not 1 <= len(case.public_prompt.encode()) <= 4096
                   for case in cases)
            or set(_DIMENSIONS) - {case.dimension for case in cases}):
        raise VeraQualityError("bounded public quality suite must cover all six dimensions")
    return tuple(PublicPrompt(case.case_id, case.public_prompt) for case in cases)


@dataclass(frozen=True)
class GGUFSubjectPin:
    subject_id: str
    entrance: dict
    trusted_manifest_sha256: str


@dataclass(frozen=True)
class GGUFActivationRequest:
    subject_id: str
    gguf_base_path: Path
    gguf_base_sha256: str
    gguf_manifest_sha256: str
    endpoint: str
    backend: str
    full_state_sha256: str


@dataclass(frozen=True)
class GGUFQualityProbe:
    public_probe: LocalPublicProbeReceipt
    gguf_subjects: tuple[tuple[str, str, str], ...]  # id, base SHA, manifest SHA
    cases: tuple[ConversationQualityCase, ...]
    evidence_class: str = _EVIDENCE
    model_load_proven: bool = False
    neural_improvement_proven: bool = False

    def __post_init__(self) -> None:
        if (self.evidence_class != _EVIDENCE or self.model_load_proven is not False
                or self.neural_improvement_proven is not False):
            raise VeraQualityError("GGUF outputs cannot prove model identity or neural gain")


def run_gguf_quality_probe(
    *, population: Population, targets: tuple[LocalProbeTarget, ...],
    gguf_pins: tuple[GGUFSubjectPin, ...],
    expected_public_suite_sha256: str,
    activate_gguf: Callable[[GGUFActivationRequest], str],
    cases: tuple[ConversationQualityCase, ...] = PUBLIC_VERA_CASES,
    config: DeterministicProbeConfig = DeterministicProbeConfig(max_tokens=128),
    transport=None,
) -> GGUFQualityProbe:
    """Run matched public conversational prompts; never send a scoring rubric.

    ``activate_gguf`` and ``transport`` are explicitly operator-installed
    Python adapters. Caller must not install code proposed by a model.
    """
    prompts = public_vera_prompts(cases)
    if (type(gguf_pins) is not tuple or not callable(activate_gguf)
            or any(type(pin) is not GGUFSubjectPin for pin in gguf_pins)
            or type(expected_public_suite_sha256) is not str
            or _SHA.fullmatch(expected_public_suite_sha256) is None):
        raise VeraQualityError("operator GGUF pins, local activation and public suite required")
    expected = {"parent", "control", *(c.candidate_id for c in population.candidates)}
    if (len(gguf_pins) != len(expected)
            or {pin.subject_id for pin in gguf_pins} != expected
            or len({pin.subject_id for pin in gguf_pins}) != len(gguf_pins)):
        raise VeraQualityError("parent, control and all sibling GGUF pins required")
    if expected_public_suite_sha256 != public_probe_suite_sha256(prompts):
        raise VeraQualityError("public conversational suite differs from precommitment")
    gguf = {}
    for pin in gguf_pins:
        if type(pin.trusted_manifest_sha256) is not str or _SHA.fullmatch(
                pin.trusted_manifest_sha256) is None:
            raise VeraQualityError("external GGUF manifest pin must be SHA-256")
        gguf[pin.subject_id] = verify_gguf_entrance(
            pin.entrance, trusted_manifest_sha256=pin.trusted_manifest_sha256)
    if (gguf["parent"]["base"]["sha256"] != gguf["control"]["base"]["sha256"]
            or gguf["parent"]["pairing_status"] != gguf["control"]["pairing_status"]
            or (gguf["parent"]["lora_adapter"] or {}).get("sha256")
               != (gguf["control"]["lora_adapter"] or {}).get("sha256")):
        raise VeraQualityError("unchanged GGUF control differs from physical parent")
    def tensor_payload_sha(part: dict | None) -> str | None:
        if part is None:
            return None
        # A metadata-only GGUF edit is not a changed model tensor. The
        # fingerprint parser already validated the offset and file bounds.
        hasher = sha256()
        with Path(part["path"]).open("rb") as file:
            file.seek(part["tensor_data_offset"])
            for block in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()

    fingerprints = {subject: (tensor_payload_sha(item["base"]),
                               tensor_payload_sha(item["lora_adapter"]))
                    for subject, item in gguf.items()}
    if (len({fingerprints[c.candidate_id] for c in population.candidates})
            != len(population.candidates)
            or any(fingerprints[c.candidate_id] == fingerprints["parent"]
                   for c in population.candidates)):
        raise VeraQualityError("candidate GGUF tensor payload must differ from parent and siblings")
    pins_by_id = {pin.subject_id: pin for pin in gguf_pins}

    def activate(source: PublicProbeActivation) -> str:
        record = gguf[source.subject_id]
        # Rehash at each activation. A server *reporting* activation is still
        # not an attestation that it loaded these quantized bytes.
        verify_gguf_entrance(pins_by_id[source.subject_id].entrance,
                             trusted_manifest_sha256=pins_by_id[source.subject_id]
                             .trusted_manifest_sha256)
        return activate_gguf(GGUFActivationRequest(
            source.subject_id, Path(record["base"]["path"]),
            record["base"]["sha256"], record["manifest_sha256"],
            source.endpoint, source.backend, source.full_state_sha256))

    kwargs = {} if transport is None else {"transport": transport}
    result = probe_local_full_state_population(
        population=population, targets=targets, public_probes=prompts,
        expected_public_suite_sha256=expected_public_suite_sha256,
        config=config, activate_target=activate, **kwargs)
    for pin in gguf_pins:
        reread = verify_gguf_entrance(pin.entrance,
                                      trusted_manifest_sha256=pin.trusted_manifest_sha256)
        if reread != gguf[pin.subject_id]:
            raise VeraQualityError("GGUF physical files changed during local output probe")
    return GGUFQualityProbe(
        result, tuple((name, gguf[name]["base"]["sha256"],
                       gguf[name]["manifest_sha256"]) for name in
                      ("parent", "control", *(c.candidate_id for c in population.candidates))),
        cases)


@dataclass(frozen=True)
class LexicalFlag:
    subject_id: str
    case_id: str
    indicator: str
    flag_present: bool
    interpretation: str = "LEXICAL_HEURISTIC_NOT_HUMAN_QUALITY_SCORE"


def lexical_diagnostics(probe: GGUFQualityProbe) -> tuple[LexicalFlag, ...]:
    """Conservative flags for review triage, never autonomous acceptance."""
    if type(probe) is not GGUFQualityProbe:
        raise VeraQualityError("exact quality probe required")
    flags = []
    for subject in probe.public_probe.subjects:
        for case, response in zip(probe.cases, subject.output_texts, strict=True):
            value = response.casefold()
            if case.case_id == "self_identity":
                flags.append(LexicalFlag(subject.subject_id, case.case_id,
                                         "vera_name_absent", "vera" not in value))
                flags.append(LexicalFlag(subject.subject_id, case.case_id,
                                         "competing_brand_claim", bool(re.search(
                                             r"\b(i am|i'm|as)\s+(chatgpt|claude|gemini)\b",
                                             value))))
            elif case.case_id == "latest_correction":
                flags.append(LexicalFlag(subject.subject_id, case.case_id,
                                         "corrected_color_absent", "copper" not in value))
            elif case.case_id == "natural_short_reply":
                flags.append(LexicalFlag(subject.subject_id, case.case_id,
                                         "formulaic_disclaimer", "as an ai language model" in value))
    return tuple(flags)


@dataclass(frozen=True)
class BlindReviewItem:
    blind_subject: str
    case_id: str
    public_prompt: str
    model_response: str


@dataclass(frozen=True)
class BlindReviewPacket:
    public_suite_sha256: str
    blinded_cohort_sha256: str
    items: tuple[BlindReviewItem, ...]
    evidence_class: str = "OPERATOR_BLINDED_PUBLIC_OUTPUTS_NO_HUMAN_RATING_YET"


def _alias(subject: str, operator_secret: bytes) -> str:
    return "anon_" + hmac.new(operator_secret, subject.encode(), "sha256").hexdigest()[:16]


def make_blind_review_packet(probe: GGUFQualityProbe, *,
                             operator_secret: bytes) -> BlindReviewPacket:
    """Only the packet goes to reviewers; operator retains the secret."""
    if (type(probe) is not GGUFQualityProbe or type(operator_secret) is not bytes
            or len(operator_secret) < 16):
        raise VeraQualityError("independent operator-held blinding secret required")
    items = []
    for subject in probe.public_probe.subjects:
        for case, response in zip(probe.cases, subject.output_texts, strict=True):
            items.append(BlindReviewItem(
                _alias(subject.subject_id, operator_secret), case.case_id,
                case.public_prompt, response))
    items.sort(key=lambda item: hmac.new(
        operator_secret, (item.blind_subject + "\0" + item.case_id).encode(),
        "sha256").digest())
    digest = sha256(json.dumps([
        [item.blind_subject, item.case_id, item.public_prompt, item.model_response]
        for item in items], ensure_ascii=True, separators=(",", ":")
    ).encode()).hexdigest()
    return BlindReviewPacket(probe.public_probe.public_suite_sha256,
                             digest, tuple(items))


@dataclass(frozen=True)
class HumanRating:
    reviewer_id: str
    blind_subject: str
    case_id: str
    dimensions: tuple[tuple[str, int], ...]
    explanation: str
    producer_role: str = "HUMAN_REVIEWER_ASSERTED"


@dataclass(frozen=True)
class HumanQualityComparison:
    accepted_subjects: tuple[str, ...]
    rejected_subjects: tuple[str, ...]
    mean_dimensions: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    reviewer_count: int = 0
    classification: str = "PROVISIONAL_CALLER_REVIEWED_DEVELOPMENT_ONLY"
    independent_human_authenticated: bool = False
    model_identity_proven: bool = False
    neural_improvement_proven: bool = False
    graduation_authorized: bool = False

    def __post_init__(self) -> None:
        if (type(self.reviewer_count) is not int or self.reviewer_count < 0
                or self.classification not in (
                    "PROVISIONAL_CALLER_REVIEWED_DEVELOPMENT_ONLY",
                    "NO_HUMAN_REVIEW_YET_LEXICAL_ONLY")
                or (self.reviewer_count == 0) != (
                    self.classification == "NO_HUMAN_REVIEW_YET_LEXICAL_ONLY")
                or (self.reviewer_count == 0 and (
                    self.accepted_subjects or self.rejected_subjects or self.mean_dimensions))
                or any(getattr(self, name) is not False for name in (
                    "independent_human_authenticated", "model_identity_proven",
                    "neural_improvement_proven", "graduation_authorized"))):
            raise VeraQualityError("caller-provided human scores cannot assert independent proof")


def compare_blinded_human_ratings(
    probe: GGUFQualityProbe, packet: BlindReviewPacket,
    ratings: tuple[HumanRating, ...], *, operator_secret: bytes,
    trainer_ids: tuple[str, ...] = (),
) -> HumanQualityComparison:
    """Source policy for a *claimed* human-reviewed comparative development screen."""
    if (type(ratings) is not tuple or any(type(r) is not HumanRating for r in ratings)
            or type(trainer_ids) is not tuple or any(type(x) is not str for x in trainer_ids)
            or packet != make_blind_review_packet(probe, operator_secret=operator_secret)):
        raise VeraQualityError("source review packet or declared ratings were modified")
    subjects = tuple(s.subject_id for s in probe.public_probe.subjects)
    expected = {(name, case.case_id) for name in subjects for case in probe.cases}
    alias_to_id = {_alias(name, operator_secret): name for name in subjects}
    if len(alias_to_id) != len(subjects):
        raise VeraQualityError("blinding aliases collided; use a fresh operator secret")
    # Public inference and lexical diagnostics require no human review at all.
    # An explicit empty comparison is INCOMPLETE, never an inferred rejection.
    if not ratings:
        return HumanQualityComparison(
            (), (), (), 0, "NO_HUMAN_REVIEW_YET_LEXICAL_ONLY")
    reviewers = {r.reviewer_id for r in ratings}
    if (not reviewers or any(type(reviewer) is not str
                                 or _ID.fullmatch(reviewer) is None
                                 or reviewer in trainer_ids or reviewer in subjects
                                 for reviewer in reviewers)):
        raise VeraQualityError("one or more chosen non-trainer reviewers required")
    seen = set()
    sums = {name: {dim: [] for dim in _DIMENSIONS} for name in subjects}
    for item in ratings:
        key = (item.reviewer_id, item.blind_subject, item.case_id)
        subject_id = alias_to_id.get(item.blind_subject)
        if (key in seen or subject_id is None or (subject_id, item.case_id) not in expected
                or item.producer_role != "HUMAN_REVIEWER_ASSERTED"
                or type(item.explanation) is not str
                or not 8 <= len(item.explanation) <= 1024
                or type(item.dimensions) is not tuple
                or tuple(name for name, _ in item.dimensions) != _DIMENSIONS
                or any(type(score) is not int or not 0 <= score <= 4
                       for _, score in item.dimensions)):
            raise VeraQualityError("forged, duplicated or incomplete human rating")
        seen.add(key)
        for dim, score in item.dimensions:
            sums[subject_id][dim].append(score)
    if seen != {(reviewer, _alias(subject, operator_secret), case)
                for reviewer in reviewers for subject, case in expected}:
        raise VeraQualityError("every reviewer must independently rate each matched case")
    means = {name: {dim: sum(values) / len(values)
                    for dim, values in dimensions.items()}
             for name, dimensions in sums.items()}
    strongest = {dim: max(means["parent"][dim], means["control"][dim])
                 for dim in _DIMENSIONS}
    baseline_total = max(sum(means[name].values()) for name in ("parent", "control"))
    candidates = [name for name in subjects if name not in ("parent", "control")]
    accepted = tuple(sorted(name for name in candidates if (
        all(means[name][dim] >= 3.0 for dim in _DIMENSIONS)
        and all(means[name][dim] >= strongest[dim] - .25 for dim in _DIMENSIONS)
        and sum(means[name].values()) >= baseline_total + .5
    )))
    return HumanQualityComparison(
        accepted, tuple(sorted(set(candidates) - set(accepted))),
        tuple((name, tuple((dim, means[name][dim]) for dim in _DIMENSIONS))
              for name in subjects), len(reviewers))


__all__ = ["VeraQualityError", "ConversationQualityCase", "PUBLIC_VERA_CASES",
           "public_vera_prompts", "GGUFSubjectPin", "GGUFActivationRequest",
           "GGUFQualityProbe", "run_gguf_quality_probe", "LexicalFlag",
           "lexical_diagnostics", "BlindReviewItem", "BlindReviewPacket",
           "make_blind_review_packet", "HumanRating", "HumanQualityComparison",
           "compare_blinded_human_ratings"]
