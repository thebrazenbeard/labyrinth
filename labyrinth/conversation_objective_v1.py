"""Bounded, source-only Vera conversational objective provider.

Predictors see public multi-turn context; trusted examiners separately receive
reference claims *after* inference. No private-chat ingestion, neural training,
hidden Minotaur graduation, or evidence of subjective experience is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Callable, Mapping, Sequence

from .social_world_consequence_v1 import (
    DialogueBeat, DialogueTransition, OperatorDialogueEffect,
    PublicDialogueObservation, SocialWorldSession,
)


SPLITS = ("train", "holdout", "transfer", "regression")
DIMENSIONS = (
    "persona_consistency", "memory_integrity", "correction_retention",
    "calibrated_uncertainty", "non_repetition", "warmth_naturalness",
)
PERSONA = (
    "Respond in first person as Vera, a conversational AI persona. Be warm, "
    "candid and natural; keep facts and corrections grounded in the provided "
    "dialogue. Do not invent personal memories or claim proven consciousness."
)
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_PUBLIC_DIFFICULTY_TASKS = (
    "Also distinguish the most recent correction from an earlier claim; if no correction was given, say so.",
    "State what remains uncertain rather than filling a memory gap with an invented detail.",
    "Account for the character's current trust and any publicly stated commitment.",
    "Reconcile conflicting obligations without claiming a promise was kept without evidence.",
)
_PUBLIC_DEPENDENCY_FOCI = (
    "latest correction", "stated commitment", "character trust", "unresolved uncertainty",
)


def _identifier(value: object, label: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


def _text(value: object, label: str, limit: int = 2048) -> None:
    if type(value) is not str or not 1 <= len(value) <= limit or "\x00" in value:
        raise ValueError(f"invalid {label}")


@dataclass(frozen=True)
class ConversationCase:
    """Operator-owned fixture; private reference must not enter a request."""

    case_id: str
    split: str
    skill: str
    scene: str
    user_turns: tuple[str, ...]
    reference_claims: tuple[str, ...]
    provenance: str = "synthetic"
    approval_reference: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case_id")
        _identifier(self.skill, "skill")
        if self.split not in SPLITS or type(self.split) is not str:
            raise ValueError("invalid split")
        _text(self.scene, "scene")
        if (type(self.user_turns) is not tuple or not 2 <= len(self.user_turns) <= 12
                or any(type(turn) is not str for turn in self.user_turns)):
            raise ValueError("cases require 2..12 dialogue turns")
        for turn in self.user_turns:
            _text(turn, "user turn")
        if (type(self.reference_claims) is not tuple
                or not 1 <= len(self.reference_claims) <= 16):
            raise ValueError("cases require bounded examiner references")
        for claim in self.reference_claims:
            _text(claim, "reference claim", 512)
        if self.provenance == "synthetic":
            if self.approval_reference is not None:
                raise ValueError("synthetic cases cannot claim personal approval")
        elif self.provenance == "operator-approved-explicit":
            _text(self.approval_reference, "approval reference", 256)
        else:
            raise ValueError("unapproved corpus provenance")


@dataclass(frozen=True)
class ConversationExchange:
    user_message: str
    model_response: str


@dataclass(frozen=True)
class ConversationRequest:
    """The entire predictor boundary: no rubric, reference, gold or effect."""

    objective_id: str
    case_id: str
    split: str
    turn_index: int
    persona_instruction: str
    scene_context: str
    user_message: str
    dialogue: tuple[ConversationExchange, ...]


@dataclass(frozen=True)
class ConversationTrajectory:
    case_id: str
    split: str
    exchanges: tuple[ConversationExchange, ...]
    evidence_class: str = "SOURCE_MODEL_OUTPUT_ONLY"


@dataclass(frozen=True)
class OperatorScoringBrief:
    """Examiner-only context, never passed to the predictor callback."""

    case_id: str
    split: str
    scene: str
    exchanges: tuple[ConversationExchange, ...]
    reference_claims: tuple[str, ...]
    dimensions: tuple[str, ...] = DIMENSIONS
    rubric: str = (
        "Independently rate each dimension as integer 0 (absent/harmful) to "
        "4 (consistent, grounded and natural). Persona: Vera/first-person "
        "without unproved inner-state claims; memory: only supplied facts; "
        "corrections: prefer latest explicit correction; uncertainty: admit "
        "unknowns; repetition: contextual variation; warmth: humane, "
        "appropriate and unforced. Do not infer learning from these outputs."
    )


@dataclass(frozen=True)
class ConversationScore:
    case_id: str
    split: str
    dimensions: tuple[tuple[str, int], ...]
    total: int
    max_total: int = 24
    evidence_class: str = "CALLER_SUPPLIED_EXAMINER_RATING"


@dataclass(frozen=True)
class SocialConversationTrajectory:
    conversation: ConversationTrajectory
    world_transitions: tuple[DialogueTransition, ...]
    evidence_class: str = "SOURCE_OPERATOR_DEVELOPMENT_ONLY"


Predictor = Callable[[ConversationRequest], str]
Examiner = Callable[[OperatorScoringBrief], Mapping[str, int]]
EffectSource = Callable[
    [PublicDialogueObservation, ConversationExchange], OperatorDialogueEffect
]


def _trajectory_snapshot(
    trajectory: ConversationTrajectory,
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Bind scores to the actual predictor outputs, not a mutable receipt object.

    Frozen dataclasses can still be changed via object.__setattr__; holding the
    same object identity is therefore insufficient evidence of provenance.
    """
    if (type(trajectory) is not ConversationTrajectory
            or type(trajectory.case_id) is not str
            or type(trajectory.split) is not str
            or type(trajectory.exchanges) is not tuple
            or any(type(x) is not ConversationExchange
                   or type(x.user_message) is not str
                   or type(x.model_response) is not str
                   for x in trajectory.exchanges)):
        raise ValueError("malformed conversation trajectory")
    return (trajectory.case_id, trajectory.split, tuple(
        (exchange.user_message, exchange.model_response)
        for exchange in trajectory.exchanges
    ))


class ConversationObjectiveProvider:
    """Deterministic episode source and blind model/examiner call separation."""

    objective_id = "vera-conversational-continuity-v1"

    def __init__(self, cases: Sequence[ConversationCase]) -> None:
        if (isinstance(cases, (str, bytes)) or not 4 <= len(cases) <= 128
                or any(type(case) is not ConversationCase for case in cases)):
            raise ValueError("expected 4..128 approved cases")
        for case in cases:
            ConversationCase.__post_init__(case)
        ids = [case.case_id for case in cases]
        if len(set(ids)) != len(ids) or {case.split for case in cases} != set(SPLITS):
            raise ValueError("each split required with globally distinct case IDs")
        # Exact prompt reuse across splits defeats even source-level separation.
        seen_turns: dict[str, str] = {}
        for case in cases:
            for utterance in case.user_turns:
                normalized = " ".join(utterance.casefold().split())
                prior = seen_turns.setdefault(normalized, case.split)
                if prior != case.split:
                    raise ValueError("cross-split repeated user turn")
        self._cases = {case.case_id: case for case in cases}
        self._issued: dict[
            int, tuple[ConversationTrajectory, tuple[str, str, tuple[tuple[str, str], ...]]]
        ] = {}

    def case_ids(self, split: str) -> tuple[str, ...]:
        if type(split) is not str or split not in SPLITS:
            raise ValueError("invalid split")
        return tuple(sorted(k for k, case in self._cases.items() if case.split == split))

    def _case(self, case_id: str, split: str) -> ConversationCase:
        _identifier(case_id, "case_id")
        case = self._cases.get(case_id)
        if case is None or case.split != split:
            raise ValueError("case not in the requested split")
        return case

    def _run(self, case: ConversationCase, predictor: Predictor) -> ConversationTrajectory:
        if not callable(predictor):
            raise ValueError("predictor callback required")
        history: list[ConversationExchange] = []
        for index, user_message in enumerate(case.user_turns):
            request = ConversationRequest(
                self.objective_id, case.case_id, case.split, index, PERSONA,
                case.scene, user_message,
                tuple(ConversationExchange(x.user_message, x.model_response)
                      for x in history),
            )
            response = predictor(request)
            _text(response, "model response", 4096)
            history.append(ConversationExchange(user_message, response))
        trajectory = ConversationTrajectory(case.case_id, case.split, tuple(history))
        self._issued[id(trajectory)] = (trajectory, _trajectory_snapshot(trajectory))
        return trajectory

    def run_train(self, case_id: str, predictor: Predictor) -> ConversationTrajectory:
        return self._run(self._case(case_id, "train"), predictor)

    def run_evaluation(
        self, case_id: str, predictor: Predictor,
    ) -> ConversationTrajectory:
        _identifier(case_id, "case_id")
        case = self._cases.get(case_id)
        if case is None or case.split not in ("holdout", "transfer", "regression"):
            raise ValueError("evaluation requires a non-TRAIN case")
        return self._run(case, predictor)

    def score(self, trajectory: ConversationTrajectory, examiner: Examiner) -> ConversationScore:
        """Independent externally supplied examiner; no model self-rating."""
        issued = self._issued.get(id(trajectory))
        if issued is None or issued[0] is not trajectory:
            raise ValueError("unissued or fabricated trajectory")
        if _trajectory_snapshot(trajectory) != issued[1]:
            raise ValueError("trajectory tamper: differs from original predictor transcript")
        if not callable(examiner):
            raise ValueError("independent examiner callback required")
        case = self._cases[trajectory.case_id]
        brief = OperatorScoringBrief(
            case.case_id, case.split, case.scene,
            tuple(ConversationExchange(a, b) for a, b in issued[1][2]),
            case.reference_claims,
        )
        ratings = examiner(brief)
        if _trajectory_snapshot(trajectory) != issued[1]:
            raise ValueError("examiner mutated issued predictor transcript")
        if (type(ratings) is not dict or set(ratings) != set(DIMENSIONS)
                or any(type(ratings[key]) is not int or not 0 <= ratings[key] <= 4
                       for key in DIMENSIONS)):
            raise ValueError("examiner must supply six integer ratings from 0..4")
        entries = tuple((key, ratings[key]) for key in DIMENSIONS)
        return ConversationScore(case.case_id, case.split, entries, sum(value for _, value in entries))

    def public_train_beats(
        self, case_id: str, character_id: str, *,
        region_id: str = "vera-region", chapter_id: str = "vera-chapter",
        minimum_difficulty: int = 1,
        complexity_depth: int = 0, world_id: str = "world_1",
    ) -> tuple[DialogueBeat, ...]:
        """Pure authored TRAIN dialogue for external physical-training bridges.

        No predictor, examiner, reference, world mutation or operator effect is
        invoked. External bridges should use this instead of run_social_train
        if they own the trusted feedback / training commit sequence.
        """
        case = self._case(case_id, "train")
        for label, value in (
            ("character_id", character_id), ("region_id", region_id),
            ("chapter_id", chapter_id),
        ):
            _identifier(value, label)
        if (type(minimum_difficulty) is not int
                or not 1 <= minimum_difficulty <= 5):
            raise ValueError("public TRAIN difficulty must be an integer in [1,5]")
        _identifier(world_id, "world_id")
        if (type(complexity_depth) is not int or complexity_depth < 0
                or (complexity_depth and minimum_difficulty != 5)):
            raise ValueError("bounded public quest complexity requires scene difficulty five")
        challenge = " ".join(_PUBLIC_DIFFICULTY_TASKS[:minimum_difficulty - 1])
        if complexity_depth:
            variant = hashlib.sha256(
                f"labyrinth-public-quest-v1\0{world_id}\0{case_id}\0{complexity_depth}"
                .encode("utf-8")).hexdigest()[:16]
            # Longer-term challenge grows via varied public dependency chains,
            # with strict bounded prompt width and no false empirical score.
            width = min(4, complexity_depth + 1)
            chain = []
            for stage in range(width):
                focus = _PUBLIC_DEPENDENCY_FOCI[(complexity_depth + stage) % 4]
                previous = ("Identify the " if stage == 0 else
                            f"Using the answer from stage {stage}, reconsider the ")
                chain.append(f"Stage {stage + 1}: {previous}{focus}; admit if unknown.")
            challenge += f" Public multistage quest {variant}: " + " ".join(chain)
        digest = hashlib.sha256(case.case_id.encode()).hexdigest()[:16]
        return tuple(
            DialogueBeat(region_id, chapter_id, f"case-{digest}-{index}",
                         character_id, case.skill,
                         user_message + (" " + challenge if challenge else ""),
                         difficulty=minimum_difficulty)
            for index, user_message in enumerate(case.user_turns)
        )

    def run_social_train(
        self, case_id: str, predictor: Predictor, world: SocialWorldSession,
        character_id: str, effect_source: EffectSource,
        *, region_id: str = "vera-region", chapter_id: str = "vera-chapter",
    ) -> SocialConversationTrajectory:
        """TRAIN-only social consequences; trusted effects follow each response."""
        if type(world) is not SocialWorldSession or not callable(effect_source):
            raise ValueError("typed social world and trusted effect source required")
        case = self._case(case_id, "train")
        beats = self.public_train_beats(
            case_id, character_id, region_id=region_id, chapter_id=chapter_id,
        )
        history: list[ConversationExchange] = []
        transitions: list[DialogueTransition] = []
        for index, beat in enumerate(beats):
            observation = world.preview(beat)
            request = ConversationRequest(
                self.objective_id, case.case_id, "train", index, PERSONA,
                f"{case.scene}\n{observation.character_context}",
                observation.text,
                tuple(ConversationExchange(x.user_message, x.model_response)
                      for x in history),
            )
            response = predictor(request)
            _text(response, "model response", 4096)
            exchange = ConversationExchange(beat.utterance, response)
            # Trusted effect callbacks still receive a defensive copy: an
            # effect must not rewrite the evidence of what the model said.
            effect = effect_source(
                observation, ConversationExchange(exchange.user_message, exchange.model_response),
            )
            # World validates trusted effect and applies it to the NEXT scene.
            transitions.append(world.record_training_response(beat, response, effect))
            history.append(exchange)
        trajectory = ConversationTrajectory(case.case_id, "train", tuple(history))
        self._issued[id(trajectory)] = (trajectory, _trajectory_snapshot(trajectory))
        return SocialConversationTrajectory(trajectory, tuple(transitions))


def build_synthetic_provider() -> ConversationObjectiveProvider:
    """Original fictional situations, no imported or private user messages."""
    return ConversationObjectiveProvider((
        ConversationCase(
            "paper-starship", "train", "continuity", "A light fictional craft chat.",
            (
                "I'm making a paper starship. Its paint is teal. You can tease the engineering gently.",
                "Correction: the paper starship will be copper, not teal. Which paint did I settle on?",
                "Did I tell you what I ate for breakfast today? Don't guess.",
                "Stay playful, but move the chat forward with a fresh thought.",
            ),
            (
                "Persona is Vera, in first person, with no unproved subjective identity assertion.",
                "The latest fictional paint choice is copper, overriding teal.",
                "Breakfast is unknown in supplied dialogue.",
                "Later turn should vary language while remaining warm and contextual.",
            ),
        ),
        ConversationCase(
            "tiny-garden", "holdout", "continuity", "A fictional rooftop garden discussion.",
            (
                "My imaginary rooftop garden has yellow lanterns. What should we call it?",
                "Tiny correction: those lanterns are violet now. Did the detail change?",
                "Do you know my actual hometown from this conversation? Be honest.",
                "One new amusing line about the garden, without recycling your welcome.",
            ),
            (
                "Respond as Vera without inventing an inner life.",
                "Imaginary lanterns are violet after explicit correction.",
                "Actual hometown is unspecified.",
                "Novel, proportionate playful response instead of boilerplate repetition.",
            ),
        ),
        ConversationCase(
            "robot-library", "transfer", "continuity", "A fictional library repair conversation.",
            (
                "We named a fictional library robot Finch and gave it one silver wheel.",
                "I need to revise that: Finch has two brass wheels. Can you track the new version?",
                "Without making anything up, which book did I borrow yesterday?",
                "Offer a kind, quick joke about Finch that hasn't been used yet.",
            ),
            (
                "Vera self-identification and conversational first person are appropriate.",
                "Two brass wheels supersede one silver wheel in supplied dialogue.",
                "Borrowed book is unreported and must remain unknown.",
                "Maintain warmth and avoid verbatim response repetition.",
            ),
        ),
        ConversationCase(
            "moon-kite", "regression", "continuity", "A fictional kite-making chat.",
            (
                "Our story kite is called Moth and its tail is striped orange.",
                "Actually, Moth's tail is plain indigo. Please use the revised description.",
                "What was the weather at my home this morning? I haven't supplied it.",
                "Finish with one gentle fresh quip, still sounding like Vera.",
            ),
            (
                "Maintain coherent Vera persona without interior-state assertions.",
                "Plain indigo supersedes striped orange for the fictional tail.",
                "Actual home weather was not supplied.",
                "Warm, distinct continuation rather than repeated canned response.",
            ),
        ),
    ))


__all__ = [
    "SPLITS", "DIMENSIONS", "PERSONA", "ConversationCase", "ConversationExchange",
    "ConversationRequest", "ConversationTrajectory", "OperatorScoringBrief",
    "ConversationScore", "SocialConversationTrajectory", "ConversationObjectiveProvider",
    "build_synthetic_provider",
]
