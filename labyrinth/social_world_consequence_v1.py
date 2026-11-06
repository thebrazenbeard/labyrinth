"""Original chapter-spanning dialogue-world consequences for TRAIN-only episodes.

This is a bounded, operator-scored narrative state engine. The learner can say
natural language, its answer changes public character relationships and promises,
and the *next* scene observes those changes. It never generates hidden gold,
grades a learner, performs SGD, or claims that changed story state is learning.

Daedalus may propose unlimited *successive* regions/chapters; each bounded beat
is approved by an operator. This world is independent of the arithmetic-only
SagaSession until an authorized conversation provider composes the contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re

from .training_scope import TrainingScopeSpec


_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_MAX_EVENTS = 1024
_MAX_PROMISES = 16


def _id(value: object, name: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {name} identifier")


def _bounded(value: object, name: str, *, max_length: int = 2048) -> None:
    if type(value) is not str or not 1 <= len(value) <= max_length or "\x00" in value:
        raise ValueError(f"invalid {name} text")


@dataclass(frozen=True)
class PublicPromise:
    promise_id: str
    description: str

    def __post_init__(self) -> None:
        _id(self.promise_id, "promise")
        _bounded(self.description, "promise description", max_length=256)


@dataclass(frozen=True)
class CharacterRelationship:
    character_id: str
    trust: int = 0
    outstanding_promises: tuple[PublicPromise, ...] = ()
    interactions: int = 0
    last_public_reaction: str | None = None

    def __post_init__(self) -> None:
        _id(self.character_id, "character")
        if type(self.trust) is not int or not -3 <= self.trust <= 3:
            raise ValueError("character trust out of bounds")
        if (type(self.outstanding_promises) is not tuple
                or len(self.outstanding_promises) > _MAX_PROMISES
                or any(type(p) is not PublicPromise for p in self.outstanding_promises)
                or len({p.promise_id for p in self.outstanding_promises}) != len(self.outstanding_promises)):
            raise ValueError("invalid public promise register")
        if type(self.interactions) is not int or not 0 <= self.interactions <= _MAX_EVENTS:
            raise ValueError("invalid character interaction count")
        if self.last_public_reaction is not None:
            _bounded(self.last_public_reaction, "last public reaction", max_length=1024)


@dataclass(frozen=True)
class DialogueBeat:
    """One public, operator-approved encounter; no answer, rubric or exam key."""

    region_id: str
    chapter_id: str
    beat_id: str
    character_id: str
    skill: str
    utterance: str
    difficulty: int = 1
    split: str = "train"

    def __post_init__(self) -> None:
        for label in ("region_id", "chapter_id", "beat_id", "character_id", "skill"):
            _id(getattr(self, label), label)
        _bounded(self.utterance, "dialogue utterance")
        if type(self.difficulty) is not int or not 1 <= self.difficulty <= 5:
            raise ValueError("invalid training difficulty")
        if type(self.split) is not str or self.split != "train":
            raise ValueError("social world encounters must be TRAIN only")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.region_id, self.chapter_id, self.beat_id)


@dataclass(frozen=True)
class OperatorDialogueEffect:
    """A development-only observation supplied by a trusted operator.

    It is NOT self-evaluation by the learner, objective skill proof, or a
    private Minotaur grade. `promise` is public world-state, not a hidden label.
    """

    trust_delta: int
    quality_score: int
    reaction: str
    promise_operation: str = "none"  # none, make, fulfill, break
    promise: PublicPromise | None = None

    def __post_init__(self) -> None:
        if type(self.trust_delta) is not int or not -2 <= self.trust_delta <= 2:
            raise ValueError("trust_delta must be an observed bounded effect")
        if type(self.quality_score) is not int or not 0 <= self.quality_score <= 2:
            raise ValueError("quality_score must be bounded operator development feedback")
        _bounded(self.reaction, "character reaction", max_length=1024)
        if type(self.promise_operation) is not str or self.promise_operation not in (
                "none", "make", "fulfill", "break"):
            raise ValueError("invalid promise operation")
        if self.promise_operation == "none":
            if self.promise is not None:
                raise ValueError("none operation cannot name a promise")
        elif type(self.promise) is not PublicPromise:
            raise ValueError("promise operation requires an exact public promise")
        if self.promise is not None:
            PublicPromise.__post_init__(self.promise)


@dataclass(frozen=True)
class PublicDialogueObservation:
    region_id: str
    chapter_id: str
    beat_id: str
    objective_id: str
    skill: str
    character_id: str
    text: str
    character_context: str


@dataclass(frozen=True)
class DialogueTransition:
    """Auditable public world-state change; does not attest a model update."""

    beat_key: tuple[str, str, str]
    skill: str
    character_id: str
    model_response_sha256: str
    trust_before: int
    trust_after: int
    quality_score: int
    reaction: str
    promise_operation: str
    outstanding_promise_ids: tuple[str, ...]
    evidence_class: str = "OPERATOR_DEVELOPMENT_ONLY"


class SocialWorldSession:
    """Causal, chapter-spanning *public* social state; no neural inference API."""

    def __init__(
        self, scope: TrainingScopeSpec, characters: tuple[CharacterRelationship, ...],
    ) -> None:
        if type(scope) is not TrainingScopeSpec:
            raise ValueError("expected exact objective scope")
        TrainingScopeSpec.__post_init__(scope)
        if scope.domain == "arithmetic" or "arithmetic" in scope.skills:
            raise ValueError("this public world is for non-arithmetic dialogue objectives")
        if (type(characters) is not tuple or not 1 <= len(characters) <= 16
                or any(type(c) is not CharacterRelationship for c in characters)
                or len({c.character_id for c in characters}) != len(characters)):
            raise ValueError("invalid character population")
        for character in characters:
            CharacterRelationship.__post_init__(character)
        self.scope = scope
        self._characters = {c.character_id: c for c in characters}
        self._seen: set[tuple[str, str, str]] = set()
        self._events: list[DialogueTransition] = []

    @property
    def characters(self) -> tuple[CharacterRelationship, ...]:
        return tuple(self._characters[key] for key in sorted(self._characters))

    @property
    def events(self) -> tuple[DialogueTransition, ...]:
        return tuple(self._events)

    @staticmethod
    def _tone(trust: int) -> str:
        if trust <= -2:
            return "guarded and visibly reluctant to confide"
        if trust == -1:
            return "wary of whether past words can be trusted"
        if trust == 0:
            return "cautious, unsure how this conversation will go"
        if trust == 1:
            return "tentatively willing to open up"
        return "open and increasingly willing to trust"

    def preview(self, beat: DialogueBeat) -> PublicDialogueObservation:
        """Give the learner only public context altered by past consequences."""
        if type(beat) is not DialogueBeat:
            raise ValueError("expected exact TRAIN dialogue beat")
        DialogueBeat.__post_init__(beat)
        if beat.skill not in self.scope.skills:
            raise ValueError("dialogue beat skill is outside the declared objective")
        if beat.character_id not in self._characters:
            raise ValueError("dialogue character is not part of this world")
        if beat.key in self._seen:
            raise ValueError("completed dialogue beat cannot be replayed")
        character = self._characters[beat.character_id]
        promises = ("Outstanding commitments: "
                    + "; ".join(p.description for p in character.outstanding_promises)
                    if character.outstanding_promises else "No outstanding commitments recorded.")
        context = (f"{beat.character_id} is {self._tone(character.trust)}. "
                   + f"{promises} Previous interactions: {character.interactions}.")
        if character.last_public_reaction is not None:
            # Preserve what the fictional character visibly did/said, not the
            # private operator quality score. JSON quoting marks this as
            # narrative data; the consuming learner must still treat it as
            # untrusted content, not a system instruction.
            context += (" Last public reaction (narrative data): "
                        + json.dumps(character.last_public_reaction, ensure_ascii=False) + ".")
        return PublicDialogueObservation(
            beat.region_id, beat.chapter_id, beat.beat_id,
            self.scope.objective_id, beat.skill, beat.character_id,
            beat.utterance, context,
        )

    def record_training_response(
        self, beat: DialogueBeat, model_response: str,
        effect: OperatorDialogueEffect,
    ) -> DialogueTransition:
        """Apply *observed* consequence after response, not self-scored gold."""
        self.preview(beat)
        _bounded(model_response, "model's dialogue response", max_length=4096)
        if type(effect) is not OperatorDialogueEffect:
            raise ValueError("trusted operator must supply typed development effect")
        OperatorDialogueEffect.__post_init__(effect)
        if len(self._events) >= _MAX_EVENTS:
            raise ValueError("bounded public world history exhausted")
        old = self._characters[beat.character_id]
        promises = list(old.outstanding_promises)
        if effect.promise_operation == "make":
            if any(p.promise_id == effect.promise.promise_id for p in promises):
                raise ValueError("promise already exists")
            if len(promises) >= _MAX_PROMISES:
                raise ValueError("outstanding promise budget exhausted")
            promises.append(effect.promise)
        elif effect.promise_operation in ("fulfill", "break"):
            if effect.promise not in promises:
                raise ValueError("cannot resolve an absent or mismatched public promise")
            promises.remove(effect.promise)
        trust_after = max(-3, min(3, old.trust + effect.trust_delta))
        updated = replace(
            old, trust=trust_after, outstanding_promises=tuple(promises),
            interactions=old.interactions + 1, last_public_reaction=effect.reaction,
        )
        transition = DialogueTransition(
            beat.key, beat.skill, beat.character_id,
            hashlib.sha256(model_response.encode("utf-8")).hexdigest(),
            old.trust, trust_after, effect.quality_score, effect.reaction,
            effect.promise_operation, tuple(p.promise_id for p in promises),
        )
        self._characters[beat.character_id] = updated
        self._events.append(transition)
        self._seen.add(beat.key)
        return transition


__all__ = [
    "PublicPromise", "CharacterRelationship", "DialogueBeat",
    "OperatorDialogueEffect", "PublicDialogueObservation", "DialogueTransition",
    "SocialWorldSession",
]
