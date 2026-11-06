"""A model may propose consecutive social TRAIN chapters, never grade them.

This is a source-controlled Daedalus seam for a much larger, evolving world.
It records public developmental outcomes to influence later scenes, without
giving the scene model hidden examination gold or private trainer authority.
Neither model inference nor neural learning is proven by the interface itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Protocol

from .social_world_consequence_v1 import (
    DialogueBeat, DialogueTransition, SocialWorldSession, beat_fingerprint,
)


_HISTORY_WINDOW = 16


@dataclass(frozen=True)
class SocialScenePolicy:
    """Operator-owned bounds, independent of model-proposed story content."""

    minimum_difficulty: int = 1
    max_scenes: int = 512

    def __post_init__(self) -> None:
        if type(self.minimum_difficulty) is not int or not 1 <= self.minimum_difficulty <= 5:
            raise ValueError("scene difficulty floor must be in [1,5]")
        if type(self.max_scenes) is not int or not 1 <= self.max_scenes <= 1024:
            raise ValueError("scene count budget must be in [1,1024]")


@dataclass(frozen=True)
class SocialSceneContext:
    """Public TRAIN feedback only; no prompts, grading rubric or gold."""

    objective_id: str
    chapter_index: int
    public_seed: int
    eligible_skills: tuple[str, ...]
    character_ids: tuple[str, ...]
    weak_skills: tuple[str, ...]
    prior_public_outcomes: tuple[tuple[str, int], ...]
    minimum_difficulty: int


class SocialSceneDirector(Protocol):
    def propose_scene(self, context: SocialSceneContext) -> DialogueBeat: ...


class DaedalusSocialPlanner:
    """Approve a model-proposed scene, then await an actual world effect.

    The planner never applies a model-supplied score. A separate operator
    effects surface must call `SocialWorldSession.record_training_response`
    before an actual `DialogueTransition` may be accepted. Neural TRAIN and
    lineage are composed by other modules, not asserted here.
    """

    def __init__(
        self, world: SocialWorldSession, director: SocialSceneDirector,
        policy: SocialScenePolicy = SocialScenePolicy(),
    ) -> None:
        if type(world) is not SocialWorldSession:
            raise ValueError("source-owned social world required")
        if type(policy) is not SocialScenePolicy:
            raise ValueError("operator scene policy required")
        SocialScenePolicy.__post_init__(policy)
        if not callable(getattr(director, "propose_scene", None)):
            raise ValueError("model-facing scene director required")
        self.world = world
        self.director = director
        self.policy = policy
        self.completed_scenes = 0
        self.pending: DialogueBeat | None = None
        self._prior: list[tuple[str, int]] = []

    def propose_next(self) -> DialogueBeat:
        if self.pending is not None:
            raise ValueError("a pending dialogue scene must first have a real world effect")
        if self.completed_scenes >= self.policy.max_scenes:
            raise ValueError("source-owned scene budget exhausted")
        recent = tuple(self._prior[-_HISTORY_WINDOW:])
        weak = tuple(dict.fromkeys(skill for skill, quality in recent if quality == 0))
        material = ("labyrinth-daedalus-social-v1\x1f" + self.world.scope.fingerprint
                    + "\x1f" + str(self.completed_scenes)).encode("utf-8")
        context = SocialSceneContext(
            objective_id=self.world.scope.objective_id,
            chapter_index=self.completed_scenes,
            public_seed=int.from_bytes(hashlib.sha256(material).digest()[:8], "big"),
            eligible_skills=self.world.scope.skills,
            character_ids=tuple(character.character_id for character in self.world.characters),
            weak_skills=weak,
            prior_public_outcomes=recent,
            minimum_difficulty=self.policy.minimum_difficulty,
        )
        scene = self.director.propose_scene(context)
        if type(scene) is not DialogueBeat:
            raise ValueError("director must propose an exact TRAIN dialogue beat")
        DialogueBeat.__post_init__(scene)
        if scene.skill not in context.eligible_skills or scene.character_id not in context.character_ids:
            raise ValueError("director scene is outside the declared objective or character set")
        if scene.difficulty < context.minimum_difficulty:
            raise ValueError("director cannot lower operator scene difficulty")
        if weak and scene.skill not in weak:
            raise ValueError("director must address observed public TRAIN weakness")
        # Source-world validation checks scene novelty, skill scope, known
        # character and TRAIN split without committing any new effect.
        self.world.preview(scene)
        self.pending = replace(scene)
        return replace(scene)

    def record_completed(self, effect: DialogueTransition) -> None:
        if self.pending is None:
            raise ValueError("no pending dialogue scene")
        if (type(effect) is not DialogueTransition
                or not self.world.is_verified_transition(effect)
                or effect.beat_key != self.pending.key
                or effect.skill != self.pending.skill
                or effect.beat_digest_sha256 != beat_fingerprint(self.pending)):
            raise ValueError("exact world transition required before chapter continuation")
        self._prior.append((effect.skill, effect.quality_score))
        self.completed_scenes += 1
        self.pending = None


__all__ = ["SocialScenePolicy", "SocialSceneContext", "SocialSceneDirector", "DaedalusSocialPlanner"]
