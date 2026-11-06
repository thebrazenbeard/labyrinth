"""Optional TRAIN-only Syllabus curriculum signal bridge.

The Labyrinth retains control of model updates, canonical room order, Ariadne
continuity, case identities, and examination custody. Syllabus is an *adviser*
for operator-led extra practice, not a maze planner, learner, or examiner.

This module deliberately has zero mandatory third-party dependencies. Its
method calls match Syllabus v0.7's Curriculum API, upstream source commit
2d3b9bdad95581ee2945499193f34dc2ef4c1507 (MIT).
"""

from __future__ import annotations

from itertools import islice
from numbers import Integral
from typing import Any

from .regenerative_maze import MazePlan, MazeRoom
from .training_scope import TrainingScopeSpec


class SyllabusBridgeUnavailable(ImportError):
    """The optional Syllabus package or one of its dependencies is absent."""


class SyllabusTrainBridge:
    """Translate known TRAIN-room outcomes into Syllabus v0.7 episode signals.

    A matching room must have been completed by an operator-trusted training
    runner. Boolean `correct` is a *development* score and does not attest any
    parameter update or grant Minotaur graduation. No exam data are accepted.

    Advisory samples are never applied to `MazePlan` or `JourneyState`; the
    operator may use them only to propose additional train-only practice.
    """

    def __init__(self, scope: TrainingScopeSpec, plan: MazePlan, curriculum: Any) -> None:
        if type(scope) is not TrainingScopeSpec or type(plan) is not MazePlan:
            raise ValueError("expected exact training scope and maze plan")
        TrainingScopeSpec.__post_init__(scope)
        MazePlan.__post_init__(plan)
        if (plan.scope_fingerprint != scope.fingerprint
                or plan.objective_id != scope.objective_id
                or plan.cycle_index >= scope.max_cycles
                or any(room.skill not in scope.skills for room in plan.rooms)):
            raise ValueError("maze is not bound to the declared TRAIN scope")
        if any(not callable(getattr(curriculum, name, None)) for name in (
                "update_task_progress", "update_on_episode", "sample")):
            raise ValueError("curriculum lacks Syllabus v0.7 episode API")
        self.scope = scope
        self.plan = plan
        self.curriculum = curriculum
        self._room_index = {room.room_id: room for room in plan.rooms}
        self._seen: set[str] = set()
        self._indeterminate = False

    @property
    def recorded_count(self) -> int:
        return len(self._seen)

    def _ensure_healthy(self) -> None:
        if self._indeterminate:
            raise RuntimeError("Syllabus feedback state indeterminate; discard this bridge")

    def record_completed_train_room(
        self, room: MazeRoom, *, correct: bool, length: int = 1,
    ) -> None:
        """Record one operator-scored *train* episode; omit prompt, gold and model text.

        Syllabus's OnlineLearningProgress requires both `update_task_progress`
        and `update_on_episode` in upstream v0.7: the latter alone does not
        increment per-task success counters. A partial upstream failure poisons
        the bridge to prevent an accidental same-process double update.
        """
        self._ensure_healthy()
        if (type(room) is not MazeRoom or room.purpose != "train"
                or self._room_index.get(room.room_id) != room):
            raise ValueError("room is not an exact canonical TRAIN room in this plan")
        if type(correct) is not bool or type(length) is not int or not 1 <= length <= 100_000:
            raise ValueError("strict boolean correctness and bounded episode length required")
        if room.room_id in self._seen:
            raise ValueError("TRAIN room already recorded in this bridge")

        task = self.scope.skills.index(room.skill)
        score = float(correct)
        # Mark before calling a mutable third-party object. If either callback
        # fails, no replay is allowed because partial changes cannot be undone.
        self._seen.add(room.room_id)
        try:
            self.curriculum.update_task_progress(task, score)
            self.curriculum.update_on_episode(score, length, task, score)
        except Exception:
            self._indeterminate = True
            raise

    def advisory_skill_sample(self, count: int = 1) -> tuple[str, ...]:
        """Decode optional Syllabus task indices into untrusted skill *suggestions*.

        This intentionally has no effect on the canonical maze, weak-skill
        evidence, journey transitions, real model weights or private exams.
        """
        self._ensure_healthy()
        if type(count) is not int or not 1 <= count <= len(self.scope.skills):
            raise ValueError("advisory count exceeds declared TRAIN skill space")
        sampled = self.curriculum.sample(k=count)
        if isinstance(sampled, (str, bytes)):
            raise ValueError("invalid Syllabus task index sequence")
        try:
            # Third-party sample providers may return generators, including
            # infinite ones. Consume at most one more than the requested count.
            values = tuple(islice(iter(sampled), count + 1))
        except TypeError as exc:
            raise ValueError("invalid Syllabus task index sequence") from exc
        if len(values) != count:
            raise ValueError("invalid Syllabus task index count")
        suggested: list[str] = []
        for value in values:
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or not 0 <= int(value) < len(self.scope.skills)):
                raise ValueError("invalid Syllabus task index")
            skill = self.scope.skills[int(value)]
            if skill not in suggested:
                suggested.append(skill)
        return tuple(suggested)


def make_syllabus_train_bridge(
    scope: TrainingScopeSpec, plan: MazePlan,
) -> SyllabusTrainBridge:
    """Create optional Syllabus v0.7 OnlineLearningProgress; do not auto-install.

    A missing Syllabus import never affects standard Labyrinth execution.
    The upstream algorithm's scores are advisory and have no custody status.
    """
    # Validate Labyrinth-owned inputs before importing optional dependencies.
    if type(scope) is not TrainingScopeSpec or type(plan) is not MazePlan:
        raise ValueError("expected exact training scope and maze plan")
    TrainingScopeSpec.__post_init__(scope)
    MazePlan.__post_init__(plan)
    try:
        from syllabus.task_space import DiscreteTaskSpace
        from syllabus.curricula import OnlineLearningProgress
    except (ImportError, ModuleNotFoundError) as exc:
        raise SyllabusBridgeUnavailable(
            "Syllabus-RL 0.7 is optional; install in an isolated compatible environment"
        ) from exc

    task_space = DiscreteTaskSpace(len(scope.skills), tasks=list(scope.skills))
    curriculum = OnlineLearningProgress(
        task_space,
        update_interval=len(plan.rooms),
        normalize_success=False,
    )
    return SyllabusTrainBridge(scope, plan, curriculum)
