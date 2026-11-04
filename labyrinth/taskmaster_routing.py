"""Opt-in training-room routing informed by bounded Minotaur proposals.

This generator concretely consumes :func:`plan_training_room` suggestions:
``difficulty`` selects an arithmetic operand magnitude and ``tactic`` selects
one of three distinct prompt structures. Neither a model nor its output may
provide an answer key, a training objective, evaluation cases, or graduation.

The callable adapter must be **operator-trusted Python**: it runs with the
caller process's privileges. Only its *returned suggestion data* is untrusted
and schema-validated by ``plan_training_room``; this is not a code sandbox.

Only the deliberately narrow arithmetic provider is currently qualified for
this routing path. Other skills fail closed until explicitly implemented and
tested. This training-side materialization is not a neural Minotaur model and
does not claim measured cognitive difficulty or general reasoning transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

from .journey_challenges import TraineeRoomView
from .maze import Challenge
from .minotaur_model import OperatorTaskmasterAdapter, RoomContext, RoomPlan, plan_training_room
from .regenerative_maze import MazePlan, MazeRoom
from .training_scope import TrainingScopeSpec


_ARITHMETIC_BOUNDS = {
    1: (1, 9),
    2: (10, 99),
    3: (100, 999),
    4: (1_000, 9_999),
    5: (10_000, 99_999),
}


@dataclass(frozen=True)
class RoutedTrainingRoom:
    """Trainer-owned result; ``challenge.answer`` is evaluator-only gold.

    Only ``trainee_view`` is safe for a learner's inference prompt. Other
    fields are development metadata and a private training supervision key.
    """

    room_id: str
    skill: str
    suggestion: RoomPlan
    challenge: Challenge

    @property
    def trainee_view(self) -> TraineeRoomView:
        return TraineeRoomView(
            room_id=self.room_id,
            skill=self.skill,
            challenge_id=self.challenge.id,
            prompt=self.challenge.prompt,
            prompt_sha256=hashlib.sha256(self.challenge.prompt.encode("utf-8")).hexdigest(),
        )


def _arithmetic_challenge(room: MazeRoom, scope: TrainingScopeSpec, choice: RoomPlan) -> Challenge:
    """Source-controlled arithmetic gold, never fabricated by taskmaster."""
    digest = hashlib.sha256(
        ("taskmaster-route-v1\x1f" + scope.fingerprint + "\x1f" + room.room_id
         + "\x1f" + str(room.case_seed) + "\x1f" + str(choice.difficulty)
         + "\x1f" + choice.tactic).encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    low, high = _ARITHMETIC_BOUNDS[choice.difficulty]
    a, b = rng.randint(low, high), rng.randint(low, high)
    operation = rng.choice(("+", "-"))
    arithmetic = a + b if operation == "+" else a - b
    instruction = "Respond with exactly one line: ANSWER: <integer>."

    if choice.tactic == "practice":
        prompt = f"Calculate {a} {operation} {b}.\n{instruction}"
        answer = arithmetic
    elif choice.tactic == "contrast":
        # This structurally differs from practice: compute both expressions
        # and compare them. The answer is not exposed in either expression.
        prompt = (
            f"Expression A is {a} + {b}. Expression B is {a} - {b}. "
            f"Calculate A minus B.\n{instruction}"
        )
        answer = 2 * b
    elif choice.tactic == "transfer":
        if operation == "+":
            prompt = (
                f"A workshop starts with {a} components and receives {b} more. "
                f"How many components does it now have?\n{instruction}"
            )
        else:
            prompt = (
                f"A workshop starts with {a} components and uses {b}. "
                f"How many components remain, allowing a negative balance?\n{instruction}"
            )
        answer = arithmetic
    else:  # The model-planning seam must already validate the tactic.
        raise ValueError("unsupported arithmetic tactic")

    return Challenge(
        # A model-selected tactic/difficulty changes case identity as well as
        # content; the room identity alone is not a content commitment.
        id=f"train-taskmaster-{room.room_id}-{digest.hex()[:12]}",
        kind="arithmetic" if choice.tactic != "contrast" else "arithmetic_contrast",
        tier=1,
        split="train",
        prompt=prompt,
        answer=str(answer),
        metadata=(
            ("generator", "taskmaster-route-v1"),
            ("tactic", choice.tactic),
            ("difficulty", str(choice.difficulty)),
            ("planner_source", choice.source),
        ),
    )


def route_training_rooms(
    scope: TrainingScopeSpec,
    plan: MazePlan,
    *,
    taskmaster: OperatorTaskmasterAdapter | None = None,
) -> tuple[RoutedTrainingRoom, ...]:
    """Materialize train-only arithmetic rooms, consuming taskmaster tactics.

    The scope is immutable and authoritative. The optional planner receives
    only predeclared metadata and returns ``difficulty``/``tactic`` proposals.
    This is not connected to private examination or the signed graduation gate.
    The resulting Challenge contains training gold and **must never** be sent
    wholesale to a trainee; use its ``trainee_view`` instead.
    """
    if type(scope) is not TrainingScopeSpec or type(plan) is not MazePlan:
        raise ValueError("expected exact TrainingScopeSpec and MazePlan")
    TrainingScopeSpec.__post_init__(scope)
    MazePlan.__post_init__(plan)
    if (plan.objective_id != scope.objective_id
            or plan.scope_fingerprint != scope.fingerprint
            or not 0 <= plan.cycle_index < scope.max_cycles
            or len(plan.rooms) != scope.total_chambers
            or set(room.skill for room in plan.rooms) != set(scope.skills)):
        raise ValueError("maze does not match declared objective and authorized training budget")
    if any(room.skill != "arithmetic" for room in plan.rooms):
        raise ValueError("taskmaster routing currently supports only arithmetic train rooms")

    routed: list[RoutedTrainingRoom] = []
    prompt_hashes: set[str] = set()
    for room in plan.rooms:
        context = RoomContext(
            objective_id=scope.objective_id,
            domain=scope.domain,
            skill=room.skill,
            room_id=room.room_id,
            cycle_index=plan.cycle_index,
            case_seed=room.case_seed,
            specificity=scope.specificity,
            scale=scope.scale,
        )
        suggestion = plan_training_room(context, taskmaster=taskmaster)
        challenge = _arithmetic_challenge(room, scope, suggestion)
        commitment = hashlib.sha256(challenge.prompt.encode("utf-8")).hexdigest()
        if commitment in prompt_hashes:
            raise ValueError("duplicate routed training prompt")
        prompt_hashes.add(commitment)
        routed.append(RoutedTrainingRoom(room.room_id, room.skill, suggestion, challenge))
    return tuple(routed)
