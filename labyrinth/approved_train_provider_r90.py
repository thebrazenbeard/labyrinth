"""Explicit, source-owned TRAIN provider bindings for declared objectives.

An ObjectiveContract can describe arbitrary learning targets, but **cannot**
install Python callbacks, define grading policy, read private exams, or grant
graduation authority. This allowlist binds only two *implemented* source-side
generator/rater pairs. Adding a domain requires a reviewed source change and
corresponding real-content tests, not a string in a user-provided contract.

Only ``RoutedObjectiveTrainRoom.trainee_view`` may reach learner inference.
``prepared`` contains private TRAIN supervision and stays on the trainer side.
The local practice score is **not** an independently heldout assessment and is
not trusted training/host provenance or an authorization token.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from types import MappingProxyType
from typing import Callable

from .evaluation import Score, score_response
from .journey import JourneyState, TRAINING, _validate_state as validate_journey
from .journey_challenges import PreparedTrainingRoom, TraineeRoomView
from .maze import Challenge
from .minotaur_model import RoomPlan
from .objective_contract import ObjectiveContract, prepare_objective_program_room
from .regenerative_maze import MazePlan, MazeRoom, generate_maze_plan
from .semantic_chambers import semantic_focus_profile
from .taskmaster_routing import _arithmetic_challenge
from .training_scope import TrainingScopeSpec


@dataclass(frozen=True)
class TrainProviderCapabilities:
    """Descriptive discovery only; the declaration confers no privileges."""

    provider_id: str
    skill: str
    allowed_domains: tuple[str, ...]
    rubric_id: str
    focus_tags: tuple[str, ...]
    has_generator: bool = True
    has_local_rater: bool = True
    split: str = "train"

    @property
    def can_access_private_exam(self) -> bool:
        return False

    @property
    def can_authorize_graduation(self) -> bool:
        return False


# Typed, trainer-side callbacks. These are source-owned bindings, not a
# registrable extension API and not a sandbox for user-supplied Python.
_Generator = Callable[
    [ObjectiveContract, MazePlan, MazeRoom, JourneyState | None], PreparedTrainingRoom
]
_LocalRater = Callable[[Challenge, str], Score]

# Source-bound retry space for arithmetic-only TRAIN generation. This is NOT
# an authorization nonce and never permits re-opening a disclosed prompt.
_MAX_OBJECTIVE_REDRAW_ATTEMPTS = 512


@dataclass(frozen=True)
class _TrustedProvider:
    capabilities: TrainProviderCapabilities
    generator: _Generator
    local_rater: _LocalRater


def _program_room(contract: ObjectiveContract, plan: MazePlan, room: MazeRoom,
                  journey: JourneyState | None) -> PreparedTrainingRoom:
    return prepare_objective_program_room(contract, plan, room, journey=journey)


def _arithmetic_room(contract: ObjectiveContract, plan: MazePlan, room: MazeRoom,
                     journey: JourneyState | None) -> PreparedTrainingRoom:
    """Wrap source-owned arithmetic generation with a fixed bounded tactic.

    Specificity selects operand magnitude 1..5. The declared single focus
    corresponds to actual integer addition/subtraction exercises, not just a
    tag appended to arbitrary content. An untrusted taskmaster never selects
    answers, generators, or examination cases on this path.
    """
    target = contract.targets[0]
    choice = RoomPlan(room_id=room.room_id, difficulty=contract.scope.specificity,
                      tactic="practice", source="approved_static_objective",
                      fallback_reason=None)
    challenge = _arithmetic_challenge(room, contract.scope, choice)
    if challenge.kind != "arithmetic" or challenge.split != "train":
        raise ValueError("arithmetic generator produced an unsupported/non-TRAIN challenge")
    challenge = replace(challenge, metadata=challenge.metadata + (
        ("objective_contract_sha256", contract.fingerprint),
        ("objective_focus_tag", target.focus_tags[0]),
        ("objective_training_rubric_marker", target.rubric_id),
    ))
    return PreparedTrainingRoom(room.room_id, room.skill, challenge)


_PROVIDERS = MappingProxyType({
    "arithmetic_static_train_v1": _TrustedProvider(
        TrainProviderCapabilities(
            provider_id="arithmetic_static_train_v1", skill="arithmetic",
            allowed_domains=("arithmetic", "reasoning"),
            rubric_id="arithmetic_exact_answer_train_v1",
            focus_tags=("integer_add_subtract",),
        ),
        generator=_arithmetic_room,
        local_rater=score_response,
    ),
    "semantic_program_train_v1": _TrustedProvider(
        TrainProviderCapabilities(
            provider_id="semantic_program_train_v1", skill="program_reasoning",
            allowed_domains=("program_reasoning", "reasoning"),
            rubric_id="program_structure_v1",
            focus_tags=tuple(semantic_focus_profile(level) for level in range(1, 6)),
        ),
        generator=_program_room,
        local_rater=score_response,
    ),
})


def approved_train_provider_capabilities() -> tuple[TrainProviderCapabilities, ...]:
    """Return a fixed source-vetted manifest, never caller-injected callbacks."""
    return tuple(binding.capabilities for binding in _PROVIDERS.values())


@dataclass(frozen=True)
class RoutedObjectiveTrainRoom:
    """Trainer-private result with a restricted prompt-only learner projection."""

    provider_id: str
    objective_fingerprint: str
    prepared: PreparedTrainingRoom
    redraw_attempt: int = 0

    @property
    def trainee_view(self) -> TraineeRoomView:
        return self.prepared.trainee_view


@dataclass(frozen=True)
class LocalTrainScore:
    """A diagnostic TRAIN result, never signed exam or graduation evidence."""

    provider_id: str
    room_id: str
    challenge_id: str
    prompt_sha256: str
    correct: bool
    points: int
    reason: str
    parsed_answer: str | None

    @property
    def evidence_split(self) -> str:
        return "train"

    @property
    def independent_exam_evidence(self) -> bool:
        return False

    @property
    def graduation_authorized(self) -> bool:
        return False


def _resolve_binding(contract: ObjectiveContract, provider_id: str | None) -> _TrustedProvider:
    if type(provider_id) not in (str, type(None)):
        raise ValueError("provider_id must be an allowlisted source provider name")
    if len(contract.targets) != 1 or len(contract.scope.skills) != 1:
        raise ValueError("objective routing requires one explicitly supported skill target")
    target = contract.targets[0]
    matches = tuple(
        binding for binding in _PROVIDERS.values()
        if binding.capabilities.skill == target.skill
    )
    if len(matches) != 1:
        raise ValueError("unsupported objective skill: no executable TRAIN provider")
    binding = matches[0]
    manifest = binding.capabilities
    if provider_id is not None and provider_id != manifest.provider_id:
        raise ValueError("caller cannot override approved TRAIN provider binding")
    if (contract.scope.domain not in manifest.allowed_domains
            or target.rubric_id != manifest.rubric_id
            or len(target.focus_tags) != 1
            or target.focus_tags[0] not in manifest.focus_tags):
        raise ValueError("unapproved objective domain, focus, or TRAIN rubric")
    return binding


def _validate_room_binding(contract: ObjectiveContract, plan: MazePlan, room: MazeRoom,
                           journey: JourneyState | None) -> None:
    if type(plan) is not MazePlan or type(room) is not MazeRoom:
        raise ValueError("expected exact validated TRAIN maze and room")
    MazePlan.__post_init__(plan)
    MazeRoom.__post_init__(room)
    scope: TrainingScopeSpec = contract.scope
    if (plan.objective_id != scope.objective_id
            or plan.scope_fingerprint != scope.fingerprint
            or not 0 <= plan.cycle_index < scope.max_cycles
            or len(plan.rooms) != scope.total_chambers
            or tuple(sorted(set(x.skill for x in plan.rooms))) != scope.skills
            or room not in plan.rooms
            or room.skill != scope.skills[0]
            or any(candidate.purpose != "train" for candidate in plan.rooms)):
        raise ValueError("TRAIN maze and room do not match the declared learning scope")
    if journey is None:
        if plan.cycle_index > 0:
            raise ValueError("regenerated objective training requires validated JourneyState")
        if plan != generate_maze_plan(scope, 0):
            raise ValueError("initial TRAIN maze must match its deterministic source plan")
    else:
        if type(journey) is not JourneyState:
            raise ValueError("expected validated JourneyState")
        validate_journey(journey)
        if journey.phase != TRAINING or journey.scope != scope or journey.maze != plan:
            raise ValueError("objective room requires the current training journey")


def route_objective_train_room(
    contract: ObjectiveContract, plan: MazePlan, room: MazeRoom,
    *, provider_id: str | None = None, journey: JourneyState | None = None,
    redraw_attempt: int = 0,
) -> RoutedObjectiveTrainRoom:
    """Bind approved TRAIN generation, never infer arbitrary skill execution.

    This is not an authorization service. No private exam or signed result can
    be requested. A returned trainer-private answer must not reach inference.
    """
    if type(contract) is not ObjectiveContract:
        raise ValueError("expected exact immutable ObjectiveContract")
    ObjectiveContract.__post_init__(contract)
    binding = _resolve_binding(contract, provider_id)
    _validate_room_binding(contract, plan, room, journey)
    if (type(redraw_attempt) is not int or not
            0 <= redraw_attempt <= _MAX_OBJECTIVE_REDRAW_ATTEMPTS):
        raise ValueError("objective TRAIN redraw attempt must be a bounded integer")
    if redraw_attempt and binding.capabilities.skill != "arithmetic":
        raise ValueError("only arithmetic TRAIN supports source-controlled redraw")
    generated_room = room
    if redraw_attempt:
        # Derive a bounded alternate generator seed from the exact approved
        # objective/room/attempt. The *admitted* MazeRoom stays unchanged; no
        # caller-supplied generator, answer or prompt is accepted. Attempt 0
        # preserves all previously qualified case and prompt bytes.
        digest = hashlib.sha256(
            ("objective-train-arithmetic-redraw-v1\x1f"
             + contract.fingerprint + "\x1f" + room.room_id + "\x1f"
             + str(room.case_seed) + "\x1f" + str(redraw_attempt))
            .encode("utf-8")
        ).digest()
        alternate_seed = int.from_bytes(digest[:8], "big") % (1 << 31)
        generated_room = replace(room, case_seed=alternate_seed)
    prepared = binding.generator(contract, plan, generated_room, journey)
    if (type(prepared) is not PreparedTrainingRoom
            or prepared.room_id != room.room_id or prepared.skill != room.skill
            or type(prepared.challenge) is not Challenge
            or prepared.challenge.split != "train"
            or not prepared.challenge.id.startswith("train-")
            or not prepared.challenge.prompt or not prepared.challenge.answer):
        raise ValueError("approved provider returned invalid or non-TRAIN content")
    metadata = dict(prepared.challenge.metadata)
    if (metadata.get("objective_contract_sha256") != contract.fingerprint
            or metadata.get("objective_focus_tag") != contract.targets[0].focus_tags[0]
            or metadata.get("objective_training_rubric_marker") != contract.targets[0].rubric_id):
        raise ValueError("approved provider did not honor objective focus and TRAIN rubric")
    prompt_sha256 = hashlib.sha256(prepared.challenge.prompt.encode("utf-8")).hexdigest()
    if journey is not None:
        if (prepared.challenge.id in set(journey.used_training_case_ids)
                | set(journey.sealed_exam_case_ids)
                or prompt_sha256 in set(journey.used_training_prompt_sha256)
                | set(journey.sealed_exam_prompt_sha256)):
            raise ValueError("approved challenge overlaps prior training or development evidence")
    return RoutedObjectiveTrainRoom(binding.capabilities.provider_id, contract.fingerprint,
                                    prepared, redraw_attempt)


def rate_objective_train_response(
    contract: ObjectiveContract, plan: MazePlan, room: MazeRoom, response: str,
    *, provider_id: str | None = None, journey: JourneyState | None = None,
    redraw_attempt: int = 0,
) -> LocalTrainScore:
    """Regenerate source-bound TRAIN gold for a local score, not an exam grant.

    The scorer receives no caller-provided answer key or provider callback;
    it reconstructs the exercise from the source-owned generator each time.
    """
    routed = route_objective_train_room(contract, plan, room,
                                        provider_id=provider_id, journey=journey,
                                        redraw_attempt=redraw_attempt)
    binding = _PROVIDERS[routed.provider_id]
    diagnostic = binding.local_rater(routed.prepared.challenge, response)
    if type(diagnostic) is not Score:
        raise ValueError("approved TRAIN rater returned an invalid diagnostic score")
    return LocalTrainScore(
        provider_id=routed.provider_id,
        room_id=routed.prepared.room_id,
        challenge_id=routed.prepared.challenge.id,
        prompt_sha256=routed.trainee_view.prompt_sha256,
        correct=diagnostic.correct,
        points=diagnostic.points,
        reason=diagnostic.reason,
        parsed_answer=diagnostic.parsed_answer,
    )
