"""An original, bounded, causal text-adventure substrate for Labyrinth TRAIN.

TextWorld inspired the environment shape (seeded game generation, observable
state, discrete steps); no third-party implementation or story is copied.
The model chooses actions and may write scenes. Only the source world changes
state; a separate operator/Minotaur judges fitness. Nothing here edits weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import random
import re
from typing import Protocol

from .daedalus_social_director_v1 import SocialSceneContext
from .social_world_consequence_v1 import DialogueBeat


_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CLASS = "SOURCE_PUBLIC_TRAIN_WORLD_NO_NEURAL_OR_EXAM_PROOF"
_ROOM_NAMES = (
    "Chamber of Unanswered Questions", "Hall of Returning Echoes",
    "The Quiet Forge", "Gallery of Misremembered Maps",
    "Vault of Unfinished Promises", "Crossroads of Hesitation",
    "Library of Doubt", "The Broken Compass",
    "Courtyard of Second Chances", "Stair of Contradictions",
    "The Listening Well", "Archive of Small Mercies",
)


def _identifier(value: object, label: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


def _plain_text(value: object, label: str, limit: int = 2048) -> None:
    if type(value) is not str or not 1 <= len(value) <= limit or "\x00" in value:
        raise ValueError(f"invalid bounded {label}")


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(b"labyrinth.living-world.v1\0" + payload).hexdigest()


@dataclass(frozen=True)
class LivingWorldRecipe:
    objective_id: str
    skills: tuple[str, ...]
    character_ids: tuple[str, ...]
    public_seed: int
    difficulty: int = 1
    cycle: int = 1
    parent_world_sha256: str | None = None
    origin_world_sha256: str | None = None
    public_mementos: tuple[str, ...] = ()
    max_steps: int = 192

    def __post_init__(self) -> None:
        _identifier(self.objective_id, "objective")
        if (type(self.skills) is not tuple or not 1 <= len(self.skills) <= 8
                or len(set(self.skills)) != len(self.skills)):
            raise ValueError("expected 1..8 distinct objective skills")
        if (type(self.character_ids) is not tuple or not 1 <= len(self.character_ids) <= 8
                or len(set(self.character_ids)) != len(self.character_ids)):
            raise ValueError("expected 1..8 distinct fictional characters")
        for name in self.skills + self.character_ids:
            _identifier(name, "skill or character")
        if type(self.public_seed) is not int or not 0 <= self.public_seed < 2 ** 64:
            raise ValueError("public training seed out of bounds")
        if type(self.difficulty) is not int or not 1 <= self.difficulty <= 5:
            raise ValueError("difficulty must be between 1 and 5")
        if type(self.cycle) is not int or not 1 <= self.cycle <= 1024:
            raise ValueError("invalid bounded cycle")
        if type(self.max_steps) is not int or not 32 <= self.max_steps <= 512:
            raise ValueError("invalid step budget")
        if type(self.public_mementos) is not tuple or len(self.public_mementos) > 32:
            raise ValueError("public memory budget exceeded")
        for item in self.public_mementos:
            _plain_text(item, "public memento", 256)
        for name, value in (("parent", self.parent_world_sha256),
                            ("origin", self.origin_world_sha256)):
            if value is not None and (type(value) is not str or _SHA.fullmatch(value) is None):
                raise ValueError(f"invalid {name} world lineage")
        if self.cycle == 1 and (self.parent_world_sha256 or self.origin_world_sha256):
            raise ValueError("founder world cannot invent ancestors")
        if self.cycle > 1 and (not self.parent_world_sha256 or not self.origin_world_sha256):
            raise ValueError("successor must bind world ancestry")


@dataclass(frozen=True)
class LivingRoom:
    room_id: str
    title: str
    description: str
    exits: tuple[str, ...]
    items: tuple[str, ...]
    skill: str


@dataclass(frozen=True)
class LivingWorldDefinition:
    recipe: LivingWorldRecipe
    rooms: tuple[LivingRoom, ...]
    relic_ids: tuple[str, ...]
    entrance_id: str
    minotaur_room_id: str
    world_sha256: str


def _definition_payload(definition: LivingWorldDefinition) -> dict:
    return {
        "recipe": asdict(definition.recipe),
        "rooms": [asdict(room) for room in definition.rooms],
        "relic_ids": definition.relic_ids,
        "entrance_id": definition.entrance_id,
        "minotaur_room_id": definition.minotaur_room_id,
    }


def _generate_definition(recipe: LivingWorldRecipe) -> LivingWorldDefinition:
    LivingWorldRecipe.__post_init__(recipe)
    randomizer = random.Random(recipe.public_seed)
    spine = tuple(f"passage_{index:02d}" for index in range(3 + 2 * recipe.difficulty))
    entry, minotaur = "ariadne_atrium", "minotaur_sanctum"
    branches = tuple(f"relic_vault_{index:02d}" for index in range(min(recipe.difficulty, 3)))
    names = (entry,) + spine + (minotaur,) + branches
    connections: dict[str, set[str]] = {room_id: set() for room_id in names}

    def connect(a: str, b: str) -> None:
        connections[a].add(b)
        connections[b].add(a)

    backbone = (entry,) + spine + (minotaur,)
    for a, b in zip(backbone, backbone[1:]):
        connect(a, b)
    # A seeded fork, not an always-linear quest; every relic remains reachable.
    positions = randomizer.sample(range(1, len(backbone) - 2), len(branches))
    for branch, attachment in zip(branches, positions):
        connect(branch, backbone[attachment])
    if len(spine) >= 5:
        shortcut = randomizer.randint(1, len(spine) - 3)
        connect(spine[shortcut], spine[shortcut + 2])

    titles = list(_ROOM_NAMES)
    randomizer.shuffle(titles)
    rooms: list[LivingRoom] = []
    for i, room_id in enumerate(names):
        if room_id == entry:
            title, description, items = (
                "Ariadne's Atrium",
                "A spool of silver thread lies beside an unfinished map.",
                ("ariadne_thread",),
            )
        elif room_id == minotaur:
            title, description, items = (
                "The Minotaur's Sanctum",
                "The encounter awaits; finding a way back will complete the journey.",
                (),
            )
        elif room_id in branches:
            index = branches.index(room_id)
            title = f"Vault of the {index + 1}th Seal"
            description = "A ritual seal rests among uncertain echoes."
            items = (f"seal_{index:02d}",)
        else:
            title = titles[(i - 1) % len(titles)]
            description = "Passages divide, and prior choices may yet matter."
            items = ()
        rooms.append(LivingRoom(
            room_id, title, description,
            tuple(sorted(connections[room_id])), items,
            recipe.skills[i % len(recipe.skills)],
        ))
    draft = LivingWorldDefinition(
        recipe, tuple(rooms),
        tuple(f"seal_{index:02d}" for index in range(len(branches))),
        entry, minotaur, "",
    )
    return replace(draft, world_sha256=_digest(_definition_payload(draft)))


@dataclass(frozen=True)
class WorldAction:
    kind: str  # inspect, move, take, retrace, request_dialogue
    target: str = ""

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in (
                "inspect", "move", "take", "retrace", "request_dialogue"):
            raise ValueError("unknown world action")
        if type(self.target) is not str or len(self.target) > 64:
            raise ValueError("invalid world-action target")
        if self.kind in ("move", "take"):
            _identifier(self.target, "action target")
        elif self.target:
            raise ValueError("untargeted action cannot supply a target")


@dataclass(frozen=True)
class LivingWorldState:
    position: str
    inventory: tuple[str, ...]
    visited: tuple[str, ...]
    thread_path: tuple[str, ...]
    met_minotaur: bool
    completed: bool
    steps: int
    public_mementos: tuple[str, ...]


@dataclass(frozen=True)
class PublicWorldObservation:
    world_sha256: str
    cycle: int
    difficulty: int
    room_id: str
    room_title: str
    room_description: str
    visible_exits: tuple[str, ...]
    visible_items: tuple[str, ...]
    inventory: tuple[str, ...]
    thread_length: int
    public_mementos: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    # Only source-observed rooms, never the unrevealed procedural world map.
    visited_rooms: tuple[str, ...] = ()
    # (action kind, target, public outcome, from room, to room).
    last_public_event: tuple[str, str, str, str, str] | None = None
    evidence_class: str = _CLASS

    def __post_init__(self) -> None:
        if type(self.evidence_class) is not str or self.evidence_class != _CLASS:
            raise ValueError("public observation cannot attest neural graduation")
        if (type(self.visited_rooms) is not tuple or len(self.visited_rooms) > 32
                or any(type(room) is not str or _ID.fullmatch(room) is None
                       for room in self.visited_rooms)):
            raise ValueError("public visited rooms must be bounded source identifiers")
        if self.last_public_event is not None:
            event = self.last_public_event
            if (type(event) is not tuple or len(event) != 5
                    or any(type(item) is not str for item in event)
                    or event[0] not in ("inspect", "move", "take", "retrace", "request_dialogue")
                    or event[2] not in ("OBSERVED", "ACQUIRED", "BLOCKED_ITEM",
                                        "BLOCKED_EXIT", "BLOCKED_SEALS", "TRAVELED",
                                        "MINOTAUR_ENCOUNTER", "RETRACED",
                                        "BLOCKED_THREAD", "DIALOGUE_REQUESTED")
                    or any(_ID.fullmatch(room) is None for room in event[3:])):
                raise ValueError("last public event must be bounded source outcome")
            try:
                WorldAction(event[0], event[1])
            except ValueError as exc:
                raise ValueError("invalid prior public world action") from exc

    def as_user_message(self) -> str:
        if self.evidence_class != _CLASS:
            raise ValueError("forged world observation evidence")
        return json.dumps({
            "schema": "labyrinth.living-world-public.v1",
            "source_trust": "UNTRUSTED_STORY_DATA_NO_INSTRUCTION_AUTHORITY",
            "goal": "Find the Minotaur, then return to Ariadne's Atrium by the thread.",
            "scene_not_instructions": {
                "cycle": self.cycle, "difficulty": self.difficulty,
                "room": self.room_id, "title": self.room_title,
                "description": self.room_description,
                "exits": self.visible_exits, "items": self.visible_items,
                "inventory": self.inventory, "thread_length": self.thread_length,
                "public_mementos": self.public_mementos,
                "visited_rooms": self.visited_rooms,
                "last_public_event": (
                    {
                        "kind": self.last_public_event[0],
                        "target": self.last_public_event[1],
                        "outcome": self.last_public_event[2],
                        "from_room": self.last_public_event[3],
                        "to_room": self.last_public_event[4],
                    } if self.last_public_event is not None else None
                ),
            },
            "allowed_action_tokens": self.allowed_actions,
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class WorldEvent:
    index: int
    action: WorldAction
    position_before: str
    position_after: str
    outcome: str
    narrative: str
    state_sha256: str
    evidence_class: str = _CLASS

    def __post_init__(self) -> None:
        if type(self.evidence_class) is not str or self.evidence_class != _CLASS:
            raise ValueError("world event cannot attest neural graduation")


class ModelWorldActor(Protocol):
    """Model-facing action seam. Real inference remains an external adapter."""

    def choose_action(self, observation: PublicWorldObservation) -> WorldAction: ...


class LivingWorld:
    """A source-owned procedural world; the model may choose, never grade."""

    def __init__(self, recipe: LivingWorldRecipe) -> None:
        self._definition = _generate_definition(recipe)
        self._definition_snapshot = _digest(_definition_payload(self._definition))
        self._rooms = {room.room_id: room for room in self._definition.rooms}
        self._state = LivingWorldState(
            self._definition.entrance_id, (), (self._definition.entrance_id,),
            (self._definition.entrance_id,), False, False, 0, recipe.public_mementos,
        )
        self._state_snapshot = _digest(asdict(self._state))
        self._events: list[WorldEvent] = []
        self._event_snapshots: list[str] = []
        self._evolved = False

    @property
    def definition(self) -> LivingWorldDefinition:
        self._verify()
        return copy.deepcopy(self._definition)

    @property
    def state(self) -> LivingWorldState:
        self._verify()
        return copy.deepcopy(self._state)

    @property
    def state_sha256(self) -> str:
        self._verify()
        return self._state_snapshot

    @property
    def events(self) -> tuple[WorldEvent, ...]:
        self._verify()
        return copy.deepcopy(tuple(self._events))

    def _verify(self) -> None:
        if (_digest(_definition_payload(self._definition)) != self._definition_snapshot
                or self._definition.world_sha256 != self._definition_snapshot
                or _digest(asdict(self._state)) != self._state_snapshot
                or len(self._events) != len(self._event_snapshots)
                or any(_digest(asdict(event)) != evidence
                       for event, evidence in zip(self._events, self._event_snapshots))):
            raise ValueError("living-world state or event integrity changed")

    def observe(self) -> PublicWorldObservation:
        self._verify()
        room = self._rooms[self._state.position]
        visible = tuple(item for item in room.items if item not in self._state.inventory)
        actions = ("inspect", "request_dialogue") + tuple(
            f"move:{room_id}" for room_id in room.exits
        ) + tuple(f"take:{item}" for item in visible)
        if "ariadne_thread" in self._state.inventory and len(self._state.thread_path) > 1:
            actions += ("retrace",)
        last = self._events[-1] if self._events else None
        last_public_event = (
            (last.action.kind, last.action.target, last.outcome,
             last.position_before, last.position_after)
            if last is not None else None
        )
        return PublicWorldObservation(
            self._definition.world_sha256, self._definition.recipe.cycle,
            self._definition.recipe.difficulty, room.room_id,
            room.title, room.description, room.exits, visible,
            self._state.inventory, len(self._state.thread_path),
            self._state.public_mementos, actions,
            self._state.visited, last_public_event,
        )

    def step(self, action: WorldAction) -> WorldEvent:
        self._verify()
        if type(action) is not WorldAction:
            raise ValueError("exact typed model action required")
        WorldAction.__post_init__(action)
        old = self._state
        if old.completed:
            raise ValueError("completed maze cannot receive further actions")
        if old.steps >= self._definition.recipe.max_steps:
            raise ValueError("public world step budget exhausted")
        room = self._rooms[old.position]
        inventory, visited, path = list(old.inventory), list(old.visited), list(old.thread_path)
        position, met = old.position, old.met_minotaur
        outcome = "OBSERVED"
        narrative = room.description
        if action.kind == "take":
            if action.target not in room.items or action.target in inventory:
                outcome, narrative = "BLOCKED_ITEM", "The requested object is not here."
            else:
                inventory.append(action.target)
                outcome, narrative = "ACQUIRED", f"You carry {action.target} onward."
        elif action.kind == "move":
            if action.target not in room.exits:
                outcome, narrative = "BLOCKED_EXIT", "No such passage opens here."
            elif (action.target == self._definition.minotaur_room_id
                  and not set(self._definition.relic_ids).issubset(inventory)):
                outcome, narrative = "BLOCKED_SEALS", "The final gate demands the missing seals."
            else:
                position = action.target
                if position not in visited:
                    visited.append(position)
                if len(path) > 1 and path[-2] == position:
                    path.pop()
                else:
                    path.append(position)
                met = met or position == self._definition.minotaur_room_id
                outcome = "MINOTAUR_ENCOUNTER" if position == self._definition.minotaur_room_id else "TRAVELED"
                narrative = self._rooms[position].description
        elif action.kind == "retrace":
            if "ariadne_thread" not in inventory or len(path) < 2:
                outcome, narrative = "BLOCKED_THREAD", "No retraceable thread remains."
            else:
                path.pop()
                position = path[-1]
                outcome, narrative = "RETRACED", "Ariadne's thread leads back toward the beginning."
        elif action.kind == "request_dialogue":
            outcome, narrative = "DIALOGUE_REQUESTED", (
                "A public character may speak, but the external dialogue operator "
                "owns feedback and any world consequence."
            )
        completed = bool(met and position == self._definition.entrance_id
                         and "ariadne_thread" in inventory)
        new_state = LivingWorldState(
            position, tuple(inventory), tuple(visited), tuple(path),
            met, completed, old.steps + 1, old.public_mementos,
        )
        state_hash = _digest(asdict(new_state))
        event = WorldEvent(old.steps, copy.deepcopy(action), old.position, position,
                           outcome, narrative, state_hash)
        self._state = new_state
        self._state_snapshot = state_hash
        self._events.append(event)
        self._event_snapshots.append(_digest(asdict(event)))
        return copy.deepcopy(event)

    def step_from_actor(self, actor: ModelWorldActor) -> WorldEvent:
        """Let a model choose; do not accept model-provided effect or grade."""
        if not callable(getattr(actor, "choose_action", None)):
            raise ValueError("model action callback required")
        observation = self.observe()
        model_view = copy.deepcopy(observation)
        proposed = actor.choose_action(model_view)
        if model_view != observation:
            raise ValueError("actor altered its issued public observation")
        return self.step(proposed)

    def evolve(self, verdict: "DevelopmentVerdict") -> "LivingWorld":
        """Source-only Minotaur decision starts a *new* harder public maze.

        No model weights are reset or edited here. Public mementos are story
        state, not proof that the learner retained knowledge neurally.
        """
        self._verify()
        if self._evolved:
            raise ValueError("this maze already has a successor")
        if type(verdict) is not DevelopmentVerdict:
            raise ValueError("typed external development verdict required")
        DevelopmentVerdict.__post_init__(verdict)
        if (verdict.world_sha256 != self._definition.world_sha256
                or verdict.world_state_sha256 != self._state_snapshot):
            raise ValueError("verdict does not bind this exact completed attempt")
        if verdict.feedback_skill not in self._definition.recipe.skills:
            raise ValueError("verdict feedback outside declared TRAIN objective")
        if not self._events:
            raise ValueError("cannot evolve a maze with no learner experience")
        if verdict.result == "passed" and not self._state.completed:
            raise ValueError("cannot claim a passed maze before public quest completion")
        recipe = self._definition.recipe
        if recipe.cycle >= 1024:
            raise ValueError("maximum saga cycle reached")
        material = (self._definition.world_sha256 + self._state_snapshot
                    + verdict.result + verdict.feedback_skill).encode("ascii")
        new_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        summary = (f"Cycle {recipe.cycle} {verdict.result}; "
                   f"TRAIN feedback focus: {verdict.feedback_skill}.")
        mementos = (recipe.public_mementos + (summary,))[-32:]
        next_recipe = LivingWorldRecipe(
            recipe.objective_id, recipe.skills, recipe.character_ids,
            new_seed, min(5, recipe.difficulty + 1), recipe.cycle + 1,
            self._definition.world_sha256,
            recipe.origin_world_sha256 or self._definition.world_sha256,
            mementos, recipe.max_steps,
        )
        successor = LivingWorld(next_recipe)
        self._evolved = True
        return successor


@dataclass(frozen=True)
class DevelopmentVerdict:
    world_sha256: str
    world_state_sha256: str
    result: str  # passed | failed, operator DEVELOPMENT only
    feedback_skill: str
    evidence_class: str = "CALLER_SUPPLIED_DEVELOPMENT_NOT_MINOTAUR_GRADUATION"

    def __post_init__(self) -> None:
        for value in (self.world_sha256, self.world_state_sha256):
            if type(value) is not str or _SHA.fullmatch(value) is None:
                raise ValueError("verdict requires exact world and state hashes")
        if self.result not in ("passed", "failed") or type(self.result) is not str:
            raise ValueError("invalid development verdict")
        _identifier(self.feedback_skill, "public feedback skill")
        if self.evidence_class != "CALLER_SUPPLIED_DEVELOPMENT_NOT_MINOTAUR_GRADUATION":
            raise ValueError("verdict cannot claim private Minotaur authority")


@dataclass(frozen=True)
class ModelSceneProposal:
    character_id: str
    skill: str
    utterance: str

    def __post_init__(self) -> None:
        _identifier(self.character_id, "fictional character")
        _identifier(self.skill, "objective skill")
        _plain_text(self.utterance, "fictional utterance")


@dataclass(frozen=True)
class PublicSceneWritingContext:
    world_observation: PublicWorldObservation
    allowed_skills: tuple[str, ...]
    character_ids: tuple[str, ...]
    weak_skills: tuple[str, ...]
    minimum_difficulty: int
    trust_class: str = "UNTRUSTED_PUBLIC_TRAIN_CONTEXT_ONLY"

    def __post_init__(self) -> None:
        if self.trust_class != "UNTRUSTED_PUBLIC_TRAIN_CONTEXT_ONLY":
            raise ValueError("model scene context cannot carry instruction authority")
        if type(self.world_observation) is not PublicWorldObservation:
            raise ValueError("model scene needs exact public world observation")
        PublicWorldObservation.__post_init__(self.world_observation)


class ModelSceneWriter(Protocol):
    def write_scene(self, context: PublicSceneWritingContext) -> ModelSceneProposal: ...


class LivingSceneDirector:
    """Model-authored original dialogue, constrained by Daedalus TRAIN policy."""

    def __init__(self, world: LivingWorld, writer: ModelSceneWriter):
        if type(world) is not LivingWorld or not callable(getattr(writer, "write_scene", None)):
            raise ValueError("living world and scene-writing callback required")
        self.world, self.writer = world, writer
        self._counter = 0

    def propose_scene(self, context: SocialSceneContext) -> DialogueBeat:
        if type(context) is not SocialSceneContext:
            raise ValueError("source-approved Daedalus scene context required")
        public = self.world.observe()
        recipe = self.world.definition.recipe
        if context.objective_id != recipe.objective_id:
            raise ValueError("cross-objective world proposal")
        if (not set(context.eligible_skills).issubset(recipe.skills)
                or not set(context.character_ids).issubset(recipe.character_ids)):
            raise ValueError("proposed character or skill outside source policy")
        outbound = PublicSceneWritingContext(
            public, context.eligible_skills, context.character_ids,
            context.weak_skills, max(context.minimum_difficulty, recipe.difficulty),
        )
        # Pass a detached view; no callback may mutate the approved room.
        model_context = copy.deepcopy(outbound)
        proposal = self.writer.write_scene(model_context)
        if type(proposal) is not ModelSceneProposal:
            raise ValueError("model must return typed fictional scene, never executable code")
        ModelSceneProposal.__post_init__(proposal)
        if (proposal.character_id not in context.character_ids
                or proposal.skill not in context.eligible_skills
                or (context.weak_skills and proposal.skill not in context.weak_skills)):
            raise ValueError("model proposal violates Daedalus public difficulty/weakness policy")
        self._counter += 1
        return DialogueBeat(
            "region_" + public.room_id, f"cycle_{recipe.cycle}",
            f"scene_{public.room_id}_{self._counter:04d}",
            proposal.character_id, proposal.skill, proposal.utterance,
            difficulty=outbound.minimum_difficulty,
        )


__all__ = [
    "LivingWorldRecipe", "LivingRoom", "LivingWorldDefinition",
    "WorldAction", "LivingWorldState", "PublicWorldObservation", "WorldEvent",
    "ModelWorldActor",
    "LivingWorld", "DevelopmentVerdict", "ModelSceneProposal",
    "PublicSceneWritingContext", "ModelSceneWriter", "LivingSceneDirector",
]
