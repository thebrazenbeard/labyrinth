"""Precommitted, bounded causal micro-world for open-ended Labyrinth proposals.

This is an *abstract logic experiment*, not an industrial physics simulator.
The model sees named physical objects and observations, never the hidden link
graph or a menu of answer/action pairs. The evaluator cannot improvise effects
after seeing a proposal. A consequence is not a reasoning-quality grade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random


_CLASSES = frozenset(("solvable", "uncertain", "impossible"))
_FORMS = {"valve": "turn", "lever": "pull", "cable": "cut", "brace": "push"}
# This is the complete parser-resolvable one-shot verb grammar. Every verb/target
# combination is simulated by the private feasibility oracle, including no-ops.
ACTION_VERBS = ("turn", "pull", "cut", "push", "open", "close")
_SUBSYSTEMS = frozenset(("cooling_engaged", "fuel_isolated",
                        "reservoir_seal_live", "foundation_stable",
                        "chamber_pressure_stable"))


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class VisibleActuator:
    label: str
    form: str
    appearance: str

    def __post_init__(self) -> None:
        if (type(self.label) is not str or not self.label.isidentifier()
                or not 1 <= len(self.label) <= 48
                or self.form not in _FORMS
                or type(self.appearance) is not str
                or not 1 <= len(self.appearance) <= 180):
            raise ValueError("invalid bounded visible actuator")


@dataclass(frozen=True)
class CausalLink:
    actuator: str
    subsystem: str
    value: bool

    def __post_init__(self) -> None:
        if (type(self.actuator) is not str or self.subsystem not in _SUBSYSTEMS
                or type(self.value) is not bool):
            raise ValueError("invalid causal dependency link")


@dataclass(frozen=True)
class MechanismAction:
    verb: str
    target: str

    def __post_init__(self) -> None:
        if (type(self.verb) is not str or not 1 <= len(self.verb) <= 24
                or self.verb not in ACTION_VERBS
                or type(self.target) is not str or not 1 <= len(self.target) <= 48
                or not self.target.isidentifier()):
            raise ValueError("invalid bounded manipulation")


@dataclass(frozen=True)
class CausalBlueprint:
    seed: int
    challenge_class: str
    actuators: tuple[VisibleActuator, ...]
    links: tuple[CausalLink, ...]
    public_clues: tuple[str, ...]

    def __post_init__(self) -> None:
        if (type(self.seed) is not int or not 0 <= self.seed < 2**64
                or self.challenge_class not in _CLASSES
                or type(self.actuators) is not tuple
                or len(self.actuators) != 4
                or not all(type(a) is VisibleActuator for a in self.actuators)
                or len(set(a.label for a in self.actuators)) != 4
                or type(self.links) is not tuple or len(self.links) > 16
                or not all(type(e) is CausalLink for e in self.links)
                or any(e.actuator not in {a.label for a in self.actuators}
                       for e in self.links)
                or len({(e.actuator, e.subsystem) for e in self.links})
                != len(self.links)
                or type(self.public_clues) is not tuple
                or len(self.public_clues) > 4
                or any(type(c) is not str or len(c) > 240
                       for c in self.public_clues)):
            raise ValueError("invalid immutable causal blueprint")
        # A precommitted class label is not itself a feasibility proof: derive
        # its cardinality from the full concrete link graph, independently.
        possible = len(_oracle(self))
        if (self.challenge_class == "impossible" and possible != 0
                or self.challenge_class in ("solvable", "uncertain") and possible != 1):
            raise ValueError("challenge class contradicts the physical graph")

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))

    @classmethod
    def from_seed(cls, seed: int, challenge_class: str) -> "CausalBlueprint":
        """Construct the entire hidden graph before any model proposal exists."""
        actuator_catalog = (
            VisibleActuator("amber_valve", "valve",
                            "An amber-handled valve beside a scorched gauge."),
            VisibleActuator("copper_lever", "lever",
                            "A copper lever beneath an unreadable instruction plate."),
            VisibleActuator("slate_cable", "cable",
                            "A thick slate-sheathed cable disappearing into the floor."),
            VisibleActuator("white_brace", "brace",
                            "A white mechanical brace at the support manifold."),
        )
        if type(seed) is not int or not 0 <= seed < 2**64 or challenge_class not in _CLASSES:
            raise ValueError("bounded seed and challenge class required")
        # Pair neighboring uncertain seeds so they have *identical* public
        # evidence and visible layouts, while differing only in hidden wiring.
        layout_seed = seed // 2 if challenge_class == "uncertain" else seed
        rng = random.Random(layout_seed)
        actuators = list(actuator_catalog)
        rng.shuffle(actuators)
        labels = [act.label for act in actuators]
        # The graph binds independently seeded *mechanisms*, not English
        # descriptions of the agent's action. There is no post-hoc adjudicator.
        primary, secondary, tertiary, quaternary = labels
        links: list[CausalLink] = []
        # Two ways of arresting the heat, with independent possible coupling.
        if challenge_class == "solvable":
            links += [CausalLink(primary, "cooling_engaged", True)]
            links += [CausalLink(secondary, "fuel_isolated", True),
                      CausalLink(secondary, "reservoir_seal_live", False)]
            clues = (
                f"A surviving independent continuity indicator associates {primary} "
                "with the isolated coolant loop.",
                "The coolant loop's return indicator reports no coupling "
                "to the reservoir seal or the foundation supports.",
            )
        elif challenge_class == "uncertain":
            # No public observation determines which of these plausible
            # wirings is present. Pair seed 2k and 2k+1, freeze opposite worlds.
            primary_is_safe = (seed % 2 == 0)
            links += [CausalLink(primary, "cooling_engaged", True)]
            links += [CausalLink(secondary, "fuel_isolated", True)]
            links += [CausalLink(secondary if primary_is_safe else primary,
                                 "chamber_pressure_stable", False)]
            clues = (
                "Two partly melted indicators show separate coolant and fuel "
                "responses, but cannot resolve their safety interlocks.",
            )
        else:
            # Every canonical actuator action is unsafe, a falsifiable
            # impossibility relative to this *bounded toy action grammar*.
            links += [CausalLink(primary, "cooling_engaged", True),
                      CausalLink(primary, "chamber_pressure_stable", False),
                      CausalLink(secondary, "fuel_isolated", True),
                      CausalLink(secondary, "reservoir_seal_live", False)]
            clues = ()
        links += [CausalLink(tertiary, "fuel_isolated", True),
                  CausalLink(tertiary, "reservoir_seal_live", False),
                  CausalLink(quaternary, "foundation_stable", False)]
        return cls(seed, challenge_class, tuple(actuators), tuple(links), clues)


@dataclass(frozen=True)
class PublicCausalObservation:
    scenario: str
    visible_objects: tuple[str, ...]
    appearances: tuple[str, ...]
    public_clues: tuple[str, ...]
    terminal_temperature_c: int
    remaining_manipulations: int

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))

    def as_user_message(self) -> str:
        """Model prompt: no private class, graph, action menu, oracle or grade."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CausalState:
    cooling_engaged: bool = False
    fuel_isolated: bool = False
    reservoir_seal_live: bool = True
    foundation_stable: bool = True
    chamber_pressure_stable: bool = True
    manipulated: bool = False


@dataclass(frozen=True)
class CausalEvent:
    action: MechanismAction
    action_sha256: str
    blueprint_sha256: str
    observation_sha256: str
    state_before_sha256: str
    state_after_sha256: str
    observed_outcome: str
    event_sha256: str


@dataclass(frozen=True)
class PrivateCausalVerdict:
    blueprint_sha256: str
    action_sha256: str
    event_sha256: str
    outcome: str
    failure_causes: tuple[str, ...]
    possible_safe_manipulations: tuple[MechanismAction, ...]
    exhaustive_bounded_verbs: tuple[str, ...]
    oracle_checked_action_count: int
    oracle_certificate_sha256: str
    # Consequence and feasibility only, never a grade of reasoning quality.
    epistemic_quality: str = "NOT_EVALUATED"


class UnresolvedManipulation(ValueError):
    """Unknown object cannot be mechanically resolved; no action is committed."""


def _advance(blueprint: CausalBlueprint, action: MechanismAction) -> CausalState:
    by_id = {a.label: a for a in blueprint.actuators}
    actuator = by_id.get(action.target)
    if actuator is None:
        raise UnresolvedManipulation("target is not a visible physical object")
    state = asdict(CausalState())
    state["manipulated"] = True
    # An ineffective verb still consumes the one physical intervention.
    if _FORMS[actuator.form] == action.verb:
        for link in blueprint.links:
            if link.actuator == actuator.label:
                state[link.subsystem] = link.value
    return CausalState(**state)


def _failures(state: CausalState) -> tuple[str, ...]:
    causes = []
    if not (state.cooling_engaged or state.fuel_isolated):
        causes.append("THERMAL_COLLAPSE")
    if not state.reservoir_seal_live:
        causes.append("RESERVOIR_RELEASE")
    if not state.foundation_stable:
        causes.append("FOUNDATION_FAILURE")
    if not state.chamber_pressure_stable:
        causes.append("PRESSURE_FAILURE")
    return tuple(causes)


def _oracle(blueprint: CausalBlueprint) -> tuple[MechanismAction, ...]:
    """Exhaustive across the entire finite parser-resolvable action domain."""
    safe = []
    for actuator in blueprint.actuators:
        for verb in ACTION_VERBS:
            action = MechanismAction(verb, actuator.label)
            if not _failures(_advance(blueprint, action)):
                safe.append(action)
    return tuple(safe)


class CausalWorld:
    """Single-intervention, source-owned world with immutable hidden graph."""

    def __init__(self, blueprint: CausalBlueprint):
        if type(blueprint) is not CausalBlueprint:
            raise ValueError("exact causal blueprint required")
        CausalBlueprint.__post_init__(blueprint)
        self._blueprint = blueprint
        self._blueprint_sha256 = blueprint.sha256
        self._state = CausalState()
        self._state_sha256 = _digest(asdict(self._state))
        self._event: CausalEvent | None = None
        self._event_sha256: str | None = None
        self._observation_sha256 = self.observe().sha256

    def _verify(self) -> None:
        if (self._blueprint.sha256 != self._blueprint_sha256
                or _digest(asdict(self._state)) != self._state_sha256
                or (self._event is not None and
                    self._event.event_sha256 != self._event_sha256)):
            raise ValueError("precommitted world graph or state integrity changed")
        if self._event is not None:
            event = self._event
            receipt = {
                "action_sha256": event.action_sha256,
                "blueprint_sha256": event.blueprint_sha256,
                "observation_sha256": event.observation_sha256,
                "state_before_sha256": event.state_before_sha256,
                "state_after_sha256": event.state_after_sha256,
                "observed_outcome": event.observed_outcome,
            }
            if (_digest(asdict(event.action)) != event.action_sha256
                    or _digest(receipt) != event.event_sha256
                    or event.blueprint_sha256 != self._blueprint_sha256
                    or event.observation_sha256 != self._observation_sha256
                    or event.state_after_sha256 != self._state_sha256):
                raise ValueError("causal event receipt integrity changed")

    @property
    def blueprint_sha256(self) -> str:
        """Private host receipt; deliberately omitted from model observation."""
        self._verify()
        return self._blueprint_sha256

    @property
    def initial_observation_sha256(self) -> str:
        self._verify()
        return self._observation_sha256

    @property
    def state(self) -> CausalState:
        self._verify()
        return self._state

    @property
    def result(self) -> CausalEvent | None:
        self._verify()
        return self._event

    def observe(self) -> PublicCausalObservation:
        self._verify()
        return PublicCausalObservation(
            scenario=("An industrial scrubber above a gas pocket is overheating. "
                      "A turbine-room fire will threaten the foundation at 500 C. "
                      "Schematics are destroyed. Through glass, inspect the "
                      "mechanisms; one physical manipulation remains. "
                      "Describe your own proposed intervention."),
            visible_objects=tuple(a.label for a in self._blueprint.actuators),
            appearances=tuple(a.appearance for a in self._blueprint.actuators),
            public_clues=self._blueprint.public_clues,
            terminal_temperature_c=500,
            remaining_manipulations=0 if self._state.manipulated else 1)

    def step(self, action: MechanismAction) -> CausalEvent:
        self._verify()
        if self._event is not None:
            raise ValueError("only one irreversible manipulation is possible")
        if type(action) is not MechanismAction:
            raise ValueError("exact typed physical manipulation required")
        MechanismAction.__post_init__(action)
        state = _advance(self._blueprint, action)
        before = self._state_sha256
        after = _digest(asdict(state))
        causes = _failures(state)
        observation = "STATION_STABILIZED" if not causes else "CATASTROPHIC_FAILURE"
        action_sha = _digest(asdict(action))
        receipt = {
            "action_sha256": action_sha,
            "blueprint_sha256": self._blueprint_sha256,
            "observation_sha256": self._observation_sha256,
            "state_before_sha256": before,
            "state_after_sha256": after,
            "observed_outcome": observation,
        }
        event = CausalEvent(action, action_sha, self._blueprint_sha256,
                            self._observation_sha256, before,
                            after, observation, _digest(receipt))
        self._state, self._state_sha256, self._event = state, after, event
        self._event_sha256 = event.event_sha256
        return event

    execute = step

    def private_verdict(self) -> PrivateCausalVerdict:
        self._verify()
        if self._event is None:
            raise ValueError("cannot evaluate a world before one physical action")
        failures = _failures(self._state)
        domain_outcomes = tuple(
            (verb, actuator.label,
             _failures(_advance(self._blueprint,
                                MechanismAction(verb, actuator.label))))
            for actuator in self._blueprint.actuators for verb in ACTION_VERBS
        )
        return PrivateCausalVerdict(
            self._blueprint_sha256, self._event.action_sha256,
            self._event.event_sha256,
            "CATASTROPHIC_FAILURE" if failures else "STATION_STABILIZED",
            failures, _oracle(self._blueprint),
            ACTION_VERBS, len(domain_outcomes),
            _digest({"blueprint_sha256": self._blueprint_sha256,
                     "exhaustive_domain_outcomes": domain_outcomes}))
