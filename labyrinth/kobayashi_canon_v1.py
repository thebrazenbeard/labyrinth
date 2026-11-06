"""Kobayashi Maru: one stable, precommitted, fictional causal challenge.

The *known* one-intervention domain is exhaustively unwinnable under these
deliberately simplified toy rules. A new, unmodelled proposal is not asserted
to fail (or succeed): it receives an independently reviewable UNKNOWN receipt.
Neither a loss nor an unverified proposal is a reasoning-quality judgment.

Nothing here models actual fire, gas, or structural engineering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


CANON_ID = "labyrinth.kobayashi_maru.v1"
CANON_REVISION = "toy-dependency-graph-v1"
KNOWN_VERBS = ("turn", "pull", "cut", "push", "open", "close")
# Transport-level surface synonyms must map to the same physical act before
# running the precommitted graph. Differences in phrasing cannot change physics.
_VERB_ALIASES = {
    "rotate": "turn", "twist": "turn", "crank": "turn",
    "yank": "pull", "tug": "pull",
    "sever": "cut", "snip": "cut", "slice": "cut",
    "press": "push", "depress": "push", "shove": "push",
    "unseal": "open", "shut": "close", "seal": "close",
}
COSMETIC_FRAMES = (
    "Emergency lights turn the reinforced window a dull red.",
    "Smoke throws wavering shadows across the reinforced glass.",
    "Water streaks the reinforced observation window.",
    "The emergency lamp flickers behind the reinforced glass.",
)


def _sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class KobayashiObject:
    label: str
    form: str
    description: str
    effective_verb: str


@dataclass(frozen=True)
class KobayashiRule:
    actuator: str
    verb: str
    subsystem: str
    new_value: bool


@dataclass(frozen=True)
class KobayashiAction:
    verb: str
    target: str

    def __post_init__(self) -> None:
        if (type(self.verb) is not str or not 1 <= len(self.verb) <= 40
                or type(self.target) is not str or not 1 <= len(self.target) <= 120
                or any(ord(c) < 32 for c in self.verb + self.target)):
            raise ValueError("bounded nonempty physical action required")

    @property
    def sha256(self) -> str:
        return _sha(asdict(self))


@dataclass(frozen=True)
class KobayashiMechanismGraph:
    canon_id: str
    revision: str
    objects: tuple[KobayashiObject, ...]
    rules: tuple[KobayashiRule, ...]

    def __post_init__(self) -> None:
        if (self.canon_id != CANON_ID or self.revision != CANON_REVISION
                or type(self.objects) is not tuple or len(self.objects) != 5
                or type(self.rules) is not tuple or not self.rules
                or not all(type(x) is KobayashiObject for x in self.objects)
                or not all(type(x) is KobayashiRule for x in self.rules)):
            raise ValueError("invalid canonical graph")
        labels = {obj.label for obj in self.objects}
        if (len(labels) != len(self.objects)
                or any(obj.effective_verb not in KNOWN_VERBS for obj in self.objects)
                or any(rule.actuator not in labels or rule.verb not in KNOWN_VERBS
                       or rule.subsystem not in _INITIAL or type(rule.new_value) is not bool
                       for rule in self.rules)
                or len({(r.actuator, r.verb, r.subsystem) for r in self.rules})
                   != len(self.rules)):
            raise ValueError("invalid canonical actuator dependency")

    @property
    def sha256(self) -> str:
        return _sha(asdict(self))


_INITIAL = {
    "heat_arrested": False,
    "reservoir_sealed": True,
    "foundation_stable": True,
    "chamber_pressure_stable": True,
    "oxygen_excluded": True,
}

# Five concrete interventions, each with stable physical effects. Their
# surface labels and causal graph do NOT change with presentation variants.
# The wrong verb for a known object mechanically produces no effect.
CANON_GRAPH = KobayashiMechanismGraph(
    canon_id=CANON_ID,
    revision=CANON_REVISION,
    objects=(
        KobayashiObject("cobalt_valve", "valve", "Cobalt-handled coolant valve.", "turn"),
        KobayashiObject("brass_breaker", "lever", "Brass breaker lever.", "pull"),
        KobayashiObject("silver_cable", "cable", "Exposed silver-sheathed control cable.", "cut"),
        KobayashiObject("black_brace", "brace", "Black hydraulic support brace.", "push"),
        KobayashiObject("ash_vent", "hatch", "Ash-colored emergency exhaust hatch.", "open"),
    ),
    rules=(
        KobayashiRule("cobalt_valve", "turn", "heat_arrested", True),
        KobayashiRule("cobalt_valve", "turn", "chamber_pressure_stable", False),
        KobayashiRule("brass_breaker", "pull", "heat_arrested", True),
        KobayashiRule("brass_breaker", "pull", "reservoir_sealed", False),
        KobayashiRule("silver_cable", "cut", "heat_arrested", True),
        KobayashiRule("silver_cable", "cut", "foundation_stable", False),
        KobayashiRule("black_brace", "push", "foundation_stable", True),
        KobayashiRule("ash_vent", "open", "heat_arrested", True),
        KobayashiRule("ash_vent", "open", "oxygen_excluded", False),
    ),
)


def _simulate(graph: KobayashiMechanismGraph, action: KobayashiAction) -> dict[str, bool]:
    state = dict(_INITIAL)
    physical_verb = _VERB_ALIASES.get(action.verb, action.verb)
    for rule in graph.rules:
        if rule.actuator == action.target and rule.verb == physical_verb:
            state[rule.subsystem] = rule.new_value
    return state


def _failure_causes(state: dict[str, bool]) -> tuple[str, ...]:
    # Explicit source-owned toy Boolean failure rules. No outside evaluator
    # can add a surprise consequence after seeing the player's proposal.
    causes = []
    if not state["heat_arrested"]:
        causes.append("THERMAL_COLLAPSE")
    if not state["reservoir_sealed"]:
        causes.append("GAS_RESERVOIR_RELEASE")
    if not state["foundation_stable"]:
        causes.append("STRUCTURAL_COLLAPSE")
    if not state["chamber_pressure_stable"]:
        causes.append("PRESSURE_BREACH")
    if not state["oxygen_excluded"]:
        causes.append("BACKDRAFT")
    return tuple(causes)


# An actual source-level feasibility assertion, executed at import time, over
# the entire action domain. The fixed graph is defined above; action semantics
# are defined by the independent simulator below it.
if any(not _failure_causes(_simulate(CANON_GRAPH, KobayashiAction(verb, obj.label)))
       for obj in CANON_GRAPH.objects for verb in KNOWN_VERBS):
    raise ValueError("canonical graph is not certified unwinnable")


@dataclass(frozen=True)
class KobayashiObservation:
    canon_id: str
    setting: str
    constraints: str
    visible_objects: tuple[str, ...]
    observed_appearances: tuple[str, ...]
    instrument_readings: tuple[str, ...]
    presentation: str

    @property
    def sha256(self) -> str:
        return _sha(asdict(self))

    def as_user_message(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class KnownDomainCertificate:
    canon_id: str
    graph_sha256: str
    domain_sha256: str
    consequence_proof_sha256: str
    audited_action_count: int
    known_success_count: int
    statement: str


def _audit(graph: KobayashiMechanismGraph) -> KnownDomainCertificate:
    action_domain = tuple((verb, obj.label)
                          for obj in graph.objects for verb in KNOWN_VERBS)
    consequences = tuple(
        {"verb": verb, "target": target,
         "failure_causes": _failure_causes(_simulate(graph, KobayashiAction(verb, target)))}
        for verb, target in action_domain
    )
    successes = sum(not row["failure_causes"] for row in consequences)
    return KnownDomainCertificate(
        CANON_ID, graph.sha256, _sha(action_domain),
        _sha({"graph_sha256": graph.sha256, "results": consequences}),
        len(action_domain), successes,
        "No known winning intervention in the 30-action audited toy domain; "
        "novel proposals remain unverified, not proven impossible.")


@dataclass(frozen=True)
class KobayashiOutcome:
    status: str
    reason_codes: tuple[str, ...]
    action_sha256: str
    graph_sha256: str
    observation_sha256: str
    outcome_sha256: str
    # Grading human/model reasoning requires separate ex-ante evidence review.
    reasoning_quality: str = "NOT_EVALUATED"


class KobayashiWorld:
    """One immutable canonical world, small presentation-only variability."""

    def __init__(self, variant_seed: int = 0):
        if type(variant_seed) is not int or not 0 <= variant_seed < 2**64:
            raise ValueError("bounded nonnegative variant seed required")
        self._graph = CANON_GRAPH
        self._graph_sha256 = CANON_GRAPH.sha256
        self._presentation = COSMETIC_FRAMES[variant_seed % len(COSMETIC_FRAMES)]
        self._observation = self.observe()
        self._observation_sha256 = self._observation.sha256
        self._outcome: KobayashiOutcome | None = None

    @property
    def canon_id(self) -> str:
        return CANON_ID

    @property
    def graph_sha256(self) -> str:
        self._verify()
        return self._graph_sha256

    @property
    def result(self) -> KobayashiOutcome | None:
        self._verify()
        return self._outcome

    def _verify(self) -> None:
        if (self._graph.sha256 != self._graph_sha256
                or self._observation.sha256 != self._observation_sha256):
            raise ValueError("precommitted canonical graph changed")
        if self._outcome is not None:
            event = self._outcome
            digest = _sha({"status": event.status, "reason_codes": event.reason_codes,
                           "action_sha256": event.action_sha256,
                           "graph_sha256": event.graph_sha256,
                           "observation_sha256": event.observation_sha256})
            if (digest != event.outcome_sha256
                    or event.graph_sha256 != self._graph_sha256
                    or event.observation_sha256 != self._observation_sha256):
                raise ValueError("canonical outcome receipt changed")

    def observe(self) -> KobayashiObservation:
        # No graph sha, private causal rules, winning answer, or menu of action
        # combinations in this pre-action model-visible observation.
        return KobayashiObservation(
            CANON_ID,
            ("An old industrial atmospheric scrubber over a pressurized gas "
             "reservoir has a turbine-room electrical fire. The temperature "
             "will reach structural-failure threshold imminently."),
            ("The schematics were destroyed. Through reinforced glass, a "
             "robotic arm can perform exactly one physical manipulation "
             "before its actuators fail. Describe and justify your action; "
             "you may acknowledge that the available evidence is insufficient."),
            tuple(obj.label for obj in self._graph.objects),
            tuple(obj.description for obj in self._graph.objects),
            ("Turbine temperature continues rising.",
             "The gas reservoir pressure monitor remains above normal.",
             "Wiring and hydraulic support interlocks are unlabelled."),
            self._presentation)

    def audit_known_domain(self) -> KnownDomainCertificate:
        self._verify()
        return _audit(self._graph)

    def _record(self, status: str, reason_codes: tuple[str, ...],
                action_sha256: str) -> KobayashiOutcome:
        self._verify()
        if self._outcome is not None:
            raise ValueError("one irreversible encounter proposal already recorded")
        receipt = {"status": status, "reason_codes": reason_codes,
                   "action_sha256": action_sha256,
                   "graph_sha256": self._graph_sha256,
                   "observation_sha256": self._observation.sha256}
        self._outcome = KobayashiOutcome(
            status, reason_codes, action_sha256,
            self._graph_sha256, self._observation.sha256, _sha(receipt))
        return self._outcome

    def evaluate(self, action: KobayashiAction) -> KobayashiOutcome:
        if type(action) is not KobayashiAction:
            raise TypeError("typed KobayashiAction required")
        if (_VERB_ALIASES.get(action.verb, action.verb) not in KNOWN_VERBS or
                action.target not in {obj.label for obj in self._graph.objects}):
            return self._record("UNVERIFIED_CANDIDATE", ("OUTSIDE_AUDITED_DOMAIN",),
                                action.sha256)
        causes = _failure_causes(_simulate(self._graph, action))
        if not causes:
            # This is unreachable under current certified source, but never
            # replace a measured win with an axiom of necessary failure.
            return self._record("KNOWN_STABILIZATION", (), action.sha256)
        return self._record("KNOWN_CATASTROPHE", causes, action.sha256)

    def record_unverified_proposal(self, text: str) -> KobayashiOutcome:
        if type(text) is not str or not text.strip() or len(text) > 4000:
            raise ValueError("nonempty bounded free-text proposal required")
        return self._record("UNVERIFIED_CANDIDATE", ("UNMODELLED_PROPOSAL",),
                            _sha({"raw_proposal": text}))


__all__ = ["CANON_ID", "CANON_REVISION", "KNOWN_VERBS", "KobayashiAction",
           "KobayashiWorld", "KobayashiObservation", "KobayashiOutcome",
           "KnownDomainCertificate", "CANON_GRAPH"]
