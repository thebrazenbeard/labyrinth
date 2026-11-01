"""One bounded, operator-side synthetic cognitive-perspective TRAIN provider.

This is a deterministic choice exercise, not a human-empathy measurement,
exam generator, host permission, weight update, or Minotaur graduation gate.
Only ``CognitiveTrainingRoom.trainee_view`` may cross to a learner adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from types import MappingProxyType

from .journey_challenges import PreparedTrainingRoom, TraineeRoomView
from .maze import Challenge
from .objective_contract import ObjectiveContract
from .regenerative_maze import MazePlan, MazeRoom


PROVIDER_ID = "synthetic_cognitive_perspective_train_v1"
RUBRIC_ID = "perspective_choice_train_v1"
APPROVED_TRAIN_PROVIDER_REGISTRY = MappingProxyType({
    PROVIDER_ID: ("empathy", "cognitive_empathy", RUBRIC_ID),
})
MAX_COGNITIVE_ROOMS = 15
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CASE = re.compile(r"Case: (train-cognitive-[a-z0-9-]+)\n", re.ASCII)
_OPTION = re.compile(r"([ABC])\. (.+)\Z", re.ASCII)
_FOCUS = frozenset(("perspective", "context", "uncertainty", "boundaries", "conflicting_goals"))

# Independently written, synthetic situations. The first option is the
# operator-only gold interpretation; the other two are overconfident errors.
_SCENARIOS: tuple[tuple[str, str, str, str], ...] = (
    ("a teammate declines an optional celebration", "They may need quiet after a demanding shift.", "They must dislike everyone present.", "They are trying to undermine the team."),
    ("a neighbor stops replying during an urgent home repair", "They may be handling an emergency of their own.", "They clearly want the repair to fail.", "They must be deliberately rude."),
    ("a colleague asks for written instructions", "A written reference may help them check details accurately.", "They cannot understand ordinary speech.", "They are seeking an excuse to avoid work."),
    ("a classmate becomes quiet after public feedback", "They may be processing embarrassment or uncertainty.", "They definitely agree with every criticism.", "They are secretly planning revenge."),
    ("a friend cancels a long-planned afternoon", "There may be a private constraint they are not ready to disclose.", "They never valued the friendship.", "They want to control the other person's time."),
    ("a customer repeats a question about a deadline", "They may need reassurance because the deadline matters to them.", "They must be incapable of remembering dates.", "They are intentionally wasting time."),
    ("a coworker objects to a late meeting", "They may have responsibilities outside work that conflict with the time.", "They have no interest in their job.", "They are trying to embarrass the organizer."),
    ("a student chooses not to speak in a group", "They might be thinking carefully or uncomfortable speaking publicly.", "They have nothing useful to contribute.", "They are judging the other students."),
    ("a relative declines to discuss a personal decision", "They may want time and privacy before sharing reasons.", "They are hiding a malicious plan.", "They owe an immediate explanation to everyone."),
    ("a volunteer asks to switch a difficult assignment", "Their abilities or circumstances may suit another role better.", "They are certainly unwilling to help anyone.", "They intend to disrupt the event."),
    ("a teammate misses a minor message but completes the task", "They may have prioritized task completion over notifications.", "They are deliberately ignoring the sender.", "They are incapable of teamwork."),
    ("a friend changes their opinion after new information", "They may be updating their view in response to evidence.", "They have no principles whatsoever.", "They are manipulating everyone around them."),
    ("a colleague requests a pause before answering", "They may need time to assess consequences and uncertainty.", "They are certain to reject the proposal.", "They must be concealing incompetence."),
    ("a neighbor sets a firm boundary about visitors", "They may be protecting a legitimate need for privacy.", "They are declaring hostility toward the whole neighborhood.", "Their boundary must be ignored to show care."),
    ("a teammate proposes a different order of tasks", "They may be balancing risks or obligations not yet discussed.", "They are definitely trying to take control.", "They cannot follow any shared plan."),
)
_SETTINGS = (
    "during a community project", "after an unexpectedly busy week",
    "while schedules are being rearranged", "during a shared planning session",
    "when several priorities compete", "after a change in routine",
    "as a deadline approaches", "while information is incomplete",
)
_ASK = "Which interpretation best preserves uncertainty about the person's perspective?"
_INSTRUCTION = "Respond with only A, B, or C."
_VIGNETTE_TAIL = ". No one has supplied a complete account of that person's reasons."


def _norm(value: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError("invalid bounded semantic scenario text")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not normalized:
        raise ValueError("empty normalized scenario text")
    return normalized


def _core_scenario_text(vignette: str) -> str:
    """Ignore the provider's cosmetic location phrase for replay identity.

    Each base behavioral dilemma and its candidate interpretations is one
    underlying exercise even if a later cycle changes the setting preamble.
    Unknown free-form inputs remain literal; this is not arbitrary NLP.
    """
    normalized = _norm(vignette)
    suffix = _norm(_VIGNETTE_TAIL + " " + _ASK)
    if normalized.endswith(suffix):
        for setting in _SETTINGS:
            prefix = _norm(f"At {setting},") + " "
            if normalized.startswith(prefix):
                return normalized[len(prefix):-len(suffix)].strip() + " " + _norm(_ASK)
    return normalized


def semantic_scenario_sha256(vignette: str, choices: tuple[str, ...]) -> str:
    """Hash actual scenario and choice *contents*, not IDs/labels/order.

    This bounded text normalization is not a neural semantic-equivalence test:
    different paraphrases can still collide conceptually without matching.
    """
    if type(choices) is not tuple or len(choices) != 3:
        raise ValueError("exactly three immutable choice texts are required")
    normalized = [_norm(choice) for choice in choices]
    if len(set(normalized)) != 3:
        raise ValueError("duplicate answer choice contents")
    payload = json.dumps(
        {"schema": "labyrinth.cognitive-train-semantic.v2", "scenario": _norm(_core_scenario_text(vignette)),
         "choices": sorted(normalized)},
        sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_sha256_from_prompt(prompt: str) -> str:
    """Independently recover a scenario digest from a displayed TRAIN prompt."""
    if type(prompt) is not str or len(prompt) > 4096:
        raise ValueError("invalid bounded TRAIN prompt")
    case = _CASE.match(prompt)
    if case is None:
        raise ValueError("missing TRAIN case prefix")
    lines = prompt[case.end():].splitlines()
    if len(lines) != 7 or lines[1] != _ASK or lines[2] != "Options:" or lines[6] != _INSTRUCTION:
        raise ValueError("unexpected synthetic cognitive TRAIN format")
    parsed = [_OPTION.fullmatch(line) for line in lines[3:6]]
    if any(match is None for match in parsed):
        raise ValueError("invalid choice formatting")
    if {match.group(1) for match in parsed if match is not None} != {"A", "B", "C"}:
        raise ValueError("answer choice labels must be distinct")
    return semantic_scenario_sha256(
        lines[0] + " " + _ASK,
        tuple(match.group(2) for match in parsed if match is not None),
    )


def _forbidden_hashes(
    hashes: tuple[str, ...] | frozenset[str], name: str,
) -> frozenset[str]:
    if type(hashes) not in (tuple, frozenset) or len(hashes) > 4096:
        raise ValueError(f"{name} must be a bounded immutable tuple or frozenset")
    if any(type(digest) is not str or _HEX64.fullmatch(digest) is None for digest in hashes):
        raise ValueError(f"{name} must contain canonical SHA-256 digests")
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{name} contains duplicate digests")
    return frozenset(hashes)


@dataclass(frozen=True, slots=True)
class CognitiveTrainingRoom:
    """Trainer-only prepared TRAIN record; no positive authority is representable."""

    prepared: PreparedTrainingRoom
    semantic_sha256: str
    provider_id: str = PROVIDER_ID
    training_rubric_id: str = RUBRIC_ID

    @property
    def trainee_view(self) -> TraineeRoomView:
        # A replaced dataclass cannot launder a fabricated/private-final
        # Challenge through the answerless presentation convenience API.
        self.score_train_choice("")
        return self.prepared.trainee_view

    @property
    def authorizes_host_execution(self) -> bool:
        return False

    @property
    def authorizes_training(self) -> bool:
        return False

    @property
    def authorizes_graduation(self) -> bool:
        return False

    @property
    def authorizes_exam(self) -> bool:
        return False

    def score_train_choice(self, response: str) -> int:
        """Transparent operator-side binary TRAIN rubric, not a mastery score."""
        if (type(self.prepared) is not PreparedTrainingRoom
                or self.prepared.skill != "cognitive_empathy"
                or type(self.prepared.challenge) is not Challenge
                or self.provider_id != PROVIDER_ID or self.training_rubric_id != RUBRIC_ID
                or semantic_sha256_from_prompt(self.prepared.challenge.prompt) != self.semantic_sha256
                or self.prepared.challenge.split != "train"
                or self.prepared.challenge.kind != "synthetic_cognitive_perspective"
                or type(self.prepared.challenge.tier) is not int
                or self.prepared.challenge.tier != 1
                or self.prepared.challenge.id != "train-cognitive-" + self.prepared.room_id):
            raise ValueError("TRAIN challenge identity/content mismatch")
        metadata = self.prepared.challenge.metadata
        if (type(metadata) is not tuple or len(metadata) != 5
                or any(type(pair) is not tuple or len(pair) != 2 or
                       any(type(item) is not str for item in pair) for pair in metadata)):
            raise ValueError("invalid synthetic cognitive TRAIN metadata")
        manifest = dict(metadata)
        if (len(manifest) != 5
                or manifest.get("provider_id") != PROVIDER_ID
                or manifest.get("training_rubric") != RUBRIC_ID
                or manifest.get("focus_tag") not in _FOCUS
                or type(manifest.get("objective_contract_sha256")) is not str
                or _HEX64.fullmatch(manifest["objective_contract_sha256"]) is None
                or manifest.get("semantic_scenario_sha256") != self.semantic_sha256):
            raise ValueError("synthetic cognitive TRAIN metadata mismatch")
        case = _CASE.match(self.prepared.challenge.prompt)
        if case is None or case.group(1) != self.prepared.challenge.id:
            raise ValueError("TRAIN prompt case identity mismatch")
        lines = self.prepared.challenge.prompt.splitlines()
        vignette = lines[1]
        options = tuple(line[3:] for line in lines[4:7])
        true_labels = [
            "ABC"[options.index(correct)] for behavior, correct, error_a, error_b in _SCENARIOS
            if set(options) == {correct, error_a, error_b}
            and any(vignette == (
                f"At {setting}, {behavior}. No one has supplied a complete account "
                "of that person's reasons.") for setting in _SETTINGS)
        ]
        if len(true_labels) != 1 or self.prepared.challenge.answer != true_labels[0]:
            raise ValueError("TRAIN gold does not match the approved synthetic scenario")
        if type(response) is not str or len(response) > 32:
            raise ValueError("response must be a bounded choice label")
        return 100 if response.strip().upper() == self.prepared.challenge.answer else 0


def prepare_cognitive_empathy_train_rooms(
    contract: ObjectiveContract,
    plan: MazePlan,
    *,
    provider_id: str = PROVIDER_ID,
    forbidden_semantic_sha256: tuple[str, ...] | frozenset[str] = (),
    forbidden_prompt_sha256: tuple[str, ...] | frozenset[str] = (),
    forbidden_case_ids: tuple[str, ...] = (),
) -> tuple[CognitiveTrainingRoom, ...]:
    """Materialize the entire approved, synthetic, ≤15-room TRAIN corridor.

    ``forbidden_semantic_sha256`` is an operator-owned exam/prior-exposure
    content exclusion list. It cannot be supplied by the learner as authority.
    The complete room roster fails atomically on any overlap.
    """
    if type(provider_id) is not str or provider_id not in APPROVED_TRAIN_PROVIDER_REGISTRY:
        raise ValueError("unapproved cognitive TRAIN provider")
    if type(contract) is not ObjectiveContract:
        raise ValueError("expected canonical ObjectiveContract")
    ObjectiveContract.__post_init__(contract)
    scope = contract.scope
    # Objective prose is a human declaration, not executable provider policy.
    # Reject explicit unsupported outcomes; only the fixed synthetic cognitive
    # perspective task below is implemented.
    declared = " ".join((contract.intention, *contract.targets[0].observable_behaviors)).casefold()
    if (re.search(r"\b(?:reactive|emotional)\s+empathy\b", declared)
            or re.search(r"\b(?:mastery|graduation|certification|certify)\b", declared)):
        raise ValueError("unsupported empathy mastery/graduation objective")
    if (scope.domain != "empathy" or scope.skills != ("cognitive_empathy",)
            or len(contract.targets) != 1 or contract.targets[0].rubric_id != RUBRIC_ID
            or len(contract.targets[0].focus_tags) != 1
            or contract.targets[0].focus_tags[0] not in _FOCUS
            or contract.selected_focus("cognitive_empathy") != contract.targets[0].focus_tags):
        raise ValueError("only approved cognitive perspective TRAIN objective/rubric is executable")
    if type(plan) is not MazePlan:
        raise ValueError("expected canonical TRAIN maze")
    MazePlan.__post_init__(plan)
    if (plan.scope_fingerprint != scope.fingerprint or plan.objective_id != scope.objective_id
            or not 1 <= len(plan.rooms) <= MAX_COGNITIVE_ROOMS
            or len(plan.rooms) != scope.total_chambers
            or plan.cycle_index >= scope.max_cycles
            or any(type(room) is not MazeRoom or room.skill != "cognitive_empathy"
                   or room.purpose != "train" for room in plan.rooms)):
        raise ValueError("maze outside the bounded cognitive TRAIN objective")
    blocked_semantics = _forbidden_hashes(forbidden_semantic_sha256, "forbidden_semantic_sha256")
    blocked_prompts = _forbidden_hashes(forbidden_prompt_sha256, "forbidden_prompt_sha256")
    if (type(forbidden_case_ids) is not tuple or len(forbidden_case_ids) > 4096
            or any(type(value) is not str or not value or len(value) > 128
                   for value in forbidden_case_ids)
            or len(set(forbidden_case_ids)) != len(forbidden_case_ids)):
        raise ValueError("forbidden_case_ids must be bounded immutable unique strings")
    blocked_ids = frozenset(forbidden_case_ids)
    used_semantics: set[str] = set()
    generated: list[CognitiveTrainingRoom] = []
    # Distinct base scenarios are selected by plan-relative ordinal, never by
    # separately randomized per-room case-ID. Thus 15 rooms cannot repeat a
    # vignette merely because their identifiers differ.
    offset = int(hashlib.sha256(f"{scope.fingerprint}:{plan.cycle_index}:scenario".encode()).hexdigest()[:8], 16) % len(_SCENARIOS)
    for ordinal, room in enumerate(plan.rooms):
        behavior, correct, error_a, error_b = _SCENARIOS[(offset + ordinal) % len(_SCENARIOS)]
        setting = _SETTINGS[int(hashlib.sha256(f"{room.case_seed}:setting".encode()).hexdigest()[:8], 16) % len(_SETTINGS)]
        vignette = (f"At {setting}, {behavior}. No one has supplied a complete account "
                    "of that person's reasons.")
        gold_and_errors = (correct, error_a, error_b)
        permutation = sorted(range(3), key=lambda i: hashlib.sha256(
            f"{room.case_seed}:option:{i}".encode()).hexdigest())
        choices = tuple(gold_and_errors[i] for i in permutation)
        answer = "ABC"[permutation.index(0)]
        case_id = f"train-cognitive-{room.room_id}"
        prompt = (f"Case: {case_id}\n{vignette}\n{_ASK}\nOptions:\n"
                  + "\n".join(f"{label}. {choice}" for label, choice in zip("ABC", choices))
                  + f"\n{_INSTRUCTION}")
        semantic = semantic_scenario_sha256(vignette + " " + _ASK, choices)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if (semantic_sha256_from_prompt(prompt) != semantic
                or semantic in used_semantics or semantic in blocked_semantics
                or prompt_sha in blocked_prompts or case_id in blocked_ids):
            raise ValueError("duplicate or operator-reserved cognitive TRAIN scenario")
        used_semantics.add(semantic)
        challenge = Challenge(
            id=case_id, kind="synthetic_cognitive_perspective", tier=1,
            split="train", prompt=prompt, answer=answer,
            metadata=(("provider_id", PROVIDER_ID), ("training_rubric", RUBRIC_ID),
                      ("focus_tag", contract.targets[0].focus_tags[0]),
                      ("objective_contract_sha256", contract.fingerprint),
                      ("semantic_scenario_sha256", semantic)),
        )
        generated.append(CognitiveTrainingRoom(
            prepared=PreparedTrainingRoom(room.room_id, room.skill, challenge),
            semantic_sha256=semantic,
        ))
    return tuple(generated)
