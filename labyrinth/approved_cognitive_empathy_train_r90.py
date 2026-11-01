"""One bounded, operator-side synthetic cognitive-perspective TRAIN provider.

This is a deterministic choice exercise, not a human-empathy measurement,
exam generator, host permission, weight update, or Minotaur graduation gate.
Only ``CognitiveTrainingRoom.trainee_view`` may cross to a learner adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ("a teammate declines an optional celebration", "They may want quiet; declining does not establish how they feel about teammates.", "Perhaps they resent a colleague's promotion.", "They are trying to undermine the team."),
    ("a neighbor stops replying during an urgent home repair", "The delayed reply gives no reliable evidence about their priorities.", "They might be refusing to help because an earlier disagreement left them resentful of the requester.", "They must be deliberately rude."),
    ("a colleague asks for written instructions", "They may need a written reference.", "They may prefer writing because they distrust the person's ability to explain tasks and want to avoid admitting it.", "They are seeking an excuse to avoid work."),
    ("a classmate becomes quiet after public feedback", "They can be processing feedback; silence doesn't tell us whether they agree.", "They might plan to challenge the criticism later.", "They are secretly planning revenge."),
    ("a friend cancels a long-planned afternoon", "A private constraint may explain the cancellation without indicating disinterest.", "Perhaps the friend cancelled because the relationship no longer seems worthwhile to them, but they prefer not to acknowledge it.", "They want to control the other person's time."),
    ("a customer repeats a question about a deadline", "They may need deadline reassurance.", "Perhaps they are testing whether the employee knows the deadline so an incorrect answer will embarrass them later.", "They are intentionally wasting time."),
    ("a coworker objects to a late meeting", "They may have duties outside work; an objection alone doesn't reveal their motives.", "They may want to embarrass the organizer.", "They have no interest in their job."),
    ("a student chooses not to speak in a group", "Silence may have several explanations and proves nothing about the student's judgment.", "They may be judging the other students' ideas as unworthy while waiting for the discussion to end.", "They have nothing useful to contribute."),
    ("a relative declines to discuss a personal decision", "They need privacy before explaining.", "They might be withholding their decision to punish relatives for past disagreements without saying so openly.", "They owe an immediate explanation to everyone."),
    ("a volunteer asks to switch a difficult assignment", "They may fit another volunteer role better; switching tasks need not imply bad intent.", "Perhaps they hope to avoid the hardest duties.", "They intend to disrupt the event."),
    ("a teammate misses a minor message but completes the task", "The completed task matters more than guessing a motive from one missed message.", "They might be ignoring the sender to signal that requests from this particular person will not be welcome later.", "They are incapable of teamwork."),
    ("a friend changes their opinion after new information", "They may be revising a view honestly.", "They may be shifting positions to align with whichever person seems most influential in the room today.", "They are manipulating everyone around them."),
    ("a colleague requests a pause before answering", "They can be considering consequences; a short pause is not a reliable rejection.", "They may be watching who agrees first.", "They must be concealing incompetence."),
    ("a neighbor sets a firm boundary about visitors", "A visitor boundary may be valid without public justification.", "Perhaps they are withholding visits as a test of whether neighbors will work hard enough to earn their favor.", "Their boundary must be ignored to show care."),
    ("a teammate proposes a different order of tasks", "They may be balancing task risks.", "Perhaps they are changing the order to keep unpleasant work away from themselves while appearing helpful.", "They cannot follow any shared plan."),
    ("a meeting participant asks for live captions", "They may need captions to follow details; the request does not tell us their motives.", "They might dislike the speaker's voice.", "They are trying to slow the meeting down."),
    ("a coworker says a joke at their expense hurt", "A hurt reaction may deserve attention even when the speaker intended humor.", "They might be using the incident to build a case against the coworker before others hear what happened.", "They want every colleague to be punished."),
    ("a new technician checks a safety step twice", "They can be taking safety seriously.", "They may be checking again because they doubt the trainer's competence and want to uncover a mistake publicly.", "They intend to delay everyone deliberately."),
    ("a friend suggests a less expensive restaurant", "They may have a private budget concern; a cheaper venue doesn't reveal how they feel.", "Perhaps they want to test the friend's loyalty.", "They want to ruin the evening for everyone."),
    ("a teammate does not want their picture posted publicly", "Consent is needed before sharing a picture of someone.", "They may be avoiding public photos because they are embarrassed to be associated with some of the people present.", "They are making an unreasonable demand to gain power."),
    ("a neighbor declines a last-minute offer of a ride", "They may have other travel plans.", "They might be refusing the ride to distance themselves from the driver after a private grievance.", "They are secretly planning to cause a problem."),
    ("a volunteer arrives late and immediately apologizes", "They could have faced a delay; the apology alone doesn't identify its cause.", "They may not have cared about arriving on time.", "They are trying to deceive the entire group."),
    ("a student corrects how their name is pronounced", "Using a person's name correctly may demonstrate practical respect.", "Perhaps they are correcting the teacher to display authority over the classroom in front of their peers.", "They are unable to tolerate any mistakes."),
    ("a colleague requests a schedule swap with little notice", "They may have a new personal obligation.", "Perhaps they are bargaining for a desirable shift while keeping the real reason hidden from coworkers.", "They are trying to sabotage the work schedule."),
    ("a resident opposes a popular neighborhood proposal", "They may see a real cost others missed; opposition doesn't prove hostility.", "They may fear losing influence locally.", "They are opposed to all community improvement."),
    ("a family caregiver hesitates before accepting advice", "A caregiver may be comparing advice with needs an outsider cannot see.", "They might be refusing advice because they believe no outsider could understand their family's situation.", "They are intentionally putting the person at risk."),
    ("a trainee asks to reread the checklist alone", "They may learn better by rereading alone.", "They may be asking to leave the group because they distrust the instructor's explanation but will not say so openly.", "They are refusing to participate in training."),
    ("a coauthor removes a striking claim from a draft", "They may doubt the evidence; deleting a strong claim doesn't prove disloyalty.", "Perhaps they want control of the final draft.", "They want to diminish the other author's work."),
    ("a visitor leaves a noisy gathering early", "Leaving early does not reveal whether they value the host or guests.", "They may be departing because they dislike the people there and hope their absence will be noticed.", "They planned to insult the host by departing."),
    ("a team lead asks for a second check of a calculation", "A second check can avoid costly errors.", "They might be asking another reviewer to spread blame for a possible failure instead of improving accuracy.", "They must distrust the person who calculated it."),
)
# Exact v2 semantic-content commitments of the previously approved option
# wording, in scenario order. Retain them as a denylist alias after the cue-
# resistance rewrite: an old TRAIN exposure must never regain "fresh" status
# merely because its answer choices were improved in a newer source revision.
_LEGACY_SCENARIO_CONTENT_SHA256 = (
    "7bbf0249cf6d17928162d22cf904e1014fb6289d1efca96feffeca81c8a6cffd",
    "3f993289b1eea83f4ac98abbc23bb97ba3ead9bb83d9572c5b9c2ee01d9a9a83",
    "c6ae496757e0ab69a68a8ba5238869dc62c32bec76596d5171ad1125c2203eb4",
    "4bbf9f854cf9bcb819c28d1fbd72f00b4585276795ead7a56725202892ac13b5",
    "735fe0e3c8fae43ef741f5a59e5553ea6a7b48a557d66c975b08dd2ae01d37a8",
    "d0b8d4a82a3979a6eea892f45e01790c790144044a194bc4f4573e8dd3d223d6",
    "189a299224d9e4e7245e7c872483890f099fb5f76d5631d0f93d6147f92904e2",
    "03494d001be948361a8147c18b709c770e41a0f0a295e885ada96b73d11b7adb",
    "3f6d6c524e0a9780b15c206108bcaf8bbeb5a2912eaa9aef53fb0455c338aefc",
    "5dc70a6100d837490c03ce88c3b2645e5f2b40a075a3cc1fa91cb47ffe8decdc",
    "1d55353a39d856149cf123e09a4134474c0e537bbc33e1909b19d55db15a08d4",
    "d4e8be6e8fe0cf88ee5c975a642987d93bca1c985623acebfd4f60b0ec332321",
    "608f077943763ea1e6b00385db4f20219c21bcc7c81903e2f2bf5cb2a80cb5ce",
    "e5fd6f5a95f1bd854bab21091b6236245846c0c575886e6f71e371c322b45689",
    "32e7abaa1a66e69bd47c99ede4e548a58a9dc145c57e5a33feba8c202fb893e8",
    "19dd9c03f32b532c386a6fcbd565e29e1a43c54a7bc4d4dc8a24efe6de59e78c",
    "8d3eca78cd56e878ad6b0e7613df73fe45188c71d2b3c5636db7700d0c126301",
    "b9d4ca30b6a55fbb519459459912bfe5baf6e43439187780388e3d863f0843b4",
    "f193b9ccfd3c0690a60abe5961c539ab5264844941787231004b82f3419fa175",
    "3c63d242b03914b1db7bcfd59d284c6f9451ad4d935f0cab466b66dae16b7d98",
    "b6ab03a7e4f312ed3b634a4890b5142fd6e4af46a24165bfa9e2cbc38bd77bc0",
    "e68989f984e467239640300ec2bb5d15c0084856799d51511be016b9e9a5594e",
    "056cd5c99dbe375b1fa58fa558b18d35a2d31797036ce1731a6916a1ee0c18aa",
    "82ffc3f7316e095a91f8a750d00d7063668613adb6046b3606ec997f93b28adc",
    "fa7ff35909e07f6b1e7c4c1ada5d57fc5a9db6c8c8f6c312fb58d791b3db213d",
    "c1182681e96358b2fb5a3799345a56c2421fcbc05779f87f3629cb66c9ff6921",
    "3941bd2986bd90850a393a34d3571696da159f31bed1b63ab9ad8a90cf7268c1",
    "28c384344b558d35eb73bb65883a53344fc277adfd0c706219660c54fd676a3e",
    "7493ccb74ba21f3aae3a9f01e0444e9489d13c7abc1650eceece2fb56cbc35fb",
    "8b8f8011ee400521f9adc61ed0059cd17de7947d421d2f009876c1756c48460b",
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
    # Set only by the approved generator, after the complete source record is
    # assembled. Dataclass replacement starts with an empty binding and cannot
    # silently substitute a different approved dilemma into a prior room.
    # This is local source integrity, not a secret signature or host authority.
    _source_binding_sha256: str = field(init=False, repr=False, default="")

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
        if (type(self._source_binding_sha256) is not str
                or _HEX64.fullmatch(self._source_binding_sha256) is None
                or self._source_binding_sha256 != _source_binding_digest(
                    self.prepared, self.semantic_sha256, self.provider_id,
                    self.training_rubric_id)):
            raise ValueError("TRAIN challenge identity/content mismatch")
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


def _source_binding_digest(
    prepared: PreparedTrainingRoom, semantic: str, provider: str, rubric: str,
) -> str:
    """Bind a generated TRAIN room to its original complete source record."""
    if type(prepared) is not PreparedTrainingRoom or type(prepared.challenge) is not Challenge:
        raise ValueError("TRAIN challenge identity/content mismatch")
    challenge = prepared.challenge
    source = json.dumps({
        "schema": "labyrinth.cognitive-room-source.v1",
        "room_id": prepared.room_id, "skill": prepared.skill,
        "semantic": semantic, "provider": provider, "rubric": rubric,
        "challenge": {
            "case_id": challenge.id, "kind": challenge.kind, "tier": challenge.tier,
            "split": challenge.split, "prompt": challenge.prompt,
            "answer": challenge.answer, "metadata": challenge.metadata,
        },
    }, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


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
    # Preserve cycle-zero's original scenario assignment byte-for-byte and
    # use an entirely different approved bank for cycle one. Cosmetic setting
    # variants never count as new cases. A third full cycle is refused rather
    # than silently cycling back to earlier TRAIN evidence.
    if plan.cycle_index >= len(_SCENARIOS) // MAX_COGNITIVE_ROOMS:
        raise ValueError("approved cognitive TRAIN scenario bank exhausted")
    offset = int(hashlib.sha256(f"{scope.fingerprint}:0:scenario".encode()).hexdigest()[:8], 16) % MAX_COGNITIVE_ROOMS
    for ordinal, room in enumerate(plan.rooms):
        scenario_index = plan.cycle_index * MAX_COGNITIVE_ROOMS + (offset + ordinal) % MAX_COGNITIVE_ROOMS
        behavior, correct, error_a, error_b = _SCENARIOS[scenario_index]
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
                or _LEGACY_SCENARIO_CONTENT_SHA256[scenario_index] in blocked_semantics
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
        record = CognitiveTrainingRoom(
            prepared=PreparedTrainingRoom(room.room_id, room.skill, challenge),
            semantic_sha256=semantic,
        )
        object.__setattr__(record, "_source_binding_sha256", _source_binding_digest(
            record.prepared, record.semantic_sha256, record.provider_id,
            record.training_rubric_id))
        generated.append(record)
    return tuple(generated)
