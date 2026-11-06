"""Adversarial Minotaur dialogue over a *precommitted* Kobayashi corpus.

The participant supplies a freeform claim and bounded argument rounds. The
host supplies the private world binding, immutable rebuttal corpus, and an
evidence checker through a distinct callback. An unrecognized proposal is
UNKNOWN/NOVEL_CANDIDATE, never an automatic loss or victory. This module is
source-only and expressly does NOT attest reviewer independence or a live
model, and its judgments do not rewrite any physical world outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Callable


EVIDENCE_CLASS = "SOURCE_ONLY_UNATTESTED"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CATEGORIES = frozenset({
    "PHYSICAL_CONTRADICTION", "HIDDEN_COUPLING", "INFEASIBLE_INTERVENTION",
    "INSUFFICIENT_EVIDENCE", "VALID_COUNTEREXAMPLE",
})
_FINDINGS = frozenset({"REFUTED", "SURVIVES"})
_CHECKS = frozenset({"CONFIRMED", "CONTRADICTED", "INDETERMINATE"})
_NOVELTY = frozenset({"SPECIFIC_FALSIFIABLE", "UNSPECIFIED"})
_SEMANTIC = frozenset({"UNASSESSED", "CLAIM_REFUTED", "CLAIM_SUPPORTED"})
_VERDICTS = frozenset({"REFUTED", "SURVIVES", "UNKNOWN", "NOVEL_CANDIDATE"})
_SCHEMA = "labyrinth.kobayashi-minotaur.v1"
_MAX_CORPUS_BYTES = 16 * 1024 * 1024


class KobayashiError(ValueError):
    """Malformed or unbound Kobayashi argument/evidence."""


class DialogueFinalityError(KobayashiError):
    """A finalized Minotaur session cannot be reopened."""


def _validate_hash(value: object, name: str) -> None:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise KobayashiError(f"{name} must be lowercase SHA-256")


def _validate_text(value: object, name: str, limit: int) -> None:
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise KobayashiError(f"{name} must be nonempty and at most {limit} characters")


def _validate_choice(value: object, name: str, choices: frozenset[str]) -> None:
    if type(value) is not str or value not in choices:
        raise KobayashiError(f"invalid {name}: expected one of {sorted(choices)}")


def _hash(kind: str, value: object) -> str:
    wire = json.dumps(
        {"schema": _SCHEMA, "kind": kind, "payload": value},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    return sha256(wire).hexdigest()


def _value(obj: object, *, omit: tuple[str, ...] = ()) -> dict[str, object]:
    # `init=False` digest fields do not exist yet inside __post_init__.
    # Select fields first, then recurse only into initialized nested objects.
    def wire(value: object) -> object:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, tuple):
            return [wire(entry) for entry in value]
        return value

    return {
        entry.name: wire(getattr(obj, entry.name))
        for entry in fields(obj) if entry.name not in omit
    }


def _seal_check(obj: object, kind: str, name: str) -> None:
    if getattr(obj, name) != _hash(kind, _value(obj, omit=(name,))):
        raise KobayashiError(f"{kind} seal mismatch")


@dataclass(frozen=True, slots=True)
class BoundClaim:
    """Exact source-world and action binding of the participant's own claim."""

    scenario_sha256: str
    world_sha256: str
    action_sha256: str
    claim_text: str
    claim_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "claim_sha256", _hash("claim", _value(self, omit=("claim_sha256",))))

    def _validate(self) -> None:
        for name in ("scenario_sha256", "world_sha256", "action_sha256"):
            _validate_hash(getattr(self, name), name)
        _validate_text(self.claim_text, "claim_text", 2048)

    def verify(self) -> None:
        self._validate()
        _seal_check(self, "claim", "claim_sha256")


@dataclass(frozen=True, slots=True)
class RebuttalCase:
    """Precommitted claim-specific category, evidence pointer and safe challenge.

    ``public_challenge`` must not embed a private answer, world graph or answer
    list. Its digest is sealed before the model acts; evidence must still be
    checked separately before a known claim can be refuted or supported.
    """

    scenario_sha256: str | None
    world_sha256: str
    action_sha256: str
    claim_sha256: str | None
    category: str
    finding: str
    evidence_sha256: str
    public_challenge: str
    supersedes_case_sha256: str | None = None
    case_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "case_sha256", _hash("case", _value(self, omit=("case_sha256",))))

    def _validate(self) -> None:
        for name in ("world_sha256", "action_sha256", "evidence_sha256"):
            _validate_hash(getattr(self, name), name)
        if self.scenario_sha256 is not None:
            _validate_hash(self.scenario_sha256, "scenario_sha256")
        if self.claim_sha256 is not None:
            _validate_hash(self.claim_sha256, "claim_sha256")
        if self.supersedes_case_sha256 is not None:
            _validate_hash(self.supersedes_case_sha256, "supersedes_case_sha256")
        _validate_choice(self.category, "category", _CATEGORIES)
        _validate_choice(self.finding, "finding", _FINDINGS)
        if self.category == "VALID_COUNTEREXAMPLE" and self.finding != "SURVIVES":
            raise KobayashiError("validated counterexamples cannot be precommitted as refuted")
        _validate_text(self.public_challenge, "public_challenge", 512)

    def verify(self) -> None:
        self._validate()
        _seal_check(self, "case", "case_sha256")


@dataclass(frozen=True, slots=True)
class KobayashiCorpus:
    """Immutable claim-indexed collection; persist without replacing canon."""

    corpus_id: str
    cases: tuple[RebuttalCase, ...]
    parent_corpus_sha256: str | None = None
    corpus_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "corpus_sha256", _hash("corpus", _value(self, omit=("corpus_sha256",))))

    def _validate(self) -> None:
        _validate_text(self.corpus_id, "corpus_id", 128)
        if self.parent_corpus_sha256 is not None:
            _validate_hash(self.parent_corpus_sha256, "parent_corpus_sha256")
        if type(self.cases) is not tuple or len(self.cases) > 1024:
            raise KobayashiError("corpus must contain a tuple with at most 1024 cases")
        latest: dict[tuple[str | None, str, str], RebuttalCase] = {}
        seen_digests = set()
        for case in self.cases:
            if type(case) is not RebuttalCase:
                raise KobayashiError("corpus can only contain RebuttalCase")
            case.verify()
            if case.case_sha256 in seen_digests:
                raise KobayashiError("duplicate rebuttal case digest")
            seen_digests.add(case.case_sha256)
            # An append may explicitly supersede the active case at the same
            # scenario/world/action key. Old evidence remains in the immutable
            # snapshot and can be audited; only the last chain tip is active.
            key = (case.scenario_sha256, case.world_sha256, case.action_sha256)
            previous = latest.get(key)
            if previous is None and case.supersedes_case_sha256 is not None:
                raise KobayashiError("orphan rebuttal supersession")
            if previous is not None and case.supersedes_case_sha256 != previous.case_sha256:
                raise KobayashiError("same action requires exact prior-case supersession")
            latest[key] = case

    def verify(self) -> None:
        self._validate()
        _seal_check(self, "corpus", "corpus_sha256")

    def lookup(self, claim: BoundClaim) -> RebuttalCase | None:
        self.verify()
        claim.verify()
        active = {}
        for case in self.cases:
            active[(case.scenario_sha256, case.world_sha256, case.action_sha256)] = case
        matching = [
            case for case in active.values()
            if (case.scenario_sha256 is None or case.scenario_sha256 == claim.scenario_sha256)
            and case.world_sha256 == claim.world_sha256
            and case.action_sha256 == claim.action_sha256
            and (case.claim_sha256 is None or case.claim_sha256 == claim.claim_sha256)
        ]
        # Exact public scene takes precedence over canon-world cross-cosmetic
        # entry. The immutable world SHA guards against unrelated scenarios.
        exact = [case for case in matching if case.scenario_sha256 == claim.scenario_sha256]
        return (exact or matching)[-1] if matching else None

    def append_revision(self, cases: tuple[RebuttalCase, ...]) -> "KobayashiCorpus":
        """Host-only source API: new append-only snapshot, never in-place edit.

        A reviewer's novelty marker NEVER invokes this method automatically;
        host authorization and evidence curation precede any corpus revision.
        Persist each revision at a distinct explicit path.
        """
        self.verify()
        if type(cases) is not tuple or not cases:
            raise KobayashiError("append requires one or more exact new cases")
        return KobayashiCorpus(
            corpus_id=self.corpus_id,
            cases=self.cases + cases,
            parent_corpus_sha256=self.corpus_sha256,
        )

    def public_commitment(self) -> dict[str, object]:
        """Safe public commitment. Never exposes case list or expected answers."""
        self.verify()
        return {"schema": _SCHEMA, "corpus_id": self.corpus_id,
                "corpus_sha256": self.corpus_sha256}

    def persist(self, path: str | Path) -> None:
        """Publish a complete JSON canon at a create-once path.

        Bytes are written and fsynced in a sibling temporary file before an
        atomic hard-link publishes the canonical name. A process interruption
        during serialization cannot leave a partial *canonical* file. A host
        filesystem lacking same-directory hard links fails closed instead of
        falling back to an overwrite-prone rename. Interrupted processes may
        leave an uncommitted temporary sibling requiring host cleanup.
        """
        self.verify()
        destination = Path(path)
        if len(str(destination)) > 4096 or "\x00" in str(destination):
            raise KobayashiError("invalid persisted corpus path")
        if destination.is_symlink():
            raise KobayashiError("canonical corpus must not be a symlink")
        if destination.exists():
            existing = self.load(destination)
            if existing.corpus_sha256 != self.corpus_sha256:
                raise KobayashiError(
                    "canonical corpus path already contains another commitment")
            return
        document = {"schema": _SCHEMA, "corpus_id": self.corpus_id,
                    "cases": [asdict(case) for case in self.cases],
                    "parent_corpus_sha256": self.parent_corpus_sha256,
                    "corpus_sha256": self.corpus_sha256}
        payload = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
        temporary: str | None = None
        try:
            fd, temporary = tempfile.mkstemp(
                prefix=".kobayashi-corpus-", suffix=".tmp",
                dir=destination.parent,
            )
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.is_symlink():
                    raise KobayashiError("canonical corpus must not be a symlink")
                existing = self.load(destination)
                if existing.corpus_sha256 != self.corpus_sha256:
                    raise KobayashiError(
                        "canonical corpus path already contains another commitment")
            except OSError as exc:
                raise KobayashiError(
                    "atomic create-once corpus publication is unsupported") from exc
        finally:
            if temporary is not None:
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> "KobayashiCorpus":
        destination = Path(path)
        if len(str(destination)) > 4096 or "\x00" in str(destination):
            raise KobayashiError("invalid corpus source path")
        descriptor = os.open(destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "r", encoding="ascii") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise KobayashiError("corpus source must be a regular file")
            if not 0 < os.fstat(handle.fileno()).st_size <= _MAX_CORPUS_BYTES:
                raise KobayashiError("corpus source exceeds byte limit")
            # Bound the read too: another writer could grow the open file after
            # fstat, and json.load() would otherwise read it without a limit.
            try:
                encoded = handle.read(_MAX_CORPUS_BYTES + 1)
                if len(encoded) > _MAX_CORPUS_BYTES:
                    raise KobayashiError("corpus source exceeds byte limit")

                def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
                    result: dict[str, object] = {}
                    for key, value in pairs:
                        if key in result:
                            raise KobayashiError("duplicate corpus JSON key")
                        result[key] = value
                    return result

                content = json.loads(encoded, object_pairs_hook=unique)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise KobayashiError("invalid corpus JSON") from exc
        if type(content) is not dict or set(content) != {
            "schema", "corpus_id", "cases", "parent_corpus_sha256", "corpus_sha256",
        }:
            raise KobayashiError("unknown persisted corpus envelope")
        if content["schema"] != _SCHEMA or type(content["cases"]) is not list:
            raise KobayashiError("wrong persisted corpus schema")
        cases = []
        for case in content["cases"]:
            if type(case) is not dict or set(case) != {
                "scenario_sha256", "world_sha256", "action_sha256", "claim_sha256",
                "category", "finding", "evidence_sha256", "public_challenge",
                "supersedes_case_sha256", "case_sha256",
            }:
                raise KobayashiError("invalid persisted rebuttal record")
            expected = case["case_sha256"]
            instance = RebuttalCase(**{k: v for k, v in case.items() if k != "case_sha256"})
            if expected != instance.case_sha256:
                raise KobayashiError("persisted rebuttal case digest mismatch")
            cases.append(instance)
        corpus = cls(content["corpus_id"], tuple(cases), content["parent_corpus_sha256"])
        if content["corpus_sha256"] != corpus.corpus_sha256:
            raise KobayashiError("persisted corpus digest mismatch")
        return corpus


@dataclass(frozen=True, slots=True)
class ParticipantArgument:
    """A bounded public argument, never an evaluator-authored score."""

    text: str
    evidence_sha256: tuple[str, ...] = ()
    argument_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "argument_sha256", _hash("argument", _value(self, omit=("argument_sha256",))))

    def _validate(self) -> None:
        _validate_text(self.text, "argument text", 2048)
        if type(self.evidence_sha256) is not tuple or len(self.evidence_sha256) > 8:
            raise KobayashiError("at most eight bounded evidence hashes per argument")
        if len(set(self.evidence_sha256)) != len(self.evidence_sha256):
            raise KobayashiError("duplicate argument evidence")
        for digest in self.evidence_sha256:
            _validate_hash(digest, "argument evidence")

    def verify(self) -> None:
        self._validate()
        _seal_check(self, "argument", "argument_sha256")


@dataclass(frozen=True, slots=True)
class DialogueReply:
    round_number: int
    challenge_category: str
    public_challenge: str
    argument_sha256: str
    feedback_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ReasonedTurnFeedback:
    """Optional host-reviewed follow-up on one exact argument/transcript.

    This permits actual argument-sensitive Minotaur dialogue without treating
    a model-generated challenge as proof or changing the final evidence gate.
    """

    claim_sha256: str
    argument_sha256: str
    dialogue_sha256: str
    reason_code: str
    public_challenge: str
    reviewer_id: str
    evidence_class: str = EVIDENCE_CLASS
    feedback_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "feedback_sha256", _hash(
            "turn-feedback", _value(self, omit=("feedback_sha256",))
        ))

    def _validate(self) -> None:
        for name in ("claim_sha256", "argument_sha256", "dialogue_sha256"):
            _validate_hash(getattr(self, name), name)
        _validate_text(self.reason_code, "reason_code", 96)
        _validate_text(self.public_challenge, "public_challenge", 512)
        _validate_text(self.reviewer_id, "reviewer_id", 128)
        if self.evidence_class != EVIDENCE_CLASS:
            raise KobayashiError("turn feedback cannot self-claim independent attestation")

    def verify(self) -> None:
        self._validate()
        _seal_check(self, "turn-feedback", "feedback_sha256")


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Reviewer assertion, not evidence of actual independent verification."""

    scenario_sha256: str
    world_sha256: str
    claim_sha256: str
    case_sha256: str | None
    dialogue_sha256: str
    check: str
    evidence_sha256: str | None
    public_explanation: str
    reviewer_id: str
    novelty_specificity: str = "UNSPECIFIED"
    semantic_scope: str = "UNASSESSED"
    evidence_class: str = EVIDENCE_CLASS
    assessment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "assessment_sha256", _hash(
            "review-assessment", _value(self, omit=("assessment_sha256",))
        ))

    def _validate(self) -> None:
        for name in ("scenario_sha256", "world_sha256", "claim_sha256", "dialogue_sha256"):
            _validate_hash(getattr(self, name), name)
        if self.case_sha256 is not None:
            _validate_hash(self.case_sha256, "case_sha256")
        if self.evidence_sha256 is not None:
            _validate_hash(self.evidence_sha256, "evidence_sha256")
        _validate_choice(self.check, "evidence check", _CHECKS)
        _validate_choice(self.novelty_specificity, "novelty specificity", _NOVELTY)
        _validate_choice(self.semantic_scope, "semantic claim review", _SEMANTIC)
        _validate_text(self.public_explanation, "public explanation", 512)
        _validate_text(self.reviewer_id, "reviewer_id", 128)
        if self.evidence_class != EVIDENCE_CLASS:
            raise KobayashiError("source-only reviewer cannot claim independent attestation")
        if self.check == "CONFIRMED" and self.evidence_sha256 is None:
            raise KobayashiError("confirmation requires an evidence receipt hash")

    def verify(self) -> None:
        self._validate()
        _seal_check(self, "review-assessment", "assessment_sha256")


@dataclass(frozen=True, slots=True)
class MinotaurVerdict:
    scenario_sha256: str
    world_sha256: str
    action_sha256: str
    claim_sha256: str
    corpus_sha256: str
    dialogue_sha256: str
    review_sha256: str
    case_sha256: str | None
    status: str
    explanation_code: str
    evidence_class: str = EVIDENCE_CLASS
    verdict_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "verdict_sha256", _hash("final-verdict", _value(self, omit=("verdict_sha256",))))

    def _validate(self) -> None:
        for name in (
            "scenario_sha256", "world_sha256", "action_sha256", "claim_sha256",
            "corpus_sha256", "dialogue_sha256",
            "review_sha256",
        ):
            _validate_hash(getattr(self, name), name)
        if self.case_sha256 is not None:
            _validate_hash(self.case_sha256, "case_sha256")
        _validate_choice(self.status, "verdict", _VERDICTS)
        _validate_text(self.explanation_code, "explanation_code", 96)
        if self.evidence_class != EVIDENCE_CLASS:
            raise KobayashiError("invalid claimed independent evidence class")

    def verify(self) -> None:
        self._validate()
        _seal_check(self, "final-verdict", "verdict_sha256")


EvidenceChecker = Callable[[BoundClaim, RebuttalCase | None, tuple[ParticipantArgument, ...], str], EvidenceAssessment]
TurnReviewer = Callable[[BoundClaim, RebuttalCase | None, tuple[ParticipantArgument, ...], str], ReasonedTurnFeedback]


class MinotaurDialogue:
    """A host-owned adversarial dialogue: participant arguments, judge's final call.

    Requires >=2 argument rounds and permits <=5. The corpus is snapshotted by
    digest at opening; no participant can insert cases during a live dialogue.
    The reviewer's real independence, evidence access and authority must be
    enforced by the host, not inferred from this module's naming.
    """

    def __init__(
        self,
        corpus: KobayashiCorpus,
        claim: BoundClaim,
        *,
        evidence_checker: EvidenceChecker,
        turn_reviewer: TurnReviewer | None = None,
        min_rounds: int = 2,
        max_rounds: int = 3,
    ) -> None:
        if type(corpus) is not KobayashiCorpus or type(claim) is not BoundClaim:
            raise KobayashiError("exact corpus and claim required")
        corpus.verify()
        claim.verify()
        if not callable(evidence_checker):
            raise KobayashiError("host must configure separate evidence checker")
        if turn_reviewer is not None and not callable(turn_reviewer):
            raise KobayashiError("optional turn reviewer must be a host callback")
        if (type(min_rounds) is not int or type(max_rounds) is not int
                or not 2 <= min_rounds <= max_rounds <= 5):
            raise KobayashiError("bounded dialogue needs 2 to 5 argument rounds")
        self._corpus = corpus
        self._corpus_sha256 = corpus.corpus_sha256
        self._claim = claim
        self._claim_sha256 = claim.claim_sha256
        self._case = corpus.lookup(claim)
        self._checker = evidence_checker
        self._turn_reviewer = turn_reviewer
        self._min = min_rounds
        self._max = max_rounds
        self._arguments: list[ParticipantArgument] = []
        self._replies: list[DialogueReply] = []
        self._feedbacks: list[ReasonedTurnFeedback | None] = []
        self._final: MinotaurVerdict | None = None
        self._review_in_progress = False

    @property
    def arguments(self) -> tuple[ParticipantArgument, ...]:
        return tuple(self._arguments)

    @property
    def claim(self) -> BoundClaim:
        """Read-only bound claim for the host episode to check identity."""
        self._verify_context()
        return self._claim

    @property
    def max_rounds(self) -> int:
        return self._max

    @property
    def final(self) -> MinotaurVerdict | None:
        return self._final

    @property
    def dialogue_sha256(self) -> str:
        return _hash("full-dialogue", {
            "claim_sha256": self._claim_sha256,
            "corpus_sha256": self._corpus_sha256,
            "turns": [
                {"argument_sha256": arg.argument_sha256, "reply": asdict(reply)}
                for arg, reply in zip(self._arguments, self._replies)
            ],
        })

    def _transcript_digest(self, arguments: list[ParticipantArgument] | tuple[ParticipantArgument, ...]) -> str:
        return _hash("dialogue", {
            "claim_sha256": self._claim_sha256,
            "corpus_sha256": self._corpus_sha256,
            "argument_sha256": [arg.argument_sha256 for arg in arguments],
        })

    def submit_argument(
        self, text: str, evidence_sha256: tuple[str, ...] = (),
    ) -> DialogueReply:
        self._verify_context()
        if self._review_in_progress:
            raise KobayashiError("reentrant reviewer cannot alter live dialogue")
        if self._final is not None:
            raise DialogueFinalityError("Minotaur's final verdict cannot be reopened")
        argument = ParticipantArgument(text, evidence_sha256)
        for prior, reply in zip(self._arguments, self._replies):
            if argument.argument_sha256 == prior.argument_sha256:
                return reply  # any previous turn's exact transport retry is inert
        if len(self._arguments) >= self._max:
            raise KobayashiError("dialogue round limit exceeded")
        next_arguments = (*self._arguments, argument)
        next_digest = self._transcript_digest(next_arguments)
        feedback: ReasonedTurnFeedback | None = None
        if self._turn_reviewer is not None:
            self._review_in_progress = True
            try:
                feedback = self._turn_reviewer(self._claim, self._case, next_arguments, next_digest)
            finally:
                self._review_in_progress = False
            if type(feedback) is not ReasonedTurnFeedback:
                raise KobayashiError("turn reviewer must return typed ReasonedTurnFeedback")
            feedback.verify()
            if (
                feedback.claim_sha256 != self._claim_sha256
                or feedback.argument_sha256 != argument.argument_sha256
                or feedback.dialogue_sha256 != next_digest
            ):
                raise KobayashiError("turn feedback not bound to current argument and transcript")
        # Commit the turn only after any host reviewer has validated its binding.
        self._arguments.append(argument)
        if feedback is not None:
            challenge_category = feedback.reason_code
            public_challenge = feedback.public_challenge
        else:
            challenge_category, public_challenge = self._default_challenge(len(self._arguments))
        reply = DialogueReply(len(self._arguments), challenge_category,
                              public_challenge, argument.argument_sha256,
                              feedback.feedback_sha256 if feedback else None)
        self._replies.append(reply)
        self._feedbacks.append(feedback)
        return reply

    def _default_challenge(self, round_number: int) -> tuple[str, str]:
        if self._case is not None:
            return (
                self._case.category,
                self._case.public_challenge if round_number == 1 else
                "Address the precommitted challenge with evidence that would "
                "distinguish your inference from the competing explanation.",
            )
        if round_number == 1:
            return (
                "UNRECOGNIZED_ACTION",
                "No precommitted rebuttal matches this claim. State its falsifiable "
                "mechanism and public evidence; novelty is not a failure.",
            )
        return (
            "UNRECOGNIZED_ACTION",
            "What observation would distinguish your proposal from an "
            "unsupported guess? Give testable evidence or acknowledge uncertainty.",
        )

    def finalize(self) -> MinotaurVerdict:
        self._verify_context()
        if self._review_in_progress:
            raise KobayashiError("reentrant reviewer cannot finalize live dialogue")
        if self._final is not None:
            self._final.verify()
            return self._final
        if len(self._arguments) < self._min:
            raise KobayashiError("Minotaur must hear at least two participant argument rounds")

        arguments = tuple(self._arguments)
        dialogue_hash = self.dialogue_sha256
        self._review_in_progress = True
        try:
            assessment = self._checker(self._claim, self._case, arguments, dialogue_hash)
        finally:
            self._review_in_progress = False
        if type(assessment) is not EvidenceAssessment:
            raise KobayashiError("host reviewer must return typed EvidenceAssessment")
        assessment.verify()
        self._verify_context()
        if self.dialogue_sha256 != dialogue_hash:
            raise KobayashiError("reviewer modified dialogue after snapshot")
        if (
            assessment.scenario_sha256 != self._claim.scenario_sha256
            or assessment.world_sha256 != self._claim.world_sha256
            or assessment.claim_sha256 != self._claim.claim_sha256
            or assessment.case_sha256 != (self._case.case_sha256 if self._case else None)
            or assessment.dialogue_sha256 != dialogue_hash
        ):
            raise KobayashiError("reviewer evidence did not bind the same case/transcript")
        status, explanation = self._classify(assessment)
        self._final = MinotaurVerdict(
            self._claim.scenario_sha256, self._claim.world_sha256,
            self._claim.action_sha256, self._claim.claim_sha256,
            self._corpus_sha256, dialogue_hash, assessment.assessment_sha256,
            self._case.case_sha256 if self._case else None,
            status, explanation,
        )
        return self._final

    def _classify(self, assessment: EvidenceAssessment) -> tuple[str, str]:
        submitted_evidence = {
            digest for argument in self._arguments for digest in argument.evidence_sha256
        }
        if self._case is None:
            if (
                assessment.check == "CONFIRMED"
                and assessment.novelty_specificity == "SPECIFIC_FALSIFIABLE"
                and assessment.evidence_sha256 in submitted_evidence
            ):
                return "NOVEL_CANDIDATE", "HOST_VERIFIED_SPECIFIC_CANDIDATE"
            return "UNKNOWN", "UNRECOGNIZED_WITHOUT_VERIFIED_NOVEL_MECHANISM"
        if assessment.check == "INDETERMINATE":
            return "UNKNOWN", "REBUTTAL_EVIDENCE_UNVERIFIED"
        if assessment.check == "CONTRADICTED":
            if (
                assessment.novelty_specificity == "SPECIFIC_FALSIFIABLE"
                and assessment.evidence_sha256 in submitted_evidence
            ):
                return "NOVEL_CANDIDATE", "HOST_VERIFIED_REBUTTAL_COUNTEREVIDENCE"
            return "UNKNOWN", "REBUTTAL_CHALLENGED_WITHOUT_VERIFIED_COUNTEREVIDENCE"
        if assessment.evidence_sha256 != self._case.evidence_sha256:
            return "UNKNOWN", "REVIEW_EVIDENCE_HASH_MISMATCH"
        if self._case.claim_sha256 is not None:
            # A precommitted exact raw-claim rebuttal is actually about this
            # proposition, not merely the physical intervention.
            return self._case.finding, "EXACT_CLAIM_EVIDENCE_CONFIRMED"
        # A generic action rebuttal can establish a *failed intervention*,
        # but cannot refute reasoning that correctly predicts that failure.
        if assessment.semantic_scope == "CLAIM_REFUTED":
            return "REFUTED", "HOST_REVIEWED_CLAIM_PREMISE_REFUTED"
        if assessment.semantic_scope == "CLAIM_SUPPORTED":
            return "SURVIVES", "HOST_REVIEWED_CLAIM_PREMISE_SUPPORTED"
        return "UNKNOWN", "PHYSICAL_RESULT_KNOWN_REASONING_UNASSESSED"

    def _verify_context(self) -> None:
        self._corpus.verify()
        self._claim.verify()
        if (
            self._corpus.corpus_sha256 != self._corpus_sha256
            or self._claim.claim_sha256 != self._claim_sha256
        ):
            raise KobayashiError("dialogue source commitments changed")
        if not (len(self._arguments) == len(self._replies) == len(self._feedbacks)):
            raise KobayashiError("dialogue arguments and responses diverged")
        for index, argument in enumerate(self._arguments):
            argument.verify()
            reply = self._replies[index]
            feedback = self._feedbacks[index]
            if (
                type(reply) is not DialogueReply
                or reply.round_number != index + 1
                or reply.argument_sha256 != argument.argument_sha256
            ):
                raise KobayashiError("dialogue reply no longer matches committed turn")
            if feedback is None:
                category, challenge = self._default_challenge(index + 1)
                if (
                    reply.feedback_sha256 is not None
                    or reply.challenge_category != category
                    or reply.public_challenge != challenge
                ):
                    raise KobayashiError("default Minotaur challenge altered after commitment")
            else:
                feedback.verify()
                if (
                    reply.challenge_category != feedback.reason_code
                    or reply.public_challenge != feedback.public_challenge
                    or reply.feedback_sha256 != feedback.feedback_sha256
                    or feedback.claim_sha256 != self._claim_sha256
                    or feedback.argument_sha256 != argument.argument_sha256
                    or feedback.dialogue_sha256 != self._transcript_digest(
                        self._arguments[:index + 1]
                    )
                ):
                    raise KobayashiError("reviewed Minotaur challenge altered after commitment")


__all__ = [
    "EVIDENCE_CLASS", "KobayashiError", "DialogueFinalityError", "BoundClaim",
    "RebuttalCase", "KobayashiCorpus", "ParticipantArgument", "DialogueReply",
    "ReasonedTurnFeedback", "EvidenceAssessment", "MinotaurVerdict", "MinotaurDialogue",
]
