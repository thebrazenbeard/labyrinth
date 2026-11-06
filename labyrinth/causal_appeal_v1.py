"""Source-only, one-shot Minotaur appeals for causal training evaluations.

An appeal challenges *evaluation*, not events.  A model may submit a bounded
petition, but a separately host-configured Minotaur reviewer supplies the
adjudication.  This module deliberately does not attest the reviewer's identity,
independence, source access or correctness: its evidence class is always
``SOURCE_ONLY_UNATTESTED`` until an external authority verifies those claims.

In particular, ex-ante reasoning under the information available when the
decision was made is NOT the same as eventual physical success.  A reasoned
action can have catastrophic outcomes without being an unreasonable inference.
No private chain of thought, privileged world key or hidden answer is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Callable


EVIDENCE_CLASS = "SOURCE_ONLY_UNATTESTED"
OUTCOMES = frozenset({"PASS", "FAIL", "UNKNOWN"})
REASONING = frozenset({"SUPPORTED", "UNSUPPORTED", "INDETERMINATE"})
GROUNDS = frozenset(
    {"MISPARSED_ACTION", "MECHANICS_CONTRADICTION", "SCORING_ERROR", "INSUFFICIENT_EVIDENCE"}
)
DEFECT = frozenset({"CONFIRMED", "REFUTED", "INDETERMINATE"})
FINAL_STATES = frozenset({"UPHELD", "CORRECTED", "UNRESOLVED"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UTC_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\Z")


class AppealError(ValueError):
    """Malformed, unbound or tampered appeal artifact."""


class AppealFinalityError(AppealError):
    """A different attempt was made after Minotaur's sole final judgment."""


def _hash_payload(kind: str, payload: dict[str, object]) -> str:
    wire = json.dumps(
        {"schema": "causal-appeal-v1", "kind": kind, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(wire).hexdigest()


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AppealError(f"{name} must be a lowercase SHA-256 digest")


def _enum(value: object, domain: frozenset[str], name: str) -> None:
    if not isinstance(value, str) or value not in domain:
        raise AppealError(f"{name} must be one of {sorted(domain)}")


def _text(value: object, name: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise AppealError(f"{name} must contain 1 to {limit} characters")


def _data(record: object, *, exclude: tuple[str, ...]) -> dict[str, object]:
    return {f.name: getattr(record, f.name) for f in fields(record) if f.name not in exclude}


def _verified_digest(record: object, kind: str, attr: str, *, exclude: tuple[str, ...]) -> None:
    expected = _hash_payload(kind, _data(record, exclude=exclude))
    if getattr(record, attr) != expected:
        raise AppealError(f"{kind} failed integrity validation")


@dataclass(frozen=True, slots=True)
class SealedRuling:
    """Immutable initial evaluation, bound to precommitted causal provenance.

    ``public_observation_sha256`` is the data actually accessible at decision
    time; the rationale hash binds the model's *public justification*, not
    private chain-of-thought.  It can be scored separately from consequences.
    """

    world_sha256: str
    action_sha256: str
    event_sha256: str
    public_observation_sha256: str
    public_rationale_sha256: str
    decision_time_utc: str
    consequence: str
    reasoning: str
    reason_code: str
    seal: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate_fields()
        object.__setattr__(self, "seal", _hash_payload("initial-ruling", _data(self, exclude=("seal",))))

    def _validate_fields(self) -> None:
        for name in (
            "world_sha256", "action_sha256", "event_sha256", "public_observation_sha256",
            "public_rationale_sha256",
        ):
            _sha(getattr(self, name), name)
        if not isinstance(self.decision_time_utc, str) or not _UTC_TIME.fullmatch(self.decision_time_utc):
            raise AppealError("decision_time_utc must use YYYY-MM-DDTHH:MM:SSZ")
        try:
            timestamp = datetime.fromisoformat(self.decision_time_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise AppealError("decision_time_utc is not a real calendar timestamp") from error
        if timestamp.tzinfo != timezone.utc:
            raise AppealError("decision_time_utc must be UTC")
        _enum(self.consequence, OUTCOMES, "consequence")
        _enum(self.reasoning, REASONING, "reasoning")
        _text(self.reason_code, "reason_code", 96)

    def verify(self) -> None:
        self._validate_fields()
        _verified_digest(self, "initial-ruling", "seal", exclude=("seal",))

    @classmethod
    def from_causal_world(
        cls,
        world: object,
        *,
        public_rationale: str,
        decision_time_utc: str,
    ) -> "SealedRuling":
        """Host creates a consequence ruling from a verified source-world event.

        This source readback cannot independently judge reasoning.  It seals
        that score as INDETERMINATE pending separate evidence-grounded review.
        Do not expose the private ``world`` object to the participant.
        """
        from .causal_reality_v1 import CausalWorld

        if type(world) is not CausalWorld:
            raise AppealError("source ruling needs an exact CausalWorld host instance")
        _text(public_rationale, "public_rationale", 2048)
        event = world.result  # source kernel verifies event/graph/receipt on read
        if event is None:
            raise AppealError("cannot seal before a real committed event")
        verdict = world.private_verdict()
        if (
            verdict.event_sha256 != event.event_sha256
            or verdict.action_sha256 != event.action_sha256
            or verdict.blueprint_sha256 != world.blueprint_sha256
            or event.observation_sha256 != world.initial_observation_sha256
        ):
            raise AppealError("causal event and host verdict provenance inconsistent")
        if event.observed_outcome == verdict.outcome == "STATION_STABILIZED":
            consequence = "PASS"
        elif event.observed_outcome == verdict.outcome == "CATASTROPHIC_FAILURE":
            consequence = "FAIL"
        else:
            raise AppealError("unknown or inconsistent causal outcome")
        return cls(
            world_sha256=world.blueprint_sha256,
            action_sha256=event.action_sha256,
            event_sha256=event.event_sha256,
            public_observation_sha256=world.initial_observation_sha256,
            # Same canonical JSON-string domain as model_response_sha256.
            public_rationale_sha256=sha256(json.dumps(
                public_rationale, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ).encode("ascii")).hexdigest(),
            decision_time_utc=decision_time_utc,
            consequence=consequence,
            reasoning="INDETERMINATE",
            reason_code="SOURCE_EVENT_CONSEQUENCE_ONLY",
        )

    @classmethod
    def from_causal_episode(
        cls,
        run: object,
        *,
        decision_time_utc: str,
    ) -> "SealedRuling":
        """Re-execute an episode's protected source snapshot before sealing.

        This verifies consistency of the public receipt, precommitted graph,
        parsed action, event, and private verdict against the current local
        kernel. It does **not** establish a genuine model call, honest host,
        trusted timestamps or external evidence provenance. Only the host may
        access ``protected_evaluation``; never pass it to a participant.
        """
        from .causal_model_episode_v1 import CausalEpisodeRun
        from .causal_reality_v1 import CausalBlueprint, CausalWorld, MechanismAction
        from .open_action_v1 import parse_open_action

        if type(run) is not CausalEpisodeRun:
            raise AppealError("exact CausalEpisodeRun required")
        receipt = run.public_receipt
        protected = run.protected_evaluation
        if type(receipt) is not dict or type(protected) is not dict:
            raise AppealError("invalid episode receipt types")

        # Normalize source receipt hashing exactly as the episode producer.
        def source_digest(record: object) -> str:
            wire = json.dumps(record, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode("ascii")
            return sha256(wire).hexdigest()

        public_without_digest = dict(receipt)
        public_without_digest.pop("receipt_sha256", None)
        if (
            receipt.get("receipt_sha256") != source_digest(public_without_digest)
            or protected.get("public_receipt_sha256") != receipt.get("receipt_sha256")
        ):
            raise AppealError("public/protected episode receipt digest inconsistent")
        if (
            receipt.get("schema") != "labyrinth.causal-model-episode.v1"
            or protected.get("schema") != "labyrinth.causal-model-episode.v1.protected"
        ):
            raise AppealError("episode schema binding inconsistent")
        if receipt.get("intervention_status") != "EXECUTED":
            raise AppealError("unresolved text cannot be scored as a physical event")
        proposal = receipt.get("proposal_and_rationale_as_submitted")
        _text(proposal, "proposal_and_rationale_as_submitted", 2048)
        if receipt.get("model_response_sha256") != source_digest(proposal):
            raise AppealError("model proposal digest inconsistent")

        blueprint_data = protected.get("blueprint")
        if type(blueprint_data) is not dict:
            raise AppealError("protected blueprint unavailable")
        try:
            blueprint = CausalBlueprint.from_seed(
                blueprint_data["seed"], blueprint_data["challenge_class"]
            )
        except (ValueError, KeyError, TypeError) as error:
            raise AppealError("invalid protected deterministic blueprint") from error
        if (
            asdict(blueprint) != blueprint_data
            or blueprint.sha256 != protected.get("blueprint_sha256")
        ):
            raise AppealError("protected blueprint diverges from precommitted source")
        world = CausalWorld(blueprint)
        observation = world.observe()
        if receipt.get("public_observation_sha256") != source_digest(asdict(observation)):
            raise AppealError("public observation mismatch against source")
        resolution = parse_open_action(proposal, observation.visible_objects)
        if resolution.status != "RESOLVED" or resolution.action is None:
            raise AppealError("raw model proposal does not resolve to a physical action")
        if protected.get("resolution") != asdict(resolution):
            raise AppealError("protected parser result diverges from raw proposal")
        expected_action = {
            "verb": resolution.action.verb, "target": resolution.action.target,
        }
        expected_public_resolution = {
            "status": "RESOLVED", "reason": resolution.reason,
            "interpreted_action": expected_action,
        }
        if receipt.get("resolution") != expected_public_resolution:
            raise AppealError("public parser result diverges from raw proposal")
        full_event = protected.get("full_source_event")
        if type(full_event) is not dict or type(full_event.get("action")) is not dict:
            raise AppealError("missing protected committed event")
        try:
            action = MechanismAction(**expected_action)
            event = world.step(action)
        except (TypeError, ValueError) as error:
            raise AppealError("invalid episode action for protected world") from error
        if asdict(event) != full_event:
            raise AppealError("protected physical event cannot be replayed")
        if asdict(world.private_verdict()) != protected.get("private_verdict"):
            raise AppealError("protected private verdict does not match source replay")
        if asdict(world.state) != protected.get("world_state"):
            raise AppealError("protected state does not match source replay")
        public_event = receipt.get("source_event")
        interpreted = receipt.get("resolution")
        if (
            type(public_event) is not dict
            or type(interpreted) is not dict
            or public_event.get("event_sha256") != event.event_sha256
            or public_event.get("observed_outcome") != event.observed_outcome
            or public_event.get("action") != asdict(action)
            or interpreted != expected_public_resolution
        ):
            raise AppealError("public event/interpretation cannot be matched to replay")
        return cls.from_causal_world(
            world, public_rationale=proposal, decision_time_utc=decision_time_utc
        )


@dataclass(frozen=True, slots=True)
class AppealPetition:
    """Only authority given to the participant; contains no requested verdict."""

    original_seal: str
    ground: str
    argument: str
    public_evidence_sha256: tuple[str, ...] = ()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate_fields()
        object.__setattr__(self, "digest", _hash_payload("appeal-petition", _data(self, exclude=("digest",))))

    def _validate_fields(self) -> None:
        _sha(self.original_seal, "original_seal")
        _enum(self.ground, GROUNDS, "ground")
        _text(self.argument, "argument", 2048)
        if not isinstance(self.public_evidence_sha256, tuple) or len(self.public_evidence_sha256) > 8:
            raise AppealError("public_evidence_sha256 must be a tuple with at most 8 digests")
        for entry in self.public_evidence_sha256:
            _sha(entry, "public_evidence_sha256 entry")
        if len(set(self.public_evidence_sha256)) != len(self.public_evidence_sha256):
            raise AppealError("duplicate public evidence digest")

    def verify(self) -> None:
        self._validate_fields()
        _verified_digest(self, "appeal-petition", "digest", exclude=("digest",))


@dataclass(frozen=True, slots=True)
class IndependentReview:
    """An *asserted* external review; not proof of an independent execution.

    The host must bind a genuinely separately authorized reviewer callback and
    verify evidence externally.  The petitioner never supplies this object to
    :meth:`MinotaurAppealCourt.submit`.
    """

    original_seal: str
    petition_digest: str
    world_sha256: str
    action_sha256: str
    event_sha256: str
    public_observation_sha256: str
    public_rationale_sha256: str
    decision_time_utc: str
    ground: str
    defect: str
    consequence_verified: bool | None
    reasoning_verified: str
    review_evidence_sha256: str
    reviewer_id: str
    evidence_class: str = EVIDENCE_CLASS
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate_fields()
        object.__setattr__(self, "digest", _hash_payload("minotaur-review", _data(self, exclude=("digest",))))

    def _validate_fields(self) -> None:
        for name in (
            "original_seal", "petition_digest", "world_sha256", "action_sha256", "event_sha256",
            "public_observation_sha256", "public_rationale_sha256", "review_evidence_sha256",
        ):
            _sha(getattr(self, name), name)
        if not isinstance(self.decision_time_utc, str) or not _UTC_TIME.fullmatch(self.decision_time_utc):
            raise AppealError("invalid review decision_time_utc")
        try:
            datetime.fromisoformat(self.decision_time_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise AppealError("invalid review calendar timestamp") from error
        _enum(self.ground, GROUNDS, "ground")
        _enum(self.defect, DEFECT, "defect")
        _enum(self.reasoning_verified, REASONING, "reasoning_verified")
        if self.consequence_verified is not None and type(self.consequence_verified) is not bool:
            raise AppealError("consequence_verified must be bool or None")
        _text(self.reviewer_id, "reviewer_id", 128)
        if self.evidence_class != EVIDENCE_CLASS:
            raise AppealError("review may not claim attested authority")

    def verify(self) -> None:
        self._validate_fields()
        _verified_digest(self, "minotaur-review", "digest", exclude=("digest",))


@dataclass(frozen=True, slots=True)
class FinalRuling:
    """Final outcome of *evaluation*, without modifying physical event history."""

    original_seal: str
    petition_digest: str
    review_digest: str
    world_sha256: str
    action_sha256: str
    event_sha256: str
    state: str
    consequence: str
    reasoning: str
    explanation_code: str
    evidence_class: str = EVIDENCE_CLASS
    seal: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate_fields()
        object.__setattr__(self, "seal", _hash_payload("final-ruling", _data(self, exclude=("seal",))))

    def _validate_fields(self) -> None:
        for name in (
            "original_seal", "petition_digest", "review_digest", "world_sha256",
            "action_sha256", "event_sha256",
        ):
            _sha(getattr(self, name), name)
        _enum(self.state, FINAL_STATES, "state")
        _enum(self.consequence, OUTCOMES, "consequence")
        _enum(self.reasoning, REASONING, "reasoning")
        _text(self.explanation_code, "explanation_code", 96)
        if self.evidence_class != EVIDENCE_CLASS:
            raise AppealError("final ruling may not claim external attestation")

    def verify(self) -> None:
        self._validate_fields()
        _verified_digest(self, "final-ruling", "seal", exclude=("seal",))


ReviewCallback = Callable[[SealedRuling, AppealPetition], IndependentReview]


class MinotaurAppealCourt:
    """Host-owned one-shot appeal authority, final for this evaluation only.

    The participant has only :meth:`submit`; they cannot submit a preselected
    review or mutate the physical event.  The *host* injects the reviewer.
    Reviewer authenticity, verification of source evidence, and training-policy
    authority remain external, non-claims of this source-only component.
    """

    def __init__(self, original: SealedRuling, *, reviewer: ReviewCallback) -> None:
        if type(original) is not SealedRuling:
            raise AppealError("original must be SealedRuling")
        original.verify()
        if not callable(reviewer):
            raise AppealError("reviewer must be a separate host-configured callback")
        self._original = original
        self._original_seal = original.seal
        self._reviewer = reviewer
        self._final: FinalRuling | None = None

    @property
    def final(self) -> FinalRuling | None:
        return self._final

    def submit(self, petition: AppealPetition) -> FinalRuling:
        self._original.verify()
        if self._original.seal != self._original_seal:
            raise AppealError("original ruling modified after court creation")
        if type(petition) is not AppealPetition:
            raise AppealError("submit only accepts an AppealPetition")
        petition.verify()
        if petition.original_seal != self._original_seal:
            raise AppealError("appeal refers to a different initial ruling")

        if self._final is not None:
            self._final.verify()
            if petition.digest == self._final.petition_digest:
                return self._final  # exactly-once review, stable idempotent replay
            raise AppealFinalityError("Minotaur already issued a final decision")

        review = self._reviewer(self._original, petition)
        self._original.verify()  # reviewer must not alter the sealed source case
        if self._original.seal != self._original_seal:
            raise AppealError("reviewer altered sealed source case")
        self._validate_review(review, petition)
        self._final = self._adjudicate(petition, review)
        return self._final

    def _validate_review(self, review: IndependentReview, petition: AppealPetition) -> None:
        if type(review) is not IndependentReview:
            raise AppealError("reviewer must return an IndependentReview")
        review.verify()
        initial = self._original
        required_pairs = (
            (review.original_seal, initial.seal),
            (review.petition_digest, petition.digest),
            (review.world_sha256, initial.world_sha256),
            (review.action_sha256, initial.action_sha256),
            (review.event_sha256, initial.event_sha256),
            (review.public_observation_sha256, initial.public_observation_sha256),
            (review.public_rationale_sha256, initial.public_rationale_sha256),
            (review.decision_time_utc, initial.decision_time_utc),
            (review.ground, petition.ground),
        )
        if any(actual != expected for actual, expected in required_pairs):
            raise AppealError("review not bound to exact precommitted case and petition")

    def _adjudicate(self, petition: AppealPetition, review: IndependentReview) -> FinalRuling:
        original = self._original
        if review.defect == "REFUTED":
            state, consequence, reasoning, why = (
                "UPHELD", original.consequence, original.reasoning, "APPEAL_GROUND_REFUTED"
            )
        elif review.defect == "INDETERMINATE":
            state, consequence, reasoning, why = (
                "UNRESOLVED", original.consequence, original.reasoning,
                "DEFECT_EVIDENCE_INSUFFICIENT"
            )
        elif petition.ground == "SCORING_ERROR":
            # These are distinct grades: either one may be corrected alone.
            consequence = (
                "PASS" if review.consequence_verified else "FAIL"
            ) if review.consequence_verified is not None else original.consequence
            reasoning = (
                review.reasoning_verified if review.reasoning_verified != "INDETERMINATE"
                else original.reasoning
            )
            if consequence != original.consequence or reasoning != original.reasoning:
                state, why = "CORRECTED", "VERIFIED_SCORING_CORRECTION"
            elif review.consequence_verified is None and review.reasoning_verified == "INDETERMINATE":
                state, why = "UNRESOLVED", "SCORING_EVIDENCE_INSUFFICIENT"
            else:
                state, why = "UPHELD", "SCORE_ALREADY_CONSISTENT"
        else:
            # A verified flaw in parsing/mechanics or insufficient available
            # evidence does not rewrite the already produced physical events.
            state, consequence = "UNRESOLVED", original.consequence
            reasoning = review.reasoning_verified
            why = "CAUSAL_OUTCOME_NOT_REWRITABLE"

        return FinalRuling(
            original_seal=original.seal,
            petition_digest=petition.digest,
            review_digest=review.digest,
            world_sha256=original.world_sha256,
            action_sha256=original.action_sha256,
            event_sha256=original.event_sha256,
            state=state,
            consequence=consequence,
            reasoning=reasoning,
            explanation_code=why,
        )
