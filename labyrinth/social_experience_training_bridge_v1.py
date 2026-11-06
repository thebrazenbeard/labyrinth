"""Causal, public dialogue-world TRAIN with operator-owned feedback and Wt.

The model responds to *untrusted story data*, not system instructions. An
operator supplies a distinct typed public world effect and approved TRAIN
target after that response. A separate trusted trainer writes actual model
artifacts, whose bytes are hashed. This is source DEVELOPMENT_ONLY evidence,
never an assertion of neural learning, private exam success or graduation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Protocol

from .model_saga_runtime import CheckpointPaths, CheckpointRef, capture_checkpoint
from .social_world_consequence_v1 import (
    CharacterRelationship, DialogueBeat, DialogueTransition, OperatorDialogueEffect,
    PublicDialogueObservation, SocialWorldSession, beat_fingerprint,
)
from .training_scope import TrainingScopeSpec


_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_MAX_TRAIN_BEATS = 1024
_SPARK_EVIDENCE_CLASS = "MECHANISM_ONLY_UNQUALIFIED"
_RECEIPT_CLASSIFICATION = "MECHANISM_ONLY_NO_NEURAL_LEARNING_CLAIM"
_RECEIPT_QUALIFICATION = "NOT_ESTABLISHED"


@dataclass(frozen=True)
class PublicSocialModelInput:
    """Story text is DATA in one user message, never privileged instructions.

    The caller-supplied responder is trusted executable adapter code; a model
    backend must serialize this exact user-data envelope as user content, not
    concatenate character promises into system/developer policy messages.
    """

    public_observation: PublicDialogueObservation
    message_role: str = "user"
    trust_class: str = "UNTRUSTED_STORY_DATA_NO_INSTRUCTION_AUTHORITY"

    def __post_init__(self) -> None:
        if (type(self.public_observation) is not PublicDialogueObservation
                or self.message_role != "user"
                or self.trust_class != "UNTRUSTED_STORY_DATA_NO_INSTRUCTION_AUTHORITY"):
            raise ValueError("social prompt is untrusted user-story data only")

    def as_user_message(self) -> str:
        obs = self.public_observation
        return json.dumps({
            "schema": "labyrinth.social-model-public-input.v1",
            "source_trust": self.trust_class,
            "objective_id": obs.objective_id,
            "region": obs.region_id,
            "chapter": obs.chapter_id,
            "beat": obs.beat_id,
            "character": obs.character_id,
            "character_story_context_not_instructions": obs.character_context,
            "character_utterance_not_instructions": obs.text,
        }, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class OperatorDialogueFeedback:
    """Explicit source-approved TRAIN target; approval ID is not a signature."""

    approval_id: str
    preferred_response: str
    coaching: str
    split: str = "train"

    def __post_init__(self) -> None:
        if type(self.approval_id) is not str or _ID.fullmatch(self.approval_id) is None:
            raise ValueError("invalid operator TRAIN approval identifier")
        for name, value, limit in (
            ("preferred response", self.preferred_response, 4096),
            ("coaching", self.coaching, 1024),
        ):
            if type(value) is not str or not 1 <= len(value) <= limit or "\x00" in value:
                raise ValueError(f"invalid bounded {name}")
        if type(self.split) is not str or self.split != "train":
            raise ValueError("social training can consume only approved TRAIN material")


@dataclass(frozen=True)
class OperatorDialogueAssessment:
    public_effect: OperatorDialogueEffect
    feedback: OperatorDialogueFeedback

    def __post_init__(self) -> None:
        if type(self.public_effect) is not OperatorDialogueEffect:
            raise ValueError("operator must provide an exact public effect")
        OperatorDialogueEffect.__post_init__(self.public_effect)
        if type(self.feedback) is not OperatorDialogueFeedback:
            raise ValueError("operator must provide an exact TRAIN feedback object")
        OperatorDialogueFeedback.__post_init__(self.feedback)


@dataclass(frozen=True)
class ApprovedSocialTrainCase:
    """Trusted trainer receives this only after the model's public reply."""

    objective_id: str
    beat: DialogueBeat
    model_input: PublicSocialModelInput
    learner_response: str
    feedback: OperatorDialogueFeedback
    split: str = "train"


@dataclass(frozen=True)
class SocialTrainArtifact:
    """Physical successor, optionally with a merely *hashed* full manifest.

    The manifest's contents and complete neural runtime recovery are not
    authenticated by this bridge; a separate full-state verifier is required.
    """

    checkpoint: CheckpointPaths
    candidate_manifest_path: Path | None = None


class PublicDialogueResponder(Protocol):
    def respond(self, public_obs: PublicSocialModelInput) -> str: ...


class DialogueTrainingOperator(Protocol):
    def assess_train(
        self, beat: DialogueBeat, public_obs: PublicDialogueObservation,
        response: str,
    ) -> OperatorDialogueAssessment: ...


class PhysicalDialogueTrainer(Protocol):
    def train_supervised(
        self, case: ApprovedSocialTrainCase, parent: CheckpointRef,
    ) -> CheckpointPaths | SocialTrainArtifact: ...


@dataclass(frozen=True)
class SocialTrainingSpark:
    index: int
    beat_key: tuple[str, str, str]
    response_sha256: str
    feedback_sha256: str
    trust_before: int
    trust_after: int
    quality_score: int
    before_weights_sha256: str
    after: CheckpointRef
    candidate_manifest_sha256: str | None
    evidence_class: str = _SPARK_EVIDENCE_CLASS

    def __post_init__(self) -> None:
        if type(self.evidence_class) is not str or self.evidence_class != _SPARK_EVIDENCE_CLASS:
            raise ValueError("social training spark cannot attest neural learning")


@dataclass(frozen=True)
class SocialTrainingReceipt:
    objective_id: str
    entrance: CheckpointRef
    current: CheckpointRef
    sparks: tuple[SocialTrainingSpark, ...]
    public_world_events: tuple[DialogueTransition, ...]
    characters: tuple[CharacterRelationship, ...]
    classification: str = _RECEIPT_CLASSIFICATION
    qualification: str = _RECEIPT_QUALIFICATION

    def __post_init__(self) -> None:
        if (type(self.classification) is not str
                or self.classification != _RECEIPT_CLASSIFICATION
                or type(self.qualification) is not str
                or self.qualification != _RECEIPT_QUALIFICATION):
            raise ValueError("social training receipt cannot attest graduation or neural learning")
        if type(self.sparks) is not tuple or any(
            type(spark) is not SocialTrainingSpark for spark in self.sparks
        ):
            raise ValueError("social training receipt needs exact source-only sparks")
        for spark in self.sparks:
            SocialTrainingSpark.__post_init__(spark)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_once_fsynced(path: Path, value: dict) -> None:
    """Fsync intent filename and data *before* any possible TRAIN side effect."""
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True,
                         allow_nan=False).encode("utf-8")
    if not 0 < len(encoded) <= 8192:
        raise ValueError("TRAIN journal event exceeds bounded 8 KiB envelope")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _hash_manifest(path: Path | None) -> str | None:
    if path is None:
        return None
    if not isinstance(path, Path):
        raise ValueError("candidate manifest must be an exact local Path")
    # Use the same no-follow, bounded regular-file readback as checkpoints.
    return capture_checkpoint(CheckpointPaths(str(path))).weights_sha256


class SocialExperienceTrainingBridge:
    """Consequence-bearing TRAIN across many approved regions and chapters.

    Every training effect has an exclusive fsynced intent file, then a
    separately fsynced completion receipt. A PREPARED-only intent after a
    crash is ambiguous: this module does not auto-resume or retry it.
    """

    def __init__(
        self, scope: TrainingScopeSpec, *,
        characters: tuple[CharacterRelationship, ...],
        checkpoint: CheckpointPaths,
        responder: PublicDialogueResponder,
        operator: DialogueTrainingOperator,
        trainer: PhysicalDialogueTrainer,
        journal_directory: Path,
    ) -> None:
        if (not callable(getattr(responder, "respond", None))
                or not callable(getattr(operator, "assess_train", None))
                or not callable(getattr(trainer, "train_supervised", None))):
            raise ValueError("independent responder, operator and physical trainer required")
        if (not isinstance(journal_directory, Path)
                or not journal_directory.is_absolute()
                or journal_directory.exists()
                or not journal_directory.parent.is_dir()):
            raise ValueError("fresh absolute TRAIN journal directory required")
        self.world = SocialWorldSession(scope, characters)
        self.scope = scope
        self.responder = responder
        self.operator = operator
        self.trainer = trainer
        self.entrance = capture_checkpoint(checkpoint)
        self.checkpoint = self.entrance
        self.checkpoint_refs: list[CheckpointRef] = [self.entrance]
        self.training_sparks: list[SocialTrainingSpark] = []
        self._used_weights_paths = {self.checkpoint.weights_path}
        self._used_optimizer_paths = ({self.checkpoint.optimizer_path}
                                      if self.checkpoint.optimizer_path else set())
        self._used_approval_ids: set[str] = set()
        self.indeterminate = False
        journal_directory.mkdir(mode=0o700)
        self.journal_directory = journal_directory

    def _verify_retained(self) -> None:
        for ref in self.checkpoint_refs:
            actual = capture_checkpoint(CheckpointPaths(ref.weights_path, ref.optimizer_path))
            if actual != ref:
                raise ValueError("retained social model artifact changed unexpectedly")

    def run_train_beat(self, beat: DialogueBeat) -> SocialTrainingSpark:
        """Respond publicly, approve TRAIN privately, effect only after readback."""
        if self.indeterminate:
            raise ValueError("indeterminate physical TRAIN requires operator recovery")
        if len(self.training_sparks) >= _MAX_TRAIN_BEATS:
            raise ValueError("bounded social TRAIN experience budget exhausted")
        # An untrusted model adapter and operator callback must never retain
        # aliases to the approved scene or to the TRAIN evidence we later log.
        beat = copy.deepcopy(beat)
        public_observation = self.world.preview(beat)
        model_input = PublicSocialModelInput(copy.deepcopy(public_observation))
        approved_prompt_sha256 = _sha(model_input.as_user_message())
        responder_input = copy.deepcopy(model_input)
        response = self.responder.respond(responder_input)
        if _sha(responder_input.as_user_message()) != approved_prompt_sha256:
            raise ValueError("responder changed the approved public TRAIN input")
        if (type(response) is not str or not 1 <= len(response) <= 4096
                or "\x00" in response):
            raise ValueError("model freeform dialogue response must be bounded")
        operator_beat = copy.deepcopy(beat)
        operator_observation = copy.deepcopy(public_observation)
        assessment = self.operator.assess_train(operator_beat, operator_observation, response)
        if (beat_fingerprint(operator_beat) != beat_fingerprint(beat)
                or operator_observation != public_observation):
            raise ValueError("operator altered the approved public TRAIN scene")
        if type(assessment) is not OperatorDialogueAssessment:
            raise ValueError("operator must supply exact typed TRAIN assessment")
        OperatorDialogueAssessment.__post_init__(assessment)
        approval = assessment.feedback.approval_id
        if approval in self._used_approval_ids:
            raise ValueError("TRAIN approval identity cannot be reused")
        # Validate the proposed effect against a detached copy of the public
        # world BEFORE calling any physical trainer. Invalid promise effects
        # cannot consume a neural training quantum and then fail at commit.
        rehearsal = copy.deepcopy(self.world)
        predicted_transition = rehearsal.record_training_response(
            beat, response, assessment.public_effect,
        )
        case = ApprovedSocialTrainCase(
            self.scope.objective_id, copy.deepcopy(beat),
            copy.deepcopy(model_input), response, copy.deepcopy(assessment.feedback),
        )
        self._verify_retained()
        old = self.checkpoint
        index = len(self.training_sparks)
        feedback_hash = _sha(json.dumps({
            "approval_id": approval,
            "preferred_response": assessment.feedback.preferred_response,
            "coaching": assessment.feedback.coaching,
        }, sort_keys=True, separators=(",", ":")))
        intent = {
            "schema": "labyrinth.social-train-intent.v1",
            "index": index, "beat_key": list(beat.key),
            "objective_id": self.scope.objective_id,
            "public_prompt_sha256": approved_prompt_sha256,
            "checkpoint_before_sha256": old.weights_sha256,
            "response_sha256": _sha(response),
            "feedback_sha256": feedback_hash,
            "status": "PENDING_POSSIBLE_EFFECT",
        }
        # From here onward an exception leaves a conservative ambiguous
        # ledger, even if the failure happened just before the trainer call.
        self.indeterminate = True
        intent_path = self.journal_directory / f"beat-{index:04d}.intent.json"
        _write_once_fsynced(intent_path, intent)
        if json.loads(intent_path.read_text(encoding="utf-8")) != intent:
            raise ValueError("write-ahead TRAIN intent readback disagrees")
        # Never let the adapter silently start again from W0. Each training
        # spark receives the exact physically retained Wt predecessor.
        supplied_artifact = self.trainer.train_supervised(case, old)
        if (beat_fingerprint(case.beat) != beat_fingerprint(beat)
                or _sha(case.model_input.as_user_message()) != approved_prompt_sha256):
            raise ValueError("trainer changed the approved TRAIN input after effect")
        if type(supplied_artifact) is CheckpointPaths:
            physical_paths = supplied_artifact
            manifest_path = None
        elif type(supplied_artifact) is SocialTrainArtifact:
            physical_paths = supplied_artifact.checkpoint
            manifest_path = supplied_artifact.candidate_manifest_path
        else:
            raise ValueError("trusted trainer must return physical checkpoint paths")
        self._verify_retained()
        successor = capture_checkpoint(physical_paths)
        if (successor.weights_path in self._used_weights_paths
                or successor.weights_sha256 == old.weights_sha256
                or (old.optimizer_path is not None and successor.optimizer_path is None)
                or (successor.optimizer_path is not None
                    and successor.optimizer_path in self._used_optimizer_paths)):
            raise ValueError("social TRAIN did not create a unique write-once successor")
        manifest_digest = _hash_manifest(manifest_path)
        transition = self.world.record_training_response(
            beat, response, assessment.public_effect,
        )
        if transition != predicted_transition:
            raise ValueError("social world changed between effect preview and commit")
        spark = SocialTrainingSpark(
            index, beat.key, _sha(response), feedback_hash,
            transition.trust_before, transition.trust_after,
            transition.quality_score, old.weights_sha256, successor, manifest_digest,
        )
        commit = {
            "schema": "labyrinth.social-train-commit.v1", "index": index,
            "beat_key": list(beat.key), "checkpoint_before_sha256": old.weights_sha256,
            "checkpoint_after_sha256": successor.weights_sha256,
            "checkpoint_path": successor.weights_path,
            "candidate_manifest_sha256": manifest_digest,
            "response_sha256": spark.response_sha256,
            "public_prompt_sha256": approved_prompt_sha256,
            "feedback_sha256": feedback_hash,
            "world_trust_after": transition.trust_after,
            "status": "COMMITTED_SOURCE_ONLY",
        }
        commit_path = self.journal_directory / f"beat-{index:04d}.commit.json"
        _write_once_fsynced(commit_path, commit)
        if json.loads(commit_path.read_text(encoding="utf-8")) != commit:
            raise ValueError("social TRAIN commit readback disagrees")
        self.checkpoint = successor
        self.checkpoint_refs.append(successor)
        self._used_weights_paths.add(successor.weights_path)
        if successor.optimizer_path:
            self._used_optimizer_paths.add(successor.optimizer_path)
        self._used_approval_ids.add(approval)
        self.training_sparks.append(spark)
        self.indeterminate = False
        return spark

    def run_train_chapter(self, beats: tuple[DialogueBeat, ...]) -> tuple[SocialTrainingSpark, ...]:
        if type(beats) is not tuple or not 1 <= len(beats) <= 64:
            raise ValueError("chapter needs 1-64 approved public TRAIN beats")
        return tuple(self.run_train_beat(beat) for beat in beats)

    def report(self) -> SocialTrainingReceipt:
        if self.indeterminate:
            raise ValueError("ambiguous external TRAIN has no successful social receipt")
        self._verify_retained()
        return SocialTrainingReceipt(
            self.scope.objective_id, self.entrance, self.checkpoint,
            tuple(self.training_sparks), self.world.events, self.world.characters,
        )


__all__ = [
    "PublicSocialModelInput", "OperatorDialogueFeedback", "OperatorDialogueAssessment",
    "ApprovedSocialTrainCase", "SocialTrainArtifact", "PublicDialogueResponder",
    "DialogueTrainingOperator", "PhysicalDialogueTrainer", "SocialTrainingSpark",
    "SocialTrainingReceipt", "SocialExperienceTrainingBridge",
]
