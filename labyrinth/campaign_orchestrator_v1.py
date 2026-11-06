"""Source-only social TRAIN -> physical ancestry -> Minotaur development loop.

Trusted operator-provided callbacks supply the model, trainer, scene director,
and *external* examination records. This is not a model runtime, safe Python
sandbox, authenticated examiner, independent evaluation or neural proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

from .ariadne_recovery_v1 import AriadneJournal, trainer_request
from .campaign_full_state_v1 import verify_campaign_state
from .conversation_objective_v1 import ConversationObjectiveProvider
from .daedalus_social_director_v1 import DaedalusSocialPlanner, SocialScenePolicy
from .evolutionary_population_v1 import (
    Candidate, Decision, EvaluationPlan, Examination, MutationKind, Population,
    deterministic_seed, prepare_population, select_generation,
)
from .model_saga_runtime import CheckpointPaths, CheckpointRef, capture_checkpoint
from .social_experience_training_bridge_v1 import (
    DialogueTrainingOperator, PhysicalDialogueTrainer, PublicDialogueResponder,
    SocialExperienceTrainingBridge, SocialTrainingReceipt,
)
from .social_world_consequence_v1 import (
    CharacterRelationship, DialogueBeat, PublicPromise,
)
from .training_scope import TrainingScopeSpec


_CLASS = "SOURCE_ONLY_SOCIAL_CAMPAIGN_NO_NEURAL_OR_INDEPENDENT_EXAM_PROOF"


class CampaignBoundaryError(ValueError):
    """An unverified or incompatible cross-module campaign boundary."""


@dataclass(frozen=True)
class CandidateLaunch:
    candidate_id: str
    case_id: str
    character_id: str
    region_id: str
    chapter_id: str
    director: object
    responder: PublicDialogueResponder
    operator: DialogueTrainingOperator
    trainer: PhysicalDialogueTrainer
    manifest_builder: Callable[[SocialTrainingReceipt, dict], Path]
    bridge_journal_directory: Path
    trainer_id: str
    max_beats: int = 1


@dataclass(frozen=True)
class CandidateExperience:
    candidate_id: str
    manifest_path: Path
    checkpoint: CheckpointRef
    social_receipt: SocialTrainingReceipt
    ariadne_commit_sha256: str
    submitted_beats: tuple[DialogueBeat, ...]


@dataclass(frozen=True)
class GenerationPendingExam:
    generation_number: int
    population: Population
    experiences: tuple[CandidateExperience, ...]
    evidence_class: str = _CLASS

    def __post_init__(self) -> None:
        if type(self.evidence_class) is not str or self.evidence_class != _CLASS:
            raise CampaignBoundaryError("campaign pending exam cannot attest neural or independent proof")


@dataclass(frozen=True)
class CampaignGenerationResult:
    decision: Decision
    journal_head_sha256: str
    selected_characters: tuple[CharacterRelationship, ...]
    evidence_class: str = _CLASS

    def __post_init__(self) -> None:
        if type(self.evidence_class) is not str or self.evidence_class != _CLASS:
            raise CampaignBoundaryError("campaign result cannot attest neural or independent proof")


def _manifest(path: Path) -> dict:
    if not isinstance(path, Path) or not path.is_absolute():
        raise CampaignBoundaryError("manifest requires an absolute canonical path")
    observed = verify_campaign_state(path, fresh_process=True)
    manifest = json.loads(path.read_bytes())
    if (manifest["state_sha256"] != observed["state_sha256"]
            or manifest["manifest_sha256"] != observed["manifest_sha256"]):
        raise CampaignBoundaryError("manifest changed during physical readback")
    return manifest


def _checkpoint_from_manifest(manifest: dict) -> CheckpointPaths:
    """Narrow single-file TRAIN view; the manifest still hashes all components."""
    role = "adapter" if manifest["mutation_kind"] == "lora_adapter" else "base"
    weights = [name for name in manifest[role]["files"] if name.endswith(".safetensors")]
    optimizer = list(manifest["optimizer"]["files"])
    if len(weights) != 1 or len(optimizer) != 1:
        raise CampaignBoundaryError(
            "source bridge supports one physical weight/adapter and optimizer file; "
            "no full-shard trainer loader is present")
    return CheckpointPaths(
        str(Path(manifest[role]["root"]) / weights[0]),
        str(Path(manifest["optimizer"]["root"]) / optimizer[0]),
    )


def _bind_checkpoint(manifest: dict, ref: CheckpointRef) -> None:
    """Require the *actual* bridge output to be inventoried in the HF genome."""
    role = "adapter" if manifest["mutation_kind"] == "lora_adapter" else "base"
    for path, digest, part in (
        (ref.weights_path, ref.weights_sha256, role),
        (ref.optimizer_path, ref.optimizer_sha256, "optimizer"),
    ):
        if path is None or digest is None:
            raise CampaignBoundaryError("physical TRAIN checkpoint requires optimizer state")
        root = Path(manifest[part]["root"])
        try:
            relative = Path(path).relative_to(root).as_posix()
        except ValueError as exc:
            raise CampaignBoundaryError("TRAIN checkpoint not inside full-state inventory") from exc
        item = manifest[part]["files"].get(relative)
        if item is None or item["sha256"] != digest:
            raise CampaignBoundaryError("TRAIN output bytes differ from full-state manifest")


def _world_state(characters: tuple[CharacterRelationship, ...]) -> dict:
    return {"characters": [{
        "character_id": c.character_id, "trust": c.trust,
        "interactions": c.interactions,
        "promises": [{"promise_id": p.promise_id, "description": p.description}
                     for p in c.outstanding_promises],
    } for c in characters]}


def _hash_public(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


class CampaignOrchestrator:
    """One active, fail-closed development campaign in a single host process.

    The journal can be inspected after process death, but resuming training
    requires an independently proven full-model loader; it is not offered here.
    No trainer is called until Ariadne has fsynced its PREPARED permit.
    """

    def __init__(self, *, journal: AriadneJournal, origin_manifest_path: Path,
                 scope: TrainingScopeSpec, provider: ConversationObjectiveProvider,
                 characters: tuple[CharacterRelationship, ...],
                 world_id: str = "world_1") -> None:
        if (type(journal) is not AriadneJournal
                or type(scope) is not TrainingScopeSpec
                or type(provider) is not ConversationObjectiveProvider
                or provider.objective_id != scope.objective_id
                or type(characters) is not tuple or not characters
                or any(type(c) is not CharacterRelationship for c in characters)):
            raise CampaignBoundaryError("operator scope, Vera TRAIN provider, world and journal required")
        scope.__post_init__()
        origin = _manifest(origin_manifest_path)
        if origin["generation_index"] != 0:
            raise CampaignBoundaryError("campaign W0 must be the physical founder")
        _bind_checkpoint(origin, capture_checkpoint(_checkpoint_from_manifest(origin)))
        self.journal = journal
        self.origin_manifest_path = origin_manifest_path
        self.scope = scope
        self.provider = provider
        self.selected_characters = characters
        self.current_world_id = world_id
        self.world_version = 1
        self._pending: GenerationPendingExam | None = None
        self._blocked = False
        self._finished_generations = 0
        journal.initialize(
            w0_manifest_path=origin_manifest_path, world_id=world_id, version=1,
            curriculum={"objective_id": scope.objective_id,
                        "train_only": True, "private_gold_included": False},
            world_state=_world_state(characters), idempotency_key="campaign_genesis")

    def run_generation(self, *, launches: tuple[CandidateLaunch, ...],
                       control_manifest_path: Path,
                       evaluation_plan: EvaluationPlan) -> GenerationPendingExam:
        """Apply approved social TRAIN to sibling physical states, then STOP.

        No examination callback is invoked. The operator must separately
        provide exact `Examination` records to `finalize_generation`.
        """
        if self._blocked or self._pending is not None:
            raise CampaignBoundaryError("campaign has pending or indeterminate work")
        if (type(launches) is not tuple or not 2 <= len(launches) <= 8
                or any(type(launch) is not CandidateLaunch for launch in launches)
                or len({launch.candidate_id for launch in launches}) != len(launches)):
            raise CampaignBoundaryError("two to eight distinct operator candidate launches required")
        self._blocked = True  # Crash/replay cannot silently duplicate effects.
        descriptor = self.journal.continuation_descriptor()
        if descriptor["status"] != "READY_FOR_EXTERNAL_FULL_STATE_LOAD":
            raise CampaignBoundaryError("Ariadne has indeterminate physical TRAIN")
        parent_path = Path(descriptor["selected_parent_manifest_path"])
        parent = _manifest(parent_path)
        origin = _manifest(self.origin_manifest_path)
        if parent["origin_state_sha256"] != origin["state_sha256"]:
            raise CampaignBoundaryError("Ariadne current parent detached from exact W0")
        parent_checkpoint = _checkpoint_from_manifest(parent)
        _bind_checkpoint(parent, capture_checkpoint(parent_checkpoint))
        generation = self._finished_generations + 1
        if descriptor["next_generation_index"] != self._finished_generations:
            raise CampaignBoundaryError("development generation disagrees with selected Ariadne ancestry")
        experiences: list[CandidateExperience] = []
        candidates: list[Candidate] = []
        for launch in launches:
            if (not callable(launch.manifest_builder)
                    or type(launch.max_beats) is not int
                    or not 1 <= launch.max_beats <= 12):
                raise CampaignBoundaryError("bounded trusted manifest builder and scene count required")
            # Source-owned provider admits TRAIN cases only. No evaluator case
            # or reference claim is passed into director, responder or trainer.
            approved = self.provider.public_train_beats(
                launch.case_id, launch.character_id,
                region_id=launch.region_id, chapter_id=launch.chapter_id)
            allowed = set(approved)
            bridge = SocialExperienceTrainingBridge(
                self.scope, characters=self.selected_characters,
                checkpoint=parent_checkpoint,
                responder=launch.responder, operator=launch.operator,
                trainer=launch.trainer,
                journal_directory=launch.bridge_journal_directory)
            planner = DaedalusSocialPlanner(
                bridge.world, launch.director,
                SocialScenePolicy(minimum_difficulty=1, max_scenes=launch.max_beats))
            # Propose before durable TRAIN intent, but only an approved public
            # TRAIN beat may be trained; model-authored code is never executed.
            first = planner.propose_next()
            if first not in allowed:
                raise CampaignBoundaryError("Daedalus proposed unapproved Vera TRAIN case")
            branch_id = "branch_" + launch.candidate_id
            self.journal.record_branch(
                branch_id=branch_id, world_id=self.current_world_id,
                experience={"observations": [
                    {"case_id": launch.case_id, "beat_key": list(first.key)}],
                    "actions": ["public_dialogue_train"], "outcomes": []},
                selected_path=[first.region_id, first.chapter_id, first.beat_id],
                idempotency_key="branch_" + launch.candidate_id)
            permit = self.journal.prepare_training(
                candidate_id=launch.candidate_id, branch_id=branch_id,
                trainer_request=trainer_request(
                    launch.trainer_id,
                    parameters={"source_only": True,
                                "scope_fingerprint": self.scope.fingerprint,
                                "public_training_seed": deterministic_seed(
                                    parent["state_sha256"], generation,
                                    launch.candidate_id)}),
                idempotency_key="prepare_" + launch.candidate_id)
            scenes = []
            for index in range(launch.max_beats):
                beat = first if index == 0 else planner.propose_next()
                if beat not in allowed or beat in scenes:
                    raise CampaignBoundaryError("Daedalus repeated or escaped approved TRAIN scenes")
                bridge.run_train_beat(beat)
                planner.record_completed(bridge.world.events[-1])
                scenes.append(beat)
            receipt = bridge.report()
            candidate_path = launch.manifest_builder(receipt, parent)
            child = _manifest(candidate_path)
            _bind_checkpoint(child, receipt.current)
            commit = self.journal.commit_training(
                permit, child_manifest_path=candidate_path,
                trainer_receipt={
                    "trainer_id": launch.trainer_id,
                    "effect_status": "completed",
                    "effect_token": permit.effect_token,
                    "details": {
                        "source_only": True, "train_sparks": len(receipt.sparks),
                        "public_world": _world_state(receipt.characters),
                        "checkpoint_sha256": receipt.current.weights_sha256,
                        "bridge_commit_receipt_sha256": _hash_public([
                            spark.after.weights_sha256 for spark in receipt.sparks]),
                    },
                }, idempotency_key="commit_" + launch.candidate_id)
            experiences.append(CandidateExperience(
                launch.candidate_id, candidate_path, receipt.current, receipt,
                commit["event_sha256"], tuple(scenes)))
            candidates.append(Candidate(
                launch.candidate_id, candidate_path,
                MutationKind(parent["mutation_kind"]), launch.trainer_id,
                commit["event_sha256"], parent["state_sha256"],
                deterministic_seed(parent["state_sha256"], generation,
                                   launch.candidate_id)))
        population = prepare_population(
            development_generation=generation,
            origin_manifest_path=self.origin_manifest_path,
            parent_manifest_path=parent_path,
            control_manifest_path=control_manifest_path,
            candidates=tuple(candidates), plan=evaluation_plan)
        pending = GenerationPendingExam(generation, population, tuple(experiences))
        self._pending = pending
        self._blocked = False
        return pending

    def finalize_generation(self, pending: GenerationPendingExam,
                            examinations: tuple[Examination, ...]) -> CampaignGenerationResult:
        """Consume separate operator-provided development receipts and select.

        The process never constructs examinations or private cohorts. A
        successful fixture selection remains unqualified for neural claims.
        """
        if (self._blocked or self._pending is not pending
                or type(pending) is not GenerationPendingExam):
            raise CampaignBoundaryError("generation must be locally pending and unfinalized")
        self._blocked = True
        decision = select_generation(pending.population, examinations)
        by_id = {exam.subject_id: exam for exam in examinations}
        for experience in pending.experiences:
            exam = by_id[experience.candidate_id]
            self.journal.record_evaluation(
                candidate_id=experience.candidate_id,
                evaluation={
                    "metric_id": "source_development_holdout",
                    "score": exam.metrics.holdout, "direction": "higher",
                    "dataset_ref": "sealed_dev_commitment",
                    "evidence": {"source_only": True,
                                 "examiner_run_sha256": exam.examiner_run_sha256,
                                 "cohort_commitment": exam.private_cohort_commitment,
                                 "not_authenticated": True},
                }, idempotency_key="eval_" + experience.candidate_id)
        for rejected in decision.rejected_candidate_ids:
            self.journal.reject_candidate(
                candidate_id=rejected,
                reason="comparative developmental thresholds not selected",
                idempotency_key="reject_" + rejected)
        selected_characters = self.selected_characters
        if decision.winner_id is not None:
            self.journal.select_candidate(
                candidate_id=decision.winner_id,
                selection_reason="source_only_comparative_development_fixture",
                idempotency_key="select_" + decision.winner_id)
            chosen = next(experience for experience in pending.experiences
                          if experience.candidate_id == decision.winner_id)
            selected_characters = chosen.social_receipt.characters
            next_world = f"world_{self.world_version + 1}"
            self.journal.record_world(
                world_id=next_world, parent_world_id=self.current_world_id,
                version=self.world_version + 1,
                curriculum={"objective_id": self.scope.objective_id,
                            "train_only": True},
                world_state=_world_state(selected_characters),
                idempotency_key="world_" + next_world)
            self.world_version += 1
            self.current_world_id = next_world
        self.selected_characters = selected_characters
        self._finished_generations += 1 if decision.winner_id else 0
        self._pending = None
        self._blocked = False
        descriptor = self.journal.continuation_descriptor()
        if descriptor["selected_parent_state_sha256"] != decision.selected_state_sha256:
            self._blocked = True
            raise CampaignBoundaryError("Ariadne selected different physical parent")
        return CampaignGenerationResult(
            decision, descriptor["event_head_sha256"], selected_characters)


__all__ = ["CampaignBoundaryError", "CandidateLaunch", "CandidateExperience",
           "GenerationPendingExam", "CampaignGenerationResult", "CampaignOrchestrator"]
