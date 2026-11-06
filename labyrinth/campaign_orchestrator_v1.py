"""Source-only social TRAIN -> physical ancestry -> Minotaur development loop.

Trusted operator-provided callbacks supply the model, trainer, scene director,
and *external* examination records. This is not a model runtime, safe Python
sandbox, authenticated examiner, independent evaluation or neural proof.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
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
    _check_plan, deterministic_seed, prepare_population, select_generation,
)
from .model_saga_runtime import CheckpointPaths, CheckpointRef, capture_checkpoint
from .social_experience_training_bridge_v1 import (
    DialogueTrainingOperator, PhysicalDialogueTrainer, PublicDialogueResponder,
    SocialExperienceTrainingBridge, SocialTrainingReceipt,
)
from .social_world_consequence_v1 import (
    CharacterRelationship, DialogueBeat, PublicPromise, SocialWorldSession,
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
    for character in characters:
        # A caller may mutate frozen records after session construction.
        # Durable world provenance must not silently accept an invalid story.
        CharacterRelationship.__post_init__(character)
    return {"characters": [{
        "character_id": c.character_id, "trust": c.trust,
        "interactions": c.interactions,
        "promises": [{"promise_id": p.promise_id, "description": p.description}
                     for p in c.outstanding_promises],
        # Omit absent optional memory: preserves old W0/WORLD_VERSION hashes.
        **({"last_public_reaction": c.last_public_reaction}
           if c.last_public_reaction is not None else {}),
    } for c in characters]}


def _hash_public(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def _experience_hash(experience: CandidateExperience) -> str:
    """Snapshot every public receipt and training-identity field at issuance."""
    return _hash_public({
        "candidate_id": experience.candidate_id,
        "manifest_path": str(experience.manifest_path),
        "checkpoint": asdict(experience.checkpoint),
        "social_receipt": asdict(experience.social_receipt),
        "ariadne_commit_sha256": experience.ariadne_commit_sha256,
        "submitted_beats": [asdict(beat) for beat in experience.submitted_beats],
    })


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
        self.selected_characters = copy.deepcopy(characters)
        self._selected_world_hash = _hash_public(_world_state(self.selected_characters))
        self.current_world_id = world_id
        self.world_version = 1
        self.minimum_difficulty = 1
        self.complexity_depth = 0
        self.maze_attempt_index = 1
        self._pending: GenerationPendingExam | None = None
        self._pending_world_hashes: dict[str, str] = {}
        self._pending_experience_hashes: dict[str, str] = {}
        self._pending_experience_ids: tuple[str, ...] = ()
        self._pending_character_copies: dict[str, tuple[CharacterRelationship, ...]] = {}
        self._blocked = False
        self._finished_generations = 0
        journal.initialize(
            w0_manifest_path=origin_manifest_path, world_id=world_id, version=1,
            curriculum={"objective_id": scope.objective_id,
                        "train_only": True, "private_gold_included": False,
                        "minimum_difficulty": self.minimum_difficulty,
                        "complexity_depth": self.complexity_depth},
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
        descriptor = self.journal.continuation_descriptor()
        if descriptor["status"] != "READY_FOR_EXTERNAL_FULL_STATE_LOAD":
            raise CampaignBoundaryError("Ariadne has indeterminate physical TRAIN")
        if descriptor["development_status"] != "NO_ACTIVE_COHORT":
            raise CampaignBoundaryError(
                "Ariadne has an unresolved current-world cohort; finish its "
                "developmental outcome before starting another generation")
        if (descriptor["maze_attempt_index"] != self.maze_attempt_index
                or descriptor["minimum_scene_difficulty"] != self.minimum_difficulty
                or descriptor["complexity_depth"] != self.complexity_depth
                or descriptor["selected_world_id"] != self.current_world_id):
            raise CampaignBoundaryError("maze attempt and difficulty disagree with durable Ariadne world")
        if _hash_public(_world_state(self.selected_characters)) != self._selected_world_hash:
            raise CampaignBoundaryError("selected public world was changed outside Ariadne")
        journal_world = next((item for item in descriptor["worlds"]
                              if item["world_id"] == self.current_world_id), None)
        if (journal_world is None
                or _hash_public(journal_world["world_state"]) != self._selected_world_hash):
            raise CampaignBoundaryError("selected world disagrees with durable Ariadne state")
        next_world_id = f"world_{self.world_version + 1}"
        if any(world["world_id"] == next_world_id for world in descriptor["worlds"]):
            raise CampaignBoundaryError(
                "next developmental world identity already exists; resolve "
                "Ariadne world collision before physical TRAIN")
        previous_candidates = {item["candidate_id"] for item in descriptor["candidates"]}
        previous_branches = {item["branch_id"] for item in descriptor["branches"]}
        if (any(launch.candidate_id in previous_candidates
                or "branch_" + launch.candidate_id in previous_branches
                for launch in launches)):
            raise CampaignBoundaryError(
                "previously attempted candidate and branch IDs cannot be retried; "
                "choose fresh IDs for the next maze")
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
        # No TRAIN side effect is allowed before the unchanged control and
        # *every* sibling's authorizations, scene and callback preflight pass.
        # This prevents a bad last sibling or missing control from consuming
        # the first sibling's irreversible GPU training quantum.
        control = _manifest(control_manifest_path)
        if (control["state_sha256"] != parent["state_sha256"]
                or control["family"] != parent["family"]
                or control["mutation_kind"] != parent["mutation_kind"]):
            raise CampaignBoundaryError("unchanged parent control failed pre-effect validation")
        # Keep the original preregistered examiner cohort detached from all
        # callback-held caller objects, including frozen-but-mutable dataclasses.
        sealed_plan = replace(evaluation_plan) if type(evaluation_plan) is EvaluationPlan else evaluation_plan
        _check_plan(sealed_plan, {l.trainer_id for l in launches}
                    | {l.candidate_id for l in launches})
        preflight: list[tuple[CandidateLaunch, DaedalusSocialPlanner,
                              DialogueBeat, set[DialogueBeat]]] = []
        seen_journal_dirs: set[Path] = set()
        for launch in launches:
            if (not callable(launch.manifest_builder)
                    or type(launch.max_beats) is not int
                    or not 1 <= launch.max_beats <= 12):
                raise CampaignBoundaryError("bounded trusted manifest builder and scene count required")
            # Source-owned provider admits TRAIN cases only. No evaluator case
            # or reference claim is passed into director, responder or trainer.
            approved = self.provider.public_train_beats(
                launch.case_id, launch.character_id,
                region_id=launch.region_id, chapter_id=launch.chapter_id,
                minimum_difficulty=self.minimum_difficulty,
                complexity_depth=self.complexity_depth,
                world_id=self.current_world_id)
            allowed = set(approved)
            if (not callable(getattr(launch.responder, "respond", None))
                    or not callable(getattr(launch.operator, "assess_train", None))
                    or not callable(getattr(launch.trainer, "train_supervised", None))):
                raise CampaignBoundaryError("all sibling callback adapters must be callable")
            journal_dir = launch.bridge_journal_directory
            if (not isinstance(journal_dir, Path) or not journal_dir.is_absolute()
                    or journal_dir.exists() or not journal_dir.parent.is_dir()
                    or journal_dir in seen_journal_dirs):
                raise CampaignBoundaryError("all sibling journals need distinct fresh absolute paths")
            seen_journal_dirs.add(journal_dir)
            # Model scene proposals are made exactly once in a pure,
            # side-effect-free public world before *any* journal directory
            # creation or training. The chosen scene is later adopted by the
            # identical fresh physical TRAIN world.
            planning_world = SocialWorldSession(self.scope, self.selected_characters)
            planner = DaedalusSocialPlanner(
                planning_world, launch.director,
                SocialScenePolicy(minimum_difficulty=self.minimum_difficulty,
                                  max_scenes=launch.max_beats))
            # Propose before durable TRAIN intent, but only an approved public
            # TRAIN beat may be trained; model-authored code is never executed.
            first = planner.propose_next()
            if first not in allowed:
                raise CampaignBoundaryError("Daedalus proposed unapproved Vera TRAIN case")
            preflight.append((replace(launch), planner, first, allowed))
        experiences: list[CandidateExperience] = []
        candidates: list[Candidate] = []
        for launch, planner, first, allowed in preflight:
            bridge = SocialExperienceTrainingBridge(
                self.scope, characters=self.selected_characters,
                checkpoint=parent_checkpoint,
                responder=launch.responder, operator=launch.operator,
                trainer=launch.trainer,
                journal_directory=launch.bridge_journal_directory)
            planner.world = bridge.world  # same state, one approved proposal
            # From the first branch event onward, errors can imply an
            # ambiguous external TRAIN; never auto-replay those effects.
            self._blocked = True
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
            candidates=tuple(candidates), plan=sealed_plan)
        pending = GenerationPendingExam(generation, population, tuple(experiences))
        # Receipt objects are frozen by convention, NOT immutable to a Python
        # callback using object.__setattr__. Retain a separate pre-examination
        # source snapshot so a caller cannot forge the world inherited by the
        # selected learner after receiving the pending campaign receipt.
        self._pending_world_hashes = {
            experience.candidate_id: _hash_public(_world_state(experience.social_receipt.characters))
            for experience in experiences
        }
        self._pending_experience_hashes = {
            experience.candidate_id: _experience_hash(experience)
            for experience in experiences
        }
        self._pending_experience_ids = tuple(x.candidate_id for x in experiences)
        self._pending_character_copies = {
            experience.candidate_id: copy.deepcopy(experience.social_receipt.characters)
            for experience in experiences
        }
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
        if tuple(x.candidate_id for x in pending.experiences) != self._pending_experience_ids:
            raise CampaignBoundaryError("pending candidate roster integrity changed")
        for experience in pending.experiences:
            if _experience_hash(experience) != self._pending_experience_hashes[experience.candidate_id]:
                raise CampaignBoundaryError("pending TRAIN receipt integrity changed")
            expected_hash = self._pending_world_hashes.get(experience.candidate_id)
            observed_hash = _hash_public(_world_state(experience.social_receipt.characters))
            if expected_hash is None or expected_hash != observed_hash:
                raise CampaignBoundaryError("pending social world receipt integrity changed")
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
        next_world = f"world_{self.world_version + 1}"
        next_difficulty = self.minimum_difficulty
        next_complexity = self.complexity_depth
        if decision.winner_id is not None:
            self.journal.select_candidate(
                candidate_id=decision.winner_id,
                selection_reason="source_only_comparative_development_fixture",
                idempotency_key="select_" + decision.winner_id)
            selected_characters = copy.deepcopy(
                self._pending_character_copies[decision.winner_id])
        else:
            if next_difficulty < 5:
                next_difficulty += 1
            else:
                next_complexity += 1
            self.journal.record_no_winner(
                next_minimum_difficulty=next_difficulty,
                next_complexity_depth=next_complexity,
                idempotency_key="no_winner_" + self.current_world_id)
        # A failed cohort advances the public challenge, never the model
        # weight lineage. Both possible selection outcomes demand a durable
        # successor world before another training entrance is considered.
        self.journal.record_world(
            world_id=next_world, parent_world_id=self.current_world_id,
            version=self.world_version + 1,
            curriculum={"objective_id": self.scope.objective_id,
                        "train_only": True,
                        "minimum_difficulty": next_difficulty,
                        "complexity_depth": next_complexity},
            world_state=_world_state(selected_characters),
            idempotency_key="world_" + next_world)
        self.world_version += 1
        self.current_world_id = next_world
        self.minimum_difficulty = next_difficulty
        self.complexity_depth = next_complexity
        self.maze_attempt_index += 1
        self.selected_characters = copy.deepcopy(selected_characters)
        self._selected_world_hash = _hash_public(_world_state(self.selected_characters))
        self._finished_generations += 1 if decision.winner_id else 0
        self._pending = None
        self._pending_world_hashes = {}
        self._pending_experience_hashes = {}
        self._pending_experience_ids = ()
        self._pending_character_copies = {}
        self._blocked = False
        descriptor = self.journal.continuation_descriptor()
        if descriptor["selected_parent_state_sha256"] != decision.selected_state_sha256:
            self._blocked = True
            raise CampaignBoundaryError("Ariadne selected different physical parent")
        if (descriptor["maze_attempt_index"] != self.maze_attempt_index
                or descriptor["selected_world_id"] != self.current_world_id
                or descriptor["minimum_scene_difficulty"] != self.minimum_difficulty
                or descriptor["complexity_depth"] != self.complexity_depth):
            self._blocked = True
            raise CampaignBoundaryError("Ariadne successor world/maze attempt readback disagrees")
        return CampaignGenerationResult(
            decision, descriptor["event_head_sha256"], copy.deepcopy(selected_characters))


__all__ = ["CampaignBoundaryError", "CandidateLaunch", "CandidateExperience",
           "GenerationPendingExam", "CampaignGenerationResult", "CampaignOrchestrator"]
