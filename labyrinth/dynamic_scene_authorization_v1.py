"""Source-only, operator-approved dynamic TRAIN scene seam.

The model may AUTHOR a dialogue encounter, never AUTHOR its authorization,
physical training effect, development score, private exam, or graduation.
This module deliberately does not change CampaignOrchestrator's existing
exact-case whitelist. It is an isolated integration candidate, not a host
capability grant, an authenticated operator, or a production deployment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import re
import secrets
from pathlib import Path
from typing import Callable

from .ariadne_recovery_v1 import AriadneJournal, trainer_request
from .campaign_orchestrator_v1 import (
    _bind_checkpoint, _checkpoint_from_manifest, _hash_public, _manifest,
    _world_state,
)
from .daedalus_social_director_v1 import SocialSceneContext
from .living_world_v1 import LivingSceneDirector, LivingWorld, PublicWorldObservation
from .model_saga_runtime import capture_checkpoint
from .social_experience_training_bridge_v1 import (
    SocialExperienceTrainingBridge, SocialTrainingReceipt,
)
from .social_world_consequence_v1 import DialogueBeat, beat_fingerprint


_SOURCE_ONLY = "SOURCE_ONLY_OPERATOR_APPROVED_DYNAMIC_TRAIN_NOT_MINOTAUR_PROOF"
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)


def _journal_key(prefix: str, value: str) -> str:
    return prefix + "_" + hashlib.sha256(value.encode("ascii")).hexdigest()[:40]


class DynamicSceneBoundaryError(ValueError):
    """A model-created scene or physical TRAIN boundary is not established."""


@dataclass(frozen=True)
class DynamicTrainPolicy:
    objective_id: str
    world_sha256: str
    skills: tuple[str, ...]
    character_ids: tuple[str, ...]
    minimum_difficulty: int = 1

    def __post_init__(self) -> None:
        if (type(self.objective_id) is not str or not self.objective_id
                or type(self.world_sha256) is not str
                or len(self.world_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.world_sha256)
                or type(self.skills) is not tuple or not self.skills
                or len(set(self.skills)) != len(self.skills)
                or type(self.character_ids) is not tuple or not self.character_ids
                or len(set(self.character_ids)) != len(self.character_ids)
                or type(self.minimum_difficulty) is not int
                or not 1 <= self.minimum_difficulty <= 5):
            raise DynamicSceneBoundaryError("invalid source-owned dynamic TRAIN policy")
        for value in self.skills + self.character_ids:
            if type(value) is not str or not value:
                raise DynamicSceneBoundaryError("invalid policy skill or character")


@dataclass(frozen=True)
class ApprovedDynamicScene:
    """Local one-use source receipt; not portable proof of operator identity."""

    approval_id: str
    beat: DialogueBeat
    scene_sha256: str
    world_sha256: str
    world_state_sha256: str
    evidence_class: str = _SOURCE_ONLY

    def __post_init__(self) -> None:
        if (type(self.evidence_class) is not str or self.evidence_class != _SOURCE_ONLY
                or type(self.beat) is not DialogueBeat
                or self.scene_sha256 != beat_fingerprint(self.beat)):
            raise DynamicSceneBoundaryError("scene receipt is not a source-approved TRAIN claim")


@dataclass(frozen=True)
class DynamicTrainResult:
    candidate_id: str
    scene_sha256: str
    parent_manifest_path: Path
    child_manifest_path: Path
    child_state_sha256: str
    social_receipt: SocialTrainingReceipt
    ariadne_commit_sha256: str
    evidence_class: str = _SOURCE_ONLY

    def __post_init__(self) -> None:
        if type(self.evidence_class) is not str or self.evidence_class != _SOURCE_ONLY:
            raise DynamicSceneBoundaryError("dynamic TRAIN cannot claim Minotaur graduation")


class DynamicSceneAuthority:
    """Local operator-callback policy for fresh model-written public scenes.

    Callback provenance is a caller responsibility; a bool is not a signature.
    Python callers in the same process are trusted, not isolated adversaries.
    """

    def __init__(self, policy: DynamicTrainPolicy) -> None:
        if type(policy) is not DynamicTrainPolicy:
            raise DynamicSceneBoundaryError("exact source policy required")
        DynamicTrainPolicy.__post_init__(policy)
        self.policy = copy.deepcopy(policy)
        self._issued: dict[str, tuple[str, str, str]] = {}

    def approve(
        self, *, world: LivingWorld, director: LivingSceneDirector,
        context: SocialSceneContext,
        operator_approves: Callable[[DialogueBeat, PublicWorldObservation], bool],
    ) -> ApprovedDynamicScene:
        if (type(world) is not LivingWorld or type(director) is not LivingSceneDirector
                or director.world is not world or type(context) is not SocialSceneContext
                or not callable(operator_approves)):
            raise DynamicSceneBoundaryError("living director, world and trusted operator required")
        policy = self.policy
        definition = world.definition
        before = world.state_sha256
        if (definition.world_sha256 != policy.world_sha256
                or definition.recipe.objective_id != policy.objective_id
                or context.objective_id != policy.objective_id
                or not set(context.eligible_skills).issubset(policy.skills)
                or not set(context.character_ids).issubset(policy.character_ids)
                or context.minimum_difficulty < policy.minimum_difficulty):
            raise DynamicSceneBoundaryError("director context is outside source-owned world policy")
        scene = director.propose_scene(copy.deepcopy(context))
        if type(scene) is not DialogueBeat:
            raise DynamicSceneBoundaryError("model did not return a typed TRAIN scene")
        before_scene_hash = beat_fingerprint(scene)
        if (scene.skill not in policy.skills or scene.character_id not in policy.character_ids
                or scene.difficulty < policy.minimum_difficulty):
            raise DynamicSceneBoundaryError("model scene escaped source-owned TRAIN limits")
        observed = world.observe()
        allowed = operator_approves(copy.deepcopy(scene), copy.deepcopy(observed))
        if (type(allowed) is not bool or not allowed
                or world.state_sha256 != before
                or beat_fingerprint(scene) != before_scene_hash):
            raise DynamicSceneBoundaryError("operator rejected or world changed during scene approval")
        # Nontransferable local registry plus full public scene hash catches
        # ordinary forgeries/replays without pretending to authenticate a host.
        approval_id = "scene_" + secrets.token_hex(24)
        self._issued[approval_id] = (before_scene_hash, policy.world_sha256, before)
        return ApprovedDynamicScene(approval_id, copy.deepcopy(scene), before_scene_hash,
                                    policy.world_sha256, before)

    def preflight_scene(self, *, receipt: ApprovedDynamicScene, world: LivingWorld,
                        bridge: SocialExperienceTrainingBridge) -> None:
        if (type(receipt) is not ApprovedDynamicScene
                or type(world) is not LivingWorld
                or type(bridge) is not SocialExperienceTrainingBridge):
            raise DynamicSceneBoundaryError("exact receipt, world and TRAIN bridge required")
        ApprovedDynamicScene.__post_init__(receipt)
        saved = self._issued.get(receipt.approval_id)
        expected = (receipt.scene_sha256, receipt.world_sha256, receipt.world_state_sha256)
        if saved is None or saved != expected:
            raise DynamicSceneBoundaryError("unknown, consumed or tampered scene approval")
        if (world.definition.world_sha256 != self.policy.world_sha256
                or world.state_sha256 != receipt.world_state_sha256
                or receipt.world_sha256 != self.policy.world_sha256
                or bridge.scope.objective_id != self.policy.objective_id
                or receipt.beat.skill not in self.policy.skills
                or receipt.beat.character_id not in self.policy.character_ids
                or receipt.beat.skill not in bridge.scope.skills
                or receipt.beat.character_id not in tuple(
                    character.character_id for character in bridge.world.characters)
                or bridge.indeterminate or bridge.training_sparks):
            raise DynamicSceneBoundaryError("live world, social scope or scene drifted")
        bridge.world.preview(copy.deepcopy(receipt.beat))

    def consume(self, receipt: ApprovedDynamicScene) -> None:
        saved = self._issued.get(receipt.approval_id)
        if saved != (receipt.scene_sha256, receipt.world_sha256, receipt.world_state_sha256):
            raise DynamicSceneBoundaryError("scene was not approved or already consumed")
        del self._issued[receipt.approval_id]


def run_one_approved_dynamic_train(
    *, authority: DynamicSceneAuthority, scene: ApprovedDynamicScene,
    world: LivingWorld, journal: AriadneJournal,
    bridge: SocialExperienceTrainingBridge, candidate_id: str, trainer_id: str,
    manifest_builder: Callable[[SocialTrainingReceipt, dict], Path],
) -> DynamicTrainResult:
    """Demonstration adapter for ONE source-only, full-state-backed TRAIN effect.

    This does not change CampaignOrchestrator, select a winner, or access
    private Minotaur cases. A real host also requires protected runtime
    launch authorization and ownership/locking outside this source module.
    """
    if (type(authority) is not DynamicSceneAuthority
            or type(journal) is not AriadneJournal
            or not callable(manifest_builder)
            or type(candidate_id) is not str or _ID.fullmatch(candidate_id) is None
            or type(trainer_id) is not str or _ID.fullmatch(trainer_id) is None):
        raise DynamicSceneBoundaryError("trusted candidate, journal, builder and trainer id required")
    # All of these checks precede any PREPARED event and all trainer calls.
    authority.preflight_scene(receipt=scene, world=world, bridge=bridge)
    descriptor = journal.continuation_descriptor()
    if (descriptor["status"] != "READY_FOR_EXTERNAL_FULL_STATE_LOAD"
            or descriptor["selected_world_id"] not in {
                item["world_id"] for item in descriptor["worlds"]
            }):
        raise DynamicSceneBoundaryError("Ariadne has no ready physically inventoried parent")
    branch_id = _journal_key("dynamic_branch", candidate_id)
    if (any(c["candidate_id"] == candidate_id for c in descriptor["candidates"])
            or any(b["branch_id"] == branch_id for b in descriptor["branches"])):
        raise DynamicSceneBoundaryError(
            "candidate or branch identity was already consumed; choose a fresh ID")
    # Assemble siblings before examination only. Once the Minotaur starts
    # judging a current-world cohort, appending fresh candidates would alter
    # the comparison population after exposure to development feedback.
    current_branches = {
        b["branch_id"] for b in descriptor["branches"]
        if b["world_id"] == descriptor["selected_world_id"]
    }
    if any(
        candidate["generation_index"] == descriptor["next_generation_index"]
        and candidate["branch_id"] in current_branches
        and candidate["fate"] != "UNEVALUATED"
        for candidate in descriptor["candidates"]
    ):
        raise DynamicSceneBoundaryError(
            "current-world candidate examination already began; "
            "fresh sibling TRAIN is not authorized")
    parent_path = Path(descriptor["selected_parent_manifest_path"])
    parent = _manifest(parent_path)
    if (parent["state_sha256"] != descriptor["selected_parent_state_sha256"]
            or parent["origin_state_sha256"] != descriptor["origin_state_sha256"]):
        raise DynamicSceneBoundaryError("Ariadne selected parent ancestry differs")
    expected_checkpoint = capture_checkpoint(_checkpoint_from_manifest(parent))
    _bind_checkpoint(parent, expected_checkpoint)
    if bridge.entrance != expected_checkpoint or bridge.checkpoint != expected_checkpoint:
        raise DynamicSceneBoundaryError("social TRAIN bridge is not bound to selected W0/parent")
    if scene.beat.region_id != "region_" + world.observe().room_id:
        raise DynamicSceneBoundaryError("approved scene no longer occurs at this world location")
    authority.consume(scene)

    # From this point a crash is indeterminate; neither scene nor trainer may
    # be automatically replayed. The physical journal owns the intent token.
    journal.record_branch(
        branch_id=branch_id, world_id=descriptor["selected_world_id"],
        experience={"observations": [{"beat_sha256": scene.scene_sha256,
                                       "world_state_sha256": scene.world_state_sha256}],
                    "actions": ["public_dynamic_dialogue_train"], "outcomes": []},
        selected_path=list(scene.beat.key), idempotency_key=branch_id,
    )
    permit = journal.prepare_training(
        candidate_id=candidate_id, branch_id=branch_id,
        trainer_request=trainer_request(trainer_id, parameters={
            "source_only": True, "scope_fingerprint": bridge.scope.fingerprint,
            "approved_public_scene_sha256": scene.scene_sha256,
            "world_sha256": scene.world_sha256,
        }), idempotency_key=_journal_key("dynamic_prepare", candidate_id),
    )
    bridge.run_train_beat(copy.deepcopy(scene.beat))
    social = bridge.report()
    candidate_path = manifest_builder(social, parent)
    child = _manifest(candidate_path)
    _bind_checkpoint(child, social.current)
    commit = journal.commit_training(
        permit, child_manifest_path=candidate_path,
        trainer_receipt={
            "trainer_id": trainer_id, "effect_status": "completed",
            "effect_token": permit.effect_token,
            "details": {"source_only": True,
                        "scene_sha256": scene.scene_sha256,
                        "train_sparks": len(social.sparks),
                        "public_world": _world_state(social.characters),
                        "checkpoint_sha256": social.current.weights_sha256,
                        "bridge_commit_receipt_sha256": _hash_public([
                            spark.after.weights_sha256 for spark in social.sparks]),
                        },
        }, idempotency_key=_journal_key("dynamic_commit", candidate_id),
    )
    return DynamicTrainResult(candidate_id, scene.scene_sha256, parent_path,
                              candidate_path, child["state_sha256"], social,
                              commit["event_sha256"])


__all__ = ["DynamicSceneBoundaryError", "DynamicTrainPolicy", "ApprovedDynamicScene",
           "DynamicTrainResult", "DynamicSceneAuthority", "run_one_approved_dynamic_train"]
