"""Bridge direct weight edits into physical Ariadne and Minotaur population seams.

SOURCE_ONLY: a trusted caller provides proposals and operator-owned tensor policy.
No model is inferred, no private exam is accessed, no candidate is evaluated,
and no lineage/evidence receipt proves a neural improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Callable

from .ariadne_recovery_v1 import AriadneJournal, _require_supported_platform, trainer_request
from .campaign_full_state_v1 import (
    capture_campaign_state, save_campaign_state_once, verify_campaign_state,
)
from .evolutionary_population_v1 import (
    Candidate, EvaluationPlan, MutationKind, Population, deterministic_seed,
    _check_plan, prepare_population,
)
from .weight_space_spark_v1 import (
    WeightSpaceEditPolicy, WeightSpaceEditProposal, WeightSpaceEditReceipt,
    WeightSpaceBoundaryError, _validate, create_weight_space_candidate,
)


_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_TRAINER_ID = "weight_space_spark"
_EVIDENCE = "SOURCE_ONLY_JOURNALED_WEIGHT_EDIT_POPULATION_NO_NEURAL_GAIN_PROOF"


def _journal_key(prefix: str, candidate_id: str) -> str:
    """Bounded collision-resistant key even for a maximal-length candidate ID."""
    return prefix + "_" + hashlib.sha256(candidate_id.encode("ascii")).hexdigest()[:40]


class WeightSpaceCampaignError(ValueError):
    """A source-only multi-sibling physical population cannot be constructed."""


@dataclass(frozen=True)
class PublicProposalContext:
    """Only the public TRAIN contract; not a model execution/sandbox boundary."""

    candidate_id: str
    parent_state_sha256: str
    public_train_source_sha256: str
    development_generation: int


@dataclass(frozen=True)
class WeightSpaceGenerationReceipt:
    origin_manifest_path: Path
    control_manifest_path: Path
    journal_root: Path
    ariadne_head_sha256: str
    candidates: tuple[WeightSpaceEditReceipt, ...]
    population: Population
    qualification: str = _EVIDENCE

    def __post_init__(self) -> None:
        if self.qualification != _EVIDENCE:
            raise WeightSpaceCampaignError("generation cannot claim neural gain")


def _control_clone(parent: dict, destination: Path) -> Path:
    """Create a distinct physical, unchanged W0 control with its own manifest."""
    destination.mkdir(mode=0o700)
    for role in ("base", "adapter", "optimizer", "scheduler", "rng"):
        item = parent[role]
        if item is not None:
            shutil.copytree(Path(item["root"]), destination / role)
    shutil.copy2(Path(parent["data_cursor"]["path"]), destination / "cursor.json")
    control = capture_campaign_state(
        base_dir=destination / "base",
        adapter_dir=destination / "adapter" if parent["adapter"] else None,
        optimizer_dir=destination / "optimizer",
        scheduler_dir=destination / "scheduler",
        rng_dir=destination / "rng",
        data_cursor_path=destination / "cursor.json",
        family=parent["family"], mutation_kind=parent["mutation_kind"],
    )
    if control["state_sha256"] != parent["state_sha256"]:
        raise WeightSpaceCampaignError("unchanged negative control diverges from W0")
    target = destination / "manifest.json"
    save_campaign_state_once(control, target)
    verify_campaign_state(target, fresh_process=True)
    return target


def run_weight_space_generation(
    *, w0_manifest_path: Path, campaign_root: Path,
    candidate_ids: tuple[str, ...],
    public_train_source_sha256: str,
    propose: Callable[[PublicProposalContext], WeightSpaceEditProposal],
    policy: WeightSpaceEditPolicy,
    evaluation_plan: EvaluationPlan,
    world_id: str = "world_1",
    max_copied_bytes: int = 512 * 1024 * 1024,
) -> WeightSpaceGenerationReceipt:
    """Make W0 control and two-to-eight distinct W1 direct-edit siblings.

    All proposal callbacks run *before* Ariadne TRAIN_PREPARED. This trusted
    callback is a dependency-injection seam, not execution of generated code.
    Physical editing begins strictly after a write-ahead intent; a crash or
    failure there leaves Ariadne INDETERMINATE, requiring investigation.

    Minotaur receives the returned Population with an immutable evaluation
    plan but **no** private outcomes, grading, promotion or inferred fitness.
    """
    if (type(candidate_ids) is not tuple or not 2 <= len(candidate_ids) <= 8
            or any(type(cid) is not str or _ID.fullmatch(cid) is None
                   or cid in ("parent", "control") for cid in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or type(public_train_source_sha256) is not str
            or _SHA.fullmatch(public_train_source_sha256) is None
            or type(world_id) is not str or _ID.fullmatch(world_id) is None
            or type(max_copied_bytes) is not int
            or not 4096 <= max_copied_bytes <= 4 * 1024 * 1024 * 1024
            or not callable(propose)
            or type(policy) is not WeightSpaceEditPolicy
            or type(evaluation_plan) is not EvaluationPlan):
        raise WeightSpaceCampaignError("exact bounded generation inputs required")
    if (not isinstance(w0_manifest_path, Path) or not w0_manifest_path.is_absolute()
            or not isinstance(campaign_root, Path) or not campaign_root.is_absolute()
            or campaign_root.exists() or campaign_root.is_symlink()
            or not campaign_root.parent.is_dir()
            or campaign_root.parent != campaign_root.parent.resolve(strict=True)
            or campaign_root != campaign_root.resolve(strict=False)):
        raise WeightSpaceCampaignError("fresh canonical campaign output and physical W0 required")
    _require_supported_platform()
    # Check the development examiner's *declaration* before any WAL or model
    # edits; this validates only a source-level plan, not a real examination.
    _check_plan(evaluation_plan, {_TRAINER_ID, *candidate_ids})
    # Frozen Python dataclasses can be altered through object.__setattr__ by
    # a callback holding the caller's object. Seal both operator policies
    # before invoking any model proposal and retain those detached values
    # through the irreversible physical edits and the final population.
    sealed_plan = replace(evaluation_plan)
    sealed_policy = replace(policy)
    verify_campaign_state(w0_manifest_path, fresh_process=True)
    parent = json.loads(w0_manifest_path.read_bytes())
    if (parent["generation_index"] != 0
            or parent["origin_state_sha256"] != parent["state_sha256"]):
        raise WeightSpaceCampaignError("this bridge starts from W0; next-generation bridge not qualified")
    source_bytes = parent["data_cursor"]["bytes"] + sum(
        record["bytes"]
        for role in ("base", "adapter", "optimizer", "scheduler", "rng")
        if parent[role] is not None
        for record in parent[role]["files"].values()
    )
    if source_bytes * (len(candidate_ids) + 1) > max_copied_bytes:
        raise WeightSpaceCampaignError("physical W0 control/sibling copies exceed explicit disk budget")
    for role in ("base", "adapter", "optimizer", "scheduler", "rng"):
        member = parent[role]
        if member is not None:
            physical_root = Path(member["root"])
            if campaign_root == physical_root or physical_root in campaign_root.parents:
                raise WeightSpaceCampaignError("campaign would recursively overlap W0 genome")
    # The proposal callback sees only public TRAIN metadata and runs before
    # any output is created. An untrusted model must be isolated by its caller.
    contexts = tuple(PublicProposalContext(
        cid, parent["state_sha256"], public_train_source_sha256, 1,
    ) for cid in candidate_ids)
    proposals = []
    requested_edits = set()
    for ctx in contexts:
        proposal = propose(replace(ctx))
        _validate(proposal, sealed_policy)
        if (proposal.proposal_id != ctx.candidate_id
                or proposal.parent_state_sha256 != ctx.parent_state_sha256
                or proposal.public_train_source_sha256 != ctx.public_train_source_sha256):
            raise WeightSpaceCampaignError("proposal differs from public bound callback context")
        edit = (proposal.shard_name, proposal.tensor_name,
                proposal.selected_columns, proposal.unit_direction, proposal.gain)
        if edit in requested_edits:
            raise WeightSpaceCampaignError("duplicate physical edit hypothesis")
        requested_edits.add(edit)
        proposals.append(replace(proposal))
    # Catch any W0 TOCTOU changes made by trusted but fallible proposal code.
    verify_campaign_state(w0_manifest_path, fresh_process=True)
    campaign_root.mkdir(mode=0o700)
    control_path = _control_clone(parent, campaign_root / "control")
    journal = AriadneJournal.create(campaign_root / "ariadne")
    journal.initialize(
        w0_manifest_path=w0_manifest_path, world_id=world_id, version=1,
        curriculum={"kind": "public_train_direct_weight_edit", "source_sha256": public_train_source_sha256},
        world_state={"model_world_evolution_proven": False},
        idempotency_key="genesis",
    )
    receipts: list[WeightSpaceEditReceipt] = []
    population_candidates: list[Candidate] = []
    for ctx, proposal in zip(contexts, proposals):
        cid = ctx.candidate_id
        branch_id = _journal_key("branch", cid)
        journal.record_branch(
            branch_id=branch_id, world_id=world_id,
            experience={"observations": [], "actions": ["propose_direct_weight_edit"], "outcomes": []},
            selected_path=["entrance", "training_chamber", branch_id],
            idempotency_key=branch_id,
        )
        proposal_hash = hashlib.sha256(_validate(proposal, sealed_policy)).hexdigest()
        permit = journal.prepare_training(
            candidate_id=cid, branch_id=branch_id,
            trainer_request=trainer_request(_TRAINER_ID, parameters={
                "method": proposal.method, "proposal_sha256": proposal_hash,
                "public_train_source_sha256": public_train_source_sha256,
                "source_only": True,
            }),
            idempotency_key=_journal_key("intent", cid),
        )
        result = create_weight_space_candidate(
            parent_manifest_path=w0_manifest_path,
            candidate_root=campaign_root / ("candidate_" + cid),
            proposal=proposal, policy=sealed_policy,
        )
        committed = journal.commit_training(
            permit, child_manifest_path=result.child_manifest_path,
            trainer_receipt={
                "trainer_id": _TRAINER_ID, "effect_status": "completed",
                "effect_token": permit.effect_token,
                "details": {
                    "method": proposal.method,
                    "source_only": True,
                    "child_state_sha256": result.child_state_sha256,
                    "proposal_sha256": result.proposal_sha256,
                    "not_neural_gain_evidence": True,
                },
            },
            idempotency_key=_journal_key("commit", cid),
        )
        receipts.append(result)
        population_candidates.append(Candidate(
            candidate_id=cid, manifest_path=result.child_manifest_path,
            mutation_kind=MutationKind(parent["mutation_kind"]),
            trainer_id=_TRAINER_ID,
            trainer_receipt_sha256=committed["event_sha256"],
            preupdate_parent_state_sha256=parent["state_sha256"],
            seed=deterministic_seed(parent["state_sha256"], 1, cid),
        ))
    population = prepare_population(
        development_generation=1,
        origin_manifest_path=w0_manifest_path,
        parent_manifest_path=w0_manifest_path,
        control_manifest_path=control_path,
        candidates=tuple(population_candidates), plan=sealed_plan,
    )
    descriptor = journal.continuation_descriptor()
    if (descriptor["status"] != "READY_FOR_EXTERNAL_FULL_STATE_LOAD"
            or descriptor["selected_parent_state_sha256"] != parent["state_sha256"]
            or len(descriptor["candidates"]) != len(candidate_ids)
            or any(child["fate"] != "UNEVALUATED" for child in descriptor["candidates"])):
        raise WeightSpaceCampaignError("Ariadne journal does not match unevaluated sibling population")
    verify_campaign_state(w0_manifest_path, fresh_process=True)
    return WeightSpaceGenerationReceipt(
        w0_manifest_path, control_path, journal.root,
        descriptor["event_head_sha256"], tuple(receipts), population,
    )


__all__ = [
    "WeightSpaceCampaignError", "PublicProposalContext",
    "WeightSpaceGenerationReceipt", "run_weight_space_generation",
]
