"""Opt-in full-optimizer resume binding for the generated MiniLM Maze.

This module composes the existing pickle-free neural full-state checkpoint
substrate with the *actual* two-cycle sentiment Maze. It binds the full ordered
TRAIN-visit schedule, assigned labels, control arm, learning rate, predecessor
checkpoint, and completed optimizer cursor before restoring any parameters.

Unlike the ordinary Maze runner, this adapter preserves AdamW moments, step
counters and Python/NumPy/Torch RNG through the existing safetensors/strict
JSON substrate. It does not alter that runner or prove that a physical MiniLM
run has used this interface. No host-owned checkpoint fence, event replay
protection, signed examiner or independent graduation authority is installed.
No optimizer scheduler is supported by this bounded runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

from .neural_resume_checkpoint import (
    FullCheckpointContext, FullCheckpointError, FullCheckpointReceipt,
    SCHEMA, _context_json, _decode_tree, _dependencies, _manifest_digest,
    _safe_snapshot_dir, _strict_json, _trainable_and_groups,
    load_full_checkpoint, save_full_checkpoint,
)
from .training_scope import TrainingScopeSpec
from scripts.qualify_maze_conditioned_minilm_cpu import (
    PREREG_ARMS, TrainingCycle, build_training_cycles,
    optimizer_learning_rate, preregistered_randomized_labels,
    select_training_batch, validate_scope,
)
from scripts import qualify_maze_conditioned_minilm_cpu as maze_runner


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_DOMAIN = b"labyrinth.minilm-maze-optimizer-resume.v1\0"


def _expected_adamw_group_options(plan: "MazeOptimizerResumePlan") -> dict[str, Any]:
    """Exact no-scheduler torch.optim.AdamW defaults used by the Maze runner.

    A new optimizer option or a non-default setting must be explicitly
    preregistered in a new contract version; neither is silently inherited.
    """
    return {
        "lr": optimizer_learning_rate(plan.arm, plan.learning_rate),
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }


def _require_adamw_group_options(group: Any, plan: "MazeOptimizerResumePlan") -> None:
    expected = _expected_adamw_group_options(plan)
    if (type(group) is not dict
            or set(group) != set(expected) | {"params"}
            or any(type(group[key]) is not type(value) or group[key] != value
                   for key, value in expected.items())):
        raise FullCheckpointError("AdamW hyperparameter configuration mismatch")


def _canonical(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


@dataclass(frozen=True)
class MazeOptimizerResumePlan:
    """Development-only fixed schedule and trust-independent checkpoint pin."""

    scope: TrainingScopeSpec
    cycles: tuple[TrainingCycle, ...]
    weak_cycle_one: tuple[str, ...]
    arm: str
    steps_per_cycle: int
    learning_rate: float
    logical_run_id: str
    base_checkpoint_sha256: str
    prior_checkpoint_sha256: str
    fence: int

    def validate(self) -> None:
        validate_scope(self.scope)
        if (type(self.weak_cycle_one) is not tuple
                or self.cycles != build_training_cycles(
                    self.scope, weak_cycle_one=self.weak_cycle_one)):
            raise ValueError("resume plan does not match authentic regenerated Maze")
        if type(self.arm) is not str or self.arm not in PREREG_ARMS:
            raise ValueError("unsupported neural control arm")
        if (type(self.steps_per_cycle) is not int
                or not 1 <= self.steps_per_cycle <= 16
                or type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate)
                or not 0 < self.learning_rate <= 0.01):
            raise ValueError("invalid bounded optimizer schedule")
        if (type(self.logical_run_id) is not str
                or not _IDENTIFIER.fullmatch(self.logical_run_id)):
            raise ValueError("invalid bounded run identity")
        if (type(self.base_checkpoint_sha256) is not str
                or _SHA.fullmatch(self.base_checkpoint_sha256) is None
                or type(self.prior_checkpoint_sha256) is not str
                or _SHA.fullmatch(self.prior_checkpoint_sha256) is None):
            raise ValueError("pinned base and learner predecessor digests required")
        if type(self.fence) is not int or not 0 <= self.fence < 2**63 - 1:
            raise ValueError("bounded external fence required")

    def _spec(self) -> dict[str, Any]:
        # Bind the two executable source files that interpret this schedule.
        # A changed trainer or resume adapter cannot silently claim the same
        # persisted optimizer identity using old source-level semantics.
        trainer_sha = sha256(Path(maze_runner.__file__).read_bytes()).hexdigest()
        adapter_sha = sha256(Path(__file__).read_bytes()).hexdigest()
        return {
            "scope_fingerprint": self.scope.fingerprint,
            "objective_id": self.scope.objective_id,
            "weak_cycle_one": list(self.weak_cycle_one),
            "arm": self.arm, "steps_per_cycle": self.steps_per_cycle,
            "learning_rate": float(self.learning_rate),
            "trainer_source_sha256": trainer_sha,
            "resume_source_sha256": adapter_sha,
        }

    @property
    def plan_sha256(self) -> str:
        self.validate()
        return sha256(_DOMAIN + _canonical(self._spec())).hexdigest()

    @property
    def checkpoint_run_id(self) -> str:
        # FullCheckpointContext has no field for arm/hyperparameter schedule.
        # Put its committed short digest in the immutable run identity.
        return f"{self.logical_run_id}:maze-{self.plan_sha256[:20]}"

    @property
    def total_steps(self) -> int:
        self.validate()
        return 2 * self.steps_per_cycle

    def exposures(self) -> tuple[tuple[str, str], ...]:
        """Full ordered visit plan; per-visit digest commits to trainer label.

        A case can be practiced repeatedly, but every occurrence ID is unique.
        An inverted/randomized-label control cannot reuse a correct-label
        checkpoint context by keeping identical prompt hashes.
        """
        self.validate()
        result = []
        for cycle in self.cycles:
            randomized = (preregistered_randomized_labels(cycle, self.scope.seed)
                          if self.arm == "randomized_labels" else None)
            for step in range(self.steps_per_cycle):
                for offset, row in enumerate(select_training_batch(cycle, step)):
                    assigned = (randomized[row.exposure_id] if randomized is not None
                                else row.label ^ int(self.arm == "inverted"))
                    label_hash = sha256(_DOMAIN + _canonical({
                        "exposure_id": row.exposure_id,
                        "prompt_sha256": sha256(row.prompt.encode("utf-8")).hexdigest(),
                        "assigned_train_label": assigned,
                    })).hexdigest()
                    occurrence = f"c{cycle.plan.cycle_index}-s{step:02d}-o{offset}-{row.exposure_id[:12]}"
                    result.append((occurrence, label_hash))
        return tuple(result)

    def context(self, completed_steps: int) -> FullCheckpointContext:
        if type(completed_steps) is not int or not 0 <= completed_steps <= self.total_steps:
            raise ValueError("resume cursor is outside precommitted Maze training")
        context = FullCheckpointContext(
            run_id=self.checkpoint_run_id,
            objective_id=self.scope.objective_id,
            base_checkpoint_sha256=self.base_checkpoint_sha256,
            prior_checkpoint_sha256=self.prior_checkpoint_sha256,
            fence=self.fence,
            cursor=completed_steps * 4,
            exposures=self.exposures(),
        )
        context.validate()
        return context


def _bundle(encoder: Any, head: Any, optimizer: Any, plan: MazeOptimizerResumePlan):
    try:
        import torch
    except ImportError as exc:
        raise FullCheckpointError("optional CPU torch dependency unavailable") from exc
    if (not isinstance(encoder, torch.nn.Module)
            or not isinstance(head, torch.nn.Module)
            or encoder is head or optimizer.__class__.__name__ != "AdamW"):
        raise FullCheckpointError("distinct encoder/head and AdamW optimizer required")
    if (not optimizer.param_groups
            or any(type(group.get("lr")) not in (int, float)
                   or group["lr"] != optimizer_learning_rate(
                       plan.arm, plan.learning_rate) for group in optimizer.param_groups)):
        raise FullCheckpointError("optimizer learning rate differs from Maze control arm")
    for group in optimizer.param_groups:
        _require_adamw_group_options(group, plan)
    combined = torch.nn.Module()
    combined.add_module("encoder", encoder)
    combined.add_module("head", head)
    if any(param.device.type != "cpu" for param in combined.parameters()):
        raise FullCheckpointError("CPU-only optimizer resume")
    return combined


def _require_complete_adamw_state(
    state_dict: Any, *, combined: Any, optimizer: Any,
    plan: MazeOptimizerResumePlan, completed_steps: int,
) -> None:
    """Reject a digest-valid snapshot that silently discards optimizer history.

    This is intentionally stricter than generic AdamW: every trainable
    parameter must receive gradients on *every* Maze optimizer step. A model
    with inactive trainables must correct its optimizer inventory rather than
    quietly resume with a partly cold optimizer. The zero-update control must
    retain an entirely empty optimizer state.
    """
    import torch

    named, names_by_group = _trainable_and_groups(combined, optimizer, torch)
    if type(state_dict) is not dict or set(state_dict) != {"state", "param_groups"}:
        raise FullCheckpointError("incomplete AdamW optimizer state envelope")
    states, saved_groups = state_dict["state"], state_dict["param_groups"]
    if (type(states) is not dict or type(saved_groups) is not list
            or len(saved_groups) != len(names_by_group)):
        raise FullCheckpointError("incomplete AdamW parameter-group inventory")
    expected_ids: set[int] = set()
    for group, names in zip(saved_groups, names_by_group):
        if type(group) is not dict or type(group.get("params")) is not list:
            raise FullCheckpointError("malformed AdamW parameter group")
        ids = group["params"]
        if (len(ids) != len(names)
                or any(type(ident) is not int or ident < 0
                       or ident in expected_ids for ident in ids)
                or len(set(ids)) != len(ids)):
            raise FullCheckpointError("AdamW group IDs or inventory mismatch")
        _require_adamw_group_options(group, plan)
        for ident, name in zip(ids, names):
            expected_ids.add(ident)
            state = states.get(ident)
            if plan.arm == "zero_update" or completed_steps == 0:
                if state not in (None, {}):
                    raise FullCheckpointError("zero-update Maze has unexpected optimizer history")
                continue
            if type(state) is not dict or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise FullCheckpointError("missing AdamW moments or step counter")
            counter = state["step"]
            if (not isinstance(counter, torch.Tensor)
                    or counter.device.type != "cpu"
                    or counter.shape != torch.Size([])
                    or counter.dtype != torch.float32
                    or not bool(torch.isfinite(counter).all())
                    or counter.item() != completed_steps):
                raise FullCheckpointError("AdamW scalar step tensor does not match Maze cursor")
            parameter = named[name]
            for moment in ("exp_avg", "exp_avg_sq"):
                tensor = state[moment]
                if (not isinstance(tensor, torch.Tensor)
                        or tensor.device.type != "cpu"
                        or tensor.shape != parameter.shape
                        or tensor.dtype != parameter.dtype
                        or not bool(torch.isfinite(tensor).all())):
                    raise FullCheckpointError("invalid AdamW moment tensor")
    if set(states) - expected_ids:
        raise FullCheckpointError("AdamW optimizer state contains unknown parameter IDs")


def _preflight_saved_optimizer(
    source: str | Path, *, pinned_sha256: str,
    context: FullCheckpointContext, combined: Any, optimizer: Any,
    plan: MazeOptimizerResumePlan, completed_steps: int,
) -> None:
    """Inspect exact pinned, pickle-free optimizer moments *before* mutation.

    The underlying loader repeats all hashes, context and tensor validation
    immediately before restore. This preflight must never be treated as an
    independent ownership/fence authority or a substitute for that loader.
    """
    source_dir = _safe_snapshot_dir(source, existing=True)
    manifest_path = source_dir / "manifest.json"
    if not 0 < manifest_path.stat().st_size <= 4 * 1024 * 1024:
        raise FullCheckpointError("checkpoint manifest exceeds byte limit")
    raw_manifest = manifest_path.read_bytes()
    if (_manifest_digest(raw_manifest) != pinned_sha256
            or type(pinned_sha256) is not str):
        raise FullCheckpointError("checkpoint digest differs from trusted identity")
    manifest = _strict_json(raw_manifest)
    if (manifest.get("schema") != SCHEMA
            or manifest.get("context") != _context_json(context)):
        raise FullCheckpointError("checkpoint context or schema mismatch")
    tensor_path = source_dir / "state.safetensors"
    if not 0 < tensor_path.stat().st_size <= 1024 * 1024 * 1024:
        raise FullCheckpointError("checkpoint tensor exceeds byte limit")
    if sha256(tensor_path.read_bytes()).hexdigest() != manifest.get("tensor_sha256"):
        raise FullCheckpointError("checkpoint tensor digest mismatch")
    _, load_file, _ = _dependencies()
    try:
        tensors = load_file(str(tensor_path), device="cpu")
    except (OSError, ValueError, RuntimeError) as exc:
        raise FullCheckpointError("invalid safetensors optimizer snapshot") from exc
    if "optimizer_tree" not in manifest:
        raise FullCheckpointError("missing AdamW optimizer tree")
    snapshot = _decode_tree(manifest["optimizer_tree"], tensors, set())
    _require_complete_adamw_state(
        snapshot, combined=combined, optimizer=optimizer, plan=plan,
        completed_steps=completed_steps,
    )


def save_maze_optimizer_snapshot(
    destination: str | Path, *, encoder: Any, head: Any,
    optimizer: Any, plan: MazeOptimizerResumePlan, completed_steps: int,
) -> FullCheckpointReceipt:
    """Save one no-overwrite full-state snapshot at a Maze optimizer step."""
    context = plan.context(completed_steps)
    combined = _bundle(encoder, head, optimizer, plan)
    _require_complete_adamw_state(
        optimizer.state_dict(), combined=combined, optimizer=optimizer,
        plan=plan, completed_steps=completed_steps,
    )
    return save_full_checkpoint(destination, model=combined,
                                optimizer=optimizer, context=context)


def load_maze_optimizer_snapshot(
    source: str | Path, *, encoder: Any, head: Any,
    optimizer: Any, plan: MazeOptimizerResumePlan, completed_steps: int,
    independently_pinned_checkpoint_sha256: str,
) -> FullCheckpointReceipt:
    """Require caller-pinned SHA + complete Maze lineage before full restore.

    This verifies local bytes and scope/fence but cannot attest that the caller
    chose the true latest checkpoint or owns the external monotonic authority.
    """
    context = plan.context(completed_steps)
    combined = _bundle(encoder, head, optimizer, plan)
    _preflight_saved_optimizer(
        source, pinned_sha256=independently_pinned_checkpoint_sha256,
        context=context, combined=combined, optimizer=optimizer,
        plan=plan, completed_steps=completed_steps,
    )
    receipt = load_full_checkpoint(
        source, model=combined, optimizer=optimizer,
        expected_sha256=independently_pinned_checkpoint_sha256,
        expected_context=context,
    )
    # Post-load assertion: any restoration contract mismatch means this
    # process must be discarded, not used to publish training evidence.
    _require_complete_adamw_state(
        optimizer.state_dict(), combined=combined, optimizer=optimizer,
        plan=plan, completed_steps=completed_steps,
    )
    return receipt
