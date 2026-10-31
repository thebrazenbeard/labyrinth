"""Opt-in, ephemeral two-cycle *toy* CPU AdamW checkpoint-continuity smoke.

This executes real Torch operations against generated Maze training examples,
but uses a randomly initialized tiny encoder, NOT pretrained MiniLM. It checks
optimizer continuation against an uninterrupted local oracle. Synthetic pins,
local journals, and a single process cannot prove host custody, independent
anti-rollback, causal generalization, or graduation authority.

Default CLI invocation is a read-only preflight. Only ``--execute`` performs
training, and all checkpoints live inside a TemporaryDirectory.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


STATUS = "EPHEMERAL_TOY_CPU_OPTIMIZER_CONTINUITY_SOURCE_ONLY"
_REQUIRED = ("torch", "safetensors")


def missing_dependencies() -> tuple[str, ...]:
    """Report availability without importing optional heavy dependencies."""
    return tuple(name for name in _REQUIRED
                 if importlib.util.find_spec(name) is None)


def _plan():
    from labyrinth.minilm_optimizer_state_resume import MazeOptimizerResumePlan
    from scripts.qualify_maze_conditioned_minilm_cpu import (
        build_training_cycles, default_scope,
    )

    scope = default_scope()
    weak = ("negative_polarity",)
    plan = MazeOptimizerResumePlan(
        scope=scope,
        cycles=build_training_cycles(scope, weak_cycle_one=weak),
        weak_cycle_one=weak,
        arm="correct",
        steps_per_cycle=4,
        learning_rate=0.002,
        logical_run_id="r22-ephemeral-cpu-toy",
        base_checkpoint_sha256="a" * 64,
        prior_checkpoint_sha256="b" * 64,
        fence=4,
    )
    plan.validate()
    return plan


def _binding(plan):
    from labyrinth.maze_crash_resume_journal import MazeJournalBinding

    return MazeJournalBinding(
        run_id=plan.logical_run_id,
        objective_id=plan.scope.objective_id,
        arm=plan.arm,
        candidate_artifact_sha256="c" * 64,
        scope_fingerprint=plan.scope.fingerprint,
        base_checkpoint_sha256=plan.base_checkpoint_sha256,
        prior_checkpoint_sha256=plan.prior_checkpoint_sha256,
        checkpoint_run_id=plan.checkpoint_run_id,
        source_sha256=plan.plan_sha256,
        fence=plan.fence,
        cycle_count=2,
    )


def _new_model(torch: Any):
    torch.manual_seed(203)
    encoder = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 4))
    for parameter in encoder[0].parameters():
        parameter.requires_grad_(False)
    head = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad]
        + list(head.parameters()), lr=0.002)
    return encoder, head, optimizer


def _train_step(torch: Any, encoder: Any, head: Any, optimizer: Any,
                cycle: Any, step: int) -> None:
    from scripts.qualify_maze_conditioned_minilm_cpu import select_training_batch

    batch = select_training_batch(cycle, step)
    # The toy feature mapping is deliberately fixed, local, and deterministic;
    # it is NOT a tokenizer or pretrained language embedding.
    features = [
        [(byte / 127.5) - 1.0 for byte in sha256(row.prompt.encode("utf-8")).digest()[:3]]
        for row in batch
    ]
    inputs = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor([row.label for row in batch], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(head(encoder(inputs)), labels)
    loss.backward()
    optimizer.step()


def _state_fingerprint(torch: Any, encoder: Any, head: Any,
                       optimizer: Any) -> str:
    """Bind model weights, AdamW moments, parameter identities and hyperparameters.

    Matching moment tensors alone cannot prove optimizer continuity: a changed
    learning rate, decay, beta, or parameter-group membership changes later
    updates. This fingerprint is still a local toy oracle, not host custody.
    """
    def canonical_option(value: Any) -> Any:
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("nonfinite AdamW group option")
            return value
        if type(value) in (tuple, list):
            return [canonical_option(item) for item in value]
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("untyped AdamW group option key")
            return {key: canonical_option(item) for key, item in value.items()}
        raise ValueError("noncanonical AdamW group option")

    def update_tensor(h, label: str, tensor: Any) -> None:
        if not all(callable(getattr(tensor, method, None))
                   for method in ("detach", "cpu", "contiguous", "numpy")):
            raise ValueError("optimizer state or parameter is not a tensor")
        value = tensor.detach().cpu().contiguous()
        shape = tuple(getattr(value, "shape", ()))
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("invalid optimizer tensor shape")
        metadata = {"label": label, "dtype": str(getattr(value, "dtype", "toy-double")),
                    "shape": shape}
        raw = value.numpy().tobytes()
        h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True).encode("ascii") + b"\0")
        h.update(len(raw).to_bytes(8, "big") + raw)

    parameter_names = {}
    actual_named_modules = all(callable(getattr(module, "named_parameters", None))
                               for module in (encoder, head))
    if actual_named_modules:
        for prefix, module in (("encoder", encoder), ("head", head)):
            for name, parameter in module.named_parameters():
                if id(parameter) in parameter_names:
                    raise ValueError("same optimizer parameter under multiple model names")
                parameter_names[id(parameter)] = prefix + "/" + name

    h = sha256(b"labyrinth.r24-toy-cpu-fullstate.v2\0")
    for name, module in (("encoder", encoder), ("head", head)):
        for key, tensor in sorted(module.state_dict().items()):
            update_tensor(h, name + "/" + key, tensor)

    seen_parameters: set[int] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        if (type(group) is not dict or "params" not in group
                or type(group["params"]) not in (list, tuple)):
            raise ValueError("invalid AdamW parameter group")
        options = {key: canonical_option(value) for key, value in group.items()
                   if key != "params"}
        if any(type(key) is not str for key in group):
            raise ValueError("untyped AdamW group key")
        members = []
        for parameter_index, parameter in enumerate(group["params"]):
            identity = id(parameter)
            if identity in seen_parameters:
                raise ValueError("AdamW parameter duplicated across groups")
            seen_parameters.add(identity)
            if actual_named_modules:
                if identity not in parameter_names:
                    raise ValueError("AdamW parameter missing from named model")
                name = parameter_names[identity]
            else:
                # A positional identifier is sufficient for lightweight
                # source test doubles; real Torch modules require names.
                name = f"test-double/group-{group_index}/param-{parameter_index}"
            members.append(name)
            state = optimizer.state.get(parameter)
            if type(state) is not dict or not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
                raise ValueError("incomplete AdamW optimizer state")
            for key, value in sorted(state.items()):
                if type(key) is not str:
                    raise ValueError("untyped AdamW state key")
                update_tensor(h, "adamw/" + name + "/" + key, value)
        h.update(json.dumps({"group": group_index, "members": members,
                             "options": options}, sort_keys=True,
                            separators=(",", ":"), ensure_ascii=True,
                            allow_nan=False).encode("ascii") + b"\0")
    return h.hexdigest()


def execute_toy_cpu_smoke() -> dict[str, object]:
    """Run fresh-source Torch, AdamW, v2 journal, and restore in /tmp only."""
    absent = missing_dependencies()
    if absent:
        raise RuntimeError("optional CPU dependencies unavailable: " + ", ".join(absent))
    if os.name == "nt":
        raise RuntimeError("Windows journal fsync has not been qualified")

    import torch
    from labyrinth.maze_crash_resume_journal import (
        MazeJournalError, MazeJournalUnknown,
        commit_maze_cycle, initialize_maze_journal,
        inspect_maze_journal, precommit_maze_cycle,
    )
    from labyrinth.minilm_optimizer_state_resume import (
        load_maze_optimizer_snapshot, save_maze_optimizer_snapshot,
    )

    plan = _plan()
    encoder, head, optimizer = _new_model(torch)
    # Independent uninterrupted toy trajectory on the exact same TRAIN batch
    # schedule. This is an optimizer parity oracle, not an independent examiner.
    for cycle in plan.cycles:
        for step in range(plan.steps_per_cycle):
            _train_step(torch, encoder, head, optimizer, cycle, step)
    oracle_sha = _state_fingerprint(torch, encoder, head, optimizer)

    with tempfile.TemporaryDirectory(prefix="labyrinth-r22-toy-cpu-") as temp:
        root = Path(temp)
        journal_root = root / "journal-v2"
        binding = _binding(plan)
        initialize_maze_journal(journal_root, binding, schema_version="v2")
        encoder, head, optimizer = _new_model(torch)
        prior_sha = plan.prior_checkpoint_sha256
        checkpoint_shas: list[str] = []
        for index, cycle in enumerate(plan.cycles):
            completed_steps = (index + 1) * plan.steps_per_cycle
            context = plan.context(
                completed_steps, cycle_index=index,
                immediate_predecessor_sha256=prior_sha)
            if index == 1:
                foreign_sha = "d" * 64
                if foreign_sha == prior_sha:
                    foreign_sha = "e" * 64
                try:
                    precommit_maze_cycle(
                        journal_root, binding, index,
                        plan.context(completed_steps, cycle_index=index,
                                     immediate_predecessor_sha256=foreign_sha),
                        schema_version="v2")
                except MazeJournalError as exc:
                    if "immediate predecessor" not in str(exc):
                        raise
                else:
                    raise RuntimeError("foreign cycle-one predecessor was admitted")
            precommit_maze_cycle(journal_root, binding, index, context,
                                 schema_version="v2")
            try:
                precommit_maze_cycle(journal_root, binding, index, context,
                                     schema_version="v2")
            except MazeJournalUnknown:
                pass
            else:
                raise RuntimeError("unresolved pending intent was replayed")
            # A real process boundary is mimicked by constructing a fresh
            # module/optimizer and loading the *prior* cycle before cycle one.
            if index == 1:
                encoder, head, optimizer = _new_model(torch)
                load_maze_optimizer_snapshot(
                    root / "cycle-0", encoder=encoder, head=head,
                    optimizer=optimizer, plan=plan,
                    completed_steps=plan.steps_per_cycle,
                    independently_pinned_checkpoint_sha256=checkpoint_shas[0],
                    cycle_index=0,
                    immediate_predecessor_sha256=plan.prior_checkpoint_sha256,
                )
            for step in range(plan.steps_per_cycle):
                _train_step(torch, encoder, head, optimizer, cycle, step)
            checkpoint_path = root / f"cycle-{index}"
            receipt = save_maze_optimizer_snapshot(
                checkpoint_path, encoder=encoder, head=head,
                optimizer=optimizer, plan=plan,
                completed_steps=completed_steps,
                cycle_index=index,
                immediate_predecessor_sha256=prior_sha,
            )
            commit_maze_cycle(
                journal_root, binding, index, context, checkpoint_path,
                receipt.sha256, schema_version="v2")
            status = inspect_maze_journal(
                journal_root, binding, schema_version="v2")
            if (status.committed_cycles != index + 1
                    or status.last_checkpoint_sha256 != receipt.sha256
                    or status.pending_cycle is not None):
                raise RuntimeError("v2 source readback differs after commit")
            checkpoint_shas.append(receipt.sha256)
            prior_sha = receipt.sha256

        resumed_sha = _state_fingerprint(torch, encoder, head, optimizer)
        if resumed_sha != oracle_sha:
            raise RuntimeError("resumed toy AdamW trajectory differs from uninterrupted training")

        # Independent fresh-module restore of the final committed checkpoint.
        restored_encoder, restored_head, restored_optimizer = _new_model(torch)
        load_maze_optimizer_snapshot(
            root / "cycle-1", encoder=restored_encoder, head=restored_head,
            optimizer=restored_optimizer, plan=plan,
            completed_steps=plan.total_steps,
            independently_pinned_checkpoint_sha256=checkpoint_shas[1],
            cycle_index=1,
            immediate_predecessor_sha256=checkpoint_shas[0],
        )
        if _state_fingerprint(torch, restored_encoder, restored_head,
                              restored_optimizer) != oracle_sha:
            raise RuntimeError("final fresh optimizer restore differs from toy oracle")

    return {
        "status": STATUS,
        "physical_cpu_toy_executed": True,
        "pretrained_model_loaded": False,
        "maze_training_prompts_executed": True,
        "optimizer_fullstate_continuity_matches_uninterrupted_toy": True,
        "final_fresh_fullstate_restore_verified": True,
        "checkpoint_cycle_receipt_sha256": checkpoint_shas,
        "toy_final_state_sha256": oracle_sha,
        "synthetic_identity_pins": True,
        "independent_host_custody_verified": False,
        "independent_antirollback_verified": False,
        "neural_generalization_verified": False,
        "graduation_authorized": False,
        "persistent_host_artifacts": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="run ephemeral synthetic two-cycle CPU training")
    args = parser.parse_args(argv)
    absent = missing_dependencies()
    if not args.execute:
        print(json.dumps({
            "status": "PREFLIGHT_ONLY_NO_EFFECT",
            "execute": False,
            "missing_dependencies": absent,
            "physical_cpu_toy_executed": False,
            "pretrained_model_loaded": False,
            "graduation_authorized": False,
        }, sort_keys=True))
        return 0
    if absent or os.name == "nt":
        print(json.dumps({
            "status": "SKIPPED_NO_PHYSICAL_EXECUTION",
            "execute": True,
            "missing_dependencies": absent,
            "reason": "Windows journal fsync unqualified" if os.name == "nt"
                      else "optional CPU dependencies unavailable",
            "physical_cpu_toy_executed": False,
            "graduation_authorized": False,
        }, sort_keys=True))
        return 3
    print(json.dumps(execute_toy_cpu_smoke(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
