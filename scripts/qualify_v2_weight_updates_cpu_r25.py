"""Opt-in physical *toy* Maze weight-update evidence, independent of replay parity.

The R22 uninterrupted-vs-resumed fingerprint comparison can pass if BOTH paths
perform zero updates. This R25 probe closes that specific evidentiary gap by
observing real trainable tensors, frozen tensors, and AdamW moments at the
start and after each generated Maze cycle. It does not load pretrained MiniLM,
establish host custody, validate generalization, or authorize graduation.

Without ``--execute`` this module performs a strictly read-only preflight.
The physical path is ephemeral and runs only when Torch is installed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import math
from typing import Any


STATUS = "EPHEMERAL_TOY_CPU_ACTUAL_WEIGHT_UPDATES_SOURCE_ONLY"


def missing_dependencies() -> tuple[str, ...]:
    return tuple(name for name in ("torch",)
                 if importlib.util.find_spec(name) is None)


def _require_finite_tensor(tensor: Any, label: str) -> None:
    """Reject NaN/Inf (or an unsupported finiteness probe) before hashing."""
    isfinite = getattr(tensor, "isfinite", None)
    if not callable(isfinite):
        raise ValueError(f"tensor finiteness unavailable: {label}")
    finite_mask = isfinite()
    all_values = getattr(finite_mask, "all", None)
    if not callable(all_values):
        raise ValueError(f"tensor finiteness unavailable: {label}")
    scalar = all_values()
    item = getattr(scalar, "item", None)
    if not callable(item):
        raise ValueError(f"tensor finiteness unavailable: {label}")
    finite = item()
    if type(finite) is not bool:
        raise ValueError(f"tensor finiteness unavailable: {label}")
    if not finite:
        raise ValueError(f"nonfinite tensor evidence: {label}")


def _digest_tensors(named: list[tuple[str, Any]]) -> str:
    if not named:
        raise ValueError("empty tensor inventory")
    digest = sha256(b"labyrinth.r25-weight-evidence.v1\0")
    for name, tensor in named:
        value = tensor.detach().cpu().contiguous()
        _require_finite_tensor(value, name)
        raw = value.numpy().tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(len(raw).to_bytes(8, "big") + raw)
    return digest.hexdigest()


def _require_finite_gradients(*modules: Any) -> None:
    observed = 0
    for module_index, module in enumerate(modules):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                raise ValueError(f"missing trainable gradient: {module_index}/{name}")
            _require_finite_tensor(parameter.grad,
                                   f"gradient/{module_index}/{name}")
            observed += 1
    if not observed:
        raise ValueError("no trainable gradients observed")


def _parameter_digests(encoder: Any, head: Any) -> tuple[str, str]:
    frozen: list[tuple[str, Any]] = []
    trainable: list[tuple[str, Any]] = []
    for prefix, module in (("encoder", encoder), ("head", head)):
        for name, parameter in module.named_parameters():
            pair = (prefix + "/" + name, parameter)
            (trainable if parameter.requires_grad else frozen).append(pair)
    return _digest_tensors(frozen), _digest_tensors(trainable)


def _optimizer_moment_digest(optimizer: Any, expected_steps: int) -> str:
    named: list[tuple[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        for param_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            if not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
                raise ValueError("AdamW moments were not initialized")
            step = state["step"]
            raw_step = (step.item() if callable(getattr(step, "item", None))
                        else step)
            if (type(raw_step) not in (int, float) or not math.isfinite(raw_step)
                    or raw_step < 0 or raw_step != int(raw_step)):
                raise ValueError("AdamW step cursor is not an exact integer")
            observed = int(raw_step)
            if observed != expected_steps:
                raise ValueError("AdamW step cursor differs from actual training plan")
            for moment in ("exp_avg", "exp_avg_sq"):
                named.append((f"{group_index}/{param_index}/{moment}",
                              state[moment]))
    return _digest_tensors(named)


def validate_observations(*, frozen: tuple[str, str, str],
                          trainable: tuple[str, str, str],
                          moments: tuple[str, str],
                          steps: tuple[int, int],
                          steps_per_cycle: int) -> None:
    """Fail closed if a no-op can masquerade as two physical learning cycles."""
    all_digests = frozen + trainable + moments
    if any(type(value) is not str or len(value) != 64
           or any(char not in "0123456789abcdef" for char in value)
           for value in all_digests):
        raise ValueError("invalid tensor evidence digest")
    if type(steps_per_cycle) is not int or steps_per_cycle <= 0:
        raise ValueError("invalid training cadence")
    if steps != (steps_per_cycle, 2 * steps_per_cycle):
        raise ValueError("optimizer step cursor does not cover both cycles")
    if not frozen[0] == frozen[1] == frozen[2]:
        raise ValueError("supposedly frozen base weights changed")
    if trainable[0] == trainable[1]:
        raise ValueError("cycle zero made no trainable tensor update")
    if trainable[1] == trainable[2]:
        raise ValueError("regenerated cycle made no trainable tensor update")
    if moments[0] == moments[1]:
        raise ValueError("optimizer moments did not advance between cycles")


def execute_weight_update_probe() -> dict[str, object]:
    absent = missing_dependencies()
    if absent:
        raise RuntimeError("optional CPU dependencies unavailable: " + ", ".join(absent))

    import torch
    from scripts.qualify_v2_checkpoint_cpu_r22 import (
        _new_model, _plan, _train_step,
    )

    torch.set_num_threads(1)
    torch.manual_seed(203)
    plan = _plan()
    encoder, head, optimizer = _new_model(torch)
    initial_frozen, initial_trainable = _parameter_digests(encoder, head)
    frozen = [initial_frozen]
    trainable = [initial_trainable]
    moments = []
    steps = []
    for index, cycle in enumerate(plan.cycles):
        for step in range(plan.steps_per_cycle):
            _train_step(torch, encoder, head, optimizer, cycle, step)
            _require_finite_gradients(encoder, head)
        frozen_sha, trained_sha = _parameter_digests(encoder, head)
        frozen.append(frozen_sha)
        trainable.append(trained_sha)
        completed_steps = (index + 1) * plan.steps_per_cycle
        moments.append(_optimizer_moment_digest(optimizer, completed_steps))
        steps.append(completed_steps)

    validate_observations(
        frozen=tuple(frozen), trainable=tuple(trainable),
        moments=tuple(moments), steps=tuple(steps),
        steps_per_cycle=plan.steps_per_cycle,
    )
    return {
        "status": STATUS,
        "physical_cpu_toy_executed": True,
        "actual_trainable_tensor_updates_both_cycles": True,
        "frozen_base_weights_unchanged": True,
        "adamw_moments_advanced_both_cycles": True,
        "frozen_tensor_sha256": frozen,
        "trainable_tensor_sha256": trainable,
        "optimizer_moments_sha256": moments,
        "optimizer_steps": steps,
        "pretrained_model_loaded": False,
        "host_custody_verified": False,
        "neural_generalization_verified": False,
        "graduation_authorized": False,
        "persistent_host_artifacts": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    absent = missing_dependencies()
    if not args.execute:
        print(json.dumps({"status": "PREFLIGHT_ONLY_NO_EFFECT",
                          "execute": False, "missing_dependencies": absent,
                          "physical_cpu_toy_executed": False,
                          "graduation_authorized": False}, sort_keys=True))
        return 0
    if absent:
        print(json.dumps({"status": "SKIPPED_NO_PHYSICAL_EXECUTION",
                          "execute": True, "missing_dependencies": absent,
                          "reason": "optional CPU dependencies unavailable",
                          "physical_cpu_toy_executed": False,
                          "graduation_authorized": False}, sort_keys=True))
        return 3
    print(json.dumps(execute_weight_update_probe(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
