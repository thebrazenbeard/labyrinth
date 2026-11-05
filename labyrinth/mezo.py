"""Safe two-sided zeroth-order step for **mutable floating** Torch parameters.

This is MeZO-style finite-difference SGD, *not* AdaMeZO, LOZO+, or AdamW.
The implementation trades one exact snapshot of **trainable** weights for
rollback guarantees. On an ordinary PEFT/LoRA model this is adapter-sized;
it is NOT the inference-memory algorithm for unrestricted full-model tuning.

See Malladi et al., NeurIPS 2023, "Fine-Tuning Language Models with Just
Forward Passes", Algorithm 1. The caller must supply a deterministic scalar
loss over the same batch for both evaluations and place the model in eval()
mode to disable dropout. No autograd graph or optimizer momentum is created.
"""

from __future__ import annotations

import math
from typing import Any, Callable


def _validate_hyperparameters(epsilon: float, learning_rate: float, seed: int) -> None:
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("MeZO epsilon must be finite and strictly positive")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("MeZO learning_rate must be finite and nonnegative")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("MeZO seed must be a nonnegative 63-bit integer")


def _trainable_parameters(model: Any, torch: Any) -> list[tuple[str, Any]]:
    selected: list[tuple[str, Any]] = []
    seen_objects: set[int] = set()
    # Include frozen Parameters in the storage map. Two distinct wrappers may
    # share underlying bytes even if only one advertises requires_grad=True.
    seen_storage: dict[tuple[str, int], bool] = {}
    for name, parameter in model.named_parameters():
        if id(parameter) in seen_objects:
            continue
        seen_objects.add(id(parameter))
        # Reject trainable-vs-trainable AND trainable-vs-frozen aliases before
        # modifying anything. Frozen-only aliasing is harmless to this step.
        storage = parameter.untyped_storage() if hasattr(parameter, "untyped_storage") else None
        if storage is not None:
            key = (str(parameter.device), storage.data_ptr())
            if key in seen_storage and (seen_storage[key] or parameter.requires_grad):
                raise ValueError("MeZO refuses overlapping trainable parameter storage")
            seen_storage[key] = bool(parameter.requires_grad)
        if not parameter.requires_grad:
            continue
        if parameter.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"MeZO needs fp32/fp64 trainable weights: {name}")
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"non-finite initial trainable parameter: {name}")
        selected.append((name, parameter))
    if not selected:
        raise ValueError("MeZO requires at least one trainable parameter")
    return selected


def mezo_step(
    model: Any,
    torch: Any,
    loss_fn: Callable[[], Any],
    *,
    epsilon: float,
    learning_rate: float,
    seed: int,
) -> dict[str, float | int]:
    """Execute one seeded symmetric two-forward finite-difference update.

    The result's ``loss`` is the *mean paired perturbed loss*, not a third
    unperturbed forward evaluation. On any failure, trainable weights are
    restored bit-for-bit to the pre-step copies. Frozen weights are untouched.
    """
    _validate_hyperparameters(epsilon, learning_rate, seed)
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    if getattr(model, "training", False):
        raise ValueError("MeZO requires model.eval() for paired deterministic losses")

    with torch.no_grad():
        parameters = _trainable_parameters(model, torch)
        originals = [(parameter, parameter.detach().clone()) for _, parameter in parameters]
        successful = False

        def noises():
            generators: dict[str, Any] = {}
            for _, parameter in parameters:
                key = str(parameter.device)
                if key not in generators:
                    generators[key] = torch.Generator(device=parameter.device).manual_seed(seed)
                yield torch.randn(parameter.shape, dtype=parameter.dtype,
                                  device=parameter.device, generator=generators[key])

        def perturb(sign: int) -> None:
            any_representable = False
            for (parameter, original), noise in zip(originals, noises()):
                parameter.copy_(original)
                parameter.add_(noise, alpha=sign * epsilon)
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError("non-finite MeZO perturbed parameter")
                if hasattr(torch, "equal"):
                    any_representable |= not bool(torch.equal(parameter, original))
                else:
                    # A minimal tensor facade need not expose torch.equal.
                    any_representable = True
            if not any_representable:
                raise RuntimeError("MeZO epsilon produces no representable perturbation")

        def scalar_loss() -> float:
            observed = loss_fn()
            value = float(observed.item() if hasattr(observed, "item") else observed)
            if not math.isfinite(value):
                raise RuntimeError("non-finite MeZO objective loss")
            return value

        try:
            perturb(+1)
            positive = scalar_loss()
            perturb(-1)
            negative = scalar_loss()
            projected_gradient = (positive - negative) / (2.0 * epsilon)
            if not math.isfinite(projected_gradient):
                raise RuntimeError("non-finite MeZO projected gradient")
            scale = -learning_rate * projected_gradient
            if not math.isfinite(scale):
                raise RuntimeError("non-finite MeZO update scale")
            paired_loss = positive / 2.0 + negative / 2.0
            if not math.isfinite(paired_loss):
                raise RuntimeError("non-finite MeZO mean paired loss")
            # RNG replay gives the identical z used in the +/- evaluations.
            for (parameter, original), noise in zip(originals, noises()):
                parameter.copy_(original)
                if learning_rate != 0.0:
                    parameter.add_(noise, alpha=scale)
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError("non-finite MeZO updated parameter")
            successful = True
            return {
                "loss": paired_loss,
                "loss_plus": positive,
                "loss_minus": negative,
                "projected_grad": projected_gradient,
                "forward_evaluations": 2,
            }
        finally:
            if not successful:
                for parameter, original in originals:
                    parameter.copy_(original)
