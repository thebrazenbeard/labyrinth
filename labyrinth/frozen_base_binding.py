"""Opt-in identity binding for *loaded* frozen neural tensors.

The full-state checkpoint substrate authenticates trainable parameters, AdamW
state, and RNG, but does not include the frozen pretrained backbone. A caller
that needs to restore those parameters must first verify a separately pinned
SHA-256 over the actual CPU-resident frozen parameters and model buffers.

This module provides that optional boundary. Existing checkpoint formats and
earlier physical development receipts are unchanged. It is NOT automatically
installed on a training host; it is not a publisher signature, does not attest
the origin of model files, and does not grant evaluation/graduation authority.
The caller must own the model exclusively while hashing/restoring it and
discard the process after any restoration failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_DOMAIN = b"labyrinth.loaded-frozen-model-tensors.v1\0"
_MAX_TENSORS = 50_000
_MAX_BYTES = 16 * 1024 * 1024 * 1024


class FrozenBaseError(ValueError):
    """The loaded base identity cannot be established or has changed."""


def _append_field(digest: Any, data: bytes | memoryview) -> None:
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def digest_frozen_model_tensors(model: Any) -> str:
    """Measure frozen parameters and every registered buffer in a CPU model.

    Includes names, tensor kinds, dtypes, shapes and raw bytes in sorted order.
    Trainable parameters are intentionally omitted so expected learning updates
    do not invalidate the frozen backbone identity. Buffers are included because
    they can affect inference even when no gradient is calculated for them.
    """
    try:
        import torch
    except ImportError as exc:
        raise FrozenBaseError("CPU PyTorch is required for loaded-model identity") from exc

    if not isinstance(model, torch.nn.Module):
        raise FrozenBaseError("expected a PyTorch module")
    candidates: list[tuple[str, str, Any]] = []
    candidates.extend(("parameter", name, p) for name, p in
                      model.named_parameters(remove_duplicate=False)
                      if not p.requires_grad)
    candidates.extend(("buffer", name, p) for name, p in
                      model.named_buffers(remove_duplicate=False))
    if not candidates or len(candidates) > _MAX_TENSORS:
        raise FrozenBaseError("invalid or empty frozen tensor inventory")

    digest = sha256(_DOMAIN)
    names: set[str] = set()
    aggregate_bytes = 0
    for kind, name, tensor in sorted(candidates, key=lambda item: (item[1], item[0])):
        if (type(name) is not str or not name or name in names
                or not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu" or tensor.layout != torch.strided):
            raise FrozenBaseError("unsafe frozen tensor inventory")
        names.add(name)
        byte_size = int(tensor.numel()) * int(tensor.element_size())
        if byte_size < 0 or byte_size > _MAX_BYTES - aggregate_bytes:
            raise FrozenBaseError("frozen tensor inventory exceeds byte limit")
        aggregate_bytes += byte_size
        # Bitcast to uint8 to support BF16 and other dtypes whose NumPy
        # conversion is unavailable. Keep a live CPU view; avoid moving tensors
        # across devices or depending on platform float formatting.
        try:
            with torch.no_grad():
                raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise FrozenBaseError("unsupported frozen tensor storage") from exc

        _append_field(digest, kind.encode("ascii"))
        _append_field(digest, name.encode("utf-8"))
        _append_field(digest, str(tensor.dtype).encode("ascii"))
        _append_field(digest, repr(tuple(tensor.shape)).encode("ascii"))
        _append_field(digest, memoryview(raw))
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenBaseIdentity:
    """SHA-256 pinned by a custodian other than the restoring worker."""

    digest: str

    def __post_init__(self) -> None:
        if type(self.digest) is not str or not _DIGEST.fullmatch(self.digest):
            raise ValueError("invalid independently pinned frozen-model SHA-256")

    def verify(self, model: Any) -> str:
        observed = digest_frozen_model_tensors(model)
        if observed != self.digest:
            raise FrozenBaseError("loaded frozen model differs from pinned tensor identity")
        return observed


def load_checkpoint_with_frozen_binding(
    source: Any, *, model: Any, optimizer: Any, expected_sha256: str,
    expected_context: Any, frozen_identity: FrozenBaseIdentity,
) -> Any:
    """Verify a frozen base before and after restoring exact trainable state.

    A new caller using this stronger contract must set its checkpoint context's
    ``base_checkpoint_sha256`` to the verified loaded-tensor digest. Old pilot
    snapshots used a different hash convention and are not silently migrated.
    """
    from labyrinth.neural_resume_checkpoint import (
        FullCheckpointContext,
        FullCheckpointError,
        load_full_checkpoint,
    )

    if type(frozen_identity) is not FrozenBaseIdentity:
        raise FrozenBaseError("an independent frozen tensor identity is required")
    if (type(expected_context) is not FullCheckpointContext
            or expected_context.base_checkpoint_sha256 != frozen_identity.digest):
        raise FrozenBaseError("checkpoint context is not bound to pinned frozen base")
    frozen_identity.verify(model)
    receipt = load_full_checkpoint(
        source, model=model, optimizer=optimizer,
        expected_sha256=expected_sha256,
        expected_context=expected_context,
    )
    # A custom model/load hook could mutate frozen tensors during restoration.
    # Detect that drift before allowing the restored learner to continue.
    try:
        frozen_identity.verify(model)
    except FrozenBaseError as exc:
        raise FullCheckpointError(
            "frozen model changed during restoration; discard the process"
        ) from exc
    return receipt
