"""Opt-in CPU full-state *development* successor, not a graduation runner.

The host supplies a separately verified signed failure event, pinned source and
maze digests, an examiner-independent journal key, and byte-attestation. These
callbacks are trust boundaries, **not** proof that a remote host performed them.
Only the initial fence-0 run is handled here; a recovery claimant needs the
separately atomic journal fence transition before restoring a later snapshot.
The existing HF adapter-only journey is unchanged by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from labyrinth.hf_development_event_journal import (
    FullStateRunBinding,
    commit_development_run_checkpoint,
    inspect_development_event,
    reserve_full_state_development_run,
)
from labyrinth.neural_resume_checkpoint import (
    FullCheckpointContext,
    FullCheckpointError,
    FullCheckpointReceipt,
    load_full_checkpoint,
    save_full_checkpoint,
)


class FullStateSuccessorError(RuntimeError):
    """No verified successor commit occurred; published snapshots may be orphans."""


@dataclass(frozen=True)
class DevelopmentStepReceipt:
    """A trainer's reported exposures, not independently attested learning."""

    visited_exposures: tuple[tuple[str, str], ...]
    optimizer_step_calls: int
    cursor_before: int
    cursor_after: int


@dataclass(frozen=True)
class FullStateSuccessorReceipt:
    run_id: str
    checkpoint_sha256: str
    fence: int
    next_cursor: int
    status: str = "PROVISIONAL_SOURCE_ONLY"
    physical_model_verified: bool = False
    examiner_custody_verified: bool = False
    graduation_authorized: bool = False


def _cpu_model(model: Any) -> None:
    """Reject mutable buffers that the v1 snapshot would silently discard."""
    parameters = getattr(model, "parameters", None)
    buffers = getattr(model, "named_buffers", None)
    if not callable(parameters) or not callable(buffers):
        raise FullStateSuccessorError("CPU model inventory cannot be checked")
    values = list(parameters())
    if not values or any(getattr(getattr(p, "device", None), "type", None) != "cpu"
                         for p in values):
        raise FullStateSuccessorError("CPU-only model parameters required")
    if list(buffers()):
        raise FullStateSuccessorError("uncheckpointed model buffers forbid full-state continuation")


def _fresh_pair(
    model_factory: Callable[[], Any], optimizer_factory: Callable[[Any], Any],
) -> tuple[Any, Any]:
    model = model_factory()
    _cpu_model(model)
    optimizer = optimizer_factory(model)
    return model, optimizer


def _neural_fingerprint(model: Any, optimizer: Any) -> tuple[str, tuple[int, ...], str]:
    """Read actual trainable bytes and every AdamW parameter's step counter.

    A changed manifest cursor is insufficient evidence of an update. Requiring
    both changed trainable tensor bytes and matching AdamW step increments
    rejects a pure metadata advance. This still cannot attest the origin or
    correctness of the optimizer computation; the host must verify that.
    """
    try:
        import torch
        from labyrinth.hf_pilot import _parameter_digest, _frozen_parameter_digest
    except ImportError as exc:
        raise FullStateSuccessorError("physical CPU Torch is required") from exc
    if optimizer.__class__.__name__ != "AdamW":
        raise FullStateSuccessorError("only AdamW neural progress is supported")
    named = sorted((name, param) for name, param in model.named_parameters()
                   if param.requires_grad)
    if not named:
        raise FullStateSuccessorError("no trainable parameters for neural progress")
    counters: list[int] = []
    for _name, param in named:
        state = optimizer.state.get(param, {})
        value = state.get("step", 0)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise FullStateSuccessorError("invalid AdamW optimizer step tensor")
            value = value.item()
        if (type(value) not in (float, int) or value < 0
                or int(value) != value):
            raise FullStateSuccessorError("invalid AdamW optimizer step count")
        counters.append(int(value))
    frozen = any(not param.requires_grad for _name, param in model.named_parameters())
    frozen_sha256 = _frozen_parameter_digest(model, torch) if frozen else ""
    return _parameter_digest(model, torch), tuple(counters), frozen_sha256


def hf_fullstate_training_step(
    *, tokenizer: Any, torch: Any, train: list[Any], labels: list[str], config: Any,
) -> Callable[[Any, Any, tuple[tuple[str, str], ...], int], DevelopmentStepReceipt]:
    """Bind HF's externally owned AdamW loop to a global exposure cursor.

    This adapter relies on the optional external optimizer/cursor contract in
    ``hf_pilot._train_steps``. It does not itself create an optimizer, load a
    checkpoint, reseed RNG, or authorize a maze-regeneration event.
    """
    def train_window(model: Any, optimizer: Any,
                     planned: tuple[tuple[str, str], ...],
                     cursor: int) -> DevelopmentStepReceipt:
        from labyrinth.hf_pilot import _train_steps
        if getattr(config, "device", None) != "cpu":
            raise FullStateSuccessorError("CPU-only full-state HF training required")
        if (type(cursor) is not int or cursor < 0
                or type(planned) is not tuple or not planned
                or type(getattr(config, "steps", None)) is not int
                or config.steps != len(planned)):
            raise FullStateSuccessorError("HF training window and exposure cursor mismatch")
        result = _train_steps(
            model, tokenizer, torch, train, labels, config,
            optimizer=optimizer, rng_policy="preserve", scheduler=None,
            visit_cursor=cursor, expected_visit_exposures=planned,
        )
        if (type(result) is not dict
                or type(result.get("visited_exposures")) not in (tuple, list)
                or type(result.get("steps")) is not int
                or type(result.get("visit_cursor_before")) is not int
                or type(result.get("visit_cursor_after")) is not int):
            raise FullStateSuccessorError("HF external optimizer report incomplete")
        reported = tuple(tuple(pair) for pair in result["visited_exposures"])
        if (reported != planned or result["steps"] != len(planned)
                or result["visit_cursor_before"] != cursor
                or result["visit_cursor_after"] != cursor + len(planned)):
            raise FullStateSuccessorError("HF external optimizer exposure receipt mismatch")
        return DevelopmentStepReceipt(reported, result["steps"],
                                      result["visit_cursor_before"],
                                      result["visit_cursor_after"])

    return train_window


def run_fullstate_development_cycle(
    *,
    binding: FullStateRunBinding,
    event_id: str,
    journal_db: Path,
    state_key: bytes,
    initial_checkpoint_dir: Path,
    successor_checkpoint_dir: Path,
    model_factory: Callable[[], Any],
    optimizer_factory: Callable[[Any], Any],
    training_step: Callable[[Any, Any, tuple[tuple[str, str], ...], int], DevelopmentStepReceipt],
    step_count: int,
    verify_signed_event: Callable[[str, FullStateRunBinding], bool] | None,
    attest_successor_bytes: Callable[[Path, FullCheckpointReceipt], bool] | None,
) -> FullStateSuccessorReceipt:
    """Reserve, train, publish, read back, and CAS-commit one source-only cycle.

    `verify_signed_event` and `attest_successor_bytes` must be host-owned,
    independent of the training model and never inferred from in-process flags.
    The caller is responsible for ensuring that the training callback actually
    executes the declared optimizer steps and preserving the claimed schedule.
    A return value never establishes model provenance or private qualification.
    """
    if type(binding) is not FullStateRunBinding:
        raise FullStateSuccessorError("host-pinned full-state binding required")
    binding.validate()
    if (type(step_count) is not int or step_count <= 0
            or binding.initial_cursor + step_count > len(binding.initial_exposures)):
        raise FullStateSuccessorError("invalid bounded successor exposure cursor")
    if (not callable(verify_signed_event)
            or not callable(attest_successor_bytes)):
        raise FullStateSuccessorError("signed event and byte attestation require host verifiers")
    if not all(isinstance(path, Path) and path.is_absolute()
               for path in (journal_db, initial_checkpoint_dir, successor_checkpoint_dir)):
        raise FullStateSuccessorError("absolute checkpoint and host journal paths required")
    if (successor_checkpoint_dir.exists() or successor_checkpoint_dir.is_symlink()
            or not successor_checkpoint_dir.parent.is_dir()):
        raise FullStateSuccessorError("successor checkpoint destination must be new")
    if verify_signed_event(event_id, binding) is not True:
        raise FullStateSuccessorError("signed development event not verified by host")

    initial_context = FullCheckpointContext(
        run_id=binding.run_id,
        objective_id=binding.objective_id,
        base_checkpoint_sha256=binding.base_checkpoint_sha256,
        prior_checkpoint_sha256=binding.prior_checkpoint_sha256,
        fence=0, cursor=binding.initial_cursor,
        exposures=binding.initial_exposures,
    )
    model, optimizer = _fresh_pair(model_factory, optimizer_factory)
    try:
        initial = load_full_checkpoint(
            initial_checkpoint_dir, model=model, optimizer=optimizer,
            expected_sha256=binding.initial_checkpoint_sha256,
            expected_context=initial_context,
        )
    except FullCheckpointError as exc:
        raise FullStateSuccessorError(
            "full-state restore failed; discard this process"
        ) from exc
    if (type(initial) is not FullCheckpointReceipt
            or initial.sha256 != binding.initial_checkpoint_sha256
            or initial.context != initial_context
            or not initial.trainable_parameters):
        raise FullStateSuccessorError("trusted initial checkpoint readback mismatch")

    fence = reserve_full_state_development_run(
        journal_db, event_id, state_key=state_key,
        binding=binding, initial_receipt=initial,
    )
    if fence != 0:
        raise FullStateSuccessorError("unexpected initial run fence")
    planned = binding.initial_exposures[
        binding.initial_cursor:binding.initial_cursor + step_count
    ]
    before_neural = _neural_fingerprint(model, optimizer)
    progress = training_step(model, optimizer, planned, binding.initial_cursor)
    if (type(progress) is not DevelopmentStepReceipt
            or progress.visited_exposures != planned
            or type(progress.optimizer_step_calls) is not int
            or progress.optimizer_step_calls != step_count
            or type(progress.cursor_before) is not int
            or type(progress.cursor_after) is not int
            or progress.cursor_before != binding.initial_cursor
            or progress.cursor_after != binding.initial_cursor + step_count):
        raise FullStateSuccessorError("training exposure, cursor or optimizer-step receipt mismatch")
    _cpu_model(model)
    after_neural = _neural_fingerprint(model, optimizer)
    if (type(before_neural) is not tuple or type(after_neural) is not tuple
            or len(before_neural) != 3 or len(after_neural) != 3
            or type(before_neural[0]) is not str or type(after_neural[0]) is not str
            or before_neural[0] == after_neural[0]
            or type(before_neural[1]) is not tuple
            or type(after_neural[1]) is not tuple
            or not before_neural[1]
            or len(before_neural[1]) != len(after_neural[1])
            or any(type(old) is not int or type(new) is not int
                   or new - old != step_count
                   for old, new in zip(before_neural[1], after_neural[1]))):
        raise FullStateSuccessorError(
            "neural parameter or AdamW optimizer step continuity not demonstrated"
        )
    if (type(before_neural[2]) is not str or type(after_neural[2]) is not str
            or before_neural[2] != after_neural[2]):
        raise FullStateSuccessorError("frozen base neural parameter bytes changed")

    successor_context = FullCheckpointContext(
        run_id=binding.run_id,
        objective_id=binding.objective_id,
        base_checkpoint_sha256=binding.base_checkpoint_sha256,
        prior_checkpoint_sha256=initial.sha256,
        fence=fence,
        cursor=binding.initial_cursor + step_count,
        exposures=binding.initial_exposures,
    )
    receipt = save_full_checkpoint(
        successor_checkpoint_dir, model=model, optimizer=optimizer,
        context=successor_context,
    )
    if (type(receipt) is not FullCheckpointReceipt
            or receipt.context != successor_context
            or receipt.trainable_parameters != initial.trainable_parameters
            or type(receipt.sha256) is not str or len(receipt.sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in receipt.sha256)
            or receipt.sha256 == initial.sha256):
        raise FullStateSuccessorError("untrusted or unchanged successor checkpoint receipt")

    # A distinct fresh model and optimizer validate the actual persisted bytes.
    # This restores the saved RNG in the readback worker too; the trainer must
    # not resume using this process after this method returns.
    readback_model, readback_optimizer = _fresh_pair(model_factory, optimizer_factory)
    try:
        readback = load_full_checkpoint(
            successor_checkpoint_dir, model=readback_model,
            optimizer=readback_optimizer, expected_sha256=receipt.sha256,
            expected_context=successor_context,
        )
    except FullCheckpointError as exc:
        raise FullStateSuccessorError(
            "successor byte readback failed; discard this process; orphan not committed"
        ) from exc
    if (type(readback) is not FullCheckpointReceipt
            or readback.sha256 != receipt.sha256
            or readback.context != successor_context
            or readback.trainable_parameters != receipt.trainable_parameters):
        raise FullStateSuccessorError("successor byte readback identity mismatch; orphan uncommitted")
    readback_neural = _neural_fingerprint(readback_model, readback_optimizer)
    if (type(readback_neural) is not tuple or len(readback_neural) != 3
            or readback_neural != after_neural):
        raise FullStateSuccessorError(
            "readback trainable, frozen or AdamW state differs; orphan uncommitted"
        )
    if attest_successor_bytes(successor_checkpoint_dir, readback) is not True:
        raise FullStateSuccessorError("independent host byte attestation missing; orphan uncommitted")

    record = inspect_development_event(journal_db, event_id, state_key=state_key)
    if (not isinstance(record, dict)
            or record.get("run_state") != "ACTIVE"
            or record.get("run_id") != binding.run_id
            or record.get("run_commitment_sha256") != binding.commitment_sha256()
            or record.get("checkpoint_sha256") != initial.sha256
            or record.get("fence") != fence):
        raise FullStateSuccessorError("fresh host journal fence mismatch; orphan uncommitted")
    # This CAS is the only point at which the new checkpoint becomes canonical.
    commit_development_run_checkpoint(
        journal_db, event_id, state_key=state_key,
        run_id=binding.run_id,
        run_commitment_sha256=binding.commitment_sha256(),
        checkpoint_sha256=initial.sha256,
        expected_fence=fence,
        next_checkpoint_sha256=receipt.sha256,
    )
    return FullStateSuccessorReceipt(binding.run_id, receipt.sha256,
                                     fence, successor_context.cursor)
