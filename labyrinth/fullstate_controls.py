"""Source contract for four isolated CPU full-state development controls.

All four arms must restore the same pinned full checkpoint *inside different OS
processes*, using fresh model/optimizer factories. This module does not launch,
attest, or supervise those processes and does not perform training. In
particular, distinct PID claims are not independently authenticated process
evidence. A host-owned runner must launch children, protect the origin and
journal, attest executed model bytes, and keep every control's output separate
from the authoritative candidate lineage.

The only neural I/O here is ``load_full_checkpoint``. There is deliberately no
checkpoint publication path. Every resulting report remains development-only;
no source-level plan or claimed PID authorizes graduation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .neural_resume_checkpoint import (
    FullCheckpointContext, FullCheckpointError, FullCheckpointReceipt,
    load_full_checkpoint,
)


ARMS = ("baseline", "zero_update", "correct_labels", "corrupted_labels")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_WORKER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def _valid_origin(receipt: FullCheckpointReceipt) -> None:
    if (type(receipt) is not FullCheckpointReceipt
            or type(receipt.sha256) is not str
            or not _DIGEST.fullmatch(receipt.sha256)
            or type(receipt.context) is not FullCheckpointContext
            or receipt.graduation_authorized is not False):
        raise ValueError("a non-graduating, pinned full checkpoint receipt is required")
    receipt.context.validate()
    names = receipt.trainable_parameters
    if (type(names) is not tuple or not names or len(set(names)) != len(names)
            or any(type(name) is not str or not name for name in names)):
        raise ValueError("invalid full checkpoint trainable inventory")


def _nonce(receipt: FullCheckpointReceipt, arm: str, worker_id: str) -> str:
    # Non-secret domain separation, *not* a signature or authentication token.
    context = receipt.context
    fields = (
        receipt.sha256, context.run_id, context.objective_id,
        context.base_checkpoint_sha256, context.prior_checkpoint_sha256,
        context.fence, context.cursor, context.exposures,
        arm, worker_id,
    )
    data = json.dumps(fields, separators=(",", ":"), ensure_ascii=False)
    return sha256(b"labyrinth.fullstate-control-fork.v1\0" + data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ControlForkSpec:
    arm: str
    worker_id: str
    fork_nonce: str
    origin_sha256: str
    origin_context: FullCheckpointContext
    training_authoritative: bool = False
    graduation_authorized: bool = False


@dataclass(frozen=True)
class ControlForkPlan:
    origin_dir: Path
    origin_receipt: FullCheckpointReceipt
    parent_pid: int
    forks: tuple[ControlForkSpec, ...]
    training_authoritative: bool = False
    process_isolation_verified: bool = False
    graduation_authorized: bool = False

    def validate(self) -> None:
        _valid_origin(self.origin_receipt)
        if (not isinstance(self.origin_dir, Path) or not self.origin_dir.is_absolute()
                or type(self.parent_pid) is not int or self.parent_pid <= 0
                or type(self.forks) is not tuple or len(self.forks) != len(ARMS)
                or any(type(f) is not ControlForkSpec for f in self.forks)
                or self.training_authoritative is not False
                or self.process_isolation_verified is not False
                or self.graduation_authorized is not False):
            raise ValueError("invalid four-arm control plan")
        if tuple(f.arm for f in self.forks) != ARMS:
            raise ValueError("control plan must contain canonical four arms")
        if len({f.worker_id for f in self.forks}) != len(ARMS):
            raise ValueError("control forks require distinct worker IDs")
        for fork in self.forks:
            if (type(fork.worker_id) is not str or not _WORKER.fullmatch(fork.worker_id)
                    or fork.worker_id == self.origin_receipt.context.run_id
                    or fork.origin_sha256 != self.origin_receipt.sha256
                    or fork.origin_context != self.origin_receipt.context
                    or fork.fork_nonce != _nonce(self.origin_receipt, fork.arm, fork.worker_id)
                    or fork.training_authoritative is not False
                    or fork.graduation_authorized is not False):
                raise ValueError("control fork differs from pinned origin")


def build_control_plan(
    origin_receipt: FullCheckpointReceipt,
    origin_dir: Path,
    worker_ids: Mapping[str, str],
) -> ControlForkPlan:
    """Create a deterministic, immutable plan; do not claim origin readback."""
    _valid_origin(origin_receipt)
    if not isinstance(origin_dir, Path) or not origin_dir.is_absolute():
        raise ValueError("control checkpoint path must be absolute")
    if (not isinstance(worker_ids, Mapping)
            or set(worker_ids) != set(ARMS)):
        raise ValueError("all four control arms require individual worker IDs")
    if (any(type(worker_ids[arm]) is not str
            or not _WORKER.fullmatch(worker_ids[arm])
            or worker_ids[arm] == origin_receipt.context.run_id for arm in ARMS)
            or len(set(worker_ids.values())) != len(ARMS)):
        raise ValueError("control arms require distinct valid worker IDs")
    forks = tuple(ControlForkSpec(
        arm, worker_ids[arm], _nonce(origin_receipt, arm, worker_ids[arm]),
        origin_receipt.sha256, origin_receipt.context,
    ) for arm in ARMS)
    plan = ControlForkPlan(origin_dir, origin_receipt, os.getpid(), forks)
    plan.validate()
    return plan


@dataclass(frozen=True)
class ControlStartReceipt:
    arm: str
    worker_id: str
    fork_nonce: str
    pid: int
    origin_sha256: str
    origin_context: FullCheckpointContext
    process_isolation_verified: bool = False
    training_authoritative: bool = False
    graduation_authorized: bool = False


@dataclass(frozen=True)
class ControlLoadedRuntime:
    # Model and optimizer are intentionally mutable *inside* the child.
    model: Any = field(repr=False, compare=False)
    optimizer: Any = field(repr=False, compare=False)
    start: ControlStartReceipt


def load_control_child(
    plan: ControlForkPlan,
    arm: str,
    model_factory: Callable[[], Any],
    optimizer_factory: Callable[[Any], Any],
) -> ControlLoadedRuntime:
    """Restore pinned state in an OS child before any control training step.

    The caller must discard the child after any restore error. This checks that
    execution occurs in a different PID, not that the process is externally
    attested or that the factories produce the claimed pretrained model.
    """
    if type(plan) is not ControlForkPlan:
        raise ValueError("invalid control plan")
    plan.validate()
    fork = next((item for item in plan.forks if item.arm == arm), None)
    if fork is None:
        raise ValueError("unknown control arm")
    pid = os.getpid()
    if pid == plan.parent_pid:
        raise ValueError("a control must load in a separate process")
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid child process identity")
    if not callable(model_factory) or not callable(optimizer_factory):
        raise ValueError("fresh model and optimizer factories are required")
    model = model_factory()
    optimizer = optimizer_factory(model)
    if model is None or optimizer is None:
        raise ValueError("factories did not create a control model and optimizer")
    loaded = load_full_checkpoint(
        plan.origin_dir, model=model, optimizer=optimizer,
        expected_sha256=plan.origin_receipt.sha256,
        expected_context=plan.origin_receipt.context,
    )
    if loaded != plan.origin_receipt:
        raise FullCheckpointError("loaded origin receipt differs from pinned control origin; discard child")
    start = ControlStartReceipt(
        fork.arm, fork.worker_id, fork.fork_nonce, pid,
        loaded.sha256, loaded.context,
    )
    return ControlLoadedRuntime(model, optimizer, start)


@dataclass(frozen=True)
class ControlStartSummary:
    arm_count: int
    origin_sha256: str
    origin_context: FullCheckpointContext
    process_isolation_verified: bool = False
    training_authoritative: bool = False
    graduation_authorized: bool = False


def verify_control_starts(
    plan: ControlForkPlan, starts: tuple[ControlStartReceipt, ...]
) -> ControlStartSummary:
    """Check completeness/consistency of **claims**, not OS or model attestation."""
    if type(plan) is not ControlForkPlan:
        raise ValueError("invalid control plan")
    plan.validate()
    if (type(starts) is not tuple or len(starts) != len(ARMS)
            or any(type(start) is not ControlStartReceipt for start in starts)):
        raise ValueError("four control start receipts are required")
    by_arm = {start.arm: start for start in starts}
    if len(by_arm) != len(ARMS) or set(by_arm) != set(ARMS):
        raise ValueError("four distinct control arms are required")
    if (len({start.pid for start in starts}) != len(ARMS)
            or any(type(start.pid) is not int or start.pid <= 0
                   or start.pid == plan.parent_pid for start in starts)):
        raise ValueError("four distinct child process IDs are required")
    for fork in plan.forks:
        start = by_arm[fork.arm]
        if (start.worker_id != fork.worker_id or start.fork_nonce != fork.fork_nonce
                or start.origin_sha256 != fork.origin_sha256
                or start.origin_context != fork.origin_context
                or start.process_isolation_verified is not False
                or start.training_authoritative is not False
                or start.graduation_authorized is not False):
            raise ValueError("control start does not match pinned fork plan")
    return ControlStartSummary(len(ARMS), plan.origin_receipt.sha256,
                               plan.origin_receipt.context)
