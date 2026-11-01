"""Reopen an externally pinned measured TRAIN schedule at the execution edge.

An in-process ``frozen=True`` schedule dataclass is still mutable through
``object.__setattr__``. The trainer must never trust a previously verified
object after crossing an execution boundary. This adapter instead accepts
exact, bounded canonical *bytes* and separate caller-supplied expected pins,
reconstructs/validates the authentic Maze at each read, and returns freshly
copied built-in scalar tuples rather than mutable model objects.

SOURCE_ONLY_UNATTESTED: this module cannot prove who controls the pins,
where the bytes were stored, whether a different process actually trained on
these rows, or that the host prevents rollback and substitution. Those require
independent host authority and process-boundary execution evidence.
"""

from __future__ import annotations

from hashlib import sha256
import re

from .maze_conditioned_minilm_cpu_experiment import (
    PREREG_ARMS, load_measured_cli_schedule,
)
from .training_scope import TrainingScopeSpec


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def resolve_source_pinned_train_batch(
    raw: bytes, *, scope: TrainingScopeSpec, steps_per_cycle: int,
    expected_raw_sha256: str, expected_schedule_sha256: str,
    expected_run_id: str, expected_candidate_artifact_sha256: str,
    expected_predecessor_checkpoint_sha256: str,
    expected_scope_fingerprint: str, expected_fence: int,
    arm: str, step: int,
) -> tuple[tuple[str, str, int], ...]:
    """Return one immutable source TRAIN batch only after fresh byte validation.

    Caller must get all ``expected_*`` values from an independent custodian;
    deriving them from ``raw`` here merely creates a self-authenticating hash.
    ``arm`` selects one of five matched visits. Label transformations for
    control arms occur later in the experiment's independently audited runner.
    """
    if (type(raw) is not bytes or not 1 <= len(raw) <= 65_536
            or type(expected_raw_sha256) is not str
            or _SHA256.fullmatch(expected_raw_sha256) is None
            or sha256(raw).hexdigest() != expected_raw_sha256):
        raise ValueError("measured TRAIN schedule differs from pinned source bytes")
    if (type(scope) is not TrainingScopeSpec
            or type(expected_scope_fingerprint) is not str
            or _SHA256.fullmatch(expected_scope_fingerprint) is None
            or scope.fingerprint != expected_scope_fingerprint):
        raise ValueError("measured TRAIN scope differs from external pin")
    if (type(steps_per_cycle) is not int
            or type(step) is not int
            or not 0 <= step < steps_per_cycle
            or type(arm) is not str or arm not in PREREG_ARMS):
        raise ValueError("invalid matched control arm or TRAIN step")

    # The loader rederives source TRAIN cycles, validates the exact canonical
    # JSON schedule and independently supplied run/candidate/predecessor/fence,
    # and rejects source authority flags. No cached dataclass is trusted.
    request, _cycles = load_measured_cli_schedule(
        raw, scope=scope, steps_per_cycle=steps_per_cycle,
        expected_schedule_sha256=expected_schedule_sha256,
        expected_run_id=expected_run_id,
        expected_candidate_artifact_sha256=expected_candidate_artifact_sha256,
        expected_predecessor_checkpoint_sha256=expected_predecessor_checkpoint_sha256,
        expected_fence=expected_fence,
    )
    batch = request.schedule.batch_for_arm(arm, step)
    scalar_rows = tuple((row.exposure_id, row.prompt, row.label) for row in batch)
    if (len(scalar_rows) != 4
            or any(type(row) is not tuple or len(row) != 3
                   or type(row[0]) is not str or type(row[1]) is not str
                   or type(row[2]) is not int or row[2] not in (0, 1)
                   for row in scalar_rows)):
        raise ValueError("noncanonical TRAIN scalar row")
    return scalar_rows
