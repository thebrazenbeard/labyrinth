"""Source-only, fail-closed Maze full-optimizer checkpoint journal.

An exclusive intent precedes each training cycle. A commit is published only
after byte-level readback of a *new*, pickle-free full-state checkpoint. A
crash between intent and commit leaves UNKNOWN, which must not be retried or
treated as a completed optimizer update. Checkpoints themselves must still be
loaded and validated with ``load_maze_optimizer_snapshot`` before execution.

This journal is NOT an external monotonic anchor, cannot prevent valid old
filesystem snapshot rollback, and cannot attest independent host key custody,
neural execution, exactly-once effects, or graduation. Its digests and lineage
are caller-pinned source checks only. In particular, its control-arm name is a
caller assertion: independent examination must separately verify that the
checkpoint run ID and committed exposure/label roster derive from the actual
precommitted MazeOptimizerResumePlan. No cleanup or implicit repair occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from .neural_resume_checkpoint import (
    SCHEMA as FULL_NEURAL_SCHEMA, FullCheckpointContext, _context_json,
    _manifest_digest, _strict_json,
)


SCHEMA = "labyrinth.maze-crash-resume-journal.v1"
SOURCE_ONLY = "SOURCE_ONLY_NO_EXTERNAL_MONOTONIC_ANCHOR"
_ARMS = frozenset(("correct", "zero_update", "zero_lr", "inverted",
                   "randomized_labels"))
_HEX = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_ENTRY = re.compile(r"cycle-([0-9]{4})\.(intent|commit)\.json\Z", re.ASCII)
_MAX_RECORD = 256 * 1024


class MazeJournalError(RuntimeError):
    """An invalid journal, blocked transition, or corrupt checkpoint."""


class MazeJournalUnknown(MazeJournalError):
    """A persisted intent has no verified commit. Do not auto-retry."""


@dataclass(frozen=True)
class MazeJournalBinding:
    # The four core keys are run, candidate artifact, scope, and cycle (intent).
    run_id: str
    objective_id: str
    arm: str
    candidate_artifact_sha256: str
    scope_fingerprint: str
    base_checkpoint_sha256: str
    prior_checkpoint_sha256: str
    checkpoint_run_id: str
    source_sha256: str
    fence: int
    cycle_count: int

    def validate(self) -> None:
        if (type(self.run_id) is not str or not _ID.fullmatch(self.run_id)
                or type(self.objective_id) is not str
                or not _ID.fullmatch(self.objective_id)
                or type(self.arm) is not str or self.arm not in _ARMS
                or type(self.checkpoint_run_id) is not str
                or not _ID.fullmatch(self.checkpoint_run_id)):
            raise ValueError("invalid Maze journal run identity")
        for field in ("candidate_artifact_sha256", "scope_fingerprint",
                      "base_checkpoint_sha256", "prior_checkpoint_sha256",
                      "source_sha256"):
            value = getattr(self, field)
            if type(value) is not str or not _HEX.fullmatch(value):
                raise ValueError("invalid Maze journal " + field)
        if type(self.fence) is not int or not 0 <= self.fence < 2**63 - 1:
            raise ValueError("invalid Maze journal fence")
        if type(self.cycle_count) is not int or not 1 <= self.cycle_count <= 128:
            raise ValueError("invalid Maze journal cycle count")


@dataclass(frozen=True)
class MazeJournalStatus:
    committed_cycles: int
    last_checkpoint_sha256: str
    last_checkpoint_path: Path | None
    pending_cycle: int | None
    qualification: str = SOURCE_ONLY

    @property
    def safe_to_begin_next_cycle(self) -> bool:
        return self.pending_cycle is None


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=True) + "\n").encode("ascii")


def _digest(value: Any) -> str:
    return sha256(b"labyrinth.maze-crash-resume-journal.v1\0" +
                  _canonical(value)).hexdigest()


def _binding_json(binding: MazeJournalBinding) -> dict[str, Any]:
    return {field: getattr(binding, field) for field in
            binding.__dataclass_fields__}


def _safe(path: Path, *, existing: bool) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("an absolute Path is required for the Maze journal")
    for component in (path, *path.parents):
        if component.is_symlink() or (hasattr(component, "is_junction")
                                      and component.is_junction()):
            raise MazeJournalError("symlink or junction journal path refused")
    if existing:
        if not path.is_dir():
            raise MazeJournalError("existing journal directory required")
    elif path.exists():
        raise MazeJournalError("journal already exists; never replace")
    elif not path.parent.is_dir():
        raise MazeJournalError("journal parent must exist")
    return path


def _fsync_directory(path: Path) -> None:
    # Directory fsync is supported on the POSIX CI environment; on Windows
    # the directory handle is not necessarily fsync-able. Explicitly refuse
    # durability claims there rather than reporting a successful commit.
    if os.name == "nt":
        raise MazeJournalError("Windows journal directory flush requires a host provider")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: dict[str, Any]) -> None:
    content = _canonical(value)
    if not 0 < len(content) <= _MAX_RECORD:
        raise MazeJournalError("journal record exceeds bound")
    # No replacement, rename-over-existing, retry, or rollback cleanup.
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600)
    except FileExistsError as exc:
        raise MazeJournalError("journal record already exists; no replay") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except OSError as exc:
        # A partial or possibly durable record remains as forensic evidence.
        raise MazeJournalUnknown("journal write outcome unknown; preserve bytes") from exc


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
        raise MazeJournalError("missing or nonregular journal record")
    if not 0 < path.stat().st_size <= _MAX_RECORD:
        raise MazeJournalError("journal record byte limit violated")
    raw = path.read_bytes()
    try:
        obj = json.loads(raw)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise MazeJournalError("partial or malformed journal record") from exc
    if type(obj) is not dict or _canonical(obj) != raw:
        raise MazeJournalError("noncanonical or corrupted journal record")
    return obj


def _cycle_paths(root: Path, cycle: int) -> tuple[Path, Path]:
    name = f"cycle-{cycle:04d}"
    return root / (name + ".intent.json"), root / (name + ".commit.json")


def _context_check(binding: MazeJournalBinding,
                   context: FullCheckpointContext) -> None:
    if type(context) is not FullCheckpointContext:
        raise ValueError("typed full-state checkpoint context required")
    context.validate()
    if (context.run_id != binding.checkpoint_run_id
            or context.objective_id != binding.objective_id
            or context.base_checkpoint_sha256 != binding.base_checkpoint_sha256
            or context.prior_checkpoint_sha256 != binding.prior_checkpoint_sha256
            or context.fence != binding.fence):
        raise MazeJournalError("context changed run, base, predecessor, or fence")


def _optimizer_moments_present(tree: Any, trainable_count: int,
                               *, zero_update: bool) -> bool:
    """Cheap encoded-state preflight; the actual Torch load is still required.

    A nonzero-cursor AdamW checkpoint with empty moments cannot serve as a
    valid optimizer continuation. This checks structure only, never tensor
    payload semantics or a model/parameter match.
    """
    if (type(tree) is not dict or set(tree) != {"type", "items"}
            or tree["type"] != "dict" or type(tree["items"]) is not list
            or len(tree["items"]) != 2):
        return False
    found: dict[str, Any] = {}
    for pair in tree["items"]:
        if (type(pair) is not list or len(pair) != 2
                or type(pair[0]) is not dict
                or pair[0].get("type") != "scalar"
                or pair[0].get("value") not in ("state", "param_groups")):
            return False
        key = pair[0]["value"]
        if key in found:
            return False
        found[key] = pair[1]
    if set(found) != {"state", "param_groups"}:
        return False
    groups, states = found["param_groups"], found["state"]
    if (type(groups) is not dict or groups.get("type") != "list"
            or type(groups.get("items")) is not list or not groups["items"]
            or type(states) is not dict or states.get("type") != "dict"
            or type(states.get("items")) is not list):
        return False
    entries = states["items"]
    if zero_update:
        return not entries
    if len(entries) != trainable_count:
        return False
    ids: set[int] = set()
    for pair in entries:
        if (type(pair) is not list or len(pair) != 2
                or type(pair[0]) is not dict
                or pair[0].get("type") != "scalar"
                or type(pair[0].get("value")) is not int
                or pair[0]["value"] < 0
                or pair[0]["value"] in ids
                or type(pair[1]) is not dict
                or pair[1].get("type") != "dict"
                or type(pair[1].get("items")) is not list):
            return False
        ids.add(pair[0]["value"])
        moments = {}
        for field in pair[1]["items"]:
            if (type(field) is not list or len(field) != 2
                    or type(field[0]) is not dict
                    or field[0].get("type") != "scalar"
                    or field[0].get("value") not in
                    ("step", "exp_avg", "exp_avg_sq")
                    or field[0]["value"] in moments
                    or type(field[1]) is not dict
                    or field[1].get("type") != "tensor"
                    or type(field[1].get("key")) is not str
                    or re.fullmatch(r"tensor_[0-9]{6}", field[1]["key"]) is None):
                return False
            moments[field[0]["value"]] = field[1]["key"]
        if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            return False
    return len(ids) == trainable_count


def _snapshot(root: Path, checkpoint_path: Path,
              expected_context_json: dict[str, Any], trusted_sha256: str,
              *, arm: str
              ) -> tuple[str, str]:
    if not isinstance(checkpoint_path, Path) or not checkpoint_path.is_absolute():
        raise ValueError("absolute checkpoint Path required")
    if type(trusted_sha256) is not str or not _HEX.fullmatch(trusted_sha256):
        raise ValueError("trusted checkpoint manifest SHA-256 required")
    _safe(checkpoint_path, existing=True)
    try:
        relative = checkpoint_path.relative_to(root.parent)
    except ValueError as exc:
        raise MazeJournalError("checkpoint must stay within the run parent") from exc
    if relative == Path(".") or any(part in ("", ".", "..") for part in relative.parts):
        raise MazeJournalError("unsafe checkpoint relative path")
    expected_files = {"manifest.json", "state.safetensors"}
    if {entry.name for entry in checkpoint_path.iterdir()} != expected_files:
        raise MazeJournalError("incomplete/extra checkpoint file inventory")
    for name in expected_files:
        target = checkpoint_path / name
        if target.is_symlink() or not stat.S_ISREG(target.lstat().st_mode):
            raise MazeJournalError("nonregular neural checkpoint file")
    manifest_raw = (checkpoint_path / "manifest.json").read_bytes()
    if _manifest_digest(manifest_raw) != trusted_sha256:
        raise MazeJournalError("checkpoint manifest not the pinned digest")
    try:
        manifest = _strict_json(manifest_raw)
    except ValueError as exc:
        raise MazeJournalError("malformed neural checkpoint manifest") from exc
    if (set(manifest) != {"schema", "context", "optimizer", "optimizer_groups",
                          "trainable_names", "model_tree", "optimizer_tree",
                          "rng_tree", "numpy_rng", "tensor_sha256"}
            or manifest["schema"] != FULL_NEURAL_SCHEMA
            or manifest["optimizer"] != "AdamW"
            or manifest["context"] != expected_context_json
            or type(manifest["trainable_names"]) is not list
            or not manifest["trainable_names"]
            or not _optimizer_moments_present(
                manifest["optimizer_tree"], len(manifest["trainable_names"]),
                zero_update=(arm == "zero_update"))
            or type(manifest.get("tensor_sha256")) is not str
            or not _HEX.fullmatch(manifest["tensor_sha256"])):
        raise MazeJournalError("checkpoint context or tensor commitment mismatch")
    tensor_path = checkpoint_path / "state.safetensors"
    if not 0 < tensor_path.stat().st_size <= 1024 * 1024 * 1024:
        raise MazeJournalError("checkpoint tensor size invalid")
    if sha256(tensor_path.read_bytes()).hexdigest() != manifest["tensor_sha256"]:
        raise MazeJournalError("checkpoint tensor integrity failure")
    return str(relative), manifest["tensor_sha256"]


def initialize_maze_journal(root: Path, binding: MazeJournalBinding) -> None:
    """Exclusively initialize a never-replaced local run journal."""
    if type(binding) is not MazeJournalBinding:
        raise ValueError("typed run binding required")
    binding.validate()
    root = _safe(root, existing=False)
    root.mkdir(mode=0o700)
    _write_once(root / "run.json", {"schema": SCHEMA,
                                    "binding": _binding_json(binding),
                                    "binding_sha256": _digest(_binding_json(binding))})
    _fsync_directory(root.parent)


def _inspect(root: Path, binding: MazeJournalBinding) -> tuple[MazeJournalStatus,
                                                               dict[str, Any] | None]:
    root = _safe(root, existing=True)
    data = _read(root / "run.json")
    if data != {"schema": SCHEMA, "binding": _binding_json(binding),
                "binding_sha256": _digest(_binding_json(binding))}:
        raise MazeJournalError("wrong or altered run binding")
    found: set[str] = set()
    for entry in root.iterdir():
        if entry.is_symlink() or not stat.S_ISREG(entry.lstat().st_mode):
            raise MazeJournalError("journal contains nonregular entry")
        if entry.name != "run.json":
            match = _ENTRY.fullmatch(entry.name)
            if match is None or int(match.group(1)) >= binding.cycle_count:
                raise MazeJournalError("unexpected, out-of-range, or partial journal entry")
        found.add(entry.name)
    last_sha = binding.prior_checkpoint_sha256
    last_path = None
    last_cursor = 0
    exposures = None
    pending = None
    last_intent = None
    committed = 0
    for cycle in range(binding.cycle_count):
        intent_path, commit_path = _cycle_paths(root, cycle)
        has_intent = intent_path.name in found
        has_commit = commit_path.name in found
        if not has_intent:
            if has_commit or any(_ENTRY.fullmatch(n) and int(_ENTRY.fullmatch(n).group(1)) > cycle
                                 for n in found):
                raise MazeJournalError("noncontiguous cycle or commit without intent")
            break
        intent = _read(intent_path)
        expected_keys = {"schema", "run_binding_sha256", "cycle_index",
                         "prior_checkpoint_sha256", "context", "context_sha256",
                         "start_cursor", "qualification"}
        if (set(intent) != expected_keys or intent["schema"] != SCHEMA
                or intent["run_binding_sha256"] != _digest(_binding_json(binding))
                or type(intent["cycle_index"]) is not int
                or intent["cycle_index"] != cycle
                or intent["prior_checkpoint_sha256"] != last_sha
                or intent["qualification"] != SOURCE_ONLY
                or type(intent["start_cursor"]) is not int
                or intent["start_cursor"] != last_cursor
                or type(intent["context"]) is not dict
                or intent["context_sha256"] != _digest(intent["context"])):
            raise MazeJournalError("invalid cycle intent or predecessor")
        try:
            context = FullCheckpointContext(
                run_id=intent["context"]["run_id"],
                objective_id=intent["context"]["objective_id"],
                base_checkpoint_sha256=intent["context"]["base_checkpoint_sha256"],
                prior_checkpoint_sha256=intent["context"]["prior_checkpoint_sha256"],
                fence=intent["context"]["fence"], cursor=intent["context"]["cursor"],
                exposures=tuple(tuple(x) for x in intent["context"]["exposures"]),
            )
            _context_check(binding, context)
        except (KeyError, TypeError, ValueError) as exc:
            raise MazeJournalError("invalid stored checkpoint context") from exc
        if (_context_json(context) != intent["context"]
                or not last_cursor < context.cursor <= len(context.exposures)
                or (exposures is not None and exposures != context.exposures)):
            raise MazeJournalError("noncontiguous cursor or exposure permutation")
        if not has_commit:
            pending = cycle
            last_intent = intent
            if any(_ENTRY.fullmatch(n) and int(_ENTRY.fullmatch(n).group(1)) > cycle
                   for n in found):
                raise MazeJournalError("later intent after unresolved cycle")
            break
        commit = _read(commit_path)
        if (set(commit) != {"schema", "intent_sha256", "cycle_index",
                            "checkpoint_sha256", "checkpoint_path", "tensor_sha256",
                            "qualification"}
                or commit["schema"] != SCHEMA
                or commit["intent_sha256"] != _digest(intent)
                or type(commit["cycle_index"]) is not int
                or commit["cycle_index"] != cycle
                or type(commit["checkpoint_sha256"]) is not str
                or not _HEX.fullmatch(commit["checkpoint_sha256"])
                or type(commit["checkpoint_path"]) is not str
                or commit["qualification"] != SOURCE_ONLY):
            raise MazeJournalError("invalid committed cycle receipt")
        cp = (root.parent / commit["checkpoint_path"]).absolute()
        rel, tensor_sha = _snapshot(root, cp, intent["context"],
                                    commit["checkpoint_sha256"], arm=binding.arm)
        if rel != commit["checkpoint_path"] or tensor_sha != commit["tensor_sha256"]:
            raise MazeJournalError("committed checkpoint bytes changed")
        last_sha = commit["checkpoint_sha256"]
        last_path = cp
        last_cursor = context.cursor
        exposures = context.exposures
        committed += 1
    return (MazeJournalStatus(committed, last_sha, last_path, pending), last_intent)


def inspect_maze_journal(root: Path, binding: MazeJournalBinding) -> MazeJournalStatus:
    """Read back a chain; UNKNOWN pending is a terminal hold, not a retry cue."""
    if type(binding) is not MazeJournalBinding:
        raise ValueError("typed run binding required")
    binding.validate()
    return _inspect(root, binding)[0]


def precommit_maze_cycle(root: Path, binding: MazeJournalBinding,
                         cycle_index: int, context: FullCheckpointContext) -> None:
    """Exclusively persist an intent before any optimizer work for this cycle."""
    binding.validate()
    _context_check(binding, context)
    if type(cycle_index) is not int or not 0 <= cycle_index < binding.cycle_count:
        raise ValueError("invalid cycle")
    status = inspect_maze_journal(root, binding)
    if status.pending_cycle is not None:
        raise MazeJournalUnknown("unresolved training cycle; do not replay")
    if status.committed_cycles != cycle_index:
        raise MazeJournalError("missing or already committed predecessor")
    previous_cursor = 0
    if cycle_index:
        last = _read(_cycle_paths(root, cycle_index - 1)[0])
        previous_cursor = last["context"]["cursor"]
        if tuple(tuple(x) for x in last["context"]["exposures"]) != context.exposures:
            raise MazeJournalError("checkpoint exposures changed across cycles")
    if not previous_cursor < context.cursor <= len(context.exposures):
        raise MazeJournalError("cycle cursor must advance on committed exposures")
    intent = {"schema": SCHEMA,
              "run_binding_sha256": _digest(_binding_json(binding)),
              "cycle_index": cycle_index,
              "prior_checkpoint_sha256": status.last_checkpoint_sha256,
              "context": _context_json(context),
              "context_sha256": _digest(_context_json(context)),
              "start_cursor": previous_cursor,
              "qualification": SOURCE_ONLY}
    _write_once(_cycle_paths(root, cycle_index)[0], intent)


def commit_maze_cycle(root: Path, binding: MazeJournalBinding,
                      cycle_index: int, context: FullCheckpointContext,
                      checkpoint_path: Path,
                      trusted_checkpoint_manifest_sha256: str) -> str:
    """Publish receipt after exact byte readback; never infer from a partial file."""
    binding.validate()
    _context_check(binding, context)
    status, pending_intent = _inspect(root, binding)
    if (type(cycle_index) is not int or status.pending_cycle != cycle_index
            or status.committed_cycles != cycle_index or pending_intent is None):
        raise MazeJournalError("no matching unresolved cycle intent")
    if pending_intent["context"] != _context_json(context):
        raise MazeJournalError("checkpoint context changed after intent")
    rel, tensor = _snapshot(root, checkpoint_path, pending_intent["context"],
                            trusted_checkpoint_manifest_sha256, arm=binding.arm)
    commit = {"schema": SCHEMA, "intent_sha256": _digest(pending_intent),
              "cycle_index": cycle_index,
              "checkpoint_sha256": trusted_checkpoint_manifest_sha256,
              "checkpoint_path": rel, "tensor_sha256": tensor,
              "qualification": SOURCE_ONLY}
    _write_once(_cycle_paths(root, cycle_index)[1], commit)
    # A successful API return is a SOURCE-ONLY local readback, not a host CAS.
    if inspect_maze_journal(root, binding).committed_cycles != cycle_index + 1:
        raise MazeJournalUnknown("commit readback not established")
    return trusted_checkpoint_manifest_sha256
