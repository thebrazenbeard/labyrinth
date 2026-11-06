"""Ariadne's write-once developmental ancestry and fail-closed TRAIN WAL.

This ledger records *intent* before a trainer effect, and full-state manifest
references after it. It cannot attest that an optimizer was actually restored,
or prove that a trainer caused the claimed child bytes to be created.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any

try:
    import fcntl
except ImportError:
    fcntl = None

from .campaign_full_state_v1 import verify_campaign_state


SCHEMA = "labyrinth.ariadne-recovery.v1"
_DOMAIN = b"labyrinth.ariadne-recovery.v1\0"
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_FILE = re.compile(r"[0-9]{12}\.json\Z", re.ASCII)
_MAX_EVENT = 256 * 1024
_SUPPORTED_PLATFORM = (
    sys.platform == "linux" and os.name == "posix" and fcntl is not None
    and hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "link") and hasattr(os, "fsync")
)


class LineageIntegrityError(ValueError):
    """Damaged, divergent, incomplete, or physically unverified ancestry."""


class IndeterminateTrainerEffect(LineageIntegrityError):
    """A persisted TRAIN intent has no trustworthy completion receipt."""


class UnsupportedJournalPlatform(RuntimeError):
    """Ariadne cannot safely synchronize and fsync this host filesystem."""


def _require_supported_platform() -> None:
    if not _SUPPORTED_PLATFORM:
        raise UnsupportedJournalPlatform(
            "Ariadne v1 journal needs Linux/WSL with flock, O_NOFOLLOW, "
            "O_DIRECTORY, hard links and fsync; native Windows unsupported"
        )


@dataclass(frozen=True)
class TrainPermit:
    """One-use permit valid only in its issuing process and journal instance."""

    candidate_id: str
    effect_token: str
    event_sha256: str
    issuing_pid: int


def _canonical(value: Any) -> bytes:
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LineageIntegrityError("event must contain bounded plain JSON") from exc
    if len(result) > _MAX_EVENT:
        raise LineageIntegrityError("Ariadne event exceeds size limit")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(_DOMAIN + _canonical(value)).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LineageIntegrityError("duplicate journal JSON key")
        result[key] = value
    return result


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise LineageIntegrityError(f"invalid {name}")
    return value


def _hex(value: Any, name: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise LineageIntegrityError(f"invalid {name}")
    return value


def _object(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise LineageIntegrityError(f"{name} must be a JSON object")
    _canonical(value)
    return value


def _fields(payload: Any, *names: str) -> dict[str, Any]:
    result = _object(payload, "event payload")
    if set(result) != set(names):
        raise LineageIntegrityError("event payload fields do not match schema")
    return result


def _manifest(path_string: Any, *, fresh_process: bool) -> dict[str, Any]:
    if type(path_string) is not str:
        raise LineageIntegrityError("full-state manifest path must be a string")
    path = Path(path_string)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path != path.resolve(strict=True)):
        raise LineageIntegrityError("full-state manifest must be canonical local file")
    try:
        observed = verify_campaign_state(path, fresh_process=fresh_process)
        item = json.loads(path.read_bytes(), object_pairs_hook=_unique_pairs)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise LineageIntegrityError("full-state manifest or physical bytes failed readback") from exc
    if (item["manifest_sha256"] != observed["manifest_sha256"]
            or item["state_sha256"] != observed["state_sha256"]):
        raise LineageIntegrityError("full-state manifest changed during verification")
    return item


def _state():
    return {
        "origin": None, "parent": None, "parent_path": None,
        "worlds": {}, "branches": {}, "candidates": {},
        "selections": [], "pending": None, "generation_index": 0,
        "selected_world_id": None, "selected_world_transition_pending": False,
        "manifest_paths": {}, "idempotency": {}, "event_ids": set(),
    }


def _apply(state: dict[str, Any], event: dict[str, Any], *, fresh_process: bool) -> None:
    kind, payload = event["kind"], event["payload"]
    if kind == "GENESIS":
        p = _fields(payload, "manifest_path", "state_sha256", "manifest_sha256",
                    "world_id", "version", "curriculum", "world_state")
        if state["origin"] is not None:
            raise LineageIntegrityError("second W0 would fork the lineage")
        manifest = _manifest(p["manifest_path"], fresh_process=fresh_process)
        _hex(p["state_sha256"], "W0 state hash")
        _hex(p["manifest_sha256"], "W0 manifest hash")
        if (manifest["generation_index"] != 0
                or manifest["state_sha256"] != p["state_sha256"]
                or manifest["manifest_sha256"] != p["manifest_sha256"]):
            raise LineageIntegrityError("W0 manifest identity or generation mismatch")
        _identifier(p["world_id"], "world ID")
        if type(p["version"]) is not int or p["version"] != 1:
            raise LineageIntegrityError("W0 curriculum version must be one")
        _object(p["curriculum"], "curriculum")
        _object(p["world_state"], "world state")
        state["origin"] = manifest["state_sha256"]
        state["parent"] = manifest
        state["parent_path"] = p["manifest_path"]
        state["manifest_paths"][manifest["state_sha256"]] = p["manifest_path"]
        state["worlds"][p["world_id"]] = {
            "world_id": p["world_id"], "parent_world_id": None,
            "version": p["version"], "curriculum": p["curriculum"],
            "world_state": p["world_state"],
        }
        state["selected_world_id"] = p["world_id"]
        return
    if state["origin"] is None:
        raise LineageIntegrityError("first event must establish physical W0")
    if kind == "WORLD_VERSION":
        p = _fields(payload, "world_id", "parent_world_id", "version",
                    "curriculum", "world_state")
        _identifier(p["world_id"], "world ID")
        _identifier(p["parent_world_id"], "world parent ID")
        parent = state["worlds"].get(p["parent_world_id"])
        if (p["world_id"] in state["worlds"] or parent is None
                or type(p["version"]) is not int
                or p["version"] != parent["version"] + 1):
            raise LineageIntegrityError("world version missing parent or forked identity")
        _object(p["curriculum"], "curriculum")
        _object(p["world_state"], "world state")
        state["worlds"][p["world_id"]] = p
        # An evolved world becomes the selected learner's next entrance
        # ONLY when it extends the world of the just-selected champion.
        # Merely creating a possible world must not change the lineage.
        if state["selected_world_transition_pending"]:
            if p["parent_world_id"] != state["selected_world_id"]:
                raise LineageIntegrityError(
                    "selected champion's successor world must extend its current world")
            state["selected_world_id"] = p["world_id"]
            state["selected_world_transition_pending"] = False
    elif kind == "BRANCH_EXPERIENCE":
        p = _fields(payload, "branch_id", "world_id", "generation_index",
                    "parent_state_sha256", "experience", "selected_path")
        _identifier(p["branch_id"], "branch ID")
        _identifier(p["world_id"], "world ID")
        _hex(p["parent_state_sha256"], "branch parent state")
        experience = _fields(p["experience"], "observations", "actions", "outcomes")
        if (any(type(experience[k]) is not list for k in experience)
                or type(p["selected_path"]) is not list
                or not p["selected_path"]
                or any(type(x) is not str or not x for x in p["selected_path"])):
            raise LineageIntegrityError("branch lacks typed experience or selected path")
        if (state["pending"] is not None or p["branch_id"] in state["branches"]
                or p["world_id"] not in state["worlds"]
                or p["generation_index"] != state["generation_index"]
                or p["parent_state_sha256"] != state["parent"]["state_sha256"]):
            raise LineageIntegrityError("branch diverges from current selected parent")
        state["branches"][p["branch_id"]] = p
    elif kind == "TRAIN_PREPARED":
        p = _fields(payload, "candidate_id", "branch_id", "generation_index",
                    "parent_state_sha256", "effect_token", "trainer_request")
        _identifier(p["candidate_id"], "candidate ID")
        _identifier(p["branch_id"], "branch ID")
        _hex(p["parent_state_sha256"], "training parent")
        _hex(p["effect_token"], "TRAIN effect token")
        req = _fields(p["trainer_request"], "trainer_id", "effect_scope",
                      "optimizer_state_required", "rng_state_required",
                      "data_cursor_required", "parameters")
        _identifier(req["trainer_id"], "trainer ID")
        if (req["effect_scope"] != "candidate_full_state"
                or any(req[key] is not True for key in (
                    "optimizer_state_required", "rng_state_required", "data_cursor_required"))):
            raise LineageIntegrityError("TRAIN must account for complete training state")
        _object(req["parameters"], "trainer parameters")
        branch = state["branches"].get(p["branch_id"])
        if (state["pending"] is not None or p["candidate_id"] in state["candidates"]
                or branch is None or branch["generation_index"] != state["generation_index"]
                or p["generation_index"] != state["generation_index"]
                or p["parent_state_sha256"] != state["parent"]["state_sha256"]):
            raise LineageIntegrityError("TRAIN effect has concurrent intent or wrong ancestry")
        state["pending"] = p
        state["candidates"][p["candidate_id"]] = {
            "candidate_id": p["candidate_id"], "branch_id": p["branch_id"],
            "generation_index": p["generation_index"], "fate": "INDETERMINATE",
            "parent_state_sha256": p["parent_state_sha256"],
            "child_state_sha256": None, "child_manifest_path": None,
            "evaluation": None, "trainer_receipt": None,
        }
    elif kind == "TRAIN_COMMITTED":
        p = _fields(payload, "candidate_id", "effect_token", "child_manifest_path",
                    "child_state_sha256", "child_manifest_sha256", "trainer_receipt")
        pending = state["pending"]
        if (pending is None or p["candidate_id"] != pending["candidate_id"]
                or p["effect_token"] != pending["effect_token"]):
            raise LineageIntegrityError("TRAIN commit lacks matching durable intent")
        _hex(p["child_state_sha256"], "child state")
        _hex(p["child_manifest_sha256"], "child manifest")
        receipt = _fields(p["trainer_receipt"], "trainer_id", "effect_status",
                          "effect_token", "details")
        if (receipt["trainer_id"] != pending["trainer_request"]["trainer_id"]
                or receipt["effect_status"] != "completed"
                or receipt["effect_token"] != p["effect_token"]):
            raise LineageIntegrityError("TRAIN receipt missing complete matching effect")
        _object(receipt["details"], "trainer receipt details")
        child = _manifest(p["child_manifest_path"], fresh_process=fresh_process)
        parent = state["parent"]
        if (child["state_sha256"] != p["child_state_sha256"]
                or child["manifest_sha256"] != p["child_manifest_sha256"]
                or child["generation_index"] != parent["generation_index"] + 1
                or child["parent_state_sha256"] != parent["state_sha256"]
                or child["origin_state_sha256"] != state["origin"]
                or child["state_sha256"] == parent["state_sha256"]
                or child["state_sha256"] in state["manifest_paths"]
                or child["family"] != parent["family"]
                or child["mutation_kind"] != parent["mutation_kind"]):
            raise LineageIntegrityError("child full-state manifest has incompatible ancestry")
        if child["mutation_kind"] == "lora_adapter":
            def adapter_weights(item):
                return {name: record for name, record in item["adapter"]["files"].items()
                        if name.endswith(".safetensors")}

            if (child["base"]["files"] != parent["base"]["files"]
                    or not adapter_weights(child)
                    or adapter_weights(child) == adapter_weights(parent)):
                raise LineageIntegrityError("LoRA child must retain base and change adapter tensor bytes")
        else:
            def non_weights(item):
                return {name: record for name, record in item["base"]["files"].items()
                        if not name.endswith(".safetensors")
                        and name != "model.safetensors.index.json"}

            def weights(item):
                return {name: record for name, record in item["base"]["files"].items()
                        if name.endswith(".safetensors")}

            if (non_weights(child) != non_weights(parent)
                    or weights(child) == weights(parent)):
                raise LineageIntegrityError("full-weight child requires real changed weights")
        candidate = state["candidates"][p["candidate_id"]]
        candidate.update(fate="UNEVALUATED", child_state_sha256=child["state_sha256"],
                         child_manifest_path=p["child_manifest_path"],
                         trainer_receipt=receipt)
        state["manifest_paths"][child["state_sha256"]] = p["child_manifest_path"]
        state["pending"] = None
    elif kind == "CANDIDATE_EVALUATED":
        p = _fields(payload, "candidate_id", "evaluation")
        candidate = state["candidates"].get(p["candidate_id"])
        e = _fields(p["evaluation"], "metric_id", "score", "direction",
                    "dataset_ref", "evidence")
        _identifier(e["metric_id"], "metric ID")
        _identifier(e["dataset_ref"], "dataset reference")
        if (type(e["score"]) not in (int, float)
                or e["direction"] not in ("higher", "lower")
                or candidate is None or candidate["fate"] != "UNEVALUATED"
                or state["pending"] is not None):
            raise LineageIntegrityError("invalid candidate evaluation or effect state")
        _object(e["evidence"], "evaluation evidence")
        candidate.update(fate="EVALUATED", evaluation=e)
    elif kind == "CANDIDATE_REJECTED":
        p = _fields(payload, "candidate_id", "reason")
        candidate = state["candidates"].get(p["candidate_id"])
        if (type(p["reason"]) is not str or not p["reason"]
                or candidate is None or candidate["fate"] != "EVALUATED"
                or candidate["generation_index"] != state["generation_index"]
                or state["pending"] is not None):
            raise LineageIntegrityError("reject requires evaluated current candidate")
        candidate["fate"] = "REJECTED"
        candidate["rejection_reason"] = p["reason"]
    elif kind == "CANDIDATE_SELECTED":
        p = _fields(payload, "candidate_id", "selection_reason")
        candidate = state["candidates"].get(p["candidate_id"])
        cohort = [x for x in state["candidates"].values()
                  if x["generation_index"] == state["generation_index"]]
        if (type(p["selection_reason"]) is not str or not p["selection_reason"]
                or candidate is None or candidate["fate"] != "EVALUATED"
                or candidate["generation_index"] != state["generation_index"]
                or len(cohort) < 2 or state["pending"] is not None
                or any(x["fate"] not in ("REJECTED", "EVALUATED") for x in cohort)
                or sum(x["fate"] == "EVALUATED" for x in cohort) != 1):
            raise LineageIntegrityError("selection needs resolved sibling fates and unique winner")
        candidate["fate"] = "SELECTED"
        state["selections"].append({
            "generation_index": state["generation_index"],
            "candidate_id": p["candidate_id"],
            "parent_state_sha256": candidate["parent_state_sha256"],
            "child_state_sha256": candidate["child_state_sha256"],
            "child_manifest_path": candidate["child_manifest_path"],
            "branch_id": candidate["branch_id"],
            "selection_reason": p["selection_reason"],
        })
        state["parent_path"] = candidate["child_manifest_path"]
        state["parent"] = _manifest(state["parent_path"], fresh_process=False)
        state["generation_index"] += 1
        state["selected_world_transition_pending"] = True
    else:
        raise LineageIntegrityError("unknown Ariadne event kind")


class AriadneJournal:
    """Cooperating writers serialize through flock and revalidate every event."""

    def __init__(self, root: Path):
        _require_supported_platform()
        if (not isinstance(root, Path) or not root.is_absolute() or root.is_symlink()
                or not root.is_dir() or root != root.resolve(strict=True)
                or not (root / "events").is_dir()
                or (root / "events").is_symlink()):
            raise LineageIntegrityError("journal root must be canonical absolute directory")
        self.root = root
        self._local_permits: dict[str, TrainPermit] = {}

    @classmethod
    def create(cls, root: Path) -> "AriadneJournal":
        _require_supported_platform()
        if (not isinstance(root, Path) or not root.is_absolute() or root.exists()
                or root.parent.is_symlink() or not root.parent.is_dir()
                or root.parent != root.parent.resolve(strict=True)):
            raise LineageIntegrityError("new journal needs unused canonical absolute path")
        root.mkdir(mode=0o700)
        (root / "events").mkdir(mode=0o700)
        descriptor = os.open(root / ".lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        cls._fsync_dir(root / "events")
        cls._fsync_dir(root)
        cls._fsync_dir(root.parent)
        return cls(root)

    @classmethod
    def open(cls, root: Path) -> "AriadneJournal":
        return cls(root)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _locked(self):
        path = self.root / ".lock"
        if path.is_symlink() or not path.is_file():
            raise LineageIntegrityError("Ariadne lock file is absent or unsafe")
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise LineageIntegrityError("Ariadne lock is not a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def _read(self, *, fresh_process: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        entries = sorted((self.root / "events").iterdir())
        if any(not _FILE.fullmatch(x.name) or x.is_symlink() or not x.is_file()
               for x in entries):
            raise LineageIntegrityError("orphan staging, unfamiliar file, or unsafe journal entry")
        events, state, head = [], _state(), "0" * 64
        for index, path in enumerate(entries):
            if path.name != f"{index:012d}.json":
                raise LineageIntegrityError("missing, duplicated, or reordered journal event")
            raw = path.read_bytes()
            if len(raw) > _MAX_EVENT or not raw.endswith(b"\n"):
                raise LineageIntegrityError("truncated or oversized journal event")
            try:
                event = json.loads(raw, object_pairs_hook=_unique_pairs)
            except (ValueError, UnicodeError) as exc:
                raise LineageIntegrityError("invalid event JSON") from exc
            if (type(event) is not dict or set(event) != {
                "schema", "event_version", "seq", "kind", "event_id", "idempotency_key", "prev_sha256",
                "payload", "event_sha256"}
                    or raw != _canonical(event) + b"\n"
                    or event["schema"] != SCHEMA
                    or type(event["event_version"]) is not int
                    or event["event_version"] != 1 or type(event["seq"]) is not int
                    or event["seq"] != index or event["prev_sha256"] != head
                    or type(event["event_id"]) is not str
                    or _SHA.fullmatch(event["event_id"]) is None
                    or event["event_sha256"] != _digest({
                        k: v for k, v in event.items() if k != "event_sha256"})):
                raise LineageIntegrityError("tampered, replayed, or divergent journal chain")
            key = _identifier(event["idempotency_key"], "idempotency key")
            if (key in state["idempotency"] or event["event_id"] in state["event_ids"]):
                raise LineageIntegrityError("replayed event ID or idempotency key in journal")
            _apply(state, event, fresh_process=fresh_process)
            state["idempotency"][key] = event
            state["event_ids"].add(event["event_id"])
            head = event["event_sha256"]
            events.append(event)
        return events, state

    def _append(self, kind: str, payload: dict[str, Any], key: str,
                *, prepare: bool = False) -> dict[str, Any]:
        _identifier(key, "idempotency key")
        fd = self._locked()
        try:
            events, state = self._read(fresh_process=False)
            existing = state["idempotency"].get(key)
            if existing is not None:
                if prepare:
                    raise IndeterminateTrainerEffect("TRAIN intent replay is never a fresh permit")
                if existing["kind"] != kind or existing["payload"] != payload:
                    raise LineageIntegrityError("idempotency key reused for different event")
                return existing
            if state["pending"] is not None and kind != "TRAIN_COMMITTED":
                raise IndeterminateTrainerEffect("uncommitted TRAIN intent: effect may have occurred")
            previous = events[-1]["event_sha256"] if events else "0" * 64
            event = {"schema": SCHEMA, "event_version": 1,
                     "seq": len(events), "kind": kind,
                     "event_id": secrets.token_hex(32), "idempotency_key": key,
                     "prev_sha256": previous, "payload": payload}
            event["event_sha256"] = _digest(event)
            _apply(state, event, fresh_process=False)
            # Never mutate/replace a committed entry. Temp is persisted before
            # a hard-link atomically publishes the write-once sequence number.
            directory = self.root / "events"
            staged = directory / (".stage-" + secrets.token_hex(16))
            target = directory / f"{len(events):012d}.json"
            stream = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                contents = _canonical(event) + b"\n"
                with os.fdopen(stream, "wb") as output:
                    output.write(contents)
                    output.flush()
                    os.fsync(output.fileno())
                os.link(staged, target, follow_symlinks=False)
                self._fsync_dir(directory)
            finally:
                staged.unlink(missing_ok=True)
            self._fsync_dir(directory)
            return event
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def initialize(self, *, w0_manifest_path: Path, world_id: str,
                   version: int, curriculum: dict[str, Any], world_state: dict[str, Any],
                   idempotency_key: str) -> dict[str, Any]:
        manifest = _manifest(str(w0_manifest_path), fresh_process=True)
        return self._append("GENESIS", {
            "manifest_path": str(w0_manifest_path),
            "state_sha256": manifest["state_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "world_id": world_id, "version": version,
            "curriculum": curriculum, "world_state": world_state,
        }, idempotency_key)

    def record_world(self, *, world_id: str, parent_world_id: str,
                     version: int, curriculum: dict[str, Any],
                     world_state: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        return self._append("WORLD_VERSION", {
            "world_id": world_id, "parent_world_id": parent_world_id,
            "version": version, "curriculum": curriculum, "world_state": world_state,
        }, idempotency_key)

    def record_branch(self, *, branch_id: str, world_id: str,
                      experience: dict[str, list[Any]], selected_path: list[str],
                      idempotency_key: str) -> dict[str, Any]:
        fd = self._locked()
        try:
            _, state = self._read(fresh_process=False)
            if state["origin"] is None:
                raise LineageIntegrityError("cannot branch before W0")
            generation = state["generation_index"]
            parent_sha = state["parent"]["state_sha256"]
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        return self._append("BRANCH_EXPERIENCE", {
            "branch_id": branch_id, "world_id": world_id,
            "generation_index": generation, "parent_state_sha256": parent_sha,
            "experience": experience, "selected_path": selected_path,
        }, idempotency_key)

    def prepare_training(self, *, candidate_id: str, branch_id: str,
                         trainer_request: dict[str, Any], idempotency_key: str) -> TrainPermit:
        # Generate the permit only after the exact PREPARED event has been
        # fsynced. The caller must not invoke a trainer before this returns.
        fd = self._locked()
        try:
            _, state = self._read(fresh_process=False)
            if state["origin"] is None:
                raise LineageIntegrityError("cannot train without W0")
            generation = state["generation_index"]
            parent_sha = state["parent"]["state_sha256"]
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        token = secrets.token_hex(32)
        event = self._append("TRAIN_PREPARED", {
            "candidate_id": candidate_id, "branch_id": branch_id,
            "generation_index": generation, "parent_state_sha256": parent_sha,
            "effect_token": token, "trainer_request": trainer_request,
        }, idempotency_key, prepare=True)
        permit = TrainPermit(candidate_id, token, event["event_sha256"], os.getpid())
        self._local_permits[token] = permit
        return permit

    def commit_training(self, permit: TrainPermit, *, child_manifest_path: Path,
                        trainer_receipt: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        if (type(permit) is not TrainPermit or permit.issuing_pid != os.getpid()
                or self._local_permits.get(permit.effect_token) != permit):
            raise IndeterminateTrainerEffect("TRAIN completion requires original live permit")
        manifest = _manifest(str(child_manifest_path), fresh_process=True)
        event = self._append("TRAIN_COMMITTED", {
            "candidate_id": permit.candidate_id,
            "effect_token": permit.effect_token,
            "child_manifest_path": str(child_manifest_path),
            "child_state_sha256": manifest["state_sha256"],
            "child_manifest_sha256": manifest["manifest_sha256"],
            "trainer_receipt": trainer_receipt,
        }, idempotency_key)
        self._local_permits.pop(permit.effect_token)
        return event

    def record_evaluation(self, *, candidate_id: str,
                          evaluation: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        return self._append("CANDIDATE_EVALUATED", {
            "candidate_id": candidate_id, "evaluation": evaluation,
        }, idempotency_key)

    def reject_candidate(self, *, candidate_id: str, reason: str,
                         idempotency_key: str) -> dict[str, Any]:
        return self._append("CANDIDATE_REJECTED", {
            "candidate_id": candidate_id, "reason": reason,
        }, idempotency_key)

    def select_candidate(self, *, candidate_id: str, selection_reason: str,
                         idempotency_key: str) -> dict[str, Any]:
        return self._append("CANDIDATE_SELECTED", {
            "candidate_id": candidate_id, "selection_reason": selection_reason,
        }, idempotency_key)

    def events(self) -> tuple[dict[str, Any], ...]:
        fd = self._locked()
        try:
            events, _ = self._read(fresh_process=True)
            return tuple(events)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def continuation_descriptor(self, *, expected_head_sha256: str | None = None) -> dict[str, Any]:
        """Re-verify every physical genome; describe work a real loader must do."""
        if expected_head_sha256 is not None:
            _hex(expected_head_sha256, "externally witnessed journal head")
        fd = self._locked()
        try:
            events, state = self._read(fresh_process=True)
            if not events:
                raise LineageIntegrityError("empty journal cannot describe W0")
            if (expected_head_sha256 is not None
                    and expected_head_sha256 != events[-1]["event_sha256"]):
                raise LineageIntegrityError("journal head differs from external witness: rollback/fork")
            pending = state["pending"]
            selected = state["selections"]
            return {
                "schema": SCHEMA,
                "status": ("INDETERMINATE_TRAINER_EFFECT" if pending else
                           "READY_FOR_EXTERNAL_FULL_STATE_LOAD"),
                "origin_state_sha256": state["origin"],
                "next_generation_index": state["generation_index"],
                "selected_parent_state_sha256": state["parent"]["state_sha256"],
                "selected_parent_manifest_path": state["parent_path"],
                "selected_world_id": state["selected_world_id"],
                "worlds": list(state["worlds"].values()),
                "branches": list(state["branches"].values()),
                "candidates": list(state["candidates"].values()),
                "selected_lineage": list(selected),
                "pending_training": pending,
                "event_count": len(events),
                "event_head_sha256": events[-1]["event_sha256"],
                "required_restore_roles": ["base", "adapter_if_lora", "optimizer",
                                           "scheduler", "rng", "data_cursor"],
                "model_loaded": False,
                "optimizer_loaded": False,
                "rng_loaded": False,
                "data_cursor_restored": False,
                "restorable_training_state_proven": False,
                "neural_update_proven": False,
                "graduation_authorized": False,
            }
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def trainer_request(trainer_id: str, *, parameters: dict[str, Any]) -> dict[str, Any]:
    """Construct a full-state-scoped request; no unknown state omission."""
    return {"trainer_id": trainer_id, "effect_scope": "candidate_full_state",
            "optimizer_state_required": True, "rng_state_required": True,
            "data_cursor_required": True, "parameters": parameters}
