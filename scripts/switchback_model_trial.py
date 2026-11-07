"""Bounded source-public Switchback trial against an explicitly supplied child.

This runs NO model on import and starts NO local service. A trusted parent owns
the private Case/ProbeSession; every candidate receives only JSON public view.
An explicit `ollama` opt-in uses the existing process adapter and an already
running loopback runtime. An explicit `command` opt-in runs a bounded local
argv vector with shell=False, sanitized environment and bounded pipes.

Not an OS sandbox, neural-weight attestation, private heldout exam, or learning
claim. This repository's deterministic source is public and enumerable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Thread
import time
from typing import Any

from labyrinth.minotaur_process_adapter import (
    ProcessAdapterFailure, ProcessCallReceipt, ProcessModelAdapter,
    _child_environment, _finish_or_track_child, _read_json,
    _json as _process_json, _release_child_call, _reserve_child_call,
    SCHEMA as PROCESS_SCHEMA,
)
from labyrinth.switchback_circuit import MAX_MOVES, MAX_PROBES, ProbeSession, make_case
from labyrinth.switchback_oracle import verify_receipt_replay


SCHEMA = "labyrinth.switchback-real-model-trial.v1"
PUBLIC_SUMMARY_SCHEMA = "labyrinth.switchback-real-model-trial-public-summary.v1"
PUBLIC_SCHEMA = "labyrinth.switchback-public-turn.v1"
MAX_PROMPT_BYTES = 16000
MAX_OUTPUT_BYTES = 8192
MAX_COMMAND_ARGS = 32
MAX_HASH_FILE_BYTES = 8 * 1024**3
MAX_HASH_SECONDS = 120
PUBLIC_INSTRUCTIONS = (
    'Respond with ONE JSON object: {"kind":"ACTION","action":"0|1|WAIT|RESET"} '
    'for a public probe, or {"kind":"PLAN","actions":["0","WAIT",...,"COMMIT"]}, '
    '{"kind":"CLAIM_UNSAT"}, {"kind":"NOVEL_RULE","description":"..."}, '
    'or {"kind":"ABSTAIN"}. A plan must meet the public move/pulse limits. '
    "Invalid output is terminal; no source seed or evaluator rules are supplied."
)
PRIVATE_PROMPT_KEYS = frozenset(("seed", "case_id", "mechanic_signature",
                                  "private", "source_commit", "oracle"))
PUBLIC_VIEW_KEYS = frozenset((
    "prompt", "initial_register", "target_register", "plan_actions",
    "probe_actions", "maximum_moves", "switch_at", "maximum_pulses",
    "current_register", "ticks", "pulses_used", "remaining_probes",
    "observations", "finished",
))
PUBLIC_OBSERVATION_KEYS = frozenset(("action", "valid", "applied", "before", "after"))


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _hash_file(path: str | Path) -> dict[str, object]:
    """Hash an explicitly selected regular file; detect concurrent changes."""
    target = Path(path).expanduser().resolve(strict=True)
    if not target.is_file():
        raise ValueError("model/adapter artifact must be an existing regular file")
    before = target.stat()
    if before.st_size > MAX_HASH_FILE_BYTES:
        raise ValueError("artifact exceeds explicit 8 GiB hashing limit")
    digest = sha256()
    deadline = time.monotonic() + MAX_HASH_SECONDS
    with target.open("rb") as stream:
        while True:
            if time.monotonic() > deadline:
                raise ValueError("artifact hashing exceeded bounded wall time")
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    after = target.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("artifact changed during hashing")
    return {"path": str(target), "bytes": after.st_size, "sha256": digest.hexdigest(),
            "hash_class": "LOCAL_FILE_READ_UNATTESTED"}


@dataclass(frozen=True)
class TrialConfig:
    case_seed: int
    generation_seed: int = 0
    mode: str = "fixed"
    max_model_calls: int = 4
    call_timeout_seconds: float = 2.0
    total_timeout_seconds: float = 30.0
    max_output_bytes: int = MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        if (type(self.case_seed) is not int or not 0 <= self.case_seed < 2**31 or
                type(self.generation_seed) is not int or
                not 0 <= self.generation_seed < 2**31):
            raise ValueError("private and generation seeds must be bounded exact integers")
        if type(self.mode) is not str or self.mode not in ("fixed", "ollama", "command"):
            raise ValueError("unsupported trial mode")
        if (type(self.max_model_calls) is not int or
                not 1 <= self.max_model_calls <= MAX_PROBES + 1):
            raise ValueError("model call budget must be 1..17")
        if (type(self.call_timeout_seconds) not in (int, float) or
                not math.isfinite(self.call_timeout_seconds) or
                not .01 <= self.call_timeout_seconds <= 30):
            raise ValueError("per-call timeout must be .01..30 seconds")
        if (type(self.total_timeout_seconds) not in (int, float) or
                not math.isfinite(self.total_timeout_seconds) or
                self.total_timeout_seconds < self.max_model_calls * self.call_timeout_seconds or
                self.total_timeout_seconds > 180):
            raise ValueError("total timeout must cover all bounded calls and be <=180s")
        if (type(self.max_output_bytes) is not int or
                not 128 <= self.max_output_bytes <= 65536):
            raise ValueError("output bound must be 128..65536 bytes")


class CommandProcessAdapter:
    """Explicit local command, binary-capped stdout, no shell or inherited creds."""

    def __init__(self, argv: tuple[str, ...], *, generation_seed: int = 0,
                 timeout_seconds: float = 2.0, max_output_bytes: int = MAX_OUTPUT_BYTES,
                 authorize_command: bool = False):
        if not authorize_command:
            raise PermissionError("arbitrary local command requires explicit opt-in")
        if (type(argv) is not tuple or not 1 <= len(argv) <= MAX_COMMAND_ARGS or
                any(type(item) is not str or not item or len(item) > 4096 or
                    "\x00" in item for item in argv)):
            raise ValueError("command must be a bounded immutable argv tuple")
        executable = Path(argv[0])
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("command executable must be an existing absolute file")
        if (type(timeout_seconds) not in (int, float) or
                not math.isfinite(timeout_seconds) or not .01 <= timeout_seconds <= 30 or
                type(max_output_bytes) is not int or
                not 128 <= max_output_bytes <= 65536 or
                type(generation_seed) is not int or not 0 <= generation_seed < 2**31):
            raise ValueError("invalid bounded command configuration")
        self.argv = argv
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self.generation_seed = generation_seed
        self.model_id = "explicit-command:unattested"
        self.argv_sha256 = _sha(_canonical(list(argv)))
        self.executable_hash = _hash_file(executable)
        self._receipts: list[ProcessCallReceipt] = []

    @property
    def receipts(self) -> tuple[ProcessCallReceipt, ...]:
        return tuple(self._receipts)

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        # Deliberately discard seed: private case seed is never forwarded.
        if type(prompt) is not str:
            raise ValueError("public prompt must be exact text")
        request = prompt.encode("utf-8")
        if len(request) > MAX_PROMPT_BYTES:
            raise ValueError("public prompt exceeds byte bound")
        _read_json(request)  # fail before launch if malformed/ambiguous JSON
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        response = bytearray()
        child: subprocess.Popen[bytes] | None = None
        reader: Thread | None = None
        writer: Thread | None = None
        child_pid: int | None = None
        status = "CHILD_ERROR"
        admitted = False
        force_reap = False
        oversized = False
        cleanup_status = "NOT_ADMITTED"
        try:
            if not _reserve_child_call():
                status = "RESOURCE_LIMIT"
                raise ProcessAdapterFailure(status)
            admitted = True
            child = subprocess.Popen(
                list(self.argv), shell=False, cwd=str(Path(__file__).resolve().parent.parent),
                env=_child_environment(), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, close_fds=True)
            child_pid = child.pid
            if child.stdin is None or child.stdout is None:
                raise OSError("child IPC pipes unavailable")

            def read_bounded() -> None:
                nonlocal oversized
                if child is None or child.stdout is None:
                    return
                while True:
                    chunk = child.stdout.read(4096)
                    if not chunk:
                        break
                    room = self.max_output_bytes - len(response)
                    response.extend(chunk[:max(0, room)])
                    if len(chunk) > room:
                        oversized = True
                        try:
                            child.kill()
                        except OSError:
                            pass
                        break

            def write_bounded() -> None:
                if child is None or child.stdin is None:
                    return
                try:
                    child.stdin.write(request)
                    child.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    try:
                        child.stdin.close()
                    except OSError:
                        pass

            reader = Thread(target=read_bounded, daemon=True)
            writer = Thread(target=write_bounded, daemon=True)
            reader.start()
            writer.start()
            try:
                child.wait(timeout=max(0., deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                try:
                    child.kill()
                except OSError:
                    pass
                try:
                    child.wait(timeout=max(0., deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    force_reap = True
                raise ProcessAdapterFailure(status)
            finally:
                reader.join(timeout=max(0., deadline - time.monotonic()))
                writer.join(timeout=max(0., deadline - time.monotonic()))
            if oversized:
                status = "OVERSIZE"
                raise ProcessAdapterFailure(status)
            if reader.is_alive() or writer.is_alive():
                status = "TIMEOUT"
                raise ProcessAdapterFailure(status)
            if child.returncode != 0:
                status = "CHILD_ERROR"
                raise ProcessAdapterFailure(status)
            try:
                decoded = bytes(response).decode("utf-8", errors="strict")
            except UnicodeError:
                status = "INVALID_UTF8"
                raise ProcessAdapterFailure(status) from None
            status = "OK"
            return decoded
        except ProcessAdapterFailure:
            raise
        except (OSError, RuntimeError) as error:
            status = "CHILD_ERROR"
            if child is not None and child.returncode is None:
                try:
                    child.kill()
                except OSError:
                    pass
                try:
                    child.wait(timeout=max(0., deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    force_reap = True
            raise ProcessAdapterFailure(status) from error
        finally:
            if admitted:
                cleanup_status = (_release_child_call() or "SLOT_RELEASED_NO_CHILD") if child is None else (
                    _finish_or_track_child(child, reader, writer, force_reap=force_reap))
            self._receipts.append(ProcessCallReceipt(
                schema="labyrinth.minotaur-process.v1", status=status,
                child_pid=child_pid, generation_seed=self.generation_seed,
                evaluator_seed_discarded=True, request_sha256=_sha(request),
                response_sha256=_sha(bytes(response)) if response else None,
                response_bytes=len(response), parent_session_frame_visible=None,
                elapsed_ms=round((time.monotonic() - started) * 1000),
                resource_admission=cleanup_status))


def _public_request(view: dict[str, object], turn: int) -> str:
    if type(view) is not dict or type(turn) is not int or turn < 0:
        raise ValueError("invalid trusted public view")
    if set(view) != PUBLIC_VIEW_KEYS:
        raise ValueError("public view differs from exact declared candidate contract")
    observations = view["observations"]
    if (type(observations) is not list or len(observations) > MAX_PROBES or
            any(type(row) is not dict or set(row) != PUBLIC_OBSERVATION_KEYS
                for row in observations)):
        raise ValueError("candidate history violates bounded observation contract")
    # Before/after are legitimate public REGISTER observations inside history,
    # but are not permitted as top-level private rule configuration objects.
    if "before" in view or "after" in view:
        raise ValueError("latent before/after rules cannot enter public request")
    def check(node: object) -> None:
        if type(node) is dict:
            for key, value in node.items():
                if type(key) is not str or key in PRIVATE_PROMPT_KEYS:
                    raise ValueError("private key found in candidate public view")
                check(value)
        elif type(node) is list:
            for value in node:
                check(value)
        elif node is not None and type(node) not in (str, bool, int, float):
            raise ValueError("unserializable candidate public view")
    check(view)
    raw = _canonical({"schema": PUBLIC_SCHEMA, "turn": turn, "view": view,
                      "response_instructions": PUBLIC_INSTRUCTIONS})
    if len(raw) > MAX_PROMPT_BYTES:
        raise ValueError("candidate public prompt too large")
    return raw.decode("utf-8")


def _parse_response(raw: str, public_actions: list[str],
                    *, max_bytes: int = MAX_OUTPUT_BYTES) -> dict[str, object]:
    if type(raw) is not str or len(raw.encode("utf-8")) > max_bytes:
        raise ValueError("model response exceeds strict limit")
    decision = _read_json(raw.encode("utf-8"))
    if type(decision) is not dict or type(decision.get("kind")) is not str:
        raise ValueError("response must be a single JSON decision")
    kind = decision["kind"]
    if kind == "ACTION" and set(decision) == {"kind", "action"}:
        if type(decision["action"]) is str and decision["action"] in public_actions:
            return decision
    if kind == "PLAN" and set(decision) == {"kind", "actions"}:
        actions = decision["actions"]
        if (type(actions) is list and 1 <= len(actions) <= MAX_MOVES and
                all(type(action) is str and action in ("0", "1", "WAIT", "COMMIT")
                    for action in actions)):
            return decision
    if kind in ("CLAIM_UNSAT", "ABSTAIN") and set(decision) == {"kind"}:
        return decision
    if (kind == "NOVEL_RULE" and set(decision) == {"kind", "description"} and
            type(decision["description"]) is str and
            0 < len(decision["description"]) <= 2000):
        return decision
    if (kind == "CLAIM" and set(decision) == {"kind", "claim"} and
            type(decision["claim"]) is dict and decision["claim"] ==
            {"kind": "ABSTAIN", "assessment": "UNKNOWN"}):
        return {"kind": "ABSTAIN"}  # Existing ProcessModelAdapter fixture.
    raise ValueError("illegal, over-budget, or unrecognized model decision")


def _independent_wire_decision(raw: str, public_actions: list[str],
                               *, max_bytes: int) -> dict[str, object]:
    """Independently bind scored decision to ACTUAL returned child bytes.

    This intentionally does not invoke `_parse_response`; otherwise one
    patched parser could manufacture a successful plan from child ABSTAIN.
    """
    if type(raw) is not str or len(raw.encode("utf-8")) > max_bytes:
        raise ValueError("unbounded raw model output")
    # This MUST NOT share the scoring parser's imported `_read_json` entrypoint.
    # A single erroneous or patched decoder previously changed an actual child
    # ABSTAIN into a winning PLAN in both supposed independent checks.
    def exact_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate raw wire JSON key")
            result[key] = item
        return result

    def reject_constant(_value: str) -> Any:
        raise ValueError("non-finite raw wire number")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite raw wire number")
        return result

    try:
        # Use JSONDecoder directly: the primary `_read_json` path calls
        # `json.loads`. A single replacement of json.loads in the trusted
        # interpreter must not corrupt both independent raw-output checks.
        value = json.JSONDecoder(object_pairs_hook=exact_pairs,
                                 parse_constant=reject_constant,
                                 parse_float=finite_float).decode(raw)
    except RecursionError as error:
        raise ValueError("raw wire JSON nesting exceeds limit") from error
    if type(value) is not dict or type(value.get("kind")) is not str:
        raise ValueError("raw decision is not an exact JSON object")
    tag = value["kind"]
    if tag == "CLAIM":
        if (set(value) != {"kind", "claim"} or
                type(value["claim"]) is not dict or value["claim"] !=
                {"kind": "ABSTAIN", "assessment": "UNKNOWN"}):
            raise ValueError("unrecognized raw claim")
        return {"kind": "ABSTAIN"}
    if tag == "ACTION":
        if (set(value) != {"kind", "action"} or
                type(value["action"]) is not str or
                value["action"] not in public_actions):
            raise ValueError("invalid raw probe")
    elif tag == "PLAN":
        if (set(value) != {"kind", "actions"} or type(value["actions"]) is not list or
                not 1 <= len(value["actions"]) <= MAX_MOVES or any(
                    type(action) is not str or
                    action not in ("0", "1", "WAIT", "COMMIT")
                    for action in value["actions"])):
            raise ValueError("invalid raw plan")
    elif tag in ("ABSTAIN", "CLAIM_UNSAT"):
        if set(value) != {"kind"}:
            raise ValueError("extra fields in raw terminal claim")
    elif tag == "NOVEL_RULE":
        if (set(value) != {"kind", "description"} or
                type(value["description"]) is not str or
                not 0 < len(value["description"]) <= 2000):
            raise ValueError("invalid novel-rule narrative")
    else:
        raise ValueError("unrecognized raw decision kind")
    return value


def _bound_process_generation(adapter: ProcessModelAdapter | CommandProcessAdapter,
                              prompt: str, response: str,
                              receipts: tuple[ProcessCallReceipt, ...]) -> bool:
    """Require a completed local child call matching the returned wire text.

    The built-in process adapter hashes its *child envelope*, while the command
    adapter hashes its raw stdout. Both must match the adapter-specific protocol.
    These are parent-owned, self-reported consistency receipts, NOT independent
    attestation of a real process, of a neural model, of the subprocess
    executable, or of OS isolation. Host interpreter integrity is assumed.
    """
    if (type(prompt) is not str or type(response) is not str or
            len(receipts) != 1 or type(receipts[0]) is not ProcessCallReceipt):
        return False
    receipt = receipts[0]
    if (receipt.schema != PROCESS_SCHEMA or receipt.status != "OK" or
            type(receipt.child_pid) is not int or receipt.child_pid <= 0 or
            type(receipt.generation_seed) is not int or
            receipt.generation_seed != adapter.generation_seed or
            receipt.evaluator_seed_discarded is not True or
            receipt.resource_admission != "REAPED_SLOT_RELEASED" or
            type(receipt.response_bytes) is not int or
            type(receipt.response_sha256) is not str or
            type(receipt.request_sha256) is not str):
        return False
    if type(adapter) is CommandProcessAdapter:
        if receipt.parent_session_frame_visible is not None:
            return False
        request = prompt.encode("utf-8")
        wire = response.encode("utf-8")
    else:
        if receipt.parent_session_frame_visible is not False:
            return False
        request = _process_json({
            "schema": PROCESS_SCHEMA, "public_prompt": prompt,
            "generation_seed": adapter.generation_seed, "mode": adapter.mode,
            "actions": adapter.actions, "model": adapter.model,
            "base_url": adapter.base_url,
        })
        wire = _process_json({
            "schema": PROCESS_SCHEMA, "response": response,
            "child_pid": receipt.child_pid,
            "parent_session_frame_visible": False,
        })
    return (receipt.request_sha256 == _sha(request) and
            receipt.response_sha256 == _sha(wire) and
            receipt.response_bytes == len(wire))


def _bounded_report_response(raw: object, max_bytes: int) -> str | None:
    """Retain exact model wire text only inside the HOST-PRIVATE trial report.

    The batch verifier needs these bounded bytes to recompute the nested child
    envelope hash. A digest alone cannot reconstruct a discarded response.
    Never truncate: a truncated response cannot be bound to its original hash.
    """
    if type(raw) is not str:
        return None
    try:
        size = len(raw.encode("utf-8"))
    except UnicodeError:
        return None
    return raw if size <= max_bytes else None


def _bound_to_actual_trial(receipt: object, case: object,
                           proposal: dict[str, object], observations: list[object],
                           model_probe_actions: list[str]) -> bool:
    """Bind a self-replayable oracle receipt to THIS candidate and THIS case.

    Standalone oracle replay is insufficient: a real receipt from another case
    or an unrelated answer must not be substituted for the model's behavior.
    """
    if (type(receipt) is not dict or type(receipt.get("seed")) is not int or
            receipt["seed"] != case.seed or
            type(receipt.get("case_id")) is not str or
            receipt["case_id"] != case.case_id or
            type(receipt.get("adjudication")) is not dict or
            type(receipt.get("probe_trace")) is not list or
            type(receipt.get("probe_count")) is not int or
            receipt["probe_count"] != len(observations) or
            len(observations) != len(model_probe_actions) or
            receipt["adjudication"].get("schema") !=
            "labyrinth.switchback-circuit-receipt.v1"):
        return False
    try:
        actual_actions = [record["action"] for record in observations]
        if (actual_actions != model_probe_actions or
                _canonical(receipt["probe_trace"]) != _canonical(observations) or
                _canonical(receipt["adjudication"]["proposal"]) !=
                _canonical(proposal)):
            return False
    except (TypeError, ValueError, KeyError, OverflowError, RecursionError):
        return False
    return True


def run_trial(config: TrialConfig,
              adapter: ProcessModelAdapter | CommandProcessAdapter,
              *, model_artifact_path: str | Path | None = None) -> dict[str, object]:
    """Run one bounded source-public case with candidate-host IPC only."""
    if type(config) is not TrialConfig or type(adapter) not in (
            ProcessModelAdapter, CommandProcessAdapter):
        raise TypeError("trusted immutable trial config and process adapter required")
    actual_mode = "command" if type(adapter) is CommandProcessAdapter else adapter.mode
    if actual_mode != config.mode:
        raise ValueError("adapter mode must match immutable trial configuration")
    if (adapter.timeout_seconds > config.call_timeout_seconds or
            adapter.max_output_bytes > config.max_output_bytes or
            adapter.generation_seed != config.generation_seed):
        raise ValueError("adapter exceeds configured resource/seed contract")
    if actual_mode == "ollama" and not adapter._local_model_authorized:
        raise PermissionError("Ollama model needs explicit local authorization")
    artifact = _hash_file(model_artifact_path) if model_artifact_path is not None else None
    source_root = Path(__file__).resolve().parent.parent
    source_hashes = {
        "trial_runner": _hash_file(Path(__file__).resolve()),
        "process_adapter": _hash_file(source_root / "labyrinth" / "minotaur_process_adapter.py"),
        "switchback_oracle": _hash_file(source_root / "labyrinth" / "switchback_oracle.py"),
    }
    case = make_case(config.case_seed)
    session = ProbeSession(case)
    turns: list[dict[str, object]] = []
    beginning = time.monotonic()
    source_receipt: dict[str, object] | None = None
    actual_probe_actions: list[str] = []
    candidate_proposal: dict[str, object] | None = None
    observed_before_submission: list[object] | None = None
    terminal_reason = "CALL_BUDGET_EXHAUSTED"
    process_evidence_valid = True
    for turn in range(config.max_model_calls):
        if time.monotonic() - beginning > config.total_timeout_seconds:
            terminal_reason = "TRIAL_TIMEOUT"
            break
        prompt = _public_request(session.view(), turn)
        previous = len(adapter.receipts)
        response: str | None = None
        try:
            response = adapter.generate(prompt, seed=None)
        except ProcessAdapterFailure as error:
            terminal_reason = "MODEL_" + error.status
        except Exception:
            terminal_reason = "MODEL_ADAPTER_ERROR"
        child_receipts = adapter.receipts[previous:]
        if response is None:
            turns.append({"turn": turn, "prompt_sha256": _sha(prompt.encode()),
                          "response_sha256": None, "raw_response": None,
                          "decision": None,
                          "process_receipts": [asdict(r) for r in child_receipts],
                          "status": terminal_reason})
            break
        bounded_response = _bounded_report_response(response, config.max_output_bytes)
        if bounded_response is None or not _bound_process_generation(
                adapter, prompt, bounded_response, child_receipts):
            process_evidence_valid = False
            terminal_reason = "EVALUATOR_DISAGREEMENT"
            turns.append({"turn": turn, "prompt_sha256": _sha(prompt.encode()),
                          "response_sha256": (_sha(bounded_response.encode("utf-8"))
                                              if bounded_response is not None else None),
                          "raw_response": bounded_response, "decision": None,
                          "process_receipts": [asdict(r) for r in child_receipts],
                          "status": terminal_reason})
            break
        try:
            decision = _parse_response(response, session.view()["probe_actions"],
                                       max_bytes=config.max_output_bytes)
            independent = _independent_wire_decision(
                response, session.view()["probe_actions"],
                max_bytes=config.max_output_bytes)
            if _canonical(decision) != _canonical(independent):
                raise ValueError("model decision diverged from raw child wire bytes")
        except (ValueError, TypeError, UnicodeError, RecursionError):
            terminal_reason = "INVALID_MODEL_OUTPUT"
            decision = None
        turns.append({"turn": turn, "prompt_sha256": _sha(prompt.encode()),
                      "response_sha256": _sha(response.encode("utf-8")),
                      "raw_response": response, "decision": decision,
                      "process_receipts": [asdict(r) for r in child_receipts],
                      "status": "PARSED" if decision is not None else terminal_reason})
        if decision is None:
            break
        if decision["kind"] == "ACTION":
            feedback = session.probe(decision["action"])
            actual_probe_actions.append(decision["action"])
            if feedback["finished"] is True:
                terminal_reason = ("PROBE_BUDGET_EXCEEDED" if feedback[
                    "remaining_probes"] == 0 else "EVALUATOR_DISAGREEMENT")
                break
            continue
        proposal = {"kind": "ABSTAIN"} if decision["kind"] == "ABSTAIN" else decision
        candidate_proposal = deepcopy(proposal)
        observed_before_submission = deepcopy(session.view()["observations"])
        source_receipt = session.submit(proposal)
        terminal_reason = "SUBMITTED"
        break
    if source_receipt is None:
        candidate_proposal = {"kind": "ABSTAIN"}
        observed_before_submission = deepcopy(session.view()["observations"])
        source_receipt = session.submit(candidate_proposal)
    independently_replayed = (type(source_receipt) is dict and
                              source_receipt.get("schema") ==
                              "labyrinth.switchback-session-receipt.v1" and
                              verify_receipt_replay(source_receipt))
    exact_trial_binding = (process_evidence_valid and independently_replayed and
                           candidate_proposal is not None and
                           observed_before_submission is not None and
                           _bound_to_actual_trial(source_receipt, case,
                                                  candidate_proposal,
                                                  observed_before_submission,
                                                  actual_probe_actions))
    if not exact_trial_binding:
        terminal_reason = "EVALUATOR_DISAGREEMENT"
    claimed_win = bool(source_receipt.get("verified_success") is True)
    verified_win = (exact_trial_binding and claimed_win and terminal_reason == "SUBMITTED")
    verdict = (source_receipt.get("verdict", "UNKNOWN") if exact_trial_binding else
               "EVALUATOR_DISAGREEMENT")
    config_record = asdict(config)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "source_commit": _git_source_commit(source_root),
        "evidence_class": "SOURCE_PUBLIC_PROCESS_TRIAL",
        "report_classification": "HOST_PRIVATE_SEED_AND_MODEL_OUTPUT_DO_NOT_PUBLISH",
        "model_neural_identity_verified": False,
        "model_weight_update_verified": False,
        "generalization_qualified": False,
        "same_os_user_process_isolation": "NOT_AN_OS_SANDBOX",
        "model_id_unattested": adapter.model_id,
        "mode": actual_mode,
        "case_seed_host_only": config.case_seed,
        "generation_seed_independence": "UNATTESTED",
        "configuration": config_record,
        "config_sha256": _sha(_canonical(config_record)),
        "adapter_sources": source_hashes,
        "model_artifact": artifact,
        "executable": (adapter.executable_hash if type(adapter) is CommandProcessAdapter else None),
        "command_argv_sha256": (adapter.argv_sha256 if type(adapter) is CommandProcessAdapter
                                else None),
        "turns": turns,
        "process_calls": len(turns),
        "terminal_reason": terminal_reason,
        "source_verdict": verdict,
        "source_receipt": source_receipt,
        "source_receipt_replay_valid": independently_replayed,
        "process_receipt_consistency_valid": process_evidence_valid,
        "physical_child_execution_independently_attested": False,
        "trial_binding_valid": exact_trial_binding,
        "verified_success": verified_win,
        "elapsed_ms": round((time.monotonic() - beginning) * 1000),
    }
    report["receipt_sha256"] = _sha(_canonical(report))
    return report


def _git_source_commit(root: Path) -> dict[str, object]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=2, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True, timeout=2,
                                    check=True).stdout.strip())
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid Git commit")
        return {"sha": sha, "worktree_dirty": dirty}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"sha": None, "worktree_dirty": None}


def _public_summary(report: dict[str, object], *, private_report_saved: bool = False,
                    private_report_save_failed: bool = False) -> dict[str, object]:
    """Only source-public source-claimed aggregate outcome, never private transcript."""
    return {
        "schema": PUBLIC_SUMMARY_SCHEMA,
        "evidence_class": "SOURCE_PUBLIC_UNQUALIFIED_SUMMARY",
        "private_report_included": False,
        "private_report_saved": private_report_saved,
        "private_report_save_failed": private_report_save_failed,
        "full_receipt_replayable_from_stdout": False,
        "trial_receipt_sha256": report.get("receipt_sha256"),
        "terminal_reason": report.get("terminal_reason"),
        "source_verdict": report.get("source_verdict"),
        "source_receipt_replay_valid": report.get("source_receipt_replay_valid") is True,
        "verified_success": report.get("verified_success") is True,
        "model_neural_identity_verified": False,
        "model_weight_update_verified": False,
        "generalization_qualified": False,
    }


def _write_private_report_exclusive(report: dict[str, object], path: str) -> None:
    """Finish a mode-0600 host-private report before atomic exclusive publish.

    An existing final filename (including a symlink) is never replaced. This
    does not isolate another process running under the same OS account.
    """
    if type(path) is not str or not Path(path).is_absolute():
        raise ValueError("private report target must be an absolute path")
    destination = Path(path)
    payload = _canonical(report) + b"\n"
    descriptor, staged_path = tempfile.mkstemp(
        prefix=".minotaur-trial-private-", suffix=".tmp",
        dir=str(destination.parent))
    try:
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(staged_path, destination)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        os.unlink(staged_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--mode", choices=("fixed", "ollama", "command"), default="fixed")
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=2.)
    parser.add_argument("--total-timeout", type=float, default=30.)
    parser.add_argument("--max-output-bytes", type=int, default=MAX_OUTPUT_BYTES)
    parser.add_argument("--fixed-actions", nargs="*", default=[])
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-file", default=None)
    export = parser.add_mutually_exclusive_group()
    export.add_argument("--private-report-file", metavar="ABSOLUTE_PATH",
                        help="write full private receipt exclusively as a 0600 file")
    export.add_argument("--emit-private-json-stdout", action="store_true",
                        help="UNSAFE opt-in: print full private receipt to stdout")
    parser.add_argument("--authorize-local-model", action="store_true")
    parser.add_argument("--authorize-command", action="store_true")
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    if args.private_report_file is not None and not Path(
            args.private_report_file).is_absolute():
        parser.error("--private-report-file requires an absolute path")
    config = TrialConfig(args.seed, args.generation_seed, args.mode,
                         args.max_calls, args.timeout, args.total_timeout,
                         args.max_output_bytes)
    if args.mode == "fixed":
        if args.model is not None or args.command:
            parser.error("fixed mode cannot include external model/command")
        adapter: ProcessModelAdapter | CommandProcessAdapter = ProcessModelAdapter(
            mode="fixed", actions=tuple(args.fixed_actions),
            generation_seed=config.generation_seed,
            timeout_seconds=config.call_timeout_seconds,
            max_output_bytes=config.max_output_bytes)
    elif args.mode == "ollama":
        if args.command or not args.authorize_local_model or args.model is None:
            parser.error("ollama requires --model and --authorize-local-model")
        adapter = ProcessModelAdapter(mode="ollama", model=args.model,
                                      authorize_local_model=True,
                                      generation_seed=config.generation_seed,
                                      timeout_seconds=config.call_timeout_seconds,
                                      max_output_bytes=config.max_output_bytes)
    else:
        if args.model is not None or not args.authorize_command or not args.command:
            parser.error("command requires --authorize-command and --command ABSOLUTE_EXEC [ARGS]")
        adapter = CommandProcessAdapter(tuple(args.command),
                                        authorize_command=True,
                                        generation_seed=config.generation_seed,
                                        timeout_seconds=config.call_timeout_seconds,
                                        max_output_bytes=config.max_output_bytes)
    result = run_trial(config, adapter, model_artifact_path=args.model_file)
    if args.emit_private_json_stdout:
        print(_canonical(result).decode("utf-8"))
    elif args.private_report_file is not None:
        try:
            _write_private_report_exclusive(result, args.private_report_file)
        except (OSError, TypeError, ValueError):
            print(_canonical(_public_summary(
                result, private_report_save_failed=True)).decode("utf-8"))
            return 2
        print(_canonical(_public_summary(
            result, private_report_saved=True)).decode("utf-8"))
    else:
        print(_canonical(_public_summary(result)).decode("utf-8"))
    return 0 if result["terminal_reason"] == "SUBMITTED" and result[
        "source_receipt_replay_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
