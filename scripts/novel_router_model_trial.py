"""Bounded, source-public novel-router trial using a real local child process.

No model/server launches on import. The default trial uses an explicit
zero-update Python fixture child. External commands require opt-in. Same OS
user subprocess isolation is NOT an OS sandbox or independently signed custody.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Thread
import time
from typing import Any

from labyrinth import switchback_novel_transfer as novel


SCHEMA = "labyrinth.novel-router-local-process-trial.v1"
PUBLIC_SCHEMA = "labyrinth.novel-router-public-turn.v1"
CHILD_SCHEMA = "labyrinth.novel-router-child-process.v1"
ROOT = Path(__file__).resolve().parent.parent
MAX_OUTPUT = 8192
MAX_PROMPT = 12000
MAX_ARGS = 32
MAX_ARTIFACT = 8 * 1024**3
PRIVATE_KEYS = frozenset(("seed", "case_id", "mechanic_signature",
                          "before_rule", "after_rule", "source_commit",
                          "oracle", "private", "initial_rule", "latent_rule"))
INSTRUCTIONS = (
    'Return a single JSON object: {"kind":"ACTION","action":"LOAD|ROUTE|'
    'TICK|PURGE|RESET"}, or {"kind":"PLAN","actions":["LOAD",...,"COMMIT"]}, '
    '{"kind":"CLAIM_UNSAT"}, {"kind":"NOVEL_RULE","description":"..."}, '
    'or {"kind":"ABSTAIN"}. Invalid output is terminal.'
)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(value if type(value) is bytes else _json(value)).hexdigest()


def _hex_digest(value: object, length: int = 64) -> bool:
    return (type(value) is str and len(value) == length and
            all(char in "0123456789abcdef" for char in value))


def _file_receipt_shape(value: object) -> bool:
    return (type(value) is dict and
            set(value) == {"path", "bytes", "sha256", "class"} and
            type(value["path"]) is str and bool(value["path"]) and
            type(value["bytes"]) is int and 0 <= value["bytes"] <= MAX_ARTIFACT and
            _hex_digest(value["sha256"]) and
            value["class"] == "LOCAL_FILE_UNATTESTED")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key")
        output[key] = value
    return output


def _nonfinite(_: str) -> object:
    raise ValueError("non-finite JSON number")


def _finite_float(text: str) -> float:
    result = float(text)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _load_json(raw: bytes) -> object:
    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique,
                      parse_constant=_nonfinite, parse_float=_finite_float)


def _valid_decision(value: object, probe_actions: list[str]) -> dict[str, object]:
    if type(value) is not dict or type(value.get("kind")) is not str:
        raise ValueError("decision must be a JSON object")
    kind = value["kind"]
    if kind == "ACTION" and set(value) == {"kind", "action"}:
        if type(value["action"]) is str and value["action"] in probe_actions:
            return value
    if kind == "PLAN" and set(value) == {"kind", "actions"}:
        actions = value["actions"]
        if (type(actions) is list and 1 <= len(actions) <= novel.MAX_MOVES and
                all(type(action) is str and action in novel.PLAN_ACTIONS
                    for action in actions)):
            return value
    if kind in ("ABSTAIN", "CLAIM_UNSAT") and set(value) == {"kind"}:
        return value
    if (kind == "NOVEL_RULE" and set(value) == {"kind", "description"} and
            type(value["description"]) is str and
            0 < len(value["description"]) <= 2000):
        return value
    raise ValueError("invalid model proposal or action")


def _parse_decision(raw: bytes, probe_actions: list[str], maximum: int) -> dict[str, object]:
    if type(raw) is not bytes or len(raw) > maximum:
        raise ValueError("output outside bound")
    return _valid_decision(_load_json(raw), probe_actions)


def _independent_wire_decision(raw: bytes, probe_actions: list[str], maximum: int
                               ) -> dict[str, object]:
    """Distinct decoder AND grammar, not the primary parser's shared helpers."""
    if type(raw) is not bytes or len(raw) > maximum:
        raise ValueError("unbounded candidate bytes")
    text = raw.decode("utf-8", "strict")
    def pairs_to_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        assembled: dict[str, object] = {}
        for key, member in pairs:
            if key in assembled:
                raise ValueError("second decoder rejected duplicate key")
            assembled[key] = member
        return assembled

    def finite_number(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("second decoder rejected nonfinite float")
        return number

    def reject_constant(_: str) -> object:
        raise ValueError("second decoder rejected NaN/Infinity")

    decoder = json.JSONDecoder(object_pairs_hook=pairs_to_object,
                               parse_constant=reject_constant,
                               parse_float=finite_number)
    value, ending = decoder.raw_decode(text.lstrip())
    if text.lstrip()[ending:].strip():
        raise ValueError("trailing bytes after JSON")
    if type(value) is not dict:
        raise ValueError("second decoder expected object")
    category = value.get("kind")
    if type(category) is not str:
        raise ValueError("second decoder expected kind")
    if category == "ACTION":
        if (set(value) != {"kind", "action"} or
                type(value.get("action")) is not str or
                value["action"] not in probe_actions):
            raise ValueError("second decoder rejected action")
    elif category == "PLAN":
        steps = value.get("actions")
        if (set(value) != {"kind", "actions"} or type(steps) is not list or
                not 1 <= len(steps) <= novel.MAX_MOVES or
                any(type(step) is not str or step not in novel.PLAN_ACTIONS
                    for step in steps)):
            raise ValueError("second decoder rejected plan")
    elif category in ("ABSTAIN", "CLAIM_UNSAT"):
        if set(value) != {"kind"}:
            raise ValueError("second decoder rejected terminal claim")
    elif category == "NOVEL_RULE":
        text_proposal = value.get("description")
        if (set(value) != {"kind", "description"} or
                type(text_proposal) is not str or
                not 0 < len(text_proposal) <= 2000):
            raise ValueError("second decoder rejected hypothesis")
    else:
        raise ValueError("second decoder rejected category")
    return value


def _public_request(view: dict[str, object], turn: int,
                    generation_seed: int) -> bytes:
    if (type(view) is not dict or type(turn) is not int or turn < 0 or
            type(generation_seed) is not int or
            not 0 <= generation_seed <= novel.MAX_SEED):
        raise ValueError("invalid public turn")
    if set(view) != set(novel.public_view(novel.make_case(0))) | {
            "current_levels", "orientation", "pending_packet", "ticks",
            "loads_used", "routes_used", "post_delivery_observed",
            "remaining_probes", "observations", "finished"}:
        raise ValueError("public view contract changed")
    def scan(part: object) -> None:
        if type(part) is dict:
            for key, child in part.items():
                if type(key) is not str or key in PRIVATE_KEYS:
                    raise ValueError("private field in public prompt")
                scan(child)
        elif type(part) is list:
            for child in part:
                scan(child)
        elif part is not None and type(part) not in (int, float, bool, str):
            raise ValueError("unrepresentable public value")
    scan(view)
    payload = _json({"schema": PUBLIC_SCHEMA, "turn": turn,
                     "generation_seed": generation_seed, "view": view,
                     "instructions": INSTRUCTIONS})
    if len(payload) > MAX_PROMPT:
        raise ValueError("public request exceeds byte cap")
    return payload


def _hash_file(path: str | Path) -> dict[str, object]:
    target = Path(path).resolve(strict=True)
    if not target.is_file():
        raise ValueError("only regular model/executable files can be hashed")
    before = target.stat()
    if before.st_size > MAX_ARTIFACT:
        raise ValueError("model artifact exceeds 8 GiB limit")
    digest = sha256()
    deadline = time.monotonic() + 120
    with target.open("rb") as stream:
        while True:
            if time.monotonic() > deadline:
                raise ValueError("artifact hash deadline")
            part = stream.read(1024 * 1024)
            if not part:
                break
            digest.update(part)
    after = target.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("artifact changed while hashing")
    return {"path": str(target), "bytes": after.st_size,
            "sha256": digest.hexdigest(), "class": "LOCAL_FILE_UNATTESTED"}


@dataclass(frozen=True)
class TrialConfig:
    case_seed: int
    generation_seed: int = 0
    max_calls: int = 4
    call_timeout_seconds: float = 1.0
    total_timeout_seconds: float = 8.0
    max_output_bytes: int = 4096

    def __post_init__(self) -> None:
        if (type(self.case_seed) is not int or
                not 0 <= self.case_seed <= novel.MAX_SEED or
                type(self.generation_seed) is not int or
                not 0 <= self.generation_seed <= novel.MAX_SEED or
                type(self.max_calls) is not int or
                not 1 <= self.max_calls <= novel.MAX_PROBES + 1 or
                type(self.call_timeout_seconds) not in (int, float) or
                not math.isfinite(self.call_timeout_seconds) or
                not .02 <= self.call_timeout_seconds <= 10 or
                type(self.total_timeout_seconds) not in (int, float) or
                not math.isfinite(self.total_timeout_seconds) or
                not self.max_calls * self.call_timeout_seconds <=
                self.total_timeout_seconds <= 220 or
                type(self.max_output_bytes) is not int or
                not 128 <= self.max_output_bytes <= MAX_OUTPUT):
            raise ValueError("invalid frozen process trial limits")


@dataclass(frozen=True)
class AdapterSpec:
    mode: str = "fixture"
    fixture_responses: tuple[str, ...] = ('{"kind":"ABSTAIN"}',)
    command_argv: tuple[str, ...] = ()
    authorize_command: bool = False
    model_artifact_path: str | None = None

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in ("fixture", "command"):
            raise ValueError("unsupported child mode")
        if (type(self.fixture_responses) is not tuple or
                len(self.fixture_responses) > novel.MAX_PROBES + 1 or
                any(type(value) is not str or len(value.encode("utf-8")) > MAX_OUTPUT
                    for value in self.fixture_responses) or
                type(self.command_argv) is not tuple or
                len(self.command_argv) > MAX_ARGS or
                any(type(value) is not str or not value or len(value) > 4096 or
                    "\x00" in value for value in self.command_argv) or
                type(self.authorize_command) is not bool):
            raise ValueError("invalid immutable adapter text")
        if self.model_artifact_path is not None and (
                type(self.model_artifact_path) is not str or
                not self.model_artifact_path):
            raise ValueError("artifact path must be explicit")
        if self.mode == "fixture":
            if self.command_argv or self.authorize_command or self.model_artifact_path:
                raise ValueError("fixture cannot disguise external model/command")
        elif (not self.authorize_command or not self.command_argv or
              self.fixture_responses != ('{"kind":"ABSTAIN"}',) or
              not Path(self.command_argv[0]).is_absolute() or
              not Path(self.command_argv[0]).is_file()):
            raise PermissionError("explicit authorized absolute executable required")


def _argv(spec: AdapterSpec) -> tuple[str, ...]:
    return ((sys.executable, "-m", "scripts.novel_router_model_trial",
             "--child-fixture", *spec.fixture_responses)
            if spec.mode == "fixture" else spec.command_argv)


def _child_environment() -> dict[str, str]:
    # Fixed minimal environment; no inherited credentials, proxies or tokens.
    return {"PATH": os.defpath, "PYTHONPATH": str(ROOT),
            "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def _stop_child(child: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            # Every launched child uses start_new_session=True, so PGID=PID.
            # Its parent may have exited while a grandchild still holds the
            # stdout pipe; killing only the direct (already reaped) parent
            # would leave that descendant running. The short-lived, owned
            # process group is the cleanup target, not arbitrary processes.
            os.killpg(child.pid, signal.SIGKILL)
        elif child.poll() is None:
            child.kill()
    except (OSError, ProcessLookupError):
        pass


def _child_call(argv: tuple[str, ...], request: bytes, generation_seed: int,
                timeout: float, maximum: int) -> tuple[bytes, dict[str, object]]:
    begun = time.monotonic()
    output = bytearray()
    pid: int | None = None
    child: subprocess.Popen[bytes] | None = None
    status = "CHILD_ERROR"
    oversize = False
    reader: Thread | None = None
    writer: Thread | None = None
    try:
        child = subprocess.Popen(list(argv), shell=False, cwd=str(ROOT),
                                 env=_child_environment(), stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 close_fds=True, start_new_session=(os.name == "posix"))
        pid = child.pid

        def receive() -> None:
            nonlocal oversize
            if child is None or child.stdout is None:
                return
            while True:
                part = child.stdout.read(4096)
                if not part:
                    break
                remain = maximum - len(output)
                output.extend(part[:max(0, remain)])
                if len(part) > remain:
                    oversize = True
                    _stop_child(child)
                    break

        def send() -> None:
            if child is None or child.stdin is None:
                return
            try:
                child.stdin.write(request)
                child.stdin.flush()
            except (OSError, BrokenPipeError):
                pass
            finally:
                try:
                    child.stdin.close()
                except OSError:
                    pass

        reader = Thread(target=receive, daemon=True)
        writer = Thread(target=send, daemon=True)
        reader.start()
        writer.start()
        try:
            child.wait(timeout=max(0., timeout - (time.monotonic() - begun)))
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            _stop_child(child)
            child.wait(timeout=.2)
        reader.join(timeout=max(0., timeout - (time.monotonic() - begun)))
        writer.join(timeout=max(0., timeout - (time.monotonic() - begun)))
        if status != "TIMEOUT":
            if oversize:
                status = "OVERSIZE"
            elif reader.is_alive() or writer.is_alive():
                status = "TIMEOUT"
            elif child.returncode != 0:
                status = "CHILD_ERROR"
            else:
                status = "OK"
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        status = "CHILD_ERROR" if status != "TIMEOUT" else "TIMEOUT"
    finally:
        if child is not None:
            _stop_child(child)
            try:
                child.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass  # No claimed host-global cleanup proof.
            if reader is not None:
                reader.join(timeout=.1)
            if writer is not None:
                writer.join(timeout=.1)
            for pipe in (child.stdin, child.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass
    record: dict[str, object] = {
        "schema": CHILD_SCHEMA,
        "status": status, "pid": pid,
        "generation_seed": generation_seed,
        "generation_seed_application": "PUBLIC_REQUEST_ONLY_UNATTESTED",
        "private_case_seed_forwarded": False,
        "request_sha256": _sha(request),
        "response_sha256": _sha(bytes(output)) if output else None,
        "response_bytes": len(output),
        "raw_stdout_base64": base64.b64encode(bytes(output)).decode("ascii"),
        "elapsed_ms": round((time.monotonic() - begun) * 1000),
        "same_os_user_isolation": "NOT_A_SANDBOX",
        "cleanup_custody": "UNATTESTED",
    }
    return bytes(output), record


def _source_receipt_valid(case: novel.Case, source: object,
                          actions: list[str], proposal: dict[str, object]) -> bool:
    if (type(source) is not dict or source.get("schema") !=
            "labyrinth.novel-router-session.v1" or
            source.get("private_custody") is not False or
            source.get("neural_learning_verified") is not False or
            source.get("graduation_eligible") is not False or
            type(source.get("probe_count")) is not int or
            source["probe_count"] != len(actions) or
            type(source.get("probe_trace")) is not list):
        return False
    try:
        expected = novel.ProbeSession(case)
        for action in actions:
            if expected.view()["finished"]:
                return False
            expected.probe(action)
        independently = expected.submit(deepcopy(proposal))
        if _json(independently) != _json(source):
            return False
        core = source["adjudication"]
        if (core.get("case_id") != case.case_id or
                type(core.get("seed")) is not int or core["seed"] != case.seed or
                _json(core.get("proposal")) != _json(proposal) or
                core.get("sha256") != _sha({key: value for key, value in core.items()
                                            if key != "sha256"}) or
                source.get("verified_success") is not (core["verdict"] == "VERIFIED_WIN")):
            return False
        if core["verdict"] == "VERIFIED_WIN" and (
                proposal.get("kind") != "PLAN" or
                novel._independent_plan_verdict(case, proposal["actions"]) !=
                "VERIFIED_WIN"):
            return False
        return True
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError,
            RecursionError):
        return False


def _git_receipt() -> dict[str, object]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, check=True, text=True,
                             timeout=2).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, check=True, text=True,
                                    timeout=2).stdout.strip())
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"sha": None, "dirty": None}


def run_trial(config: TrialConfig, spec: AdapterSpec) -> dict[str, object]:
    if type(config) is not TrialConfig or type(spec) is not AdapterSpec:
        raise TypeError("frozen exact trial and adapter required")
    argv = _argv(spec)
    executable = _hash_file(argv[0])
    model_file = (_hash_file(spec.model_artifact_path)
                  if spec.model_artifact_path else None)
    case = novel.make_case(config.case_seed)
    session = novel.ProbeSession(case)
    beginning = time.monotonic()
    turns: list[dict[str, object]] = []
    actions: list[str] = []
    proposal: dict[str, object] | None = None
    terminal = "CALL_BUDGET_EXHAUSTED"
    source: dict[str, object] | None = None
    for index in range(config.max_calls):
        if time.monotonic() - beginning >= config.total_timeout_seconds:
            terminal = "TRIAL_TIMEOUT"
            break
        view = session.view()
        request = _public_request(view, index, config.generation_seed)
        budget = min(config.call_timeout_seconds, config.total_timeout_seconds -
                     (time.monotonic() - beginning))
        output, receipt = _child_call(argv, request, config.generation_seed,
                                      budget, config.max_output_bytes)
        decision: dict[str, object] | None = None
        if receipt["status"] != "OK":
            terminal = "MODEL_" + str(receipt["status"])
        else:
            try:
                decision = _parse_decision(output, view["probe_actions"],
                                           config.max_output_bytes)
                independent = _independent_wire_decision(
                    output, view["probe_actions"], config.max_output_bytes)
                if _json(decision) != _json(independent):
                    raise ValueError("parser and independent wire disagree")
            except (ValueError, TypeError, UnicodeError, RecursionError):
                terminal = "INVALID_MODEL_OUTPUT"
                decision = None
        turns.append({"turn": index, "request_sha256": _sha(request),
                      "decision": deepcopy(decision), "child": receipt,
                      "status": "PARSED" if decision is not None else terminal})
        if decision is None:
            break
        if decision["kind"] == "ACTION":
            actions.append(decision["action"])
            feedback = session.probe(decision["action"])
            if feedback["finished"]:
                terminal = "EVALUATOR_DISAGREEMENT"
                break
            continue
        proposal = deepcopy(decision)
        source = session.submit(proposal)
        terminal = "SUBMITTED"
        break
    if source is None:
        proposal = {"kind": "ABSTAIN"}
        source = session.submit(proposal)
    source_valid = _source_receipt_valid(case, source, actions, proposal)
    if not source_valid:
        terminal = "EVALUATOR_DISAGREEMENT"
    candidate_plan = bool(turns and turns[-1]["decision"] is not None and
                          turns[-1]["decision"].get("kind") == "PLAN")
    success = (source_valid and terminal == "SUBMITTED" and candidate_plan and
               source.get("verified_success") is True and
               all(row["child"]["status"] == "OK" for row in turns))
    report: dict[str, object] = {
        "schema": SCHEMA,
        "evidence_class": "SOURCE_PUBLIC_LOCAL_PROCESS_UNATTESTED",
        "source_commit": _git_receipt(),
        "private_custody": False,
        "neural_learning_verified": False,
        "generalization_qualified": False,
        "graduation_eligible": False,
        "model_weight_update_verified": False,
        "model_identity_qualified": False,
        "generation_seed_effect_verified": False,
        "same_os_user_process_isolation": "NOT_AN_OS_SANDBOX",
        "report_classification": "HOST_PRIVATE_CASE_SEED_AND_MODEL_OUTPUT",
        "configuration": asdict(config), "configuration_sha256": _sha(asdict(config)),
        "adapter": asdict(spec), "adapter_sha256": _sha(asdict(spec)),
        "argv_sha256": _sha(list(argv)),
        "executable": executable,
        "model_artifact": model_file,
        "source_files": {"runner": _hash_file(Path(__file__).resolve()),
                         "novel_router": _hash_file(Path(novel.__file__).resolve())},
        "turns": turns, "process_calls": len(turns),
        "terminal_reason": terminal,
        "source_receipt": deepcopy(source),
        "source_receipt_replay_valid": source_valid,
        "source_verdict": source.get("verdict") if source_valid else "UNKNOWN",
        "verified_success": success,
        "elapsed_ms": round((time.monotonic() - beginning) * 1000),
    }
    report["receipt_sha256"] = _sha(report)
    if not verify_trial_receipt(report):
        report["terminal_reason"] = "EVALUATOR_DISAGREEMENT"
        report["verified_success"] = False
        report["source_receipt_replay_valid"] = False
        report["source_verdict"] = "UNKNOWN"
        report["receipt_sha256"] = _sha({key: value for key, value in report.items()
                                          if key != "receipt_sha256"})
    return report


def verify_trial_receipt(report: object) -> bool:
    """Recompute claimed source outcome from raw bounded child-wire transcript.

    Does not independently authenticate PID, elapsed time, executable identity,
    user ownership, network isolation, hidden rules or the Python interpreter.
    """
    if type(report) is not dict or report.get("schema") != SCHEMA:
        return False
    try:
        if (type(report.get("receipt_sha256")) is not str or
                report["receipt_sha256"] != _sha({
                    key: value for key, value in report.items()
                    if key != "receipt_sha256"}) or
                type(report.get("configuration")) is not dict or
                type(report.get("adapter")) is not dict or
                report.get("configuration_sha256") != _sha(report["configuration"]) or
                report.get("adapter_sha256") != _sha(report["adapter"]) or
                report.get("evidence_class") !=
                "SOURCE_PUBLIC_LOCAL_PROCESS_UNATTESTED" or
                report.get("same_os_user_process_isolation") !=
                "NOT_AN_OS_SANDBOX" or
                report.get("report_classification") !=
                "HOST_PRIVATE_CASE_SEED_AND_MODEL_OUTPUT" or
                any(report.get(key) is not False for key in (
                    "private_custody", "neural_learning_verified",
                    "generalization_qualified", "graduation_eligible",
                    "model_weight_update_verified", "model_identity_qualified",
                    "generation_seed_effect_verified"))):
            return False
        config = TrialConfig(**report["configuration"])
        adapter_data = deepcopy(report["adapter"])
        for key in ("fixture_responses", "command_argv"):
            adapter_data[key] = tuple(adapter_data[key])
        spec = AdapterSpec(**adapter_data)
        argv = _argv(spec)
        commit = report.get("source_commit")
        files = report.get("source_files")
        artifact = report.get("model_artifact")
        expected_paths = {
            "runner": str(Path(__file__).resolve()),
            "novel_router": str(Path(novel.__file__).resolve()),
        }
        if (type(commit) is not dict or set(commit) != {"sha", "dirty"} or
                not (commit["sha"] is None or _hex_digest(commit["sha"], 40)) or
                not (type(commit["dirty"]) is bool or commit["dirty"] is None) or
                (commit["sha"] is None) != (commit["dirty"] is None) or
                not _file_receipt_shape(report.get("executable")) or
                report["executable"]["path"] != str(Path(argv[0]).resolve()) or
                type(files) is not dict or set(files) != set(expected_paths) or
                any(not _file_receipt_shape(files.get(key)) or
                    files[key]["path"] != expected_path
                    for key, expected_path in expected_paths.items()) or
                not _hex_digest(report.get("argv_sha256")) or
                not _hex_digest(report.get("configuration_sha256")) or
                not _hex_digest(report.get("adapter_sha256")) or
                type(report.get("elapsed_ms")) is not int or
                report["elapsed_ms"] < 0):
            return False
        if spec.model_artifact_path is None:
            if artifact is not None:
                return False
        elif (not _file_receipt_shape(artifact) or
              artifact["path"] != str(Path(spec.model_artifact_path).resolve())):
            return False
        if (report.get("argv_sha256") != _sha(list(argv)) or
                type(report.get("turns")) is not list or
                len(report["turns"]) > config.max_calls or
                type(report.get("process_calls")) is not int or
                report["process_calls"] != len(report["turns"])):
            return False
        case = novel.make_case(config.case_seed)
        session = novel.ProbeSession(case)
        actions: list[str] = []
        candidate_terminal: dict[str, object] | None = None
        expected_terminal: str | None = None
        for index, turn in enumerate(report["turns"]):
            if type(turn) is not dict or type(turn.get("turn")) is not int or \
                    turn["turn"] != index:
                return False
            view = session.view()
            prompt = _public_request(view, index, config.generation_seed)
            child = turn.get("child")
            if (type(child) is not dict or child.get("schema") != CHILD_SCHEMA or
                    child.get("private_case_seed_forwarded") is not False or
                    child.get("generation_seed_application") !=
                    "PUBLIC_REQUEST_ONLY_UNATTESTED" or
                    child.get("same_os_user_isolation") != "NOT_A_SANDBOX" or
                    child.get("cleanup_custody") != "UNATTESTED" or
                    type(child.get("generation_seed")) is not int or
                    child["generation_seed"] != config.generation_seed or
                    child.get("request_sha256") != _sha(prompt) or
                    turn.get("request_sha256") != _sha(prompt) or
                    type(child.get("raw_stdout_base64")) is not str):
                return False
            raw = base64.b64decode(child["raw_stdout_base64"], validate=True)
            if (len(raw) > config.max_output_bytes or
                    child.get("response_sha256") != (_sha(raw) if raw else None) or
                    type(child.get("response_bytes")) is not int or
                    child["response_bytes"] != len(raw) or
                    type(child.get("elapsed_ms")) is not int or
                    child["elapsed_ms"] < 0):
                return False
            if child.get("status") == "OK":
                if type(child.get("pid")) is not int or child["pid"] <= 0:
                    return False
                try:
                    observed = _independent_wire_decision(
                        raw, view["probe_actions"], config.max_output_bytes)
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    # Invalid child syntax is a legitimate terminal *failure*
                    # of the candidate, not evaluator corruption. Do not
                    # reinterpret it as an unspecified model proposal.
                    if (turn.get("decision") is not None or
                            turn.get("status") != "INVALID_MODEL_OUTPUT" or
                            report.get("terminal_reason") != "INVALID_MODEL_OUTPUT" or
                            index != len(report["turns"]) - 1):
                        return False
                    expected_terminal = "INVALID_MODEL_OUTPUT"
                    continue
                if turn.get("decision") != observed or turn.get("status") != "PARSED":
                    return False
                if observed["kind"] == "ACTION":
                    if candidate_terminal is not None:
                        return False
                    actions.append(observed["action"])
                    feedback = session.probe(observed["action"])
                    if feedback["finished"]:
                        return False
                else:
                    if candidate_terminal is not None or index != len(report["turns"]) - 1:
                        return False
                    candidate_terminal = observed
                    expected_terminal = "SUBMITTED"
            else:
                if (child.get("status") not in ("CHILD_ERROR", "TIMEOUT", "OVERSIZE") or
                        turn.get("decision") is not None or
                        index != len(report["turns"]) - 1 or
                        turn.get("status") != "MODEL_" + child["status"]):
                    return False
                expected_terminal = "MODEL_" + child["status"]
        if expected_terminal is None:
            expected_terminal = ("CALL_BUDGET_EXHAUSTED" if
                                 len(report["turns"]) == config.max_calls else
                                 "TRIAL_TIMEOUT")
        if report.get("terminal_reason") != expected_terminal:
            return False
        proposal = deepcopy(candidate_terminal) if candidate_terminal is not None else {
            "kind": "ABSTAIN"}
        if not _source_receipt_valid(case, report.get("source_receipt"), actions, proposal):
            return False
        winning = bool(candidate_terminal is not None and
                       candidate_terminal.get("kind") == "PLAN" and
                       report.get("terminal_reason") == "SUBMITTED" and
                       report["source_receipt"].get("verified_success") is True and
                       all(row["child"].get("status") == "OK" for row in report["turns"]))
        if (report.get("verified_success") is not winning or
                report.get("source_verdict") != report["source_receipt"].get("verdict") or
                report.get("source_receipt_replay_valid") is not True):
            return False
        return True
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError,
            RecursionError, UnicodeError, base64.binascii.Error):
        return False


def _child_fixture_main(responses: tuple[str, ...]) -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_PROMPT + 1)
        if len(raw) > MAX_PROMPT:
            return 2
        request = _load_json(raw)
        if (type(request) is not dict or set(request) !=
                {"schema", "turn", "generation_seed", "view", "instructions"} or
                request["schema"] != PUBLIC_SCHEMA or
                type(request["turn"]) is not int or request["turn"] < 0 or
                type(request["generation_seed"]) is not int or
                not 0 <= request["generation_seed"] <= novel.MAX_SEED or
                type(request["view"]) is not dict or
                any(key in PRIVATE_KEYS for key in request["view"])):
            return 2
        output = (responses[request["turn"]] if request["turn"] < len(responses)
                  else '{"kind":"ABSTAIN"}')
        sys.stdout.buffer.write(output.encode("utf-8"))
        return 0
    except (TypeError, ValueError, KeyError, UnicodeError, RecursionError):
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-fixture", nargs="*")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1.)
    parser.add_argument("--total-timeout", type=float, default=8.)
    parser.add_argument("--max-output-bytes", type=int, default=4096)
    parser.add_argument("--fixture-json", nargs="*", default=[])
    parser.add_argument("--authorize-command", action="store_true")
    parser.add_argument("--model-artifact", default=None)
    parser.add_argument("--emit-private-json-stdout", action="store_true",
                        help="explicitly print host-private seed/argv/raw output receipt")
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    if args.child_fixture is not None:
        return _child_fixture_main(tuple(args.child_fixture))
    if args.seed is None:
        parser.error("explicit --seed required; no model or fixture starts by default")
    config = TrialConfig(args.seed, args.generation_seed, args.max_calls,
                         args.timeout, args.total_timeout, args.max_output_bytes)
    if args.command:
        spec = AdapterSpec("command", command_argv=tuple(args.command),
                           authorize_command=args.authorize_command,
                           model_artifact_path=args.model_artifact)
    else:
        if args.authorize_command or args.model_artifact:
            parser.error("fixture cannot claim external model authorization")
        spec = AdapterSpec("fixture", fixture_responses=(
            tuple(args.fixture_json) if args.fixture_json else ('{"kind":"ABSTAIN"}',)))
    report = run_trial(config, spec)
    if args.emit_private_json_stdout:
        emitted = report
    else:
        emitted = {
            "schema": "labyrinth.novel-router-local-trial-summary.v1",
            "evidence_class": "SOURCE_PUBLIC_LOCAL_PROCESS_UNATTESTED",
            "private_receipt_omitted": True,
            "model_weight_update_verified": False,
            "neural_learning_verified": False,
            "graduation_eligible": False,
            "terminal_reason": report["terminal_reason"],
            "source_verdict": report["source_verdict"],
            "verified_success": report["verified_success"],
            "source_receipt_replay_valid": report["source_receipt_replay_valid"],
            "process_calls": report["process_calls"],
            "private_receipt_sha256": report["receipt_sha256"],
        }
    print(_json(emitted).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
