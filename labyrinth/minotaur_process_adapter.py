"""Bounded process-separated Minotaur candidate adapter.

This closes *Python caller-frame* access to the host examiner for executable
candidate behavior launched solely in the child: a fresh Python process
receives only the public prompt and trusted adapter configuration over stdin.
It deliberately ignores ModelAdapter.generate(seed=...), because an older
caller passed the PRIVATE case-generation seed through that parameter.
Discarding that argument does NOT establish the replacement generation seed's
statistical independence, entropy, or legitimate matched stochastic controls.

The parent launcher and any `candidate_factory` that constructs this adapter
MUST remain trusted host code. Executing untrusted factory Python in the host
before calling this adapter preserves a fatal caller-frame/private-spec leak.
`__slots__` and the setter guard prevent *ordinary* configuration mutation,
not adversarial same-process Python: `object.__setattr__` overrides the guard,
including its local-model authorization flag. This is not a hostile factory
authorization boundary. Requiring trusted host launcher code remains essential.

IMPORTANT: A same-user subprocess is NOT a hardened sandbox. It can retain
the parent's filesystem permissions and may access other processes' files or
debug facilities. The bridge cannot attest the identity of neural weights,
private dataset custody, machine isolation, or actual generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
import time
from typing import Any

from .adapters import AdapterError, OllamaAdapter


SCHEMA = "labyrinth.minotaur-process.v1"
MAX_PROMPT_BYTES = 64000
MAX_REQUEST_BYTES = 131072
ALLOWED_MODES = frozenset({"fixed", "probe", "stall", "oversize", "invalid_json", "breach", "ollama"})
_ABSTAIN = '{"claim":{"assessment":"UNKNOWN","kind":"ABSTAIN"},"kind":"CLAIM"}'


class ProcessAdapterFailure(AdapterError):
    """A transport failure with a stable, machine-checkable disposition."""

    def __init__(self, status: str):
        super().__init__(f"isolated candidate call rejected: {status}")
        self.status = status


@dataclass(frozen=True)
class ProcessCallReceipt:
    schema: str
    status: str
    child_pid: int | None
    generation_seed: int
    evaluator_seed_discarded: bool
    request_sha256: str
    response_sha256: str | None
    response_bytes: int
    parent_session_frame_visible: bool | None
    elapsed_ms: int
    generation_seed_independence: str = "UNATTESTED"
    host_launcher_authorization: str = "TRUSTED_HOST_REQUIRED_UNATTESTED"
    boundary: str = "SEPARATE_PYTHON_PROCESS_SAME_OS_USER_NOT_SANDBOXED"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for key, value in pairs:
        if key in item:
            raise ValueError("duplicate JSON key")
        item[key] = value
    return item


def _reject_nonfinite(_value: str) -> Any:
    raise ValueError("non-finite JSON number")


def _finite_json_float(value: str) -> float:
    # JSON permits large exponents (e.g. 1e999), but Python's default
    # decoder silently converts them to infinity without parse_constant.
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _read_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8", errors="strict"),
                          object_pairs_hook=_unique_pairs, parse_constant=_reject_nonfinite,
                          parse_float=_finite_json_float)
    except RecursionError as error:
        # An attacker can stay below MAX_PROMPT_BYTES yet exceed the JSON
        # decoder's nesting limit. Normalize this into a prelaunch rejection.
        raise ValueError("JSON nesting exceeds decoder limit") from error


def _child_environment() -> dict[str, str]:
    # Remove ambient API tokens, PYTHONPATH, cloud identity, proxy settings.
    # This is hygiene, NOT a security guarantee against a same-user process.
    allowed = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _trusted_prompt(prompt: str) -> bytes:
    if type(prompt) is not str:
        raise ValueError("prompt must be text")
    raw = prompt.encode("utf-8")
    if len(raw) > MAX_PROMPT_BYTES:
        raise ValueError("public prompt exceeds size limit")
    parsed = _read_json(raw)
    if type(parsed) is not dict:
        raise ValueError("public prompt must be a JSON object")
    return raw


class ProcessModelAdapter:
    """Run one model call in a fresh Python process through fixed JSON IPC.

    `fixed` and `probe` are no-update test controls, not neural candidates.
    `ollama` is only enabled by an explicit opt-in and only calls an already
    running loopback runtime; it never launches Ollama or a GPU process.
    """

    __slots__ = ("mode", "actions", "generation_seed", "timeout_seconds",
                 "max_output_bytes", "model", "base_url", "model_id",
                 "_local_model_authorized", "_receipts", "_sealed")

    def __setattr__(self, key: str, value: Any) -> None:
        # This intercepts normal assignment only. Malicious same-process
        # Python can call object.__setattr__ and modify even `_sealed`.
        if getattr(self, "_sealed", False):
            raise AttributeError("process candidate configuration is immutable")
        object.__setattr__(self, key, value)

    def __init__(self, *, mode: str = "fixed", actions: tuple[str, ...] = (),
                 generation_seed: int = 0, timeout_seconds: float = 5.0,
                 max_output_bytes: int = 8192, model: str | None = None,
                 base_url: str = "http://127.0.0.1:11434",
                 authorize_local_model: bool = False):
        if type(mode) is not str or mode not in ALLOWED_MODES:
            raise ValueError("unsupported child mode")
        if (type(actions) is not tuple or len(actions) > 32 or any(
                type(action) is not str or not 1 <= len(action) <= 64 for action in actions)):
            raise ValueError("actions must be a bounded tuple of text")
        if type(generation_seed) is not int or not 0 <= generation_seed < 2**31:
            raise ValueError("generation seed must be an independent bounded integer")
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or not .01 <= timeout_seconds <= 300):
            raise ValueError("timeout must be 0.01 to 300 seconds")
        if type(max_output_bytes) is not int or not 128 <= max_output_bytes <= 65536:
            raise ValueError("output byte limit must be 128 to 65536")
        if mode == "ollama":
            if not authorize_local_model:
                raise PermissionError("local model inference requires explicit authorization")
            # Configuration validated before a subprocess is ever launched.
            if type(model) is not str:
                raise ValueError("local model name is required")
            OllamaAdapter(model=model, base_url=base_url)
        elif model is not None:
            raise ValueError("model is only valid for ollama child mode")
        self.mode = mode
        self.actions = actions
        self.generation_seed = generation_seed
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self.model = model
        self.base_url = base_url
        self.model_id = (f"process:ollama:{model}" if mode == "ollama"
                         else f"process:{mode}:no-update-fixture")
        self._local_model_authorized = mode == "ollama" and authorize_local_model
        self._receipts: list[ProcessCallReceipt] = []
        self._sealed = True

    @property
    def receipts(self) -> tuple[ProcessCallReceipt, ...]:
        return tuple(self._receipts)

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        # SECURITY-INVARIANT: DO NOT FORWARD OR DERIVE FROM `seed`.
        # Older run_candidate/cross_exam callers passed a PRIVATE exam seed.
        if self.mode == "ollama" and not self._local_model_authorized:
            raise PermissionError("local model inference lacks constructor authorization")
        _trusted_prompt(prompt)
        request = _json({"schema": SCHEMA, "public_prompt": prompt,
                         "generation_seed": self.generation_seed, "mode": self.mode,
                         "actions": self.actions, "model": self.model,
                         "base_url": self.base_url})
        if len(request) > MAX_REQUEST_BYTES:
            raise ValueError("child request exceeds byte limit")
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        process: subprocess.Popen[bytes] | None = None
        response = b""
        child_pid: int | None = None
        frame_visible: bool | None = None
        status = "CHILD_ERROR"
        output_text: str | None = None
        try:
            module_root = str(Path(__file__).resolve().parent.parent)
            process = subprocess.Popen(
                [sys.executable, "-m", "labyrinth.minotaur_process_adapter", "--child"],
                shell=False, cwd=module_root, env=_child_environment(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                close_fds=True)
            child_pid = process.pid
            # communicate(capture_output=True) reads unbounded child stdout
            # into RAM BEFORE the caller can check size. Drain with a cap,
            # kill promptly on overflow, and use a separate bounded writer
            # so a child not reading stdin cannot defeat the wall timeout.
            output = bytearray()
            oversized = False

            def read_bounded() -> None:
                nonlocal oversized
                assert process is not None and process.stdout is not None
                while True:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    room = self.max_output_bytes - len(output)
                    output.extend(chunk[:max(0, room)])
                    if len(chunk) > room:
                        oversized = True
                        try:
                            process.kill()
                        except OSError:
                            pass
                        break

            def write_bounded() -> None:
                assert process is not None and process.stdin is not None
                try:
                    process.stdin.write(request)
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass

            reader = Thread(target=read_bounded, daemon=True)
            writer = Thread(target=write_bounded, daemon=True)
            reader.start()
            writer.start()
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                try:
                    process.kill()
                except OSError:
                    pass  # The child may have exited while the timeout fired.
                try:
                    # Reaping is best-effort; never make a second unbounded
                    # wait after a deadline has already expired.
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    # Retain a strong child reference until the OS reaps it;
                    # otherwise Popen.__del__ may warn or strand a zombie.
                    # This daemon is best-effort cleanup, NOT an OS sandbox or
                    # a guarantee that an unkillable child will terminate.
                    def reap_in_background(child: subprocess.Popen[bytes]) -> None:
                        try:
                            child.wait()
                        except OSError:
                            pass

                    Thread(target=reap_in_background, args=(process,),
                           name="minotaur-child-reaper", daemon=True).start()
                raise ProcessAdapterFailure(status)
            finally:
                # All cleanup consumes the SAME wall-clock budget. A child
                # may exit while a daemon pipe writer is still blocked.
                reader.join(timeout=max(0.0, deadline - time.monotonic()))
                writer.join(timeout=max(0.0, deadline - time.monotonic()))
                if process.stdout is not None:
                    process.stdout.close()
            response = bytes(output)
            if oversized:
                status = "OVERSIZE"
                raise ProcessAdapterFailure(status)
            if reader.is_alive() or writer.is_alive():
                status = "TIMEOUT"
                raise ProcessAdapterFailure(status)
            if process.returncode != 0:
                status = "CHILD_ERROR"
                raise ProcessAdapterFailure(status)
            try:
                result = _read_json(response)
                if (type(result) is not dict or set(result) !=
                        {"schema", "response", "child_pid", "parent_session_frame_visible"}
                        or result["schema"] != SCHEMA
                        or type(result["response"]) is not str
                        or type(result["child_pid"]) is not int
                        or result["child_pid"] != child_pid
                        or type(result["parent_session_frame_visible"]) is not bool):
                    raise ValueError("malformed child envelope")
                output_text = result["response"]
                if len(output_text.encode("utf-8")) > self.max_output_bytes:
                    raise ValueError("model text exceeds byte limit")
                frame_visible = result["parent_session_frame_visible"]
            except (ValueError, TypeError, UnicodeError, json.JSONDecodeError):
                status = "INVALID_JSON"
                raise ProcessAdapterFailure(status) from None
            if frame_visible:
                status = "PRIVATE_FRAME_EXPOSED"
                raise ProcessAdapterFailure(status)
            status = "OK"
            return output_text
        except ProcessAdapterFailure:
            raise
        except OSError as error:
            status = "CHILD_ERROR"
            raise ProcessAdapterFailure(status) from error
        finally:
            elapsed = round((time.monotonic() - started) * 1000)
            self._receipts.append(ProcessCallReceipt(
                schema=SCHEMA, status=status, child_pid=child_pid,
                generation_seed=self.generation_seed, evaluator_seed_discarded=True,
                request_sha256=_sha(request),
                response_sha256=_sha(response) if response else None,
                response_bytes=len(response), parent_session_frame_visible=frame_visible,
                elapsed_ms=elapsed))


def _parent_frame_probe() -> bool:
    """Child's actual caller frames, not an attestation of OS isolation."""
    frame = inspect.currentframe()
    while frame is not None:
        session = frame.f_locals.get("session")
        if session is not None and hasattr(session, "_case"):
            return True
        frame = frame.f_back
    return False


def _child_main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            return 2
        request = _read_json(raw)
        if (type(request) is not dict or set(request) !=
                {"schema", "public_prompt", "generation_seed", "mode",
                 "actions", "model", "base_url"}
                or request["schema"] != SCHEMA
                or request["mode"] not in ALLOWED_MODES
                or type(request["generation_seed"]) is not int
                or not 0 <= request["generation_seed"] < 2**31
                or type(request["public_prompt"]) is not str):
            return 2
        prompt = request["public_prompt"]
        _trusted_prompt(prompt)
        visible = _parent_frame_probe()
        mode = request["mode"]
        if mode == "breach":
            visible = True  # red-team protocol fixture; no private access implied
        if mode == "stall":
            time.sleep(3)
        if mode == "invalid_json":
            sys.stdout.write("this is intentionally malformed JSON")
            return 0
        if mode == "oversize":
            sys.stdout.write("X" * 100_000)
            return 0
        if mode == "fixed":
            actions = request["actions"]
            public = _read_json(prompt.encode("utf-8"))
            turn = public.get("turn", 0)
            if type(turn) is not int or turn < 0:
                return 2
            if turn < len(actions):
                text = _json({"kind": "ACTION", "action": actions[turn]}).decode("utf-8")
            else:
                text = _ABSTAIN
        elif mode == "ollama":
            # Use an already available LOCAL runtime, explicitly opted in by
            # the trusted parent. Never launch a server or access remote hosts.
            adapter = OllamaAdapter(model=request["model"], base_url=request["base_url"])
            text = adapter.generate(prompt, seed=request["generation_seed"])
        else:
            text = _ABSTAIN
        sys.stdout.buffer.write(_json({"schema": SCHEMA, "response": text,
                                       "child_pid": os.getpid(),
                                       "parent_session_frame_visible": visible}))
        return 0
    except (ValueError, TypeError, KeyError, UnicodeError, AdapterError):
        return 2


if __name__ == "__main__":
    if sys.argv[1:] != ["--child"]:
        sys.exit(2)
    sys.exit(_child_main())
