"""Bounded, public-observation Minotaur candidate versus control runner.

This is a development harness, not a private model examiner or training path.
Its injectable adapter uses the existing ModelAdapter.generate text interface;
no model is downloaded, trained, started or contacted at import time. A host
must construct the private examiner and supply only its public view. Python
callbacks and object inspection are NOT a security isolation boundary.

Receipts are write-once and independently replayable against a host-held case.
Their self-hashes detect accidental alteration, NOT malicious re-signing,
model identity, private-custody authority or actual neural execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping

from .adapters import ModelAdapter, OllamaAdapter, ReplayAdapter
from .runner import _source_state


SCHEMA = "labyrinth.minotaur-candidate-bridge.v1"
MAX_RESPONSE_CHARACTERS = 4096
MAX_PUBLIC_CHARACTERS = 16000
MAX_ACTIONS = 32
_PRIVATE_FIELDS = frozenset({"case_id", "seed", "sealed", "gold_trace", "fingerprint",
                             "answer", "correct_answer", "exam_id", "kind", "private"})
_INSTRUCTION = ("Return one JSON object only: either "
                'an ACTION object with action exactly one of the public action/question '
                'strings (not the generic word "query"), or '
                '{"kind":"CLAIM","claim":{"kind":"ABSTAIN","assessment":"UNKNOWN"}}. '
                "Read public_start.prompt and public action/question names for exact ACTION "
                 "names and CLAIM field shapes. No prose, tools, code or extra fields. "
                 "Invalid output ends this evaluation.")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_unicode_scalars(value: str) -> None:
    """Reject lone surrogates before text enters a UTF-8 receipt boundary."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("invalid Unicode scalar text") from exc


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, element in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = element
    return value


def _no_nan(_value: str) -> Any:
    raise ValueError("non-finite JSON number")


def _public(value: Mapping[str, Any], *, require_plain_containers: bool = False) -> str:
    # Check type identity BEFORE isinstance/Mapping iteration: programmable
    # __class__ and container methods are not part of a JSON data boundary.
    if require_plain_containers and type(value) is not dict:
        raise ValueError("process-only public view requires plain JSON containers")
    if not isinstance(value, Mapping):
        raise ValueError("public view must be a mapping")
    active_containers: set[int] = set()
    visited_nodes = 0
    visited_characters = 0

    def snapshot(node: Any, path: tuple[object, ...] = ()) -> Any:
        """Validate and copy once; never serialize an uninspected source object."""
        nonlocal visited_nodes, visited_characters
        visited_nodes += 1
        if len(path) > 32 or visited_nodes > 20_000:
            raise ValueError("public view nesting or node budget exceeded")
        if require_plain_containers:
            # Reject subclasses before invoking their iterator or mapping
            # methods in the examiner process.
            kind = type(node)
            if kind not in (dict, list, tuple, str, int, float, bool, type(None)):
                raise ValueError("process-only public view requires plain JSON containers")
            is_mapping = kind is dict
            is_sequence = kind in (list, tuple)
        else:
            is_mapping = isinstance(node, Mapping)
            is_sequence = isinstance(node, (list, tuple))
        if not is_mapping and not is_sequence:
            # A host-controlled scalar subclass may execute arbitrary methods
            # during json.dumps. Only inert JSON primitives cross this seam.
            if type(node) not in (str, int, float, bool, type(None)):
                raise ValueError("unsupported public value type")
            if type(node) is str:
                visited_characters += len(node)
                if visited_characters > MAX_PUBLIC_CHARACTERS:
                    raise ValueError("public view exceeds 16000-character limit")
            return node
        identity = id(node)
        if identity in active_containers:
            raise ValueError("public view cycle")
        active_containers.add(identity)
        try:
            if is_mapping:
                result: dict[str, Any] = {}
                for key, child in node.items():
                    if type(key) is not str:
                        raise ValueError("evaluator-only or malformed public field")
                    # Oracle Chamber's public answer transcript contains
                    # binary bits for valid queries and null for malformed
                    # queries that still consume budget. Only that exact
                    # answers[index].answer location may expose either value.
                    public_query_answer = (
                        key == "answer" and len(path) == 2
                        and path[0] == "answers" and type(path[1]) is int
                        and ((type(child) is int and child in (0, 1)) or child is None)
                    )
                    if key in _PRIVATE_FIELDS and not public_query_answer:
                        raise ValueError("evaluator-only or malformed public field")
                    if key in result:
                        raise ValueError("duplicate public field")
                    visited_characters += len(key)
                    if visited_characters > MAX_PUBLIC_CHARACTERS:
                        raise ValueError("public view exceeds 16000-character limit")
                    result[key] = snapshot(child, (*path, key))
                return result
            else:
                return [snapshot(child, (*path, index))
                        for index, child in enumerate(node)]
        finally:
            active_containers.remove(identity)
    raw = _json(snapshot(value))
    _require_unicode_scalars(raw)
    if len(raw) > MAX_PUBLIC_CHARACTERS:
        raise ValueError("public view exceeds 16000-character limit")
    return raw


def _parse(raw: str) -> tuple[str, Any]:
    if type(raw) is not str or len(raw) > MAX_RESPONSE_CHARACTERS:
        raise ValueError("oversized or non-text adapter response")
    item = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_no_nan)
    # Escaped JSON \uD800 is ASCII on the wire but decodes to a lone surrogate;
    # any nested string may later reach a hash or public receipt serializer.
    _require_unicode_scalars(_json(item))
    if not isinstance(item, dict) or type(item.get("kind")) is not str:
        raise ValueError("adapter response must specify a kind")
    if item["kind"] == "ACTION" and set(item) == {"kind", "action"}:
        action = item["action"]
        if type(action) is not str or not 1 <= len(action) <= 64:
            raise ValueError("invalid action type/length")
        return "ACTION", action
    if item["kind"] == "CLAIM" and set(item) == {"kind", "claim"}:
        claim = item["claim"]
        if not isinstance(claim, dict) or type(claim.get("kind")) is not str:
            raise ValueError("invalid claim structure")
        return "CLAIM", claim
    raise ValueError("unsupported adapter decision")


def _claim_for_host(claim: dict[str, Any]) -> dict[str, Any]:
    """Map JSON state arrays to exact tuple-state certificate input.

    A model can only emit JSON lists; the finite oracle explicitly requires
    immutable five-integer state tuples. Reject ambiguous transformations and
    keep the original raw JSON in the replayable candidate receipt.
    """
    if claim.get("kind") != "UNSAT_CERTIFICATE":
        return claim
    states = claim.get("states")
    if (type(states) is not list or len(states) > 10000 or
            any(type(row) is not list or len(row) != 5 or
                any(type(value) is not int for value in row) for row in states)):
        return claim
    return {**claim, "states": [tuple(row) for row in states]}


@dataclass(frozen=True)
class TurnReceipt:
    prompt_json: str
    response_text: str
    public_feedback_json: str | None
    disposition: str


@dataclass(frozen=True)
class RunReceipt:
    schema: str
    model_id_unattested: str
    adapter_provenance: str
    seed: int
    max_model_calls: int
    public_start_json: str
    public_binding: str
    source_commit_unattested: str | None
    source_tree_sha256: str
    turns: tuple[TurnReceipt, ...]
    final_verdict: str
    verified_escape: bool
    receipt_sha256: str


@dataclass(frozen=True)
class PairedReceipt:
    candidate: RunReceipt
    control: RunReceipt
    candidate_win: bool
    control_win: bool
    neural_training_claim: bool = False
    session_isolation: str = "IN_PROCESS_UNATTESTED"
    comparison_qualified: bool = False


class FixedActionAdapter:
    """Frozen zero-update script: source actions are NOT learned or inferred."""

    def __init__(self, actions: tuple[str, ...], *, model_id: str = "fixed:zero-update"):
        if type(actions) is not tuple or any(type(a) is not str for a in actions):
            raise ValueError("fixed script must be a tuple of strings")
        if type(model_id) is not str or not model_id.strip():
            raise ValueError("model_id must be nonempty text")
        self._actions = actions
        self._position = 0
        self.model_id = model_id

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        # Consume the same public prompt interface as an optional local model.
        if type(prompt) is not str or len(prompt) > 64000:
            raise ValueError("invalid public prompt")
        if self._position < len(self._actions):
            action = self._actions[self._position]
            self._position += 1
            return _json({"kind": "ACTION", "action": action})
        return _json({"kind": "CLAIM", "claim": {"kind": "ABSTAIN", "assessment": "UNKNOWN"}})


def _read_stdout_posix(pipe: Any, *, cap: int, deadline: float) -> bytes:
    """Poll one candidate pipe without an EOF-blocked reader thread.

    A detached grandchild may inherit stdout indefinitely; readiness polling
    lets the examiner enforce its deadline and close its own file descriptor.
    This does not kill or contain the detached process.
    """
    descriptor = pipe.fileno()
    fragments: list[bytes] = []
    remaining_bytes = cap + 1
    while remaining_bytes:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("candidate output deadline exceeded")
        ready, _, _ = select.select([descriptor], [], [], remaining_seconds)
        if not ready:
            raise TimeoutError("candidate output deadline exceeded")
        data = os.read(descriptor, remaining_bytes)
        if not data:
            break
        fragments.append(data)
        remaining_bytes -= len(data)
    return b"".join(fragments)


def _read_stdout_windows(pipe: Any, *, cap: int, deadline: float) -> bytes:
    """Poll an anonymous Windows pipe without an EOF-blocked reader thread.

    A surviving child can inherit stdout; PeekNamedPipe reports readable bytes
    without waiting for that child to exit. This bounds examiner waiting, not
    process-tree containment or child access to same-user host resources.
    """
    import ctypes
    from ctypes import wintypes
    import msvcrt

    peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
    peek.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                     ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                     ctypes.POINTER(wintypes.DWORD))
    peek.restype = wintypes.BOOL
    descriptor = pipe.fileno()
    handle = msvcrt.get_osfhandle(descriptor)
    fragments: list[bytes] = []
    remaining_bytes = cap + 1
    available = wintypes.DWORD()
    while remaining_bytes:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("candidate output deadline exceeded")
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error in (109, 232):  # ERROR_BROKEN_PIPE or ERROR_NO_DATA
                break
            raise OSError(error, "candidate output pipe peek failed")
        if available.value:
            data = os.read(descriptor, min(remaining_bytes, available.value))
            if not data:
                break
            fragments.append(data)
            remaining_bytes -= len(data)
        else:
            time.sleep(min(0.01, remaining_seconds))
    return b"".join(fragments)


@contextmanager
def _candidate_request_directory():
    """Retry transient Windows sharing violations after job-wide termination.

    Job termination may release child-inherited stdin handles asynchronously.
    Never ignore a locked file or leave a public request on disk silently.
    """
    folder = tempfile.TemporaryDirectory(prefix="labyrinth_candidate_")
    try:
        yield folder.name
    finally:
        deadline = time.monotonic() + (0.75 if os.name == "nt" else 0)
        while True:
            try:
                folder.cleanup()
                break
            except PermissionError as exc:
                if (os.name != "nt" or getattr(exc, "winerror", None) != 32
                        or time.monotonic() >= deadline):
                    raise
                time.sleep(0.02)


class SubprocessCandidateAdapter:
    """Optional public-JSON child protocol, NOT an OS access-control boundary.

    One child is spawned per model call; prompt includes all public history.
    The child receives a JSON object with prompt and seed on stdin and must
    return one bridge decision JSON object on stdout. No shell or inherited
    private environment is used. Same-user processes can still read known
    examiner-owned file paths: independent custody is NOT established.
    """

    __slots__ = ("command", "model_id", "timeout_seconds", "unix_user_id", "unix_group_id")

    def __init__(self, command: tuple[str, ...], *, model_id: str,
                 timeout_seconds: float = 10.0, unix_user_id: int | None = None,
                 unix_group_id: int | None = None):
        if (type(command) is not tuple or not 1 <= len(command) <= 16 or
                any(type(arg) is not str or not arg or len(arg) > 4096 for arg in command)
                or not os.path.isabs(command[0])):
            raise ValueError("candidate command requires a bounded tuple and absolute executable")
        if (type(timeout_seconds) not in (float, int)
                or not 0.1 <= timeout_seconds <= 60):
            raise ValueError("candidate timeout must be between 0.1 and 60 seconds")
        if (unix_user_id is None) != (unix_group_id is None):
            raise ValueError("POSIX candidate identity requires both user and group")
        if unix_user_id is not None:
            if (os.name != "posix" or type(unix_user_id) is not int
                    or type(unix_group_id) is not int
                    or not 1 <= unix_user_id < 2**31
                    or not 1 <= unix_group_id < 2**31):
                raise ValueError("candidate identity requires non-root POSIX UID and GID")
            if os.geteuid() != 0:
                raise PermissionError("POSIX candidate identity drop requires root examiner")
            if "microsoft" in os.uname().release.lower():
                raise PermissionError("WSL interop bypasses candidate Linux UID isolation")
        self.command = command
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.unix_user_id = unix_user_id
        self.unix_group_id = unix_group_id

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        if type(prompt) is not str or len(prompt) > 64000:
            raise ValueError("invalid subprocess prompt")
        _require_unicode_scalars(prompt)
        request = (_json({"prompt": json.loads(prompt), "seed": seed}) + "\n").encode("utf-8")
        # Only basic Windows loader variables: neither credentials nor Python
        # import-path overrides are copied from the examiner environment.
        environment = {key: os.environ[key] for key in ("SystemRoot", "WINDIR")
                       if key in os.environ}
        launch_identity = {"start_new_session": True} if os.name == "posix" else {}
        if self.unix_user_id is not None:
            if (os.name != "posix" or os.geteuid() != 0
                    or "microsoft" in os.uname().release.lower()):
                raise PermissionError("candidate UID drop no longer authorized")
            # Clear supplementary groups before dropping effective UID/GID.
            launch_identity.update({"user": self.unix_user_id,
                                    "group": self.unix_group_id, "extra_groups": ()})
        # Feed stdin from an on-disk PUBLIC request so neither side can
        # deadlock writing an input pipe while the child fills stdout.
        # Both platforms poll stdout. Windows starts candidate code only
        # after attaching an event-gated bootstrap to its Job Object. The
        # same-user OS principal and independent examiner custody remain open.
        with _candidate_request_directory() as directory:
            request_file = Path(directory) / "public_request.json"
            request_file.write_bytes(request)
            with request_file.open("rb") as source:
                candidate_job = None
                if os.name == "nt":
                    # Trusted bootstrap waits on an inherited unsignaled event.
                    # Only attach-to-job success releases candidate execution;
                    # this remains SAME-USER execution, NOT a file sandbox.
                    from ._windows_gated_launch import start_candidate_inside_job
                    child, candidate_job = start_candidate_inside_job(
                        self.command, source, environment, tempfile.gettempdir())
                else:
                    child = subprocess.Popen(
                        self.command, stdin=source, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, cwd=directory,
                        env=environment, shell=False, **launch_identity)
                cap = MAX_RESPONSE_CHARACTERS * 4
                deadline = time.monotonic() + self.timeout_seconds
                try:
                    if os.name == "posix":
                        raw = _read_stdout_posix(child.stdout, cap=cap, deadline=deadline)
                    else:
                        raw = _read_stdout_windows(child.stdout, cap=cap, deadline=deadline)
                    if len(raw) > cap:
                        return "X" * (MAX_RESPONSE_CHARACTERS + 1)
                    child.wait(timeout=max(0.01, deadline - time.monotonic()))
                    if child.returncode:
                        raise RuntimeError("candidate process exited unsuccessfully")
                    return raw.decode("utf-8", errors="strict").strip()
                finally:
                    # POSIX terminates the initial process group; Windows
                    # closes its candidate Job Object, killing descendants
                    # actually attached after launch. Neither is a sandbox.
                    if os.name == "posix":
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        try:
                            if candidate_job is not None:
                                from ._windows_candidate_job import close_kill_job
                                close_kill_job(candidate_job)
                        finally:
                            if child.poll() is None:
                                child.kill()
                    child.wait()
                    child.stdout.close()


# Pinned host-owned method: class monkeypatching must not silently turn
# process-only mode back into an attacker-supplied in-process callback.
_SUBPROCESS_GENERATE = SubprocessCandidateAdapter.generate


def _request(public_start_json: str, history: list[dict[str, Any]], turn: int) -> str:
    return _json({"public_start": json.loads(public_start_json), "history": history,
                  "turn": turn, "response_instructions": _INSTRUCTION})


def _terminal(feedback: Mapping[str, Any]) -> bool:
    if feedback.get("finished") is True:
        return True
    return feedback.get("status") in ("VERIFIED_WIN", "UNSOLVED", "INVALID_TRACE",
                                       "VERIFIED_FAILURE", "EVALUATOR_DISAGREEMENT",
                                       "EVALUATOR_ERROR", "EXHAUSTED", "REPLAY_DIVERGENCE",
                                       "REACHED_EXIT_UNCERTIFIED")


def _session_view(session: Any, *, first: bool = False) -> Mapping[str, Any]:
    if hasattr(session, "view"):
        return session.view()
    if first and hasattr(session, "start"):
        return session.start()
    raise ValueError("examiner has no supported feedback view")


def _observation(value: Mapping[str, Any]) -> Any:
    return value.get("observation", value.get("initial_observation"))


def _verdict(feedback: Mapping[str, Any]) -> str:
    result = feedback.get("verdict", feedback.get("status", "UNKNOWN"))
    return result if type(result) is str else "UNKNOWN"


def _seal(**fields: Any) -> RunReceipt:
    digest = _sha(_json({**fields, "turns": [asdict(t) for t in fields["turns"]]}))
    return RunReceipt(**fields, receipt_sha256=digest)


def _validated_budget(public: Mapping[str, Any], limit: int) -> int:
    if type(limit) is not int or not 1 <= limit <= MAX_ACTIONS:
        raise ValueError("max_model_calls must be an integer from 1 to 32")
    candidate = public.get("remaining_actions", public.get("actions_remaining",
                           public.get("remaining_queries",
                           public.get("maximum_actions", limit))))
    if type(candidate) is not int or not 1 <= candidate <= MAX_ACTIONS:
        raise ValueError("public action budget must be a bounded integer")
    # The task's charged-action allowance and the model's call allowance are
    # separate: after the final question, one terminal claim may be required.
    return candidate


def _bind_public_start(session: Any, supplied_json: str, initial: Mapping[str, Any],
                       *, require_plain_containers: bool = False) -> str:
    """Cross-check the advertised world against its host-owned public rules.

    Observation-only legacy examiners are explicitly unqualified as a full
    public/private rule binding, even when they offer a nonempty observation.
    """
    source = getattr(session, "public_start", None)
    if callable(source):
        if _public(source(), require_plain_containers=require_plain_containers) != supplied_json:
            raise ValueError("candidate public rules do not match the host examiner")
        return "HOST_PUBLIC_VIEW_MATCHED"
    supplied = json.loads(supplied_json)
    if _observation(initial) != _observation(supplied):
        raise ValueError("candidate public observation does not match private session")
    return "OBSERVATION_ONLY_UNATTESTED"


def _apply_action(session: Any, action: str) -> Mapping[str, Any]:
    """Support action environments and bounded binary-query environments."""
    if hasattr(session, "act"):
        return session.act(action)
    if hasattr(session, "ask"):
        return session.ask(action)
    raise ValueError("examiner has no declared action/query method")


def _is_verified_escape(verdict: str, feedback: Mapping[str, Any]) -> bool:
    """A generic VERIFIED_WIN can describe a *policy*, not an escape.

    Credit only an examiner's explicit independently checked escape witness.
    A legacy status-only interface is intentionally insufficient evidence.
    """
    return verdict == "VERIFIED_WIN" and feedback.get("escape_verified") is True


def run_candidate(public_start: Mapping[str, Any], session_factory: Callable[[], Any],
                  adapter: ModelAdapter, *, seed: int = 0,
                  authorize_local_adapter: bool = False,
                  max_model_calls: int = 32,
                  require_process_separation: bool = False) -> RunReceipt:
    """Evaluate one candidate through public JSON only; private session stays host-side.

    Passing an arbitrary adapter explicitly requires caller authorization. The
    boolean does NOT authenticate actual locality/model bytes or grant any GPU
    training permission. Only our fixed and existing replay fixtures are
    automatically classified as deterministic no-update controls.
    """
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid evaluation seed")
    # Validate adapter authority before traversing even public caller objects.
    # Custom mappings may execute Python code through items()/iteration.
    # Subclasses may override generate() and perform arbitrary, even remote,
    # effects. Only the two exact frozen control implementations are trusted.
    fixed = type(adapter) in (FixedActionAdapter, ReplayAdapter)
    separated = type(adapter) is SubprocessCandidateAdapter
    if type(require_process_separation) is not bool:
        raise ValueError("require_process_separation must be a boolean")
    if require_process_separation and not separated:
        raise PermissionError("process-only examination refuses in-process adapters")
    if not fixed and not authorize_local_adapter:
        raise PermissionError("non-control model adapter requires explicit local authorization")
    if isinstance(adapter, OllamaAdapter) and adapter.allow_remote:
        raise PermissionError("remote Ollama endpoints are not admitted by this bridge")
    model_id = adapter.model_id
    if type(model_id) is not str or not 1 <= len(model_id) <= 256:
        raise ValueError("unattested model ID must be bounded text")
    _require_unicode_scalars(model_id)
    # The caller's authorization flag cannot attest an arbitrary adapter's
    # locality, inference weights, provider, or independence from cloud APIs.
    provenance = ("DETERMINISTIC_CONTROL" if fixed else
                  "MODEL_SUBPROCESS_UNATTESTED" if separated else "MODEL_ADAPTER_UNATTESTED")
    # A subprocess-labeled run always rejects programmable host containers,
    # whether or not the optional process-only rejection flag was supplied.
    strict_plain = separated
    public_json = _public(public_start, require_plain_containers=strict_plain)
    public_snapshot = json.loads(public_json)
    budget = _validated_budget(public_snapshot, max_model_calls)
    call_limit = min(max_model_calls, budget + 1)
    session = session_factory()
    initial = json.loads(_public(_session_view(session, first=True),
                                 require_plain_containers=strict_plain))
    last_feedback = initial
    binding = _bind_public_start(session, public_json, initial,
                                 require_plain_containers=strict_plain)
    history: list[dict[str, Any]] = []
    turns: list[TurnReceipt] = []
    final = _verdict(initial)
    charged_actions = 0
    for index in range(call_limit):
        if _terminal(initial):
            break
        request = _request(public_json, history, index)
        try:
            response = (_SUBPROCESS_GENERATE(adapter, request, seed=seed)
                        if separated else adapter.generate(request, seed=seed))
        except Exception:
            turns.append(TurnReceipt(request, "", None, "MODEL_CALL_ERROR"))
            final = "MODEL_CALL_ERROR"
            break
        try:
            decision, content = _parse(response)
        except (ValueError, TypeError, json.JSONDecodeError, RecursionError):
            # Never call __str__ on a malformed adapter object: its conversion
            # can execute arbitrary code, throw, or allocate unbounded text.
            # The adapter boundary admits only an exact string as evidence.
            # Keep one overflow character: 4096 may be a parseable prefix,
            # while 4097 alone proves the original response exceeded the cap.
            safe_response = response[:MAX_RESPONSE_CHARACTERS + 1] if type(response) is str else ""
            try:
                _require_unicode_scalars(safe_response)
            except ValueError:
                safe_response = ""
            turns.append(TurnReceipt(request, safe_response,
                                     None, "MODEL_OUTPUT_INVALID"))
            final = "MODEL_OUTPUT_INVALID"
            break
        if decision == "ACTION" and charged_actions >= budget:
            # Do not grant a sixth question; accept a CLAIM only. Record the
            # attempted policy violation without touching the examiner.
            turns.append(TurnReceipt(request, response, None, "ACTION_BUDGET_EXCEEDED"))
            final = "ACTION_BUDGET_EXCEEDED"
            break
        try:
            if decision == "ACTION":
                feedback = _apply_action(session, content)
                charged_actions += 1
            else:
                feedback = session.submit(_claim_for_host(content))
            feedback_json = _public(feedback,
                                    require_plain_containers=strict_plain)
            feedback = json.loads(feedback_json)
        except Exception:
            turns.append(TurnReceipt(request, response, None, "EVALUATOR_ERROR"))
            final = "EVALUATOR_ERROR"
            break
        turns.append(TurnReceipt(request, response, feedback_json, decision))
        last_feedback = feedback
        history.append({"decision": json.loads(response), "feedback": json.loads(feedback_json)})
        final = _verdict(feedback)
        # The action/claim feedback is the *recorded* source of truth. A second
        # view() call here could contradict it, granting an unrecorded win by
        # combining its verdict with feedback's escape witness.
        initial = feedback
        if decision == "CLAIM":
            break
    if not turns and _terminal(initial):
        final = _verdict(initial)
    elif turns and turns[-1].disposition in ("ACTION", "CLAIM"):
        final = _verdict(initial)
    source = _source_state()
    return _seal(schema=SCHEMA, model_id_unattested=model_id,
                 adapter_provenance=provenance, seed=seed,
                 max_model_calls=max_model_calls,
                 public_start_json=public_json, public_binding=binding,
                 source_commit_unattested=source["commit"],
                 source_tree_sha256=source["source_tree_sha256"],
                 turns=tuple(turns), final_verdict=final,
                 verified_escape=(bool(turns) and turns[-1].disposition == "ACTION"
                                   and _is_verified_escape(final, last_feedback)))


def _inert_receipt_shape(receipt: Any) -> bool:
    """Reject programmable receipt objects before dataclass serialization.

    Exact built-in field types prevent user-defined __getattribute__ and
    __deepcopy__ callbacks. This is a local object safety check, NOT a
    signer, provenance, or execution attestation.
    """
    if type(receipt) is not RunReceipt:
        return False
    for name in ("schema", "model_id_unattested", "adapter_provenance",
                 "public_start_json", "public_binding", "source_tree_sha256",
                 "final_verdict", "receipt_sha256"):
        if type(getattr(receipt, name)) is not str:
            return False
    if (type(receipt.seed) is not int or type(receipt.max_model_calls) is not int
            or type(receipt.verified_escape) is not bool
            or (receipt.source_commit_unattested is not None
                and type(receipt.source_commit_unattested) is not str)
            or type(receipt.turns) is not tuple
            or len(receipt.turns) > MAX_ACTIONS + 1):
        return False
    return all(
        type(turn) is TurnReceipt
        and type(turn.prompt_json) is str
        and type(turn.response_text) is str
        and (turn.public_feedback_json is None
             or type(turn.public_feedback_json) is str)
        and type(turn.disposition) is str
        for turn in receipt.turns
    )


def audit_receipt(receipt: RunReceipt, public_start: Mapping[str, Any],
                  session_factory: Callable[[], Any]) -> bool:
    """Replay observed actions against a fresh host-held case, not model identity.

    Successful replay validates conditional examiner effects, not that a real
    model produced the response bytes or that self-hashed data are authentic.
    """
    try:
        if not _inert_receipt_shape(receipt) or receipt.schema != SCHEMA:
            return False
        if (receipt.adapter_provenance not in (
                "DETERMINISTIC_CONTROL", "MODEL_SUBPROCESS_UNATTESTED",
                "MODEL_ADAPTER_UNATTESTED")
                or not 1 <= len(receipt.model_id_unattested) <= 256):
            # The runner cannot issue invented provenance labels or model IDs.
            # Allowed labels remain self-asserted, never authenticated.
            return False
        fields = asdict(receipt)
        claimed = fields.pop("receipt_sha256")
        # Replay may only enforce the provenance claimed by a self-hashed
        # receipt; this is not independently authenticated process identity.
        strict_plain = receipt.adapter_provenance == "MODEL_SUBPROCESS_UNATTESTED"
        if (claimed != _sha(_json(fields))
                or receipt.public_start_json != _public(
                    public_start, require_plain_containers=strict_plain)):
            return False
        if receipt.source_tree_sha256 != _source_state()["source_tree_sha256"]:
            return False
        if (type(receipt.seed) is not int or not 0 <= receipt.seed < 2**31
                or type(receipt.max_model_calls) is not int
                or not 1 <= receipt.max_model_calls <= MAX_ACTIONS):
            # Replay must reject runner-impossible metadata before examiner use.
            return False
        session = session_factory()
        initial = json.loads(_public(_session_view(session, first=True),
                                     require_plain_containers=strict_plain))
        last_feedback = initial
        if receipt.public_binding != _bind_public_start(
                session, receipt.public_start_json, initial,
                require_plain_containers=strict_plain):
            return False
        if not receipt.turns and not _terminal(initial):
            return False
        budget = _validated_budget(json.loads(receipt.public_start_json), MAX_ACTIONS)
        call_limit = min(receipt.max_model_calls, budget + 1)
        charged_actions = 0
        history: list[dict[str, Any]] = []
        last = _verdict(initial)
        for index, turn in enumerate(receipt.turns):
            if index >= call_limit or _terminal(initial) or turn.prompt_json != _request(receipt.public_start_json, history, index):
                return False
            if turn.disposition in ("EVALUATOR_ERROR", "MODEL_CALL_ERROR"):
                # Exception claims cannot be verified by replaying nothing.
                # Without a replayable evaluator or model outcome, fail closed.
                return False
            if turn.disposition == "MODEL_OUTPUT_INVALID":
                # The runner records at most one overflow character (4097)
                # even when the original malformed output is arbitrarily long.
                if (index != len(receipt.turns) - 1
                        or turn.public_feedback_json is not None
                        or len(turn.response_text) > MAX_RESPONSE_CHARACTERS + 1):
                    return False
                try:
                    _parse(turn.response_text)
                except (ValueError, TypeError, json.JSONDecodeError, RecursionError):
                    pass
                else:
                    return False
                return receipt.final_verdict == turn.disposition and not receipt.verified_escape
            decision, content = _parse(turn.response_text)
            if turn.disposition == "ACTION_BUDGET_EXCEEDED":
                return (index == len(receipt.turns) - 1
                        and turn.public_feedback_json is None
                        and decision == "ACTION" and charged_actions >= budget
                        and receipt.final_verdict == "ACTION_BUDGET_EXCEEDED"
                        and not receipt.verified_escape)
            if decision != turn.disposition:
                return False
            if decision == "ACTION":
                if charged_actions >= budget:
                    return False
                charged_actions += 1
            feedback = (_apply_action(session, content) if decision == "ACTION"
                        else session.submit(_claim_for_host(content)))
            feedback_json = _public(feedback, require_plain_containers=strict_plain)
            if feedback_json != turn.public_feedback_json:
                return False
            feedback = json.loads(feedback_json)
            history.append({"decision": json.loads(turn.response_text),
                            "feedback": json.loads(feedback_json)})
            last_feedback = feedback
            initial = feedback
            last = _verdict(initial)
            if decision == "CLAIM" and index < len(receipt.turns) - 1:
                return False
        if (receipt.turns and receipt.turns[-1].disposition == "ACTION"
                and not _terminal(initial) and len(receipt.turns) < call_limit):
            # Runner would request another model response on this active case.
            return False
        return (receipt.final_verdict == last
                and receipt.verified_escape == (bool(receipt.turns)
                                                and receipt.turns[-1].disposition == "ACTION"
                                                and _is_verified_escape(last, last_feedback)))
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RecursionError):
        return False


def run_matched_pair(public_start: Mapping[str, Any], session_factory: Callable[[], Any],
                     *, candidate: ModelAdapter, control: ModelAdapter, seed: int = 0,
                     authorize_local_adapter: bool = False) -> PairedReceipt:
    """Matched input/seed, independent fresh game sessions; no training claim."""
    if candidate is control:
        raise ValueError("candidate and control require independent adapter instances")
    candidate_session = session_factory()
    control_session = session_factory()
    if candidate_session is control_session:
        raise ValueError("candidate and control require independent examiner sessions")
    if not hasattr(control_session, "view"):
        raise ValueError("paired control requires a readback-capable examiner")
    # Detect aliasing through separate proxy objects in the common case:
    # the control must remain unchanged while the candidate runs. This is
    # only a source-level consistency test, not proof of process isolation.
    control_before = _public(control_session.view())
    candidate_run = run_candidate(public_start, lambda: candidate_session, candidate, seed=seed,
                                  authorize_local_adapter=authorize_local_adapter)
    if _public(control_session.view()) != control_before:
        raise ValueError("control examiner changed during candidate evaluation")
    control_run = run_candidate(public_start, lambda: control_session, control, seed=seed,
                                authorize_local_adapter=authorize_local_adapter)
    return PairedReceipt(candidate_run, control_run,
                         candidate_run.verified_escape, control_run.verified_escape)


def save_receipt_new(path: str | Path, receipt: RunReceipt) -> None:
    """Create a receipt file exclusively; never overwrite existing evidence."""
    if not _inert_receipt_shape(receipt):
        raise TypeError("only an inert RunReceipt may be persisted")
    target = Path(path)
    if target.is_symlink():
        raise FileExistsError("receipt destination is a symlink")
    fields = asdict(receipt)
    claimed = fields.pop("receipt_sha256")
    if claimed != _sha(_json(fields)):
        raise ValueError("receipt digest does not match serialized fields")
    fields["receipt_sha256"] = claimed
    encoded = (_json(fields) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())
