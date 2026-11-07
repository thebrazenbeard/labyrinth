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

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
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
                '{"kind":"ACTION","action":"one action or query from the public list"} or '
                '{"kind":"CLAIM","claim":{"kind":"ABSTAIN","assessment":"UNKNOWN"}}. '
                "No prose, tools, code or extra fields. Invalid output ends this evaluation.")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, element in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = element
    return value


def _no_nan(_value: str) -> Any:
    raise ValueError("non-finite JSON number")


def _public(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("public view must be a mapping")
    def inspect(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if type(key) is not str or key in _PRIVATE_FIELDS:
                    raise ValueError("evaluator-only or malformed public field")
                inspect(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                inspect(child)
    inspect(value)
    raw = _json(dict(value))
    if len(raw) > MAX_PUBLIC_CHARACTERS:
        raise ValueError("public view exceeds 16000-character limit")
    return raw


def _parse(raw: str) -> tuple[str, Any]:
    if type(raw) is not str or len(raw) > MAX_RESPONSE_CHARACTERS:
        raise ValueError("oversized or non-text adapter response")
    item = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_no_nan)
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
    public_start_json: str
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
    return min(candidate, limit)


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
                  max_model_calls: int = 32) -> RunReceipt:
    """Evaluate one candidate through public JSON only; private session stays host-side.

    Passing an arbitrary adapter explicitly requires caller authorization. The
    boolean does NOT authenticate actual locality/model bytes or grant any GPU
    training permission. Only our fixed and existing replay fixtures are
    automatically classified as deterministic no-update controls.
    """
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid evaluation seed")
    public_json = _public(public_start)
    budget = _validated_budget(public_start, max_model_calls)
    # Subclasses may override generate() and perform arbitrary, even remote,
    # effects. Only the two exact frozen control implementations are trusted.
    fixed = type(adapter) in (FixedActionAdapter, ReplayAdapter)
    if not fixed and not authorize_local_adapter:
        raise PermissionError("non-control model adapter requires explicit local authorization")
    if isinstance(adapter, OllamaAdapter) and adapter.allow_remote:
        raise PermissionError("remote Ollama endpoints are not admitted by this bridge")
    model_id = adapter.model_id
    if type(model_id) is not str or not 1 <= len(model_id) <= 256:
        raise ValueError("unattested model ID must be bounded text")
    # The caller's authorization flag cannot attest an arbitrary adapter's
    # locality, inference weights, provider, or independence from cloud APIs.
    provenance = "DETERMINISTIC_CONTROL" if fixed else "MODEL_ADAPTER_UNATTESTED"
    session = session_factory()
    initial = _session_view(session, first=True)
    last_feedback = initial
    if _observation(initial) != _observation(public_start):
        raise ValueError("candidate public view does not match private session observation")
    history: list[dict[str, Any]] = []
    turns: list[TurnReceipt] = []
    final = _verdict(initial)
    for index in range(budget):
        if _terminal(initial):
            break
        request = _request(public_json, history, index)
        try:
            response = adapter.generate(request, seed=seed)
        except Exception:
            turns.append(TurnReceipt(request, "", None, "MODEL_CALL_ERROR"))
            final = "MODEL_CALL_ERROR"
            break
        try:
            decision, content = _parse(response)
        except (ValueError, TypeError, json.JSONDecodeError, RecursionError):
            turns.append(TurnReceipt(request, str(response)[:MAX_RESPONSE_CHARACTERS],
                                     None, "MODEL_OUTPUT_INVALID"))
            final = "MODEL_OUTPUT_INVALID"
            break
        try:
            if decision == "ACTION":
                feedback = _apply_action(session, content)
            else:
                feedback = session.submit(_claim_for_host(content))
            feedback_json = _public(feedback)
        except Exception:
            turns.append(TurnReceipt(request, response, None, "EVALUATOR_ERROR"))
            final = "EVALUATOR_ERROR"
            break
        turns.append(TurnReceipt(request, response, feedback_json, decision))
        last_feedback = feedback
        history.append({"decision": json.loads(response), "feedback": json.loads(feedback_json)})
        final = _verdict(feedback)
        initial = (session.view() if hasattr(session, "view") else feedback)
        if decision == "CLAIM":
            break
    if not turns and _terminal(initial):
        final = _verdict(initial)
    elif turns and turns[-1].disposition in ("ACTION", "CLAIM"):
        final = _verdict(initial)
    source = _source_state()
    return _seal(schema=SCHEMA, model_id_unattested=model_id,
                 adapter_provenance=provenance, seed=seed,
                 public_start_json=public_json, source_commit_unattested=source["commit"],
                 source_tree_sha256=source["source_tree_sha256"],
                 turns=tuple(turns), final_verdict=final,
                 verified_escape=_is_verified_escape(final, last_feedback))


def audit_receipt(receipt: RunReceipt, public_start: Mapping[str, Any],
                  session_factory: Callable[[], Any]) -> bool:
    """Replay observed actions against a fresh host-held case, not model identity.

    Successful replay validates conditional examiner effects, not that a real
    model produced the response bytes or that self-hashed data are authentic.
    """
    try:
        if not isinstance(receipt, RunReceipt) or receipt.schema != SCHEMA:
            return False
        fields = asdict(receipt)
        claimed = fields.pop("receipt_sha256")
        if claimed != _sha(_json(fields)) or receipt.public_start_json != _public(public_start):
            return False
        if receipt.source_tree_sha256 != _source_state()["source_tree_sha256"]:
            return False
        session = session_factory()
        initial = _session_view(session, first=True)
        last_feedback = initial
        if _observation(initial) != _observation(public_start):
            return False
        if not receipt.turns and not _terminal(initial):
            return False
        budget = _validated_budget(public_start, MAX_ACTIONS)
        history: list[dict[str, Any]] = []
        last = _verdict(initial)
        for index, turn in enumerate(receipt.turns):
            if index >= budget or _terminal(initial) or turn.prompt_json != _request(receipt.public_start_json, history, index):
                return False
            if turn.disposition in ("MODEL_CALL_ERROR", "MODEL_OUTPUT_INVALID", "EVALUATOR_ERROR"):
                if index != len(receipt.turns) - 1 or turn.public_feedback_json is not None:
                    return False
                if turn.disposition == "MODEL_OUTPUT_INVALID":
                    try:
                        _parse(turn.response_text)
                    except (ValueError, TypeError, json.JSONDecodeError, RecursionError):
                        pass
                    else:
                        return False
                return receipt.final_verdict == turn.disposition and not receipt.verified_escape
            decision, content = _parse(turn.response_text)
            if decision != turn.disposition:
                return False
            feedback = (_apply_action(session, content) if decision == "ACTION"
                        else session.submit(_claim_for_host(content)))
            feedback_json = _public(feedback)
            if feedback_json != turn.public_feedback_json:
                return False
            history.append({"decision": json.loads(turn.response_text),
                            "feedback": json.loads(feedback_json)})
            last_feedback = feedback
            initial = (session.view() if hasattr(session, "view") else feedback)
            last = _verdict(initial)
            if decision == "CLAIM" and index < len(receipt.turns) - 1:
                return False
        return (receipt.final_verdict == last
                and receipt.verified_escape == _is_verified_escape(last, last_feedback))
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
    if not isinstance(receipt, RunReceipt):
        raise TypeError("only a RunReceipt may be persisted")
    target = Path(path)
    if target.is_symlink():
        raise FileExistsError("receipt destination is a symlink")
    encoded = (_json(asdict(receipt)) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())
