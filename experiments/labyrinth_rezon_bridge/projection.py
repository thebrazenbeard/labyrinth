"""Source-only projection of Rezon collective text into Labyrinth review input.

This experimental adapter consumes the *serialized* shape of Rezon's
``CollectiveSnapshot.to_dict``. It intentionally does not import Rezon, execute
remote workers, update model weights, or grant Minotaur graduation authority.
The caller's source pins are NOT an authenticated custody mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any


class RezonProjectionError(ValueError):
    """The provided source projection contradicts its declared exact pins."""


_HEAD = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_AGENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z", re.ASCII)
_TOP = frozenset({"episode_id", "prompt", "agents", "answers", "reviews",
                 "syntheses", "missing_agents", "complete", "input_digest"})
_CONTRIBUTION = frozenset({"id", "agent_id", "kind", "content",
                           "source_refs", "target_ids"})
_SYNTHESIS = frozenset({"synthesis_id", "integrator_id", "input_digest",
                        "content", "status", "is_current"})


@dataclass(frozen=True)
class ReviewProjection:
    rezon_head: str
    episode_id: str
    source_input_digest: str
    advisory_text: tuple[str, ...]
    advisory_sha256: str
    evidence_class: str = "SOURCE_ONLY_UNVERIFIED_TEXT"
    neural_weight_update_verified: bool = False
    distributed_tensor_execution_verified: bool = False
    can_authorize_training_or_graduation: bool = False


def _exact_dict(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if (type(value) is not dict or any(type(key) is not str for key in value)
            or set(value) != fields):
        raise RezonProjectionError(f"invalid {label} shape")
    return value


def _exact_str(value: Any, label: str, *, max_len: int = 32768) -> str:
    if type(value) is not str or not value or len(value) > max_len:
        raise RezonProjectionError(f"invalid {label}")
    if label in ("prompt", "contribution content", "synthesis content") and not value.strip():
        raise RezonProjectionError(f"blank {label}")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise RezonProjectionError(f"invalid {label} encoding") from exc
    return value


def _sequence(value: Any, label: str, maximum: int = 32) -> list[Any]:
    if type(value) is not list or len(value) > maximum:
        raise RezonProjectionError(f"invalid {label}")
    return value


def _sha(value: Any, label: str, *, git_head: bool = False) -> str:
    if type(value) is not str or (_HEAD if git_head else _DIGEST).fullmatch(value) is None:
        raise RezonProjectionError(f"invalid {label}")
    return value


def _refs(value: Any, label: str) -> list[str]:
    refs = _sequence(value, label)
    checked = [_exact_str(item, label, max_len=1024) for item in refs]
    if len(checked) != len(set(checked)):
        raise RezonProjectionError(f"duplicate {label}")
    return checked


def _contribution(value: Any, kind: str, roster: tuple[str, ...]) -> dict[str, Any]:
    raw = _exact_dict(value, _CONTRIBUTION, f"{kind} contribution")
    agent = _exact_str(raw["agent_id"], "agent ID", max_len=64)
    if _AGENT.fullmatch(agent) is None or agent not in roster:
        raise RezonProjectionError("non-roster contributor")
    if type(raw["kind"]) is not str or raw["kind"] != kind:
        raise RezonProjectionError("contribution kind mismatch")
    if type(raw["id"]) is not str or raw["id"] != f"{kind}:{agent}":
        raise RezonProjectionError("contribution identity mismatch")
    content = _exact_str(raw["content"], "contribution content")
    refs = _refs(raw["source_refs"], "source_refs")
    targets = _refs(raw["target_ids"], "target_ids")
    if kind == "answer" and targets:
        raise RezonProjectionError("answer may not target peers")
    return {"contribution_id": raw["id"], "agent_id": agent,
            "kind": kind, "content": content, "source_refs": refs,
            "target_ids": targets}


def project_current_collective_synthesis(
    snapshot: object,
    *,
    rezon_head: str,
    independently_expected_head: str,
    independently_expected_episode: str,
    independently_expected_input_digest: str,
) -> ReviewProjection:
    """Check exact source shape, stale syntheses and hashes; emit review-only text.

    The independently named pins are *caller supplied*: callers must never
    interpret a match as proof that custody or a real model execution occurred.
    """
    _sha(rezon_head, "Rezon head", git_head=True)
    _sha(independently_expected_head, "expected Rezon head", git_head=True)
    if rezon_head != independently_expected_head:
        raise RezonProjectionError("Rezon source head changed")
    _exact_str(independently_expected_episode, "expected episode", max_len=128)
    _sha(independently_expected_input_digest, "expected input digest")
    body = _exact_dict(snapshot, _TOP, "Rezon collective snapshot")
    episode = _exact_str(body["episode_id"], "episode", max_len=128)
    if episode != independently_expected_episode:
        raise RezonProjectionError("episode binding mismatch")
    prompt = _exact_str(body["prompt"], "prompt", max_len=16384)
    agents = _sequence(body["agents"], "agents")
    roster = tuple(_exact_str(agent, "agent", max_len=64) for agent in agents)
    if (not roster or tuple(sorted(roster)) != roster or len(set(roster)) != len(roster)
            or any(_AGENT.fullmatch(agent) is None for agent in roster)):
        raise RezonProjectionError("invalid ordered participant roster")
    if type(body["complete"]) is not bool or not body["complete"]:
        raise RezonProjectionError("incomplete collective episode")
    if _sequence(body["missing_agents"], "missing agents"):
        raise RezonProjectionError("missing participants")

    answers = [_contribution(item, "answer", roster)
               for item in _sequence(body["answers"], "answers")]
    reviews = [_contribution(item, "review", roster)
               for item in _sequence(body["reviews"], "reviews")]
    if (len(answers) != len(roster) or
            tuple(item["agent_id"] for item in answers) != roster or
            tuple(item["agent_id"] for item in reviews) !=
            tuple(sorted(item["agent_id"] for item in reviews)) or
            len({item["agent_id"] for item in reviews}) != len(reviews)):
        raise RezonProjectionError("collective answer/review roster mismatch")
    answer_ids = {item["contribution_id"] for item in answers}
    for review in reviews:
        if (not review["target_ids"] or
                not set(review["target_ids"]).issubset(answer_ids) or
                f"answer:{review['agent_id']}" in review["target_ids"]):
            raise RezonProjectionError("invalid peer review targets")

    # This is the exact Rezon CollectiveStore._snapshot digest preimage.
    basis = {"episode_id": episode, "prompt": prompt, "agents": roster,
             "contributions": answers + reviews}
    calculated = sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode("utf-8")).hexdigest()
    declared = _sha(body["input_digest"], "snapshot input digest")
    if calculated != declared or declared != independently_expected_input_digest:
        raise RezonProjectionError("stale or forged input digest")

    advisory: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw in _sequence(body["syntheses"], "syntheses"):
        item = _exact_dict(raw, _SYNTHESIS, "synthesis")
        historical_digest = _sha(item["input_digest"], "synthesis input digest")
        if (type(item["is_current"]) is not bool
                or item["is_current"] != (historical_digest == calculated)
                or type(item["status"]) is not str
                or item["status"] != "unverified_model_synthesis"):
            raise RezonProjectionError("forged currentness or elevated synthesis")
        integrator = _exact_str(item["integrator_id"], "integrator", max_len=64)
        if _AGENT.fullmatch(integrator) is None:
            raise RezonProjectionError("invalid integrator")
        expected_id = f"synthesis:{historical_digest}:{integrator}"
        if type(item["synthesis_id"]) is not str or item["synthesis_id"] != expected_id:
            raise RezonProjectionError("synthesis identity mismatch")
        if (integrator, historical_digest) in seen:
            raise RezonProjectionError("duplicate integrator")
        seen.add((integrator, historical_digest))
        text = _exact_str(item["content"], "synthesis content")
        if item["is_current"]:
            advisory.append(text)
    if not advisory:
        raise RezonProjectionError("no current model synthesis")
    advisory_digest = sha256(json.dumps(advisory, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode("utf-8")).hexdigest()
    return ReviewProjection(rezon_head=rezon_head, episode_id=episode,
                            source_input_digest=declared,
                            advisory_text=tuple(advisory), advisory_sha256=advisory_digest)
