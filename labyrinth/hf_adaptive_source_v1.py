"""Opt-in, source-only adaptive TRAIN manifest for a future HF pilot path.

All source material is reconstructed from a public scope and registered room
choices. The public manifest contains *neither gold answers nor hashes of
gold answers* (low-domain arithmetic gold is dictionary-reversible). This is
an internal consistency contract, not signed custody, pretrained execution,
an independent examiner, or permission to resume or graduate.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .hf_pilot import PilotConfig, _training_visit_exposures
from .journey_policy_runner import TrainingCorridorRunner
from .maze import Challenge, generate_maze
from .minotaur_hf_roster_bridge import MinotaurHfTrainRoster
from .minotaur_model import TACTICS
from .regenerative_maze import generate_maze_plan
from .training_scope import TrainingScopeSpec


ADAPTIVE_SCHEMA = "labyrinth.hf-lora-adaptive-source.v1"
MANIFEST_SCHEMA = "labyrinth.hf-adaptive-source-manifest.v1"
_HASH_DOMAIN = b"labyrinth.hf-adaptive-source-manifest.v1\x00"
_EVAL_DOMAIN = b"labyrinth.hf-adaptive-public-eval.v1\x00"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CASE_ID = re.compile(r"train-taskmaster-([0-9a-f]{32})-[0-9a-f]{12}\Z", re.ASCII)
_INSTRUCTION = r"\nRespond with exactly one line: ANSWER: <integer>\."
_PRACTICE = re.compile(r"Calculate (\d+) ([+\-]) (\d+)\." + _INSTRUCTION, re.ASCII)
_TRANSFER_ADD = re.compile(
    r"A workshop starts with (\d+) components and receives (\d+) more\. "
    r"How many components does it now have\?" + _INSTRUCTION, re.ASCII,
)
_TRANSFER_SUB = re.compile(
    r"A workshop starts with (\d+) components and uses (\d+)\. "
    r"How many components remain, allowing a negative balance\?" + _INSTRUCTION,
    re.ASCII,
)
_CONTRAST = re.compile(
    r"Expression A is (\d+) \+ (\d+)\. Expression B is (\d+) - (\d+)\. "
    r"Calculate A minus B\." + _INSTRUCTION, re.ASCII,
)
_ROOM_FIELDS = frozenset({
    "room_id", "case_id", "skill", "tactic", "difficulty", "planner_source",
    "prompt_sha256",
})
_PAYLOAD_FIELDS = frozenset({
    "schema", "scope", "cycle_index", "pilot_budget", "train_rooms",
    "ordered_optimizer_exposures", "initial_checkpoint_sha256",
    "adaptive_feedback_rows",
})
_MANIFEST_FIELDS = _PAYLOAD_FIELDS | {"source_sha256"}
_BUDGET_FIELDS = frozenset({"seed", "train_count", "eval_count", "tiers", "steps"})


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("adaptive source manifest is not canonical JSON") from exc
    if len(encoded) > 131072:
        raise ValueError("adaptive source manifest exceeds 128 KiB")
    return encoded


def _source_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_HASH_DOMAIN + _canonical_bytes(payload)).hexdigest()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _budget(config: PilotConfig) -> dict[str, Any]:
    if (type(config) is not PilotConfig
            or type(config.seed) is not int or not 0 <= config.seed < 2**31
            or type(config.train_count) is not int or not 2 <= config.train_count <= 32
            or type(config.eval_count) is not int or not 2 <= config.eval_count <= 32
            or type(config.steps) is not int or not 1 <= config.steps <= 128
            or type(config.tiers) is not tuple or config.tiers != (1,)):
        raise ValueError("adaptive HF source requires bounded tier-one pilot budget")
    return {
        "seed": config.seed, "train_count": config.train_count,
        "eval_count": config.eval_count, "tiers": [1], "steps": config.steps,
    }


def _arith_atom(operation: str, left: int, right: int) -> tuple[str, int, int]:
    if operation == "+":
        left, right = sorted((left, right))
    return operation, left, right


def arithmetic_semantic_atoms(case: Challenge) -> frozenset[tuple[str, int, int]]:
    """Strict bounded grammar/answer oracle for all three Minotaur tactics.

    A contrast task contains two arithmetic source expressions. Reject public
    heldout cases that reuse either expression, not just exact prompt bytes.
    This is not a general mathematical equivalence checker.
    """
    if (type(case) is not Challenge or type(case.tier) is not int or case.tier != 1
            or type(case.prompt) is not str or type(case.answer) is not str
            or type(case.kind) is not str or case.kind not in ("arithmetic", "arithmetic_contrast")):
        raise ValueError("unsupported adaptive arithmetic source case")
    prompt = case.prompt
    match = _PRACTICE.fullmatch(prompt)
    if match is not None and case.kind == "arithmetic":
        left, right = int(match[1]), int(match[3])
        op = match[2]
        expected = left + right if op == "+" else left - right
        atoms = frozenset({_arith_atom(op, left, right)})
    else:
        match = _TRANSFER_ADD.fullmatch(prompt)
        if match is not None and case.kind == "arithmetic":
            left, right, op = int(match[1]), int(match[2]), "+"
            expected = left + right
            atoms = frozenset({_arith_atom(op, left, right)})
        else:
            match = _TRANSFER_SUB.fullmatch(prompt)
            if match is not None and case.kind == "arithmetic":
                left, right, op = int(match[1]), int(match[2]), "-"
                expected = left - right
                atoms = frozenset({_arith_atom(op, left, right)})
            else:
                match = _CONTRAST.fullmatch(prompt)
                if (match is None or case.kind != "arithmetic_contrast"
                        or match[1] != match[3] or match[2] != match[4]):
                    raise ValueError("unrecognized adaptive arithmetic template")
                left, right = int(match[1]), int(match[2])
                expected = 2 * right
                atoms = frozenset({
                    _arith_atom("+", left, right), _arith_atom("-", left, right),
                })
    if case.answer != str(expected):
        raise ValueError("adaptive arithmetic gold differs from its canonical prompt")
    return atoms


def _replay_rooms(config: PilotConfig, payload: dict[str, Any]) -> tuple[TrainingScopeSpec, list[Challenge]]:
    scope = TrainingScopeSpec.from_dict(payload["scope"])
    recorded_budget = payload["pilot_budget"]
    expected_budget = _budget(config)
    if (type(recorded_budget) is not dict or set(recorded_budget) != _BUDGET_FIELDS
            or any(type(recorded_budget[key]) is not type(value)
                   for key, value in expected_budget.items())
            or type(recorded_budget["tiers"]) is not list
            or any(type(tier) is not int for tier in recorded_budget["tiers"])):
        raise ValueError("adaptive source budget has noncanonical field types")
    if (scope.skills != ("arithmetic",) or payload["cycle_index"] != 0
            or type(payload["cycle_index"]) is not int
            or recorded_budget != expected_budget
            or scope.total_chambers != config.train_count):
        raise ValueError("adaptive scope, cycle, or HF budget differs from source manifest")
    plan = generate_maze_plan(scope, 0)
    entries = payload["train_rooms"]
    feedback_rows = payload["adaptive_feedback_rows"]
    initial = payload["initial_checkpoint_sha256"]
    if (type(entries) is not list or len(entries) != len(plan.rooms)
            or type(feedback_rows) is not list or len(feedback_rows) != len(entries)
            or type(initial) is not str or _DIGEST.fullmatch(initial) is None):
        raise ValueError("adaptive source manifest omits a planned TRAIN room")
    # Replay the actual deterministic Minotaur TRAIN policy rather than merely
    # accepting an arbitrary tactic/difficulty that generates self-consistent
    # gold. A synthetic correct or incorrect response reproduces precisely
    # the feedback bit consumed by the policy; original learner response text
    # is neither required nor contained in this public source manifest.
    runner = TrainingCorridorRunner.begin(
        scope, plan, checkpoint_sha256=initial,
    )
    train: list[Challenge] = []
    for index, (entry, raw_feedback) in enumerate(zip(entries, feedback_rows)):
        if type(entry) is not dict or set(entry) != _ROOM_FIELDS:
            raise ValueError("adaptive TRAIN admission row is malformed")
        if (type(raw_feedback) is not list or len(raw_feedback) != 6
                or any(type(part) is not str for pos, part in enumerate(raw_feedback)
                       if pos != 3)
                or type(raw_feedback[3]) is not int):
            raise ValueError("adaptive TRAIN feedback row is malformed")
        public_view = runner.next_training_prompt()
        pending = runner.state.pending
        if pending is None:
            raise ValueError("adaptive TRAIN route lacks a pending chamber")
        room_id = entry["room_id"]
        if (type(room_id) is not str or room_id != pending.room_id
                or raw_feedback[0] != pending.room_id
                or raw_feedback[1] != entry["case_id"]
                or raw_feedback[2] != entry["prompt_sha256"]):
            raise ValueError("adaptive TRAIN route diverges from feedback-derived chamber")
        if (type(entry["skill"]) is not str or entry["skill"] != "arithmetic"
                or type(entry["tactic"]) is not str or entry["tactic"] not in TACTICS
                or type(entry["difficulty"]) is not int or not 1 <= entry["difficulty"] <= 5
                or type(entry["planner_source"]) is not str
                or entry["planner_source"] != (
                    "initial_train_bootstrap" if index == 0 else "policy_plus_adaptive_train"
                )
                or type(entry["case_id"]) is not str
                or type(entry["prompt_sha256"]) is not str
                or _DIGEST.fullmatch(entry["prompt_sha256"]) is None):
            raise ValueError("adaptive TRAIN admission fields are invalid")
        case = runner.operator_pending_train_case(public_view)
        approval = runner.training_registry[-1]
        if (case.id != entry["case_id"]
                or _prompt_hash(case.prompt) != entry["prompt_sha256"]
                or case.split != "train"
                or approval.room_id != room_id
                or approval.case_id != entry["case_id"]
                or approval.skill != entry["skill"]
                or approval.tactic != entry["tactic"]
                or approval.difficulty != entry["difficulty"]
                or approval.prompt_sha256 != entry["prompt_sha256"]
                or dict(case.metadata).get("planner_source") != entry["planner_source"]):
            raise ValueError("adaptive TRAIN admission differs from policy-generated source")
        arithmetic_semantic_atoms(case)
        train.append(case)
        if (raw_feedback[3] not in (0, 1)
                or raw_feedback[4] != runner.state.checkpoint_sha256
                or raw_feedback[1] != case.id
                or raw_feedback[2] != public_view.prompt_sha256):
            raise ValueError("adaptive feedback differs from exact pre-score TRAIN source")
        response = ("ANSWER: " + case.answer) if raw_feedback[3] else "ANSWER: invalid"
        outcome = runner.submit_training_response(
            response, checkpoint_after_sha256=raw_feedback[5],
        )
        if outcome.correct is not bool(raw_feedback[3]):
            raise ValueError("adaptive source scorer does not match committed feedback bit")
    if not runner.is_complete:
        raise ValueError("adaptive TRAIN route did not consume the declared budget")
    return scope, train


def _public_diagnostic(config: PilotConfig, scope: TrainingScopeSpec,
                       digest: str, train: list[Challenge]) -> list[Challenge]:
    source_atoms: set[tuple[str, int, int]] = set()
    source_ids = {case.id for case in train}
    source_prompts = {case.prompt for case in train}
    for case in train:
        source_atoms.update(arithmetic_semantic_atoms(case))
    derived_seed = int.from_bytes(hashlib.sha256(
        _EVAL_DOMAIN + scope.fingerprint.encode("ascii") + b"\x00"
        + str(config.seed).encode("ascii") + b"\x00" + digest.encode("ascii"),
    ).digest()[:4], "big") & (2**31 - 1)
    for attempt in range(128):
        heldout = generate_maze(
            seed=derived_seed ^ attempt, count=config.eval_count,
            tiers=(1,), split="eval",
        )
        if (not source_ids.intersection(c.id for c in heldout)
                and not source_prompts.intersection(c.prompt for c in heldout)
                and all(not source_atoms.intersection(arithmetic_semantic_atoms(c))
                        for c in heldout)):
            return heldout
    raise ValueError("adaptive public diagnostic cannot avoid TRAIN semantic source")


def reconstruct_adaptive_source(
    config: PilotConfig, manifest: dict[str, Any],
) -> tuple[list[Challenge], list[Challenge], tuple[tuple[str, str], ...]]:
    """Reject tampered/malformed manifests, replay exact TRAIN gold privately.

    The digest detects intra-report inconsistency only. A caller can fabricate
    the entire scope, source history and hash; real custody needs a separate
    owner-pinned authenticated service. In particular, an independently
    unchecked *last* TRAIN success bit can be changed together with the
    source hash: no subsequent routing step can cross-check that final score,
    and original learner responses are intentionally absent from this public
    manifest. This must never become an exam or graduation authority.
    """
    _budget(config)
    if type(manifest) is not dict or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("adaptive source manifest fields missing or unexpected")
    payload = {key: manifest[key] for key in _PAYLOAD_FIELDS}
    if payload["schema"] != MANIFEST_SCHEMA or type(payload["schema"]) is not str:
        raise ValueError("unsupported adaptive source manifest schema")
    if (type(payload["pilot_budget"]) is not dict
            or set(payload["pilot_budget"]) != _BUDGET_FIELDS
            or type(manifest["source_sha256"]) is not str
            or not _DIGEST.fullmatch(manifest["source_sha256"])
            or _source_hash(payload) != manifest["source_sha256"]):
        raise ValueError("adaptive source manifest commitment mismatch")
    scope, train = _replay_rooms(config, payload)
    planned = _training_visit_exposures(train, seed=config.seed, steps=config.steps)
    recorded = payload["ordered_optimizer_exposures"]
    if (type(recorded) is not list or len(recorded) != len(planned)
            or any(type(pair) is not list or len(pair) != 2
                   or tuple(pair) != expected
                   for pair, expected in zip(recorded, planned))):
        raise ValueError("adaptive optimizer visit schedule differs from committed source")
    heldout = _public_diagnostic(config, scope, manifest["source_sha256"], train)
    return train, heldout, planned


def build_adaptive_manifest(
    config: PilotConfig, scope: TrainingScopeSpec, roster: MinotaurHfTrainRoster,
) -> dict[str, Any]:
    """Bind a staged private TRAIN roster to a public source-only manifest."""
    _budget(config)
    if (type(scope) is not TrainingScopeSpec or type(roster) is not MinotaurHfTrainRoster
            or scope.skills != ("arithmetic",) or roster.cycle_index != 0
            or roster.scope_sha256 != scope.fingerprint
            or len(roster.train_cases) != config.train_count):
        raise ValueError("adaptive roster does not match current scope and cycle-zero budget")
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(roster.train_cases):
        if (type(case) is not Challenge or type(case.id) is not str
                or _CASE_ID.fullmatch(case.id) is None
                or type(case.metadata) is not tuple
                or any(type(pair) is not tuple or len(pair) != 2
                       or any(type(part) is not str for part in pair)
                       for pair in case.metadata)):
            raise ValueError("adaptive staged TRAIN case is malformed")
        meta = dict(case.metadata)
        if (len(meta) != 4 or set(meta) != {
            "generator", "tactic", "difficulty", "planner_source",
        } or meta["generator"] != "taskmaster-route-v1"):
            raise ValueError("adaptive staged TRAIN metadata is not approved taskmaster source")
        try:
            difficulty = int(meta["difficulty"])
        except ValueError as exc:
            raise ValueError("adaptive staged difficulty is not an integer") from exc
        rows.append({
            "room_id": _CASE_ID.fullmatch(case.id)[1],
            "case_id": case.id,
            "skill": "arithmetic", "tactic": meta["tactic"],
            "difficulty": difficulty, "planner_source": meta["planner_source"],
            "prompt_sha256": _prompt_hash(case.prompt),
        })
    planned = _training_visit_exposures(
        list(roster.train_cases), seed=config.seed, steps=config.steps,
    )
    if planned != roster.ordered_optimizer_exposures:
        raise ValueError("adaptive staged optimizer visit schedule changed")
    feedback = getattr(roster, "adaptive_feedback_rows", None)
    initial = getattr(roster, "initial_checkpoint_sha256", None)
    if (type(initial) is not str or _DIGEST.fullmatch(initial) is None
            or type(feedback) is not tuple or len(feedback) != len(rows)
            or any(type(row) is not tuple or len(row) != 6 for row in feedback)):
        raise ValueError("adaptive staged route provenance is missing or malformed")
    payload = {
        "schema": MANIFEST_SCHEMA, "scope": scope.to_dict(), "cycle_index": 0,
        "pilot_budget": _budget(config), "train_rooms": rows,
        "ordered_optimizer_exposures": [list(pair) for pair in planned],
        "initial_checkpoint_sha256": initial,
        "adaptive_feedback_rows": [list(row) for row in feedback],
    }
    manifest = {**payload, "source_sha256": _source_hash(payload)}
    replayed, _heldout, _planned = reconstruct_adaptive_source(config, manifest)
    if replayed != list(roster.train_cases):
        raise ValueError("adaptive staged TRAIN gold differs from replayed approved source")
    return manifest
