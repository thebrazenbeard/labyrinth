"""Small, persistent, TRAIN-only Minotaur routing-policy learner.

This is a **tabular online developmental taskmasker**, not a pretrained
language model, Minotaur examiner, independent test custodian, or subjective
competence model. It learns a per-skill/tactic success histogram from
operator-supplied *training* outcomes and updates subsequent training-room
difficulty and tactic proposals. Those correctness observations are not
authenticated. It never sees evaluation cases, gold labels, private exams,
model weights, or memory/identity/consent state.

Python adapters executing in the host process remain operator-trusted. The
`propose_room` result is merely untrusted data passed through the existing
`plan_training_room` validator; this code is NOT a sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
from dataclasses import dataclass

from .minotaur_model import RoomContext, TACTICS
from .training_scope import TrainingScopeSpec


SCHEMA = "labyrinth.minotaur-taskmasker-policy.v1"
MAX_TRAIN_OUTCOMES = 2048
MAX_POLICY_JSON_BYTES = 262_144
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ROOM_METADATA = frozenset((
    "objective_id", "domain", "skill", "room_id", "cycle_index", "case_seed", "specificity", "scale",
))
_MODES = ("correct", "zero_update", "corrupt_label")


def _id(value: str, label: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be an ASCII identifier <=128 chars")


def _name(value: str, label: str) -> None:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be an ASCII name <=80 chars")


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class TrainRoomOutcome:
    """No prompt, answer key, or examiner score; TRAIN-only observation."""

    objective_id: str
    room_id: str
    case_id: str
    skill: str
    tactic: str
    difficulty: int
    prompt_sha256: str
    was_correct: bool
    split: str = "train"

    def __post_init__(self) -> None:
        for label in ("objective_id", "room_id", "case_id", "skill"):
            (_id if label in ("room_id", "case_id") else _name)(getattr(self, label), label)
        if type(self.tactic) is not str or self.tactic not in TACTICS:
            raise ValueError("unsupported training tactic")
        if type(self.difficulty) is not int or not 1 <= self.difficulty <= 5:
            raise ValueError("training difficulty must be in [1,5]")
        if type(self.prompt_sha256) is not str or not _SHA256.fullmatch(self.prompt_sha256):
            raise ValueError("invalid prompt SHA-256 commitment")
        if type(self.was_correct) is not bool:
            raise ValueError("was_correct must be an actual boolean")
        if self.split != "train":
            raise ValueError("only TRAIN outcomes can update the taskmaster")


@dataclass(frozen=True)
class TrainCaseApproval:
    """Trainer-owned exact TRAIN admission, independent of reported success.

    The authorized host must generate this registry from approved TRAIN-only
    challenge materialization *before* accepting learner results. This type
    cannot independently authenticate that source or an evaluator identity.
    """

    room_id: str
    case_id: str
    skill: str
    tactic: str
    difficulty: int
    prompt_sha256: str
    split: str = "train"

    def __post_init__(self) -> None:
        _id(self.room_id, "approved room_id")
        _id(self.case_id, "approved case_id")
        _name(self.skill, "approved skill")
        if self.tactic not in TACTICS or type(self.tactic) is not str:
            raise ValueError("unsupported approved tactic")
        if type(self.difficulty) is not int or not 1 <= self.difficulty <= 5:
            raise ValueError("invalid approved difficulty")
        if type(self.prompt_sha256) is not str or not _SHA256.fullmatch(self.prompt_sha256):
            raise ValueError("invalid approved prompt commitment")
        if self.split != "train":
            raise ValueError("only TRAIN cases may be approved")


class MinotaurTrainingPolicy:
    """Bounded tabular competence observations; not neural model weights.

    Compatible with the operator-installed `OperatorTaskmasterAdapter`
    protocol. The stateless deterministic rule selects practice first, moves
    to contrast/transfer after observed train successes, and adjusts
    difficulty against a declared specificity baseline. It only operates on
    training feedback supplied by a trusted trainer, never exam outcomes.
    """

    def __init__(
        self, *, scope: TrainingScopeSpec,
        approved_training_cases: tuple[TrainCaseApproval, ...],
    ) -> None:
        if type(scope) is not TrainingScopeSpec:
            raise ValueError("an exact frozen TrainingScopeSpec is required")
        TrainingScopeSpec.__post_init__(scope)
        self.scope = scope
        self.objective_id = scope.objective_id
        self.domain = scope.domain
        self.skills = scope.skills
        self.seed = scope.seed
        self._approved_cases = approved_training_cases
        self._counts = {
            skill: {tactic: {"correct": 0, "incorrect": 0} for tactic in TACTICS}
            for skill in self.skills
        }
        self._case_ids: list[str] = []
        self._prompt_sha256: list[str] = []
        self._registry_digest()

    def _registry_digest(self) -> str:
        """Validate exact immutable training registry and fingerprint it."""
        cases = self._approved_cases
        if type(cases) is not tuple or not 1 <= len(cases) <= MAX_TRAIN_OUTCOMES:
            raise ValueError("nonempty approved TRAIN registry required")
        seen_ids: set[str] = set()
        seen_prompts: set[str] = set()
        serializable: list[dict[str, str | int]] = []
        for approval in cases:
            if type(approval) is not TrainCaseApproval:
                raise ValueError("registry must contain exact TrainCaseApproval values")
            TrainCaseApproval.__post_init__(approval)
            if approval.skill not in self.skills:
                raise ValueError("registry skill is outside authorized objective")
            if approval.case_id in seen_ids or approval.prompt_sha256 in seen_prompts:
                raise ValueError("registry contains duplicate case or prompt commitment")
            seen_ids.add(approval.case_id)
            seen_prompts.add(approval.prompt_sha256)
            serializable.append({
                "room_id": approval.room_id,
                "case_id": approval.case_id,
                "skill": approval.skill,
                "tactic": approval.tactic,
                "difficulty": approval.difficulty,
                "prompt_sha256": approval.prompt_sha256,
                "split": approval.split,
            })
        return hashlib.sha256(_canonical({"approved_training_cases": serializable})).hexdigest()

    def extend_training_registry(self, new_cases: tuple[TrainCaseApproval, ...]) -> str:
        """Explicit trusted-trainer admission for a later regenerated maze.

        This is an authority-sensitive host operation, not an autonomous
        learner action. The new registry is validated before any mutation.
        It deliberately changes the policy state digest while keeping all
        learned histograms and prior anti-reuse commitments unchanged.
        """
        self._validated_payload()
        if type(new_cases) is not tuple or not new_cases:
            raise ValueError("new authorized TRAIN cases required")
        candidate = self._approved_cases + new_cases
        old = self._approved_cases
        self._approved_cases = candidate
        try:
            digest = self._registry_digest()
            self._validated_payload()
        except Exception:
            self._approved_cases = old
            raise
        return digest

    def _validated_payload(self) -> dict[str, Any]:
        """Reject corruption of mutable Python internals before save/hash/use."""
        if type(self.scope) is not TrainingScopeSpec:
            raise ValueError("missing frozen training scope")
        TrainingScopeSpec.__post_init__(self.scope)
        if (self.objective_id != self.scope.objective_id
                or self.domain != self.scope.domain
                or self.skills != self.scope.skills or self.seed != self.scope.seed):
            raise ValueError("policy metadata diverged from declared scope")
        _name(self.objective_id, "objective_id")
        _name(self.domain, "domain")
        if (type(self.skills) is not tuple or not 1 <= len(self.skills) <= 16
                or len(set(self.skills)) != len(self.skills)):
            raise ValueError("invalid skills")
        for skill in self.skills:
            _name(skill, "skill")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("invalid seed")
        registry_digest = self._registry_digest()
        if (type(self._counts) is not dict or set(self._counts) != set(self.skills)
                or type(self._case_ids) is not list or type(self._prompt_sha256) is not list):
            raise ValueError("invalid routing-policy state")
        if (len(self._case_ids) != len(self._prompt_sha256)
                or not 0 <= len(self._case_ids) <= MAX_TRAIN_OUTCOMES
                or len(set(self._case_ids)) != len(self._case_ids)
                or len(set(self._prompt_sha256)) != len(self._prompt_sha256)):
            raise ValueError("duplicate or mismatched training evidence")
        for case_id, prompt in zip(self._case_ids, self._prompt_sha256):
            _id(case_id, "case_id")
            if type(prompt) is not str or not _SHA256.fullmatch(prompt):
                raise ValueError("invalid training prompt commitment")
        registered = {case.case_id: case.prompt_sha256 for case in self._approved_cases}
        if any(registered.get(case_id) != prompt for case_id, prompt in
               zip(self._case_ids, self._prompt_sha256)):
            raise ValueError("observation history is not bound to an approved TRAIN registry")
        total = 0
        for skill in self.skills:
            tactics = self._counts[skill]
            if type(tactics) is not dict or set(tactics) != set(TACTICS):
                raise ValueError("invalid tactic statistics")
            for tactic in TACTICS:
                counts = tactics[tactic]
                if type(counts) is not dict or set(counts) != {"correct", "incorrect"}:
                    raise ValueError("invalid success/failure counts")
                for key in ("correct", "incorrect"):
                    count = counts[key]
                    if type(count) is not int or not 0 <= count <= MAX_TRAIN_OUTCOMES:
                        raise ValueError("invalid training count")
                    total += count
        if total != len(self._case_ids):
            raise ValueError("observed cases and histograms disagree")
        return {
            "schema": SCHEMA,
            "objective_id": self.objective_id,
            "domain": self.domain,
            "skills": list(self.skills),
            "seed": self.seed,
            "scope_fingerprint": self.scope.fingerprint,
            "specificity": self.scope.specificity,
            "scale": self.scope.scale,
            "max_cycles": self.scope.max_cycles,
            "approved_registry_sha256": registry_digest,
            "counts": self._counts,
            "case_ids": list(self._case_ids),
            "prompt_sha256": list(self._prompt_sha256),
        }

    @property
    def model_hash(self) -> str:
        """Local state fingerprint; not evidence of training provenance."""
        return hashlib.sha256(_canonical(self._validated_payload())).hexdigest()

    def observe(self, outcome: TrainRoomOutcome, *, mode: str = "correct") -> dict[str, Any]:
        """Consume TRAIN-only scored behavior; controls opt in explicitly.

        Corrupt-label and zero-update modes are experiment controls, not
        production policy. `was_correct` is caller-supplied, so this is not
        an independently scored proof. A zero-update visit leaves all state
        and sealed IDs unchanged, as needed for the sham comparison.
        """
        self._validated_payload()
        if type(outcome) is not TrainRoomOutcome:
            raise ValueError("expected a TrainRoomOutcome")
        TrainRoomOutcome.__post_init__(outcome)
        if outcome.objective_id != self.objective_id or outcome.skill not in self.skills:
            raise ValueError("training result is outside the authorized objective")
        if type(mode) is not str or mode not in _MODES:
            raise ValueError("unsupported training-control mode")
        registered = {entry.case_id: entry for entry in self._approved_cases}
        admission = registered.get(outcome.case_id)
        if (admission is None or admission.room_id != outcome.room_id
                or admission.prompt_sha256 != outcome.prompt_sha256
                or admission.skill != outcome.skill
                or admission.tactic != outcome.tactic
                or admission.difficulty != outcome.difficulty):
            raise ValueError("outcome is not bound to an approved TRAIN case")
        if (outcome.case_id in self._case_ids
                or outcome.prompt_sha256 in self._prompt_sha256):
            raise ValueError("reused training case or prompt commitment")
        before = self.model_hash
        if mode == "zero_update":
            return {"mode": mode, "updated": False, "before_hash": before, "after_hash": before}
        if len(self._case_ids) >= MAX_TRAIN_OUTCOMES:
            raise ValueError("taskmaster observation budget exhausted")
        success = outcome.was_correct if mode == "correct" else not outcome.was_correct
        key = "correct" if success else "incorrect"
        self._counts[outcome.skill][outcome.tactic][key] += 1
        self._case_ids.append(outcome.case_id)
        self._prompt_sha256.append(outcome.prompt_sha256)
        after = self.model_hash
        return {"mode": mode, "updated": True, "before_hash": before, "after_hash": after}

    def propose_room(self, metadata: Mapping[str, str | int]) -> dict[str, object]:
        """Suggest next TRAIN-only room tactic and difficulty, without gold."""
        self._validated_payload()
        if not isinstance(metadata, Mapping) or set(metadata) != _ROOM_METADATA:
            raise ValueError("room metadata must be the exact public training view")
        try:
            context = RoomContext(**dict(metadata))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid public room metadata") from exc
        if (context.objective_id != self.objective_id
                or context.domain != self.domain or context.skill not in self.skills
                or context.specificity != self.scope.specificity
                or context.scale != self.scope.scale
                or context.cycle_index >= self.scope.max_cycles):
            raise ValueError("room outside taskmaster's predeclared scope")
        counts = self._counts[context.skill]
        practice = counts["practice"]
        contrast = counts["contrast"]

        def mastered(bucket: dict[str, int]) -> bool:
            trials = bucket["correct"] + bucket["incorrect"]
            # Laplace/Beta(1,1) posterior mean avoids overreacting to one case.
            return trials >= 2 and (bucket["correct"] + 1) / (trials + 2) >= 0.7

        tactic = "practice"
        if mastered(practice):
            tactic = "contrast"
            if mastered(contrast):
                tactic = "transfer"
        # Difficulty must follow evidence for the *current* tactic. Otherwise
        # many old practice wins can keep a failing contrast/transfer chamber
        # maximally difficult. For a newly introduced tactic, use the mastered
        # prerequisite only until that tactic produces its own observations.
        active = counts[tactic]
        if active["correct"] + active["incorrect"] == 0 and tactic != "practice":
            prerequisite = "practice" if tactic == "contrast" else "contrast"
            active = counts[prerequisite]
        wins = active["correct"]
        losses = active["incorrect"]
        difference = wins - losses
        adjustment = min(2, difference // 2) if difference >= 0 else -min(2, (-difference + 1) // 2)
        difficulty = max(1, min(5, context.specificity + adjustment))
        return {"difficulty": difficulty, "tactic": tactic}

    def save_policy(self, destination: str | Path) -> str:
        """Persist write-once, bounded local checkpoint; no destination overwrite."""
        path = Path(destination)
        payload = self._validated_payload()
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        wrapped = {**payload, "integrity_sha256": digest}
        encoded = _canonical(wrapped) + b"\n"
        if len(encoded) > MAX_POLICY_JSON_BYTES:
            raise ValueError("taskmaster checkpoint exceeds size cap")
        with path.open("xb") as handle:
            handle.write(encoded)
        return digest


def load_policy(
    source: str | Path, *, scope: TrainingScopeSpec,
    approved_training_cases: tuple[TrainCaseApproval, ...]
) -> MinotaurTrainingPolicy:
    """Require external scope and approved TRAIN registry to reload the policy.

    Checkpoint data cannot invent its own authority or mutate the declaration.
    The caller must present the exact original scope and admission manifest;
    locally recomputable digests are still not trusted signatures.
    """
    path = Path(source)
    if path.is_symlink():
        raise ValueError("symlink taskmaster checkpoint refused")
    # O_NOFOLLOW adds a final-component protection on platforms supporting it.
    descriptor = os.open(
        str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("taskmaster checkpoint must be an ordinary regular file")
        raw = handle.read(MAX_POLICY_JSON_BYTES + 1)
    if not raw or len(raw) > MAX_POLICY_JSON_BYTES:
        raise ValueError("taskmaster checkpoint exceeds size cap or is empty")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate checkpoint key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"nonfinite number in taskmaster checkpoint: {value}")

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                          parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid taskmaster checkpoint JSON") from exc
    required = {
        "schema", "objective_id", "domain", "skills", "seed", "counts",
        "case_ids", "prompt_sha256", "approved_registry_sha256", "integrity_sha256",
        "scope_fingerprint", "specificity", "scale", "max_cycles",
    }
    if type(data) is not dict or set(data) != required or data["schema"] != SCHEMA:
        raise ValueError("unsupported or ambiguous taskmaster checkpoint schema")
    expected = data.pop("integrity_sha256")
    if type(expected) is not str or not _SHA256.fullmatch(expected):
        raise ValueError("invalid checkpoint digest")
    if hashlib.sha256(_canonical(data)).hexdigest() != expected:
        raise ValueError("checkpoint digest mismatch")
    if (type(data["skills"]) is not list or type(data["case_ids"]) is not list
            or type(data["prompt_sha256"]) is not list):
        raise ValueError("malformed checkpoint arrays")
    policy = MinotaurTrainingPolicy(
        scope=scope, approved_training_cases=approved_training_cases,
    )
    policy._counts = data["counts"]
    policy._case_ids = data["case_ids"]
    policy._prompt_sha256 = data["prompt_sha256"]
    if policy.model_hash != expected:
        raise ValueError("restored state does not match checkpoint")
    return policy
