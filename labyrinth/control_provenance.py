"""Custodian-authenticated metadata for matched neural-learning controls.

The secret MAC key and independently precommitted label schedules must remain
outside the training process. This module authenticates *claims*, not actual
model execution or trained parameter values. Its outputs cannot graduate a
learner. Consumers must separately open and hash checkpoint/report bytes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import hmac
import json
import re
from typing import Any


SCHEMA = "labyrinth.control-provenance.v1"
ARMS = ("baseline", "zero_update", "correct_labels", "corrupted_labels")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_SEAL_KEYS = frozenset({"schema", "manifest", "manifest_sha256", "mac_sha256"})
_MAC_CONTEXT = b"labyrinth.control-provenance.mac.v1\0"
_LABEL_CONTEXT = b"labyrinth.control-provenance.labels.v1\0"


def _id(value: Any, name: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")


def _digest(value: Any, name: str) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")


def _key(key: bytes) -> None:
    if type(key) is not bytes or len(key) < 16:
        raise ValueError("custodian key must be at least 16 secret bytes")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def commit_label_schedule(schedule: tuple[tuple[str, str], ...], key: bytes) -> str:
    """Keyed, ordered label precommit; no guessable label hashes are published."""
    _key(key)
    if type(schedule) is not tuple or len(schedule) > 100000:
        raise ValueError("invalid label schedule")
    for pair in schedule:
        if (type(pair) is not tuple or len(pair) != 2
                or type(pair[1]) is not str or not 1 <= len(pair[1]) <= 4096):
            raise ValueError("invalid label schedule entry")
        _id(pair[0], "label schedule case ID")
    return hmac.new(key, _LABEL_CONTEXT + _canonical(schedule), "sha256").hexdigest()


@dataclass(frozen=True)
class ArmManifest:
    experiment_id: str
    objective_id: str
    source_tree_sha256: str
    base_model_sha256: str
    initial_adapter_sha256: str
    final_adapter_sha256: str
    exam_precommit_sha256: str
    report_sha256: str
    arm: str
    label_commitment: str
    training_visits: tuple[tuple[str, str, str], ...]  # room, case, prompt SHA
    training_structure_exposures: tuple[tuple[str, str, str], ...]  # case, prompt SHA, structure
    compute_steps: int
    optimizer_updates: int

    @property
    def graduation_authorized(self) -> bool:
        return False

    def validate(self) -> None:
        _id(self.experiment_id, "experiment ID")
        _id(self.objective_id, "objective ID")
        if self.arm not in ARMS:
            raise ValueError("invalid control arm")
        for name in (
            "source_tree_sha256", "base_model_sha256", "initial_adapter_sha256",
            "final_adapter_sha256", "exam_precommit_sha256", "report_sha256",
            "label_commitment",
        ):
            _digest(getattr(self, name), name)
        if (type(self.compute_steps) is not int
                or type(self.optimizer_updates) is not int
                or not 0 <= self.optimizer_updates <= self.compute_steps <= 100000):
            raise ValueError("invalid optimizer/compute counts")
        if (type(self.training_visits) is not tuple
                or len(self.training_visits) > 100000):
            raise ValueError("invalid training visits")
        for visit in self.training_visits:
            if type(visit) is not tuple or len(visit) != 3:
                raise ValueError("invalid training visit")
            _id(visit[0], "room ID")
            _id(visit[1], "case ID")
            _digest(visit[2], "prompt commitment")
        if (type(self.training_structure_exposures) is not tuple
                or len(self.training_structure_exposures) > 4096):
            raise ValueError("invalid structural exposure manifest")
        seen_case: set[str] = set()
        seen_prompt: set[str] = set()
        for entry in self.training_structure_exposures:
            if type(entry) is not tuple or len(entry) != 3:
                raise ValueError("invalid structural exposure entry")
            _id(entry[0], "structural exposure case ID")
            _digest(entry[1], "structural exposure prompt commitment")
            _id(entry[2], "structural exposure key")
            if entry[0] in seen_case or entry[1] in seen_prompt:
                raise ValueError("duplicate structural exposure")
            seen_case.add(entry[0])
            seen_prompt.add(entry[1])
        if (set((v[1], v[2]) for v in self.training_visits)
                != {(entry[0], entry[1]) for entry in self.training_structure_exposures}):
            raise ValueError("structural exposure does not cover authenticated training visits")
        if self.arm == "baseline" and (self.training_visits or self.compute_steps):
            raise ValueError("baseline cannot claim training visits")
        if self.arm != "baseline" and len(self.training_visits) != self.compute_steps:
            raise ValueError("training visits must equal compute steps")


def seal_arm_manifest(manifest: ArmManifest, key: bytes) -> dict[str, Any]:
    """Seal claims under an external custodian key; never persist that key."""
    _key(key)
    if type(manifest) is not ArmManifest:
        raise ValueError("invalid arm manifest")
    manifest.validate()
    payload = asdict(manifest)
    encoded = _canonical(payload)
    return {"schema": SCHEMA, "manifest": payload,
            "manifest_sha256": sha256(encoded).hexdigest(),
            "mac_sha256": hmac.new(key, _MAC_CONTEXT + encoded, "sha256").hexdigest()}


def verify_arm_manifest(
    seal: dict[str, Any], key: bytes, *, expected_arm: str,
    expected_label_commitment: str,
) -> ArmManifest:
    """Authenticate exact arm and expected evaluator-owned label commitment."""
    _key(key)
    if (type(seal) is not dict or set(seal) != _SEAL_KEYS
            or seal.get("schema") != SCHEMA or type(seal["manifest"]) is not dict):
        raise ValueError("invalid sealed control manifest")
    payload = seal["manifest"]
    if set(payload) != {field.name for field in fields(ArmManifest)}:
        raise ValueError("invalid arm manifest fields")
    _digest(seal["manifest_sha256"], "manifest SHA")
    _digest(seal["mac_sha256"], "manifest MAC")
    encoded = _canonical(payload)
    if (not hmac.compare_digest(seal["manifest_sha256"], sha256(encoded).hexdigest())
            or not hmac.compare_digest(
                seal["mac_sha256"],
                hmac.new(key, _MAC_CONTEXT + encoded, "sha256").hexdigest(),
            )):
        raise ValueError("manifest authentication failed")
    try:
        manifest = ArmManifest(**{
            **payload,
            "training_visits": tuple(
                tuple(visit) if type(visit) is list else visit
                for visit in payload["training_visits"]
            ),
            "training_structure_exposures": tuple(
                tuple(entry) if type(entry) is list else entry
                for entry in payload["training_structure_exposures"]
            ),
        })
        manifest.validate()
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("invalid authenticated manifest content") from exc
    if manifest.arm != expected_arm:
        raise ValueError("control arm mismatch")
    _digest(expected_label_commitment, "expected label commitment")
    if manifest.label_commitment != expected_label_commitment:
        raise ValueError("label commitment mismatch")
    return manifest


def verify_matched_group(
    seals: tuple[dict[str, Any], ...], key: bytes, *,
    correct_label_commitment: str, corrupted_label_commitment: str,
    empty_label_commitment: str, expected_exam_precommit: str,
) -> tuple[ArmManifest, ...]:
    """Validate four authenticated, compute/exposure-matched control claims.

    Actual neural training, label correctness, checkpoint bytes, evaluator
    independence, and fresh-process reload remain separate external gates.
    """
    _digest(expected_exam_precommit, "expected exam precommitment")
    for name, value in (("correct", correct_label_commitment),
                        ("corrupted", corrupted_label_commitment),
                        ("empty", empty_label_commitment)):
        _digest(value, f"{name} label commitment")
    if len({correct_label_commitment, corrupted_label_commitment, empty_label_commitment}) != 3:
        raise ValueError("control label commitments are not distinct")
    if type(seals) is not tuple or len(seals) != len(ARMS):
        raise ValueError("four control receipts required")
    by_arm: dict[str, dict[str, Any]] = {}
    for seal in seals:
        if type(seal) is not dict or type(seal.get("manifest")) is not dict:
            raise ValueError("invalid control receipt")
        arm = seal["manifest"].get("arm")
        if arm not in ARMS or arm in by_arm:
            raise ValueError("duplicate or invalid control arm")
        by_arm[arm] = seal
    if set(by_arm) != set(ARMS):
        raise ValueError("missing control arm")
    expected_labels = dict(
        baseline=empty_label_commitment,
        zero_update=correct_label_commitment,
        correct_labels=correct_label_commitment,
        corrupted_labels=corrupted_label_commitment,
    )
    manifests = tuple(verify_arm_manifest(
        by_arm[arm], key, expected_arm=arm,
        expected_label_commitment=expected_labels[arm],
    ) for arm in ARMS)
    if any(m.exam_precommit_sha256 != expected_exam_precommit for m in manifests):
        raise ValueError("exam precommitment mismatch")
    for field_name in (
        "experiment_id", "objective_id", "source_tree_sha256",
        "base_model_sha256", "initial_adapter_sha256", "exam_precommit_sha256",
    ):
        if len({getattr(m, field_name) for m in manifests}) != 1:
            raise ValueError("control arms must share same base and experiment lineage")
    baseline, sham, correct, corrupt = manifests
    if (baseline.optimizer_updates or baseline.final_adapter_sha256 != baseline.initial_adapter_sha256
            or sham.optimizer_updates or sham.final_adapter_sha256 != sham.initial_adapter_sha256):
        raise ValueError("baseline and zero-update control must remain frozen")
    if (correct.optimizer_updates < 1 or corrupt.optimizer_updates < 1
            or correct.optimizer_updates != corrupt.optimizer_updates):
        raise ValueError("trained arms require matched optimizer updates")
    if (not correct.training_visits
            or not (sham.training_visits == correct.training_visits == corrupt.training_visits)
            or not (sham.training_structure_exposures
                    == correct.training_structure_exposures
                    == corrupt.training_structure_exposures)
            or not (sham.compute_steps == correct.compute_steps == corrupt.compute_steps)):
        raise ValueError("control training visits and compute must match")
    if (correct.final_adapter_sha256 == correct.initial_adapter_sha256
            or corrupt.final_adapter_sha256 == corrupt.initial_adapter_sha256):
        raise ValueError("trained adapters report no weight change")
    return manifests
