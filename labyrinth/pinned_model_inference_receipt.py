"""Source-only evaluator-invoked logits custody for a future sealed examiner.

No caller-supplied prediction map or gold label enters this API: predictions
are derived from logits returned by an evaluator-selected runtime. Three
component files and the ordered prompt roster are read back against pinned
digests before and after invocation. The runtime itself, its use of those
files, key custody, one-shot authority, and physical execution are NOT
attested by Python source checks. In particular, this is not a private final
examination or a graduation credential.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable


SCHEMA = "labyrinth.pinned-model-inference-source.v1"
_ROSTER_DOMAIN = b"labyrinth.evaluator-private-roster.v1\0"
_PRIVATE_DOMAIN = b"labyrinth.evaluator-source-inference.private.v1\0"
_PUBLIC_DOMAIN = b"labyrinth.evaluator-source-inference.public.v1\0"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_ARTIFACT_BYTES = 16 * 1024**3


class InferenceSourceError(ValueError):
    """Pinned source-inference input, artifact, or output was invalid."""


@dataclass(frozen=True)
class PrivateInferenceCase:
    """Private to the evaluator; notably contains no correct answer."""

    case_id: str
    prompt: str


@dataclass(frozen=True)
class PinnedComponent:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ThreeComponentModel:
    pretrained: PinnedComponent
    adapter: PinnedComponent
    head: PinnedComponent


@dataclass(frozen=True)
class EvaluatorInferenceIntent:
    exam_id: str
    event_id: str
    run_id: str
    arm: str
    seed: int
    steps_per_cycle: int
    stage: str  # after-training or prior-skill-before
    candidate_artifact_sha256: str
    prior_skill_checkpoint_sha256: str
    scope_sha256: str
    source_sha256: str
    roster_commitment: str
    component_set_sha256: str


@dataclass(frozen=True)
class SignedSourceInference:
    """Private evidence never proves which model a host actually ran."""

    public_bytes: bytes
    private_bytes: bytes
    public_mac_hex: str
    private_mac_hex: str
    physical_execution_authenticated: bool = False
    independent_evaluator_host_verified: bool = False
    graduation_authorized: bool = False


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _ident(value: Any, name: str) -> None:
    if type(value) is not str or _IDENT.fullmatch(value) is None:
        raise InferenceSourceError(f"invalid {name}")


def _hex(value: Any, name: str) -> None:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise InferenceSourceError(f"invalid {name} SHA-256")


def _keys(evaluator_key: bytes, roster_key: bytes) -> None:
    if (type(evaluator_key) is not bytes or len(evaluator_key) < 32
            or type(roster_key) is not bytes or len(roster_key) < 32
            or hmac.compare_digest(evaluator_key, roster_key)):
        raise InferenceSourceError("distinct evaluator and roster custody keys required")


def _cases(cases: Any) -> tuple[PrivateInferenceCase, ...]:
    if (type(cases) is not tuple or not 1 <= len(cases) <= 400
            or any(type(case) is not PrivateInferenceCase for case in cases)):
        raise InferenceSourceError("complete typed evaluator prompt roster required")
    ids: set[str] = set()
    for case in cases:
        _ident(case.case_id, "private case")
        if (type(case.prompt) is not str or not 1 <= len(case.prompt) <= 4096
                or case.case_id in ids):
            raise InferenceSourceError("private prompt or case roster malformed")
        ids.add(case.case_id)
    return cases


def evaluator_roster_commitment(cases: tuple[PrivateInferenceCase, ...], *,
                                roster_key: bytes) -> str:
    """Keyed roster digest must be pinned BEFORE any candidate inference."""
    if type(roster_key) is not bytes or len(roster_key) < 32:
        raise InferenceSourceError("evaluator-held roster key required")
    return hmac.new(roster_key, _ROSTER_DOMAIN + _canonical([
        [case.case_id, sha256(case.prompt.encode("utf-8")).hexdigest()]
        for case in _cases(cases)
    ]), sha256).hexdigest()


def model_component_set_sha256(model: ThreeComponentModel) -> str:
    """Precommit the exact three named file digests, separate from run claims."""
    if type(model) is not ThreeComponentModel:
        raise InferenceSourceError("typed model component set required")
    named = {}
    for name in ("pretrained", "adapter", "head"):
        part = getattr(model, name)
        if type(part) is not PinnedComponent:
            raise InferenceSourceError("typed model component pin required")
        _hex(part.sha256, name)
        named[name] = part.sha256
    return sha256(b"labyrinth.evaluator-component-set.v1\0" + _canonical(named)).hexdigest()


def candidate_component_binding_sha256(candidate_artifact_sha256: str,
                                       component_set_sha256: str) -> str:
    """Source digest of a candidate-to-components map; requires external pin."""
    _hex(candidate_artifact_sha256, "candidate artifact")
    _hex(component_set_sha256, "model component set")
    return sha256(b"labyrinth.evaluator-candidate-components.v1\0" + _canonical({
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "component_set_sha256": component_set_sha256,
    })).hexdigest()


def _intent(intent: Any) -> None:
    if type(intent) is not EvaluatorInferenceIntent:
        raise InferenceSourceError("typed evaluator inference intent required")
    for name in ("exam_id", "event_id", "run_id"):
        _ident(getattr(intent, name), name)
    if (intent.arm not in ("correct", "zero_update", "inverted", "zero_lr",
                           "randomized_labels")
            or type(intent.seed) is not int or not 0 <= intent.seed < 2**31
            or type(intent.steps_per_cycle) is not int
            or not 1 <= intent.steps_per_cycle <= 64
            or intent.stage not in ("after-training", "prior-skill-before")):
        raise InferenceSourceError("invalid model-control or stage identity")
    for name in ("candidate_artifact_sha256", "prior_skill_checkpoint_sha256",
                 "scope_sha256", "source_sha256", "roster_commitment",
                 "component_set_sha256"):
        _hex(getattr(intent, name), name)


def _component_probe(value: Any, name: str) -> tuple[int, int, int, int, int]:
    if type(value) is not PinnedComponent or not isinstance(value.path, Path):
        raise InferenceSourceError(f"typed {name} component pin required")
    _hex(value.sha256, name)
    if not value.path.is_absolute():
        raise InferenceSourceError(f"{name} path must be absolute")
    for part in (value.path, *value.path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise InferenceSourceError(f"linked {name} component or ancestor")
    try:
        descriptor = os.open(value.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (not stat.S_ISREG(before.st_mode)
                    or not 1 <= before.st_size <= _MAX_ARTIFACT_BYTES):
                raise InferenceSourceError(f"invalid {name} component size/type")
            digest = sha256()
            for piece in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(piece)
            after = os.fstat(stream.fileno())
        final = value.path.lstat()
    except OSError as exc:
        raise InferenceSourceError(f"{name} component inaccessible") from exc
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_ctime_ns)
    if (identity(before) != identity(after) or identity(before) != identity(final)
            or not hmac.compare_digest(digest.hexdigest(), value.sha256)):
        raise InferenceSourceError(f"{name} component changed or differs from pin")
    return identity(before)


def _components(pins: Any) -> dict[str, tuple[int, int, int, int, int]]:
    if type(pins) is not ThreeComponentModel:
        raise InferenceSourceError("three pinned component files required")
    result = {name: _component_probe(getattr(pins, name), name)
              for name in ("pretrained", "adapter", "head")}
    if len({(value[0], value[1]) for value in result.values()}) != 3:
        raise InferenceSourceError("model components must be distinct file identities")
    return result


def _logits(value: Any) -> tuple[float, float]:
    if (type(value) is not tuple or len(value) != 2
            or any(type(item) is not float or not math.isfinite(item)
                   for item in value) or value[0] == value[1]):
        raise InferenceSourceError("two finite non-tied detached float logits required")
    return value


def run_source_pinned_evaluator_inference(
    *, intent: EvaluatorInferenceIntent,
    cases: tuple[PrivateInferenceCase, ...],
    model: ThreeComponentModel, externally_pinned_roster_commitment: str,
    externally_pinned_component_set_sha256: str,
    externally_pinned_candidate_component_binding_sha256: str,
    evaluator_key: bytes, roster_key: bytes,
    evaluator_selected_driver_factory: Callable[[ThreeComponentModel], Any],
) -> SignedSourceInference:
    """Derive predictions from logits; never accept trainee predictions/gold.

    The driver is still only an in-process, operator-selected Python object;
    neither this API nor its HMAC can prove host custody or physical model use.
    """
    _intent(intent)
    _keys(evaluator_key, roster_key)
    _hex(externally_pinned_roster_commitment, "external roster")
    _hex(externally_pinned_component_set_sha256, "external component set")
    _hex(externally_pinned_candidate_component_binding_sha256,
         "external candidate-component binding")
    cases = _cases(cases)
    # A frozen dataclass is not a security boundary against an in-process
    # driver using object.__setattr__. Snapshot source identities before
    # invoking the evaluator-selected factory and logits callbacks; calculate
    # from that snapshot, and reject changes to the caller's original inputs.
    original_inputs = (intent, cases, model)
    intent, cases, model = deepcopy(original_inputs)
    committed = evaluator_roster_commitment(cases, roster_key=roster_key)
    if (not hmac.compare_digest(committed, intent.roster_commitment)
            or not hmac.compare_digest(committed,
                                      externally_pinned_roster_commitment)):
        raise InferenceSourceError("ordered private roster differs from pinned intent")
    component_set = model_component_set_sha256(model)
    if (not hmac.compare_digest(component_set, intent.component_set_sha256)
            or not hmac.compare_digest(component_set,
                                      externally_pinned_component_set_sha256)):
        raise InferenceSourceError("model component set differs from external precommit")
    candidate_binding = candidate_component_binding_sha256(
        intent.candidate_artifact_sha256, component_set)
    if not hmac.compare_digest(candidate_binding,
                               externally_pinned_candidate_component_binding_sha256):
        raise InferenceSourceError("candidate-to-components map differs from external precommit")
    before = _components(model)
    if not callable(evaluator_selected_driver_factory):
        raise InferenceSourceError("evaluator-selected model loader required")
    driver = evaluator_selected_driver_factory(model)
    predictor = getattr(driver, "predict_logits", None)
    if not callable(predictor):
        raise InferenceSourceError("driver must perform prompted logits inference")
    rows: list[dict[str, Any]] = []
    for case in cases:
        raw = _logits(predictor(case.prompt))
        rows.append({"case_id": case.case_id,
                     "logits_float_hex": [raw[0].hex(), raw[1].hex()],
                     "predicted_label": int(raw[1] > raw[0])})
    if _components(model) != before:
        raise InferenceSourceError("model artifact identity changed during inference")
    if original_inputs != (intent, cases, model):
        raise InferenceSourceError("evaluator intent, roster or model pins changed during inference")
    identity = {
        "exam_id": intent.exam_id, "event_id": intent.event_id,
        "run_id": intent.run_id, "arm": intent.arm,
        "seed": intent.seed, "steps_per_cycle": intent.steps_per_cycle,
        "stage": intent.stage,
        "candidate_artifact_sha256": intent.candidate_artifact_sha256,
        "prior_skill_checkpoint_sha256": intent.prior_skill_checkpoint_sha256,
        "scope_sha256": intent.scope_sha256,
        "source_sha256": intent.source_sha256,
        "roster_commitment": committed,
        "component_set_sha256": component_set,
        "candidate_component_binding_sha256": candidate_binding,
        "pretrained_sha256": model.pretrained.sha256,
        "adapter_sha256": model.adapter.sha256,
        "head_sha256": model.head.sha256,
    }
    private = _canonical({
        "schema": SCHEMA + ".private.v1", **identity,
        "predictions_derived_from_logits": rows,
        "driver_provenance": "SOURCE_SELECTED_IN_PROCESS_NOT_HOST_ATTESTED",
        "physical_execution_authenticated": False,
        "independent_evaluator_host_verified": False,
        "graduation_authorized": False,
    })
    public = _canonical({
        "schema": SCHEMA + ".public.v1", **identity,
        "case_count": len(cases),
        "private_prediction_record_sha256": sha256(private).hexdigest(),
        "driver_provenance": "SOURCE_SELECTED_IN_PROCESS_NOT_HOST_ATTESTED",
        "physical_execution_authenticated": False,
        "independent_evaluator_host_verified": False,
        "graduation_authorized": False,
    })
    return SignedSourceInference(
        public, private,
        hmac.new(evaluator_key, _PUBLIC_DOMAIN + public, sha256).hexdigest(),
        hmac.new(evaluator_key, _PRIVATE_DOMAIN + private, sha256).hexdigest(),
    )


def verify_source_inference_receipt(receipt: SignedSourceInference, *,
                                    intent: EvaluatorInferenceIntent,
                                    evaluator_key: bytes,
                                    expected_cases: tuple[PrivateInferenceCase, ...],
                                    roster_key: bytes,
                                    externally_pinned_roster_commitment: str,
                                    externally_pinned_candidate_component_binding_sha256: str,
                                    ) -> tuple[tuple[str, int], ...]:
    """Verify evidence bytes, not host execution; keep results evaluator-side."""
    _intent(intent)
    _keys(evaluator_key, roster_key)
    _hex(externally_pinned_roster_commitment, "external roster")
    _hex(externally_pinned_candidate_component_binding_sha256,
         "external candidate-component binding")
    expected_cases = _cases(expected_cases)
    roster = evaluator_roster_commitment(expected_cases, roster_key=roster_key)
    if (not hmac.compare_digest(roster, intent.roster_commitment)
            or not hmac.compare_digest(roster,
                                      externally_pinned_roster_commitment)):
        raise InferenceSourceError("evaluator expected case roster not independently pinned")
    if (type(receipt) is not SignedSourceInference
            or receipt.physical_execution_authenticated is not False
            or receipt.independent_evaluator_host_verified is not False
            or receipt.graduation_authorized is not False):
        raise InferenceSourceError("untrusted or elevated inference receipt")
    for data, signed, domain in ((receipt.public_bytes, receipt.public_mac_hex,
                                 _PUBLIC_DOMAIN),
                                (receipt.private_bytes, receipt.private_mac_hex,
                                 _PRIVATE_DOMAIN)):
        if (type(data) is not bytes or not 0 < len(data) <= 262144
                or type(signed) is not str or _HEX.fullmatch(signed) is None
                or not hmac.compare_digest(
                    hmac.new(evaluator_key, domain + data, sha256).hexdigest(),
                    signed)):
            raise InferenceSourceError("evaluator receipt authenticity failed")
    try:
        public = json.loads(receipt.public_bytes)
        private = json.loads(receipt.private_bytes)
        if (_canonical(public) != receipt.public_bytes
                or _canonical(private) != receipt.private_bytes):
            raise ValueError("noncanonical")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise InferenceSourceError("invalid canonical inference evidence") from exc
    if (public.get("schema") != SCHEMA + ".public.v1"
            or private.get("schema") != SCHEMA + ".private.v1"
            or public.get("private_prediction_record_sha256")
            != sha256(receipt.private_bytes).hexdigest()):
        raise InferenceSourceError("private/public inference record mismatch")
    common = {
        "exam_id", "event_id", "run_id", "arm", "seed", "steps_per_cycle",
        "stage", "candidate_artifact_sha256", "prior_skill_checkpoint_sha256",
        "scope_sha256", "source_sha256", "roster_commitment",
        "component_set_sha256", "candidate_component_binding_sha256",
        "pretrained_sha256", "adapter_sha256",
        "head_sha256", "driver_provenance", "physical_execution_authenticated",
        "independent_evaluator_host_verified", "graduation_authorized",
    }
    if (type(public) is not dict or type(private) is not dict
            or set(public) != common | {"schema", "case_count",
                                            "private_prediction_record_sha256"}
            or set(private) != common | {"schema", "predictions_derived_from_logits"}):
        raise InferenceSourceError("unrecognized or missing source evidence field")
    for key in ("exam_id", "event_id", "run_id", "arm", "seed",
                "steps_per_cycle", "stage", "candidate_artifact_sha256",
                "prior_skill_checkpoint_sha256", "scope_sha256", "source_sha256",
                "roster_commitment", "component_set_sha256",
                "candidate_component_binding_sha256",
                "pretrained_sha256", "adapter_sha256",
                "head_sha256"):
        if public.get(key) != private.get(key):
            raise InferenceSourceError("private/public inference identity mismatch")
        if hasattr(intent, key) and public.get(key) != getattr(intent, key):
            raise InferenceSourceError("inference receipt differs from pinned intent")
    components = {
        "pretrained": public["pretrained_sha256"],
        "adapter": public["adapter_sha256"],
        "head": public["head_sha256"],
    }
    for name, value in components.items():
        _hex(value, name)
    computed_component_set = sha256(
        b"labyrinth.evaluator-component-set.v1\0" + _canonical(components)
    ).hexdigest()
    if not hmac.compare_digest(computed_component_set,
                               intent.component_set_sha256):
        raise InferenceSourceError("inference model component set is inconsistent")
    computed_candidate_binding = candidate_component_binding_sha256(
        intent.candidate_artifact_sha256, computed_component_set)
    if (not hmac.compare_digest(computed_candidate_binding,
                               public["candidate_component_binding_sha256"])
            or not hmac.compare_digest(computed_candidate_binding,
                                      externally_pinned_candidate_component_binding_sha256)):
        raise InferenceSourceError("inference candidate-to-components mapping not independently pinned")
    for envelope in (public, private):
        if (envelope.get("physical_execution_authenticated") is not False
                or envelope.get("independent_evaluator_host_verified") is not False
                or envelope.get("graduation_authorized") is not False
                or envelope.get("driver_provenance")
                != "SOURCE_SELECTED_IN_PROCESS_NOT_HOST_ATTESTED"):
            raise InferenceSourceError("source evidence claims unsupported authority")
    rows = private.get("predictions_derived_from_logits")
    if (type(rows) is not list or len(rows) != public.get("case_count")
            or len(rows) != len(expected_cases)):
        raise InferenceSourceError("inference prediction roster incomplete")
    predictions: list[tuple[str, int]] = []
    ids: set[str] = set()
    for expected_case, row in zip(expected_cases, rows):
        if type(row) is not dict or set(row) != {
            "case_id", "logits_float_hex", "predicted_label"}:
            raise InferenceSourceError("inference row malformed")
        _ident(row["case_id"], "prediction case")
        if row["case_id"] != expected_case.case_id or row["case_id"] in ids:
            raise InferenceSourceError("prediction case differs from pinned ordered roster")
        ids.add(row["case_id"])
        values = row["logits_float_hex"]
        if (type(values) is not list or len(values) != 2
                or any(type(value) is not str for value in values)):
            raise InferenceSourceError("malformed detached logits")
        try:
            logits = _logits(tuple(float.fromhex(value) for value in values))
        except ValueError as exc:
            raise InferenceSourceError("invalid detached logits") from exc
        if (type(row["predicted_label"]) is not int
                or row["predicted_label"] != int(logits[1] > logits[0])):
            raise InferenceSourceError("prediction label was not derived from logits")
        predictions.append((row["case_id"], row["predicted_label"]))
    return tuple(predictions)
