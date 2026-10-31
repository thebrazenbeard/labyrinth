"""Guarded opt-in PEFT adapter *resume* seam for a future neural journey.

This module checks the exact predecessor pilot report, local base checkpoint,
PEFT adapter bytes, source identity and LoRA parameter values before returning
a *trainable* resumed adapter. It does not run SGD/MeZO, create output files,
use private evaluation, or grant graduation. The caller must separately bind
fresh train-only MazePlan cases and equal-start controls to the training engine.

The learning context authorizes no effects outside ``LOCAL_WEIGHTS_ONLY``;
the ``authority_ref`` is a label, not a credential or proof of consent. There
is no memory/identity/personality rewrite. Digests detect observable changes
but do not authenticate a hostile filesystem or operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

from .hf_pilot import (
    PilotConfig, SCHEMA, _adapter_manifest, _assert_adapter_unchanged,
    _assert_checkpoint_unchanged, _buffer_digest, _checkpoint_manifest,
    _deps, _device_headroom, _frozen_parameter_digest, _load_base,
    _lora_value_digest, _parameter_digest, _sha_file, _source_state,
    validate_config,
)
from .runner import load_json
from .training_scope import TrainingScopeSpec


_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
SCHEMA_RESUME = "labyrinth.hf-resume-bridge.v1"
MAX_PREDECESSOR_REPORT_BYTES = 8 * 1024 * 1024
MAX_PREDECESSOR_ADAPTER_BYTES = 256 * 1024 * 1024
MAX_PREDECESSOR_ADAPTER_FILES = 64


def _check_digest(value: str, name: str) -> None:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ResumeContext:
    """Operator-owned learning context; declarative, not a signed permission.

    The fingerprint identifies the predeclared training objective, not an
    implemented semantic generator or evaluation rubric. Cycle zero must use
    fresh initialization; this adapter resume seam begins with cycle one.
    """

    scope_fingerprint: str
    cycle_index: int
    predecessor_report_sha256: str
    predecessor_lora_value_sha256: str
    authority_ref: str
    state_family: str = "LOCAL_WEIGHTS_ONLY"
    # Only an explicit cycle-one operator decision may import unrelated
    # legacy v3 LoRA weights as the *initial* state of the declared objective.
    # This does not grant a same-objective learning-lineage claim.
    allow_legacy_unscoped: bool = False

    def __post_init__(self) -> None:
        _check_digest(self.scope_fingerprint, "scope_fingerprint")
        _check_digest(self.predecessor_report_sha256, "predecessor_report_sha256")
        _check_digest(self.predecessor_lora_value_sha256, "predecessor_lora_value_sha256")
        if type(self.cycle_index) is not int or not 1 <= self.cycle_index <= 7:
            raise ValueError("resume cycle_index must be in [1,7]")
        if type(self.authority_ref) is not str or _IDENTIFIER.fullmatch(self.authority_ref) is None:
            raise ValueError("authority_ref must be an opaque bounded identifier, not credentials")
        if self.state_family != "LOCAL_WEIGHTS_ONLY":
            raise ValueError("only LOCAL_WEIGHTS_ONLY adapter continuation is permitted")
        if type(self.allow_legacy_unscoped) is not bool:
            raise ValueError("allow_legacy_unscoped must be an explicit boolean")
        if self.allow_legacy_unscoped and self.cycle_index != 1:
            raise ValueError("legacy cross-scope bootstrap is permitted only at cycle 1")


@dataclass(frozen=True)
class ResumeSource:
    """Validated source and retained hashes (internal integrity, not custody)."""

    model_path: Path
    output_dir: Path
    base_weights: tuple[Path, ...]
    base_manifest: dict[str, dict[str, str]]
    predecessor_run: Path
    predecessor_adapter: Path
    predecessor_adapter_manifest: dict[str, str]
    predecessor_lora_value_sha256: str
    predecessor_frozen_sha256: str
    predecessor_buffers_sha256: str
    predecessor_report_sha256: str
    predecessor_scope_status: str
    same_objective_predecessor: bool
    predecessor_cycle_index: int | None


def _predecessor_scope_binding(report: dict[str, Any], context: ResumeContext) -> tuple[str, bool, int | None]:
    """Check a persisted predecessor's *declared* objective and cycle.

    This checks report integrity against the separately operator-pinned
    report SHA-256. It does not make an unsigned report an independently
    authenticated training attestation.
    """
    prior = report.get("resumed_journey")
    if prior is None:
        if context.cycle_index != 1 or not context.allow_legacy_unscoped:
            raise ValueError("unscoped predecessor needs explicit cycle-1 legacy bootstrap authorization")
        return "LEGACY_UNSCOPED_OPERATOR_BOOTSTRAP_NOT_SAME_OBJECTIVE_PROOF", False, None
    if context.allow_legacy_unscoped:
        raise ValueError("legacy bootstrap authorization cannot be used for scoped predecessor")
    if type(prior) is not dict:
        raise ValueError("predecessor scoped journey receipt is malformed")
    try:
        scope = TrainingScopeSpec.from_dict(prior.get("scope"))
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("predecessor scoped journey declaration is invalid") from exc
    prior_cycle = prior.get("cycle_index")
    if (prior.get("scope_fingerprint") != scope.fingerprint
            or scope.fingerprint != context.scope_fingerprint):
        raise ValueError("predecessor scoped objective differs from declared training objective")
    if type(prior_cycle) is not int or prior_cycle != context.cycle_index - 1:
        raise ValueError("predecessor scoped cycle must be exactly one before the resumed cycle")
    if not 0 <= prior_cycle < scope.max_cycles - 1:
        raise ValueError("predecessor scoped cycle is beyond declared objective budget")
    if prior.get("predecessor_scope_status") not in (
        "NEW_SCOPED_JOURNEY_INITIALIZATION", "SAME_DECLARED_SCOPE_RECORDED",
        "LEGACY_UNSCOPED_OPERATOR_BOOTSTRAP_NOT_SAME_OBJECTIVE_PROOF",
    ):
        raise ValueError("predecessor scoped lineage status is absent or unrecognized")
    if (prior_cycle == 0 and prior["predecessor_scope_status"] != "NEW_SCOPED_JOURNEY_INITIALIZATION"
            or prior_cycle > 0 and prior["predecessor_scope_status"] == "NEW_SCOPED_JOURNEY_INITIALIZATION"):
        raise ValueError("predecessor scoped lineage status contradicts cycle index")
    nested = prior.get("predecessor_resume_receipt")
    if prior_cycle == 0:
        if nested is not None:
            raise ValueError("fresh scoped predecessor cycle zero cannot claim an inherited adapter")
    else:
        # A cycle-1+ report must bind its own resume receipt, not just assert
        # a scope label retrospectively after training. Prior cycle one may
        # itself have begun via an explicitly marked *unscoped* bootstrap.
        if (type(nested) is not dict or nested.get("schema") != SCHEMA_RESUME
                or nested.get("state_family") != "LOCAL_WEIGHTS_ONLY"
                or nested.get("objective_scope_fingerprint") != scope.fingerprint
                or type(nested.get("cycle_index")) is not int
                or nested["cycle_index"] != prior_cycle
                or nested.get("this_cycle_initial_lora_sha256") != report.get("initial_lora_parameter_sha256")):
            raise ValueError("predecessor scoped receipt does not bind its own objective/cycle/init")
        expected_same_objective = prior["predecessor_scope_status"] == "SAME_DECLARED_SCOPE_RECORDED"
        if (type(nested.get("same_objective_predecessor")) is not bool
                or nested["same_objective_predecessor"] != expected_same_objective):
            raise ValueError("predecessor scoped receipt contradicts recorded lineage status")
    return "MATCHED_SCOPED_PREDECESSOR_REPORT_NOT_INDEPENDENTLY_ATTESTED", True, prior_cycle


@dataclass(frozen=True)
class ResumedTrainableAdapter:
    """Loaded model and tokenizer for operator-owned training on new cases."""

    model: Any
    tokenizer: Any
    receipt: dict[str, Any]


def preflight_resume(
    config: PilotConfig, predecessor_run_dir: str | Path, context: ResumeContext,
) -> ResumeSource:
    """Verify predecessor bytes and new output path before loading weights.

    Requires the previous local v3 HF pilot report. This is intentionally
    strict: pilot v2 and PR #60's changed v4 schema are not silently accepted.
    An explicit adapter-metadata migration/requalification is needed later.
    """
    if type(context) is not ResumeContext:
        raise ValueError("expected ResumeContext")
    ResumeContext.__post_init__(context)
    model_path, output_dir, weights = validate_config(config, output_must_be_new=True)
    raw_root = Path(predecessor_run_dir).expanduser()
    if raw_root.is_symlink():
        raise ValueError("predecessor run root must not be a symlink")
    root = raw_root.resolve(strict=True)
    if not root.is_dir() or root == output_dir or root in output_dir.parents or output_dir in root.parents:
        raise ValueError("new output directory must be distinct from predecessor run")
    source_file = root / "pilot_report.json"
    if (source_file.is_symlink() or not source_file.is_file()
            or not 0 < source_file.stat().st_size <= MAX_PREDECESSOR_REPORT_BYTES):
        raise ValueError("prior report must be a regular non-symlink file")
    observed_report_digest = _sha_file(source_file)
    if observed_report_digest != context.predecessor_report_sha256:
        raise ValueError("prior report bytes differ from operator-pinned identity")
    report = load_json(source_file, strict=True)
    if type(report) is not dict or report.get("schema") != SCHEMA:
        raise ValueError("only current v3 LoRA pilot reports can be resumed")
    if (type(report.get("source")) is not dict
            or report["source"].get("source_tree_sha256") != _source_state()["source_tree_sha256"]):
        raise ValueError("predecessor source implementation differs from current source")
    scope_status, same_objective, prior_cycle = _predecessor_scope_binding(report, context)
    prior_config = report.get("config")
    if (type(prior_config) is not dict
            or not all(type(prior_config.get(key)) is type(getattr(config, key))
                       for key in ("model_path", "output_dir", "lora_rank", "lora_alpha", "answer_only"))):
        raise ValueError("predecessor LoRA configuration is missing or malformed")
    if (Path(prior_config["model_path"]).expanduser().resolve(strict=True) != model_path
            or Path(prior_config.get("output_dir", "")).expanduser().resolve(strict=False) != root
            or prior_config["lora_rank"] != config.lora_rank
            or prior_config["lora_alpha"] != config.lora_alpha
            or prior_config["answer_only"] != config.answer_only):
        raise ValueError("base checkpoint or prior LoRA setup does not match next-cycle training")
    actual_base = _checkpoint_manifest(model_path, weights)
    expected_base = {"weights": report.get("base_weights_sha256"),
                     "metadata": report.get("model_metadata_sha256")}
    if actual_base != expected_base:
        raise ValueError("predecessor base weight/tokenizer/config bytes changed")
    adapter_dir = root / "candidate_adapter"
    if adapter_dir.is_symlink():
        raise ValueError("predecessor adapter root must not be a symlink")
    # Bound untrusted legacy run-directory size before digesting adapter files.
    adapter_entries: list[Path] = []
    for entry in adapter_dir.rglob("*"):
        adapter_entries.append(entry)
        if len(adapter_entries) > MAX_PREDECESSOR_ADAPTER_FILES:
            raise ValueError("predecessor adapter exceeds bounded file-count or byte budget")
    if any(entry.is_symlink() or not (entry.is_file() or entry.is_dir())
           for entry in adapter_entries):
        raise ValueError("predecessor adapter contains symlink or nonregular entry")
    adapter_files = [entry for entry in adapter_entries if entry.is_file()]
    if (len(adapter_entries) > MAX_PREDECESSOR_ADAPTER_FILES
            or not adapter_files
            or sum(path.stat().st_size for path in adapter_files) > MAX_PREDECESSOR_ADAPTER_BYTES):
        raise ValueError("predecessor adapter exceeds bounded file-count or byte budget")
    # run_pilot persists PeftModel.save_pretrained(..., safe_serialization=True).
    # An interrupted save can leave a hashed directory with arbitrary files,
    # or a zero-byte tensor file, but no PEFT-loadable candidate checkpoint.
    # Reject this before importing the neural stack or accepting its lineage.
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if (adapter_weights.is_symlink() or not adapter_weights.is_file()
            or adapter_weights.stat().st_size == 0):
        raise ValueError("predecessor requires nonempty adapter_model.safetensors")
    actual_adapters = _adapter_manifest(adapter_dir)
    if actual_adapters != report.get("candidate_adapter_sha256"):
        raise ValueError("predecessor adapter files changed after the recorded training run")
    adapter_config = load_json(adapter_dir / "adapter_config.json", strict=True)
    if (type(adapter_config) is not dict
            or adapter_config.get("peft_type") != "LORA"
            or adapter_config.get("task_type") != "CAUSAL_LM"
            or adapter_config.get("bias") != "none"
            or type(adapter_config.get("r")) is not int
            or type(adapter_config.get("lora_alpha")) is not int
            or adapter_config["r"] != config.lora_rank
            or adapter_config["lora_alpha"] != config.lora_alpha):
        raise ValueError("PEFT adapter config does not match pinned LoRA rank, scaling and task")
    prior_value = report.get("candidate_lora_tensor_value_sha256")
    if prior_value != report.get("candidate_lora_parameter_sha256"):
        raise ValueError("prior candidate trainable/LoRA digest fields disagree")
    frozen = report.get("initial_frozen_parameter_sha256")
    buffers = report.get("initial_persistent_buffers_sha256")
    for value, label in ((prior_value, "predecessor LoRA value"),
                         (frozen, "predecessor frozen parameter"),
                         (buffers, "predecessor persistent buffer")):
        _check_digest(value, label)
    if prior_value != context.predecessor_lora_value_sha256:
        raise ValueError("expected prior LoRA tensor-value digest mismatches report")
    if _sha_file(source_file) != observed_report_digest:
        raise RuntimeError("predecessor report changed during preflight")
    return ResumeSource(
        model_path=model_path, output_dir=output_dir, base_weights=tuple(weights),
        base_manifest=actual_base, predecessor_run=root,
        predecessor_adapter=adapter_dir, predecessor_adapter_manifest=actual_adapters,
        predecessor_lora_value_sha256=prior_value,
        predecessor_frozen_sha256=frozen, predecessor_buffers_sha256=buffers,
        predecessor_report_sha256=observed_report_digest,
        predecessor_scope_status=scope_status,
        same_objective_predecessor=same_objective,
        predecessor_cycle_index=prior_cycle,
    )


def load_trainable_resume(
    config: PilotConfig, predecessor_run_dir: str | Path, context: ResumeContext,
) -> ResumedTrainableAdapter:
    """Reload trainable LoRA and prove initial values equal prior saved state.

    The caller must *not* count this function as an optimization step or a
    completed neural journey. No report/checkpoint/output is written here.
    """
    source = preflight_resume(config, predecessor_run_dir, context)
    torch, auto_model, auto_tokenizer, peft_parts, device = _deps(config.device)
    _device_headroom(torch, device, list(source.base_weights))
    _, peft_model, _ = peft_parts
    base, tokenizer = _load_base(source.model_path, device, torch=torch,
                                 auto_model=auto_model, auto_tokenizer=auto_tokenizer)
    _assert_checkpoint_unchanged(source.model_path, list(source.base_weights), source.base_manifest)
    model = peft_model.from_pretrained(base, str(source.predecessor_adapter),
                                      is_trainable=True, local_files_only=True)
    _assert_adapter_unchanged(source.predecessor_adapter, source.predecessor_adapter_manifest)
    actual_adapter_values = _lora_value_digest(model, torch)
    trainable_values = _parameter_digest(model, torch)
    if (actual_adapter_values != source.predecessor_lora_value_sha256
            or trainable_values != actual_adapter_values):
        raise RuntimeError("resumed trainable LoRA values differ from predecessor checkpoint")
    if (_frozen_parameter_digest(model, torch) != source.predecessor_frozen_sha256
            or _buffer_digest(model, torch) != source.predecessor_buffers_sha256):
        raise RuntimeError("base frozen parameters or persistent buffers changed across resume")
    _assert_adapter_unchanged(source.predecessor_adapter, source.predecessor_adapter_manifest)
    _assert_checkpoint_unchanged(source.model_path, list(source.base_weights), source.base_manifest)
    return ResumedTrainableAdapter(model=model, tokenizer=tokenizer, receipt={
        "schema": SCHEMA_RESUME,
        "state_family": context.state_family,
        "authority_ref": context.authority_ref,
        "objective_scope_fingerprint": context.scope_fingerprint,
        "cycle_index": context.cycle_index,
        "predecessor_cycle_index": source.predecessor_cycle_index,
        "predecessor_scope_status": source.predecessor_scope_status,
        "same_objective_predecessor": source.same_objective_predecessor,
        "allow_legacy_unscoped": context.allow_legacy_unscoped,
        "predecessor_report_sha256": source.predecessor_report_sha256,
        "predecessor_adapter_sha256": source.predecessor_adapter_manifest,
        "previous_cycle_final_lora_sha256": source.predecessor_lora_value_sha256,
        "this_cycle_initial_lora_sha256": actual_adapter_values,
        "frozen_parameter_sha256": source.predecessor_frozen_sha256,
        "persistent_buffers_sha256": source.predecessor_buffers_sha256,
        "base_checkpoint_sha256": source.base_manifest,
        "phase": "TRAINABLE_RESUME_READY_ONLY",
        "optimizer_steps": 0,
        "optimizer_moments": "NOT_RETAINED_BY_THIS_BRIDGE",
        "private_graduation": False,
        "qualification": "NOT_ESTABLISHED",
        "limitations": [
            "Source/parameter digests are integrity observations, not independent attestation.",
            "No train-only new maze was consumed, no optimizer step run, no model persisted.",
            "No memory, identity, permissions, or broader state family was modified.",
            "Scoped lineage is a report-bound operator declaration, not independently authenticated learning evidence.",
        ],
    })
