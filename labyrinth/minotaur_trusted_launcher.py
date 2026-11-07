"""Trusted-host, configuration-only entry point for Minotaur examinations.

No candidate-supplied Python factory, adapter, callback or arbitrary command
crosses the host-side evaluator stack in this path. The only candidate code
is the fixed child-process adapter selected by an exact, primitive-only config.

The legacy `run_cross_exam(candidate_factory=...)` remains available for
development but its Python factory MUST be trusted host code: even if passed
only a public mapping, it can inspect caller frames and steal private ExamSpec.

This path does NOT authenticate a local model, prevent same-OS-user filesystem
access, establish independent challenge custody, prove model-seed independence,
or establish matched stochastic controls. Frozen dataclasses and strict types
are host validation conveniences, not defenses against malicious Python code
already running in the host interpreter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import secrets
from typing import Any, Mapping

from .minotaur_cross_exam import CrossCaseReceipt, ExamSpec, run_cross_exam
from .minotaur_candidate_bridge import RunReceipt, TurnReceipt
from .minotaur_process_adapter import ProcessCallReceipt, ProcessModelAdapter


_MODES = frozenset({"fixed", "probe", "ollama"})
_PUBLIC_VERDICTS = frozenset({
    "UNKNOWN", "VERIFIED_WIN", "VERIFIED_FAILURE", "NO_KNOWN_WIN",
    "PROVEN_UNSAT_UNDER_DECLARED_RULES", "NOVEL_RULESET_CHALLENGE",
    "UNSOLVED", "INVALID_TRACE", "EXHAUSTED", "REACHED_EXIT_UNCERTIFIED",
    "REPLAY_DIVERGENCE", "EVALUATOR_ERROR", "EVALUATOR_DISAGREEMENT",
    "MODEL_CALL_ERROR", "MODEL_OUTPUT_INVALID", "ACTION_BUDGET_EXCEEDED",
    "IN_PROGRESS", "NOT_STARTED",
})


@dataclass(frozen=True, slots=True)
class ProcessCandidateConfig:
    """Primitive-only model declaration; never a Python code plugin/factory.

    A fixed generation seed supports narrow fixture reproducibility, not
    statistical independence. With None, a fresh host CSPRNG seed is requested
    per case but external RNG authority and exact runtime weights are NOT
    attested. Neither path claims matched candidate/control stochastic draws.
    """

    mode: str = "probe"
    actions: tuple[str, ...] = ()
    generation_seed: int | None = None
    timeout_seconds: float = 5.0
    max_output_bytes: int = 8192
    model: str | None = None
    base_url: str = "http://127.0.0.1:11434"
    authorize_local_model: bool = False

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True, slots=True)
class TrustedCaseReceipt:
    """HOST-PRIVATE receipt: wraps the legacy case's secret generation seed.

    Never pass, publish or serialize this object to a candidate. For a
    deliberately narrow after-exam summary, use candidate_public_summary().
    """

    cross_case: CrossCaseReceipt
    process_calls: tuple[ProcessCallReceipt, ...]
    candidate_kind: str
    model_seed_source: str
    model_authentication: str = "UNATTESTED"
    private_custody: str = "UNQUALIFIED_SAME_OS_USER"
    host_factory: str = "TRUSTED_CONFIG_ONLY_NO_CANDIDATE_CALLBACK"
    stochastic_comparison_matched: bool = False
    neural_training_performed: bool = False
    receipt_visibility: str = "HOST_PRIVATE_NOT_FOR_CANDIDATE"


def _validate_config(config: ProcessCandidateConfig) -> None:
    if type(config) is not ProcessCandidateConfig:
        raise TypeError("candidate configuration must be an exact ProcessCandidateConfig")
    if type(config.mode) is not str or config.mode not in _MODES:
        raise ValueError("trusted launcher supports fixed, probe, or ollama mode only")
    if type(config.actions) is not tuple or any(type(a) is not str for a in config.actions):
        raise ValueError("actions must be a primitive tuple of strings")
    if config.mode != "fixed" and config.actions:
        raise ValueError("only fixed no-update controls may carry an action script")
    if (config.generation_seed is not None and
            (type(config.generation_seed) is not int or
             not 0 <= config.generation_seed < 2**31)):
        raise ValueError("generation_seed must be a bounded integer or None")
    if type(config.timeout_seconds) not in (int, float):
        raise ValueError("timeout must be a primitive number")
    if type(config.max_output_bytes) is not int:
        raise ValueError("max_output_bytes must be an integer")
    if type(config.authorize_local_model) is not bool:
        raise ValueError("authorize_local_model must be a boolean")
    if type(config.base_url) is not str:
        raise ValueError("base_url must be a primitive string")
    if config.model is not None and type(config.model) is not str:
        raise ValueError("model must be primitive text")
    if config.mode == "ollama":
        if not config.authorize_local_model:
            raise ValueError("local Ollama mode requires explicit authorization")
    elif config.authorize_local_model or config.model is not None:
        raise ValueError("model authorization/name is restricted to Ollama mode")
    # Reuse the adapter's own bounds and loopback URL validation. This does
    # NOT contact Ollama, start a process, or authenticate model artifacts.
    ProcessModelAdapter(
        mode=config.mode, actions=config.actions, generation_seed=0,
        timeout_seconds=config.timeout_seconds,
        max_output_bytes=config.max_output_bytes, model=config.model,
        base_url=config.base_url, authorize_local_model=config.authorize_local_model)


def _validate_specs(specs: tuple[ExamSpec, ...]) -> None:
    if type(specs) is not tuple or not 1 <= len(specs) <= 64:
        raise ValueError("exam specs must be a bounded exact tuple")
    for spec in specs:
        if type(spec) is not ExamSpec:
            raise TypeError("examiner specs must be exact host ExamSpec instances")
        if (type(spec.family) is not str or type(spec.seed) is not int
                or type(spec.variant) is not int or type(spec.sealed) is not bool
                or type(spec.split) is not str or type(spec.weak_cell) is not bool):
            raise ValueError("examiner fields must be exact primitive types")


def run_trusted_cross_exam(specs: tuple[ExamSpec, ...],
                           config: ProcessCandidateConfig) -> tuple[TrustedCaseReceipt, ...]:
    """Run public exams without accepting ANY untrusted host-side callback.

    This module supplies its OWN trusted candidate_factory closure to the
    legacy runner, never a callback passed by a caller or model. The child
    still runs under the same OS user, so physical custody is unqualified.
    """
    _validate_config(config)  # also recheck object.__setattr__ field mutations
    _validate_specs(specs)
    adapters: list[ProcessModelAdapter] = []
    kind = ("DETERMINISTIC_ZERO_UPDATE_CONTROL" if config.mode in ("fixed", "probe")
            else "LOCAL_MODEL_REQUEST_UNATTESTED")
    seed_source = ("FIXED_CONFIG_UNATTESTED" if config.generation_seed is not None
                   else "HOST_CSPRNG_REQUEST_UNATTESTED")

    def trusted_host_factory(_public_view: Mapping[str, Any]) -> ProcessModelAdapter:
        # Deliberately neither inspect frames nor execute any caller logic.
        # This Python function is owned by this trusted module, and an
        # untrusted candidate cannot supply or override it through this API.
        model_seed = (config.generation_seed if config.generation_seed is not None
                      else secrets.randbelow(2**31))
        adapter = ProcessModelAdapter(
            mode=config.mode, actions=config.actions,
            generation_seed=model_seed, timeout_seconds=config.timeout_seconds,
            max_output_bytes=config.max_output_bytes, model=config.model,
            base_url=config.base_url, authorize_local_model=config.authorize_local_model)
        adapters.append(adapter)
        return adapter

    # The legacy engine runs its baseline and examiner in the host; only our
    # own trusted closure runs there for the candidate. Each model call is in
    # a fresh child process. Permission flag is scoped to this trusted adapter.
    cases = run_cross_exam(specs, candidate_factory=trusted_host_factory,
                           authorize_local_adapter=True,
                           trust_host_python_factory=True)
    if len(adapters) != len(cases):
        raise RuntimeError("trusted adapter/examiner case count mismatch")
    return tuple(TrustedCaseReceipt(
        cross_case=case, process_calls=adapter.receipts,
        candidate_kind=kind, model_seed_source=seed_source)
        for case, adapter in zip(cases, adapters))


def candidate_public_summary(
    receipts: tuple[TrustedCaseReceipt, ...],
) -> tuple[dict[str, Any], ...]:
    """Minimal after-evaluation summary with no case seed or raw receipts.

    Intended only AFTER a candidate has finished all evaluated instances;
    it does not attest release authorization, private custody, or inference.
    It intentionally excludes host-private case identity, model RNG seeds,
    prompts, transcripts, hashes and examiner session/certificate internals.
    """
    if type(receipts) is not tuple or any(type(r) is not TrustedCaseReceipt for r in receipts):
        raise TypeError("public summaries require a tuple of exact trusted receipts")
    result = []
    for item in receipts:
        _validate_host_receipt_structure(item)
        case = item.cross_case
        summary = {
            "family": case.family,
            # These are HOST-REPORTED, self-hashed claims. Not independent
            # verification, model authentication, or graduation evidence.
            "host_reported_candidate_verdict": case.candidate.final_verdict,
            "host_reported_candidate_escape": case.candidate.verified_escape,
            "host_reported_baseline_escape": case.baseline_escape_verified,
            "candidate_kind": item.candidate_kind,
            "evidence_status": "UNATTESTED_HOST_SELF_HASH",
        }
        # Keep this boundary explicitly JSON primitive-only, so no Python
        # reference to CrossCaseReceipt or a host-held source object escapes.
        result.append(json.loads(json.dumps(summary, allow_nan=False)))
    return tuple(result)


def _validate_host_receipt_structure(item: TrustedCaseReceipt) -> None:
    """Reject malformed/altered local receipts, WITHOUT claiming custody.

    Self-hash verification only detects inconsistent serialized fields: a
    hostile host can reconstruct a new self-hash. Private examiner replay and
    independently signed authority are intentionally NOT asserted here.
    """
    case = item.cross_case
    if type(case) is not CrossCaseReceipt:
        raise TypeError("cross-family receipt must be an exact host record")
    if (type(case.family) is not str or case.family not in ("constant", "diverse", "timegate", "oracle")
            or type(case.seed) is not int
            or case.baseline_kind != "ZERO_UPDATE_PUBLIC_ONLY"
            or type(case.baseline_escape_verified) is not bool
            or type(case.baseline_policy_verified) is not bool
            or type(case.baseline_certificate_verified) is not bool
            or type(case.candidate) is not RunReceipt
            or type(case.baseline) is not RunReceipt):
        raise ValueError("cross-family receipt has inconsistent public fields")
    if type(item.candidate_kind) is not str or item.candidate_kind not in (
            "DETERMINISTIC_ZERO_UPDATE_CONTROL", "LOCAL_MODEL_REQUEST_UNATTESTED"):
        raise ValueError("candidate kind must be a declared primitive category")
    if (type(item.process_calls) is not tuple or
            any(type(call) is not ProcessCallReceipt for call in item.process_calls)
            or len(item.process_calls) != len(case.candidate.turns)):
        raise ValueError("candidate process receipts do not match model call count")
    if (item.receipt_visibility != "HOST_PRIVATE_NOT_FOR_CANDIDATE"
            or item.model_authentication != "UNATTESTED"
            or item.private_custody != "UNQUALIFIED_SAME_OS_USER"
            or item.host_factory != "TRUSTED_CONFIG_ONLY_NO_CANDIDATE_CALLBACK"
            or item.model_seed_source not in ("FIXED_CONFIG_UNATTESTED",
                                              "HOST_CSPRNG_REQUEST_UNATTESTED")
            or item.stochastic_comparison_matched is not False
            or item.neural_training_performed is not False):
        raise ValueError("unqualified host evidence cannot be upgraded")
    for run in (case.candidate, case.baseline):
        if (type(run.turns) is not tuple or
                any(type(turn) is not TurnReceipt for turn in run.turns)
                or type(run.verified_escape) is not bool
                or type(run.final_verdict) is not str
                or run.final_verdict not in _PUBLIC_VERDICTS):
            raise ValueError("candidate or baseline run is malformed")
        fields = asdict(run)
        claimed_sha = fields.pop("receipt_sha256")
        expected_sha = hashlib.sha256(json.dumps(
            fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        if type(claimed_sha) is not str or claimed_sha != expected_sha:
            raise ValueError("run receipt self-hash does not match its fields")
    if (case.candidate.verified_escape and case.candidate.final_verdict != "VERIFIED_WIN"):
        raise ValueError("escape and verdict claims disagree")
    if case.baseline_escape_verified != case.baseline.verified_escape:
        raise ValueError("baseline escape flag does not match its run")
