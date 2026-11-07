"""Precommitted, bounded, source-public Switchback multi-case process trial.

Every listed case remains in the denominator, including skipped/error cases.
Only trusted built-in process adapters are constructed (no arbitrary candidate
factory in the examiner). Public source means NO private holdout, model-weight
update, learning, semantic independence or graduation qualification.

Example zero-update control:
  python3 -m scripts.switchback_batch_trial --seeds 0 1 2 --mode fixed \
      --fixed-actions 0 RESET --max-calls-per-case 3 --max-total-calls 9 \
      --max-total-seconds 15

No model, service, network or host resource is contacted at import time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from labyrinth.minotaur_process_adapter import ProcessModelAdapter
from labyrinth.switchback_circuit import MAX_MOVES, MAX_PROBES, make_case
from labyrinth.switchback_oracle import verify_receipt_replay
from scripts.switchback_model_trial import (
    CommandProcessAdapter, TrialConfig, _canonical, _git_source_commit,
    run_trial,
)


SCHEMA = "labyrinth.switchback-batch-process.v1"
PUBLIC_SUMMARY_SCHEMA = "labyrinth.switchback-batch-public-summary.v1"
MAX_CASES = 32
MAX_BATCH_CALLS = MAX_CASES * (MAX_PROBES + 1)
CASE_ADMISSION_OVERHEAD_SECONDS = 1.0


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _source_commit_shape_valid(source: object) -> bool:
    """Check claimed source metadata shape; this is NOT Git attestation."""
    if type(source) is not dict or set(source) != {"sha", "worktree_dirty"}:
        return False
    rev = source["sha"]
    dirty = source["worktree_dirty"]
    if rev is None:
        return dirty is None
    return (type(rev) is str and len(rev) == 40 and
            all(char in "0123456789abcdef" for char in rev) and
            type(dirty) is bool)


def _batch_decision_from_wire(raw: str, max_output_bytes: int) -> dict[str, object]:
    """Reparse child bytes without trusting the lower runner's decision field.

    This independently interprets an already bounded wire response. It does
    not attest an OS process, private model, or signed custody.
    """
    if type(raw) is not str or len(raw.encode("utf-8")) > max_output_bytes:
        raise ValueError("unbounded child response")

    def unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
        obj: dict[str, object] = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate child JSON key")
            obj[key] = value
        return obj

    def reject_nonfinite(_raw: str) -> object:
        raise ValueError("nonfinite child JSON")

    def bounded_float(raw_float: str) -> float:
        number = float(raw_float)
        if not math.isfinite(number):
            raise ValueError("nonfinite child JSON float")
        return number

    parsed = json.JSONDecoder(object_pairs_hook=unique_fields,
                              parse_constant=reject_nonfinite,
                              parse_float=bounded_float).decode(raw)
    if type(parsed) is not dict or type(parsed.get("kind")) is not str:
        raise ValueError("child decision must be an object with kind")
    kind = parsed["kind"]
    if kind == "ACTION":
        if (set(parsed) != {"kind", "action"} or
                type(parsed["action"]) is not str or
                parsed["action"] not in ("0", "1", "WAIT", "RESET")):
            raise ValueError("bad raw child action")
    elif kind == "PLAN":
        if (set(parsed) != {"kind", "actions"} or
                type(parsed["actions"]) is not list or
                not 1 <= len(parsed["actions"]) <= MAX_MOVES or
                any(type(item) is not str or
                    item not in ("0", "1", "WAIT", "COMMIT")
                    for item in parsed["actions"])):
            raise ValueError("bad raw child plan")
    elif kind in ("ABSTAIN", "CLAIM_UNSAT"):
        if set(parsed) != {"kind"}:
            raise ValueError("bad raw child terminal claim")
    elif kind == "NOVEL_RULE":
        if (set(parsed) != {"kind", "description"} or
                type(parsed["description"]) is not str or
                not 0 < len(parsed["description"]) <= 2000):
            raise ValueError("bad raw child rule narrative")
    elif kind == "CLAIM":
        if (set(parsed) != {"kind", "claim"} or
                type(parsed["claim"]) is not dict or
                parsed["claim"] != {"kind": "ABSTAIN", "assessment": "UNKNOWN"}):
            raise ValueError("bad raw child structured claim")
        return {"kind": "ABSTAIN"}
    else:
        raise ValueError("unsupported raw child decision")
    return parsed


@dataclass(frozen=True)
class BatchConfig:
    case_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    max_calls_per_case: int = 3
    max_total_calls: int = 16
    call_timeout_seconds: float = 1.0
    max_total_seconds: float = 40.0
    max_output_bytes: int = 8192

    def __post_init__(self) -> None:
        if (type(self.case_seeds) is not tuple or
                not 1 <= len(self.case_seeds) <= MAX_CASES or
                any(type(seed) is not int or not 0 <= seed < 2**31
                    for seed in self.case_seeds)):
            raise ValueError("precommitted case seeds must be 1..32 exact bounded integers")
        if len(set(self.case_seeds)) != len(self.case_seeds):
            raise ValueError("duplicate seeds cannot inflate distinct-case evidence")
        if (type(self.generation_seeds) is not tuple or
                len(self.generation_seeds) != len(self.case_seeds) or
                any(type(seed) is not int or not 0 <= seed < 2**31
                    for seed in self.generation_seeds)):
            raise ValueError("one exact generation seed per precommitted case required")
        if (type(self.max_calls_per_case) is not int or
                not 1 <= self.max_calls_per_case <= MAX_PROBES + 1 or
                type(self.max_total_calls) is not int or
                not 1 <= self.max_total_calls <= MAX_BATCH_CALLS):
            raise ValueError("invalid global/per-case bounded model call budget")
        if (type(self.call_timeout_seconds) not in (int, float) or
                not math.isfinite(self.call_timeout_seconds) or
                not .01 <= self.call_timeout_seconds <= 10 or
                self.call_timeout_seconds * self.max_calls_per_case > 179):
            raise ValueError("per-call timeout and per-case horizon exceed bounds")
        if (type(self.max_total_seconds) not in (int, float) or
                not math.isfinite(self.max_total_seconds) or
                not .01 <= self.max_total_seconds <= 3600):
            raise ValueError("global reservation time must be .01..3600 seconds")
        if (type(self.max_output_bytes) is not int or
                not 128 <= self.max_output_bytes <= 65536):
            raise ValueError("invalid bounded candidate output bytes")


@dataclass(frozen=True)
class AdapterSpec:
    mode: str = "fixed"
    fixed_actions: tuple[str, ...] = ()
    command_argv: tuple[str, ...] = ()
    ollama_model: str | None = None
    model_artifact_path: str | None = None
    explicit_authorization: bool = False

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in ("fixed", "command", "ollama"):
            raise ValueError("only built-in process modes are supported")
        if (type(self.fixed_actions) is not tuple or len(self.fixed_actions) > 32 or
                any(type(action) is not str or not 1 <= len(action) <= 64
                    for action in self.fixed_actions) or
                type(self.command_argv) is not tuple or
                any(type(part) is not str for part in self.command_argv)):
            raise ValueError("adapter source configuration must be immutable bounded text")
        if (self.model_artifact_path is not None and
                (type(self.model_artifact_path) is not str or
                 not self.model_artifact_path)):
            raise ValueError("model artifact path must be explicit text")
        if type(self.explicit_authorization) is not bool:
            raise ValueError("authorization must be an exact bool")
        if self.mode == "fixed":
            if self.command_argv or self.ollama_model is not None:
                raise ValueError("frozen control cannot bundle a model/command")
        elif self.mode == "command":
            if (not self.explicit_authorization or not self.command_argv or
                    self.fixed_actions or self.ollama_model is not None):
                raise PermissionError("command requires explicit authorization and argv")
        else:
            if (not self.explicit_authorization or self.command_argv or
                    self.fixed_actions or type(self.ollama_model) is not str or
                    not self.ollama_model):
                raise PermissionError("Ollama requires explicit local opt-in and model name")


def _make_adapter(spec: AdapterSpec, generation_seed: int,
                  timeout: float, max_output: int
                  ) -> ProcessModelAdapter | CommandProcessAdapter:
    # Trusted, source-owned construction; never call a model-supplied factory.
    if spec.mode == "fixed":
        return ProcessModelAdapter(mode="fixed", actions=spec.fixed_actions,
                                   generation_seed=generation_seed,
                                   timeout_seconds=timeout,
                                   max_output_bytes=max_output)
    if spec.mode == "command":
        return CommandProcessAdapter(spec.command_argv,
                                     generation_seed=generation_seed,
                                     authorize_command=True,
                                     timeout_seconds=timeout,
                                     max_output_bytes=max_output)
    return ProcessModelAdapter(mode="ollama", model=spec.ollama_model,
                               authorize_local_model=True,
                               generation_seed=generation_seed,
                               timeout_seconds=timeout,
                               max_output_bytes=max_output)


def _wire_turn_valid(turn: object, mode: str, generation_seed: int,
                     max_output_bytes: int) -> bool:
    """Reconstruct bounded child stdout independently of its claimed SHA.

    A verified digest still has same-user source custody, not a cryptographic
    signature or OS measurement. Fail closed when the lower-level trial did not
    preserve enough source-public bytes to reconstruct the child output.
    """
    if type(turn) is not dict or type(turn.get("process_receipts")) is not list or \
            len(turn["process_receipts"]) != 1:
        return False
    receipt = turn["process_receipts"][0]
    if (type(receipt) is not dict or
            receipt.get("schema") != "labyrinth.minotaur-process.v1" or
            type(receipt.get("generation_seed")) is not int or
            receipt["generation_seed"] != generation_seed or
            receipt.get("evaluator_seed_discarded") is not True or
            type(receipt.get("request_sha256")) is not str or
            len(receipt["request_sha256"]) != 64 or
            type(receipt.get("response_bytes")) is not int or
            not 0 <= receipt["response_bytes"] <= max_output_bytes or
            type(receipt.get("elapsed_ms")) is not int or
            receipt["elapsed_ms"] < 0 or
            receipt.get("resource_scope") != "PER_INTERPRETER_NOT_HOST_GLOBAL" or
            receipt.get("generation_seed_independence") != "UNATTESTED" or
            receipt.get("boundary") !=
            "SEPARATE_PYTHON_PROCESS_SAME_OS_USER_NOT_SANDBOXED"):
        return False
    response = turn.get("raw_response")
    if response is None:
        # Failed calls never support success. They have no inspectable raw
        # response; mark this particular trial unqualified for wire replay.
        return False
    if (type(response) is not str or len(response.encode("utf-8")) > max_output_bytes or
            receipt.get("status") != "OK" or
            type(receipt.get("child_pid")) is not int or
            receipt["child_pid"] <= 0 or
            type(turn.get("response_sha256")) is not str or
            turn["response_sha256"] != hashlib.sha256(
                response.encode("utf-8")).hexdigest()):
        return False
    if mode == "command":
        if receipt.get("parent_session_frame_visible") is not None:
            return False
        child_wire = response.encode("utf-8")
        if receipt["request_sha256"] != turn.get("prompt_sha256"):
            return False
    else:
        if receipt.get("parent_session_frame_visible") is not False:
            return False
        child_wire = json.dumps({
            "schema": "labyrinth.minotaur-process.v1",
            "response": response,
            "child_pid": receipt["child_pid"],
            "parent_session_frame_visible": False,
        }, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    if (receipt.get("response_bytes") != len(child_wire) or
            receipt.get("response_sha256") != hashlib.sha256(child_wire).hexdigest()):
        return False
    return True


def _trial_reverified(raw: object, case_seed: int, generation_seed: int,
                      adapter_spec: AdapterSpec, authorized_calls: int,
                      timeout: float, max_output_bytes: int) -> tuple[bool, str]:
    """Demand independent source proof, THIS case, THIS model transcript.

    The lower-level runner is same-interpreter source code, not authenticated
    process custody. Batch checks prevent trivial 100%-claim injection and
    reused receipts, not arbitrary compromise of the final trusted verifier.
    """
    if type(raw) is not dict or raw.get("schema") != "labyrinth.switchback-real-model-trial.v1":
        return False, "NOT_A_PROCESS_TRIAL_RECEIPT"
    if (type(raw.get("case_seed_host_only")) is not int or
            raw["case_seed_host_only"] != case_seed or
            not _source_commit_shape_valid(raw.get("source_commit")) or
            raw.get("mode") != adapter_spec.mode or
            raw.get("trial_binding_valid") is not True or
            raw.get("source_receipt_replay_valid") is not True or
            raw.get("model_weight_update_verified") is not False or
            raw.get("model_neural_identity_verified") is not False or
            raw.get("generalization_qualified") is not False):
        return False, "TRIAL_CUSTODY_OR_CASE_MISMATCH"
    if (type(raw.get("configuration")) is not dict or
            type(raw["configuration"].get("case_seed")) is not int or
            raw["configuration"]["case_seed"] != case_seed or
            type(raw["configuration"].get("generation_seed")) is not int or
            raw["configuration"]["generation_seed"] != generation_seed or
            type(raw["configuration"].get("max_model_calls")) is not int or
            raw["configuration"]["max_model_calls"] != authorized_calls or
            raw["configuration"].get("call_timeout_seconds") != timeout or
            raw["configuration"].get("max_output_bytes") != max_output_bytes or
            raw.get("config_sha256") != _sha(raw["configuration"])):
        return False, "WRONG_FROZEN_TRIAL_CONFIG"
    if type(raw.get("receipt_sha256")) is not str:
        return False, "UNHASHED_PROCESS_TRIAL"
    try:
        if raw["receipt_sha256"] != _sha({key: value for key, value in raw.items()
                                         if key != "receipt_sha256"}):
            return False, "MUTATED_PROCESS_TRIAL_RECEIPT"
    except (ValueError, TypeError, OverflowError, RecursionError):
        return False, "INVALID_PROCESS_RECEIPT_SERIALIZATION"
    turns = raw.get("turns")
    if (type(turns) is not list or len(turns) > authorized_calls or
            type(raw.get("process_calls")) is not int or
            raw["process_calls"] != len(turns)):
        return False, "MODEL_CALL_BUDGET_OR_TRANSCRIPT_MISMATCH"
    observations: list[str] = []
    terminal_decision: dict[str, object] | None = None
    for index, turn in enumerate(turns):
        if (type(turn) is not dict or type(turn.get("turn")) is not int or
                turn["turn"] != index or type(turn.get("process_receipts")) is not list or
                len(turn["process_receipts"]) != 1):
            return False, "MALFORMED_PROCESS_TURN"
        if not _wire_turn_valid(turn, adapter_spec.mode, generation_seed,
                                max_output_bytes):
            return False, "CHILD_PROCESS_WIRE_UNRECONSTRUCTIBLE_OR_MISMATCH"
        decision = turn.get("decision")
        try:
            wire_decision = _batch_decision_from_wire(
                turn["raw_response"], max_output_bytes)
        except (ValueError, TypeError, UnicodeError, OverflowError,
                RecursionError):
            return False, "CHILD_PROCESS_RAW_DECISION_INVALID"
        if (type(decision) is not dict or
                _canonical(decision) != _canonical(wire_decision) or
                turn.get("status") != "PARSED"):
            return False, "CHILD_PROCESS_DECISION_DOES_NOT_MATCH_RAW_BYTES"
        if decision is None:
            continue
        if type(decision) is not dict or type(decision.get("kind")) is not str:
            return False, "INVALID_RECORDED_DECISION"
        if decision["kind"] == "ACTION":
            if (terminal_decision is not None or type(decision.get("action")) is not str or
                    decision["action"] not in ("0", "1", "WAIT", "RESET")):
                return False, "INVALID_RECORDED_ACTION"
            observations.append(decision["action"])
        else:
            if terminal_decision is not None or index != len(turns) - 1:
                return False, "MULTIPLE_OR_NONTERMINAL_CLAIMS"
            terminal_decision = decision
    source = raw.get("source_receipt")
    if (type(source) is not dict or source.get("schema") !=
            "labyrinth.switchback-session-receipt.v1" or
            type(source.get("seed")) is not int or source["seed"] != case_seed or
            source.get("case_id") != make_case(case_seed).case_id or
            type(source.get("probe_trace")) is not list or
            type(source.get("probe_count")) is not int or
            source["probe_count"] != len(observations)):
        return False, "WRONG_SOURCE_CASE_OR_PROBE_COUNT"
    if not verify_receipt_replay(source):
        return False, "FAILED_INDEPENDENT_ORACLE_REPLAY"
    if [record.get("action") for record in source["probe_trace"]
            if type(record) is dict] != observations:
        return False, "MODEL_PROBES_DO_NOT_MATCH_ORACLE_TRACE"
    actual_proposal = {"kind": "ABSTAIN"} if terminal_decision is None else (
        {"kind": "ABSTAIN"} if terminal_decision.get("kind") == "ABSTAIN"
        else terminal_decision)
    if (type(source.get("adjudication")) is not dict or
            _canonical(source["adjudication"].get("proposal")) !=
            _canonical(actual_proposal)):
        return False, "MODEL_TERMINAL_PROPOSAL_MISMATCH"
    if (raw.get("source_verdict") != source.get("verdict") or
            raw.get("verified_success") is not (
                raw.get("terminal_reason") == "SUBMITTED" and
                terminal_decision is not None and terminal_decision.get("kind") == "PLAN" and
                source.get("verified_success") is True)):
        return False, "MODEL_VERDICT_DOES_NOT_MATCH_INDEPENDENT_WITNESS"
    if raw.get("verified_success") is True:
        for turn in turns:
            process_receipt = turn["process_receipts"][0]
            if type(process_receipt) is not dict or process_receipt.get("status") != "OK":
                return False, "WIN_WITH_FAILED_CHILD_PROCESS"
    if adapter_spec.mode == "fixed":
        for turn in turns:
            number = turn["turn"]
            expected = ({"kind": "ACTION", "action": adapter_spec.fixed_actions[number]}
                        if number < len(adapter_spec.fixed_actions) else
                        {"kind": "ABSTAIN"})
            if turn.get("decision") != expected:
                return False, "FROZEN_ZERO_UPDATE_CONTROL_WAS_REWRITTEN"
    return True, "SOURCE_PROCESS_TRIAL_REPLAY_VALID"


def run_batch(config: BatchConfig, adapter_spec: AdapterSpec) -> dict[str, object]:
    if type(config) is not BatchConfig or type(adapter_spec) is not AdapterSpec:
        raise TypeError("only exact immutable batch configuration/adapter spec admitted")
    manifest = {"schema": SCHEMA, "config": asdict(config),
                "adapter": asdict(adapter_spec),
                "seed_order_committed": True, "source_public": True}
    manifest_sha = _sha(manifest)
    remaining_calls = config.max_total_calls
    remaining_reserved_seconds = float(config.max_total_seconds)
    start = time.monotonic()
    records: list[dict[str, object]] = []
    for index, (case_seed, generation_seed) in enumerate(zip(
            config.case_seeds, config.generation_seeds)):
        row: dict[str, object] = {
            "index": index, "case_seed": case_seed,
            "generation_seed": generation_seed,
            "verified_win": False, "trial_receipt_valid": False,
            "trial": None, "reserved_calls": 0, "reserved_seconds": 0.0,
            "admission": "SKIPPED",
        }
        if time.monotonic() - start > config.max_total_seconds:
            row["admission"] = "SKIPPED_GLOBAL_WALL_TIMEOUT"
        elif remaining_calls < config.max_calls_per_case:
            row["admission"] = "SKIPPED_GLOBAL_CALL_BUDGET"
        else:
            reserve = config.max_calls_per_case * config.call_timeout_seconds + \
                CASE_ADMISSION_OVERHEAD_SECONDS
            if remaining_reserved_seconds < reserve:
                row["admission"] = "SKIPPED_GLOBAL_TIME_RESERVATION"
            else:
                # Preallocate against the immutable manifest; do NOT recycle
                # unused calls based on which cases happened to win quickly.
                remaining_calls -= config.max_calls_per_case
                remaining_reserved_seconds -= reserve
                row["reserved_calls"] = config.max_calls_per_case
                row["reserved_seconds"] = reserve
                row["admission"] = "ADMITTED"
                try:
                    single = TrialConfig(
                        case_seed=case_seed, generation_seed=generation_seed,
                        mode=adapter_spec.mode,
                        max_model_calls=config.max_calls_per_case,
                        call_timeout_seconds=config.call_timeout_seconds,
                        total_timeout_seconds=reserve,
                        max_output_bytes=config.max_output_bytes)
                    process_adapter = _make_adapter(
                        adapter_spec, generation_seed, config.call_timeout_seconds,
                        config.max_output_bytes)
                    result = run_trial(single, process_adapter,
                                       model_artifact_path=adapter_spec.model_artifact_path)
                    valid, reason = _trial_reverified(
                        result, case_seed, generation_seed, adapter_spec,
                        config.max_calls_per_case, config.call_timeout_seconds,
                        config.max_output_bytes)
                    row["trial"] = result
                    row["trial_receipt_valid"] = valid
                    row["verified_win"] = valid and result["verified_success"] is True
                    row["admission"] = "COMPLETED" if valid else "INVALID_TRIAL_RECEIPT"
                    row["review_reason"] = reason
                except Exception as error:
                    row["admission"] = "RUNNER_ERROR"
                    row["review_reason"] = "BOUNDED_CASE_EXCEPTION"
                    row["exception_class"] = type(error).__name__
        records.append(row)
    wins = sum(record["verified_win"] is True for record in records)
    completed = sum(record["trial_receipt_valid"] is True for record in records)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "source_commit": _git_source_commit(Path(__file__).resolve().parent.parent),
        "manifest": manifest, "manifest_sha256": manifest_sha,
        "evidence_class": "SOURCE_PUBLIC_PRECOMMITTED_PROCESS_BATCH",
        "private_heldout_qualified": False,
        "model_weight_update_verified": False,
        "neural_learning_verified": False,
        "generalization_qualified": False,
        "comparison_qualified": False,
        "wire_model_attribution": "SAME_OS_USER_UNATTESTED",
        "global_budget_scope": "PRE_ADMISSION_NOT_KERNEL_QUOTA",
        "cases": records,
        "aggregate": {
            "precommitted_case_count": len(config.case_seeds),
            "source_verified_wins": wins,
            "verified_fraction_planned": {"numerator": wins,
                                          "denominator": len(config.case_seeds)},
            "receipts_valid": completed,
            "skipped_or_invalid": len(config.case_seeds) - completed,
            "calls_reserved": config.max_total_calls - remaining_calls,
            "calls_executed": sum(len(row["trial"].get("turns", []))
                                  for row in records if type(row["trial"]) is dict),
            "seconds_reserved": round(config.max_total_seconds -
                                      remaining_reserved_seconds, 6),
            "wall_elapsed_ms": round((time.monotonic() - start) * 1000),
        },
    }
    report["batch_receipt_sha256"] = _sha(report)
    return report


def verify_batch_receipt(batch: object) -> bool:
    if (type(batch) is not dict or batch.get("schema") != SCHEMA or
            type(batch.get("batch_receipt_sha256")) is not str or
            not _source_commit_shape_valid(batch.get("source_commit")) or
            type(batch.get("manifest")) is not dict or
            type(batch.get("cases")) is not list or
            type(batch.get("aggregate")) is not dict):
        return False
    try:
        if batch["batch_receipt_sha256"] != _sha({
                key: value for key, value in batch.items()
                if key != "batch_receipt_sha256"}):
            return False
        if batch.get("manifest_sha256") != _sha(batch["manifest"]):
            return False
        if (batch.get("evidence_class") !=
                "SOURCE_PUBLIC_PRECOMMITTED_PROCESS_BATCH" or
                batch.get("wire_model_attribution") != "SAME_OS_USER_UNATTESTED" or
                batch.get("global_budget_scope") != "PRE_ADMISSION_NOT_KERNEL_QUOTA" or
                any(batch.get(name) is not False for name in (
                    "private_heldout_qualified", "model_weight_update_verified",
                    "neural_learning_verified", "generalization_qualified",
                    "comparison_qualified"))):
            return False
        manifest = batch["manifest"]
        if (manifest.get("schema") != SCHEMA or
                manifest.get("seed_order_committed") is not True or
                manifest.get("source_public") is not True or
                type(manifest.get("config")) is not dict or
                type(manifest.get("adapter")) is not dict):
            return False
        config = BatchConfig(**{key: tuple(value) if key in (
            "case_seeds", "generation_seeds") else value
            for key, value in manifest["config"].items()})
        adapter_data = dict(manifest["adapter"])
        for key in ("fixed_actions", "command_argv"):
            adapter_data[key] = tuple(adapter_data[key])
        spec = AdapterSpec(**adapter_data)
        rows = batch["cases"]
        if len(rows) != len(config.case_seeds):
            return False
        aggregate = batch["aggregate"]
        if (any(type(aggregate.get(name)) is not int for name in (
                "precommitted_case_count", "source_verified_wins",
                "receipts_valid", "skipped_or_invalid", "calls_reserved",
                "calls_executed", "wall_elapsed_ms")) or
                type(aggregate.get("seconds_reserved")) is not float or
                type(aggregate.get("verified_fraction_planned")) is not dict or
                any(type(aggregate["verified_fraction_planned"].get(name)) is not int
                    for name in ("numerator", "denominator"))):
            return False
        observed_wins = 0
        observed_valid = 0
        reserved_calls = 0
        reserved_seconds = 0.0
        executed_calls = 0
        case_reservation = (config.max_calls_per_case *
                            config.call_timeout_seconds + CASE_ADMISSION_OVERHEAD_SECONDS)
        for index, row in enumerate(rows):
            if (type(row) is not dict or type(row.get("index")) is not int or
                    row["index"] != index or type(row.get("case_seed")) is not int or
                    row["case_seed"] != config.case_seeds[index] or
                    type(row.get("generation_seed")) is not int or
                    row["generation_seed"] != config.generation_seeds[index] or
                    type(row.get("reserved_calls")) is not int or
                    type(row.get("reserved_seconds")) is not float or
                    type(row.get("verified_win")) is not bool):
                return False
            admission = row.get("admission")
            if type(admission) is not str or type(row.get("trial_receipt_valid")) is not bool:
                return False
            admitted = admission in ("COMPLETED", "INVALID_TRIAL_RECEIPT", "RUNNER_ERROR")
            if admitted:
                if (row.get("reserved_calls") != config.max_calls_per_case or
                        row.get("reserved_seconds") != case_reservation or
                        reserved_calls + config.max_calls_per_case > config.max_total_calls or
                        reserved_seconds + case_reservation > config.max_total_seconds + 1e-8):
                    return False
                reserved_calls += config.max_calls_per_case
                reserved_seconds += case_reservation
            else:
                if (admission not in ("SKIPPED_GLOBAL_WALL_TIMEOUT",
                                      "SKIPPED_GLOBAL_CALL_BUDGET",
                                      "SKIPPED_GLOBAL_TIME_RESERVATION") or
                        row.get("reserved_calls") != 0 or
                        row.get("reserved_seconds") != 0.0 or
                        row.get("trial") is not None or
                        row.get("trial_receipt_valid") is not False):
                    return False
                if (admission == "SKIPPED_GLOBAL_CALL_BUDGET" and
                        reserved_calls + config.max_calls_per_case <= config.max_total_calls):
                    return False
                if (admission == "SKIPPED_GLOBAL_TIME_RESERVATION" and
                        (reserved_calls + config.max_calls_per_case > config.max_total_calls or
                         reserved_seconds + case_reservation <= config.max_total_seconds)):
                    return False
            if admission == "COMPLETED" and row.get("trial_receipt_valid") is not True:
                return False
            if admission != "COMPLETED" and row.get("trial_receipt_valid") is not False:
                return False
            if row.get("trial_receipt_valid") is True:
                valid, _ = _trial_reverified(
                    row.get("trial"), config.case_seeds[index],
                    config.generation_seeds[index], spec,
                    config.max_calls_per_case, config.call_timeout_seconds,
                    config.max_output_bytes)
                if not valid or row.get("verified_win") is not (
                        row["trial"].get("verified_success") is True):
                    return False
                observed_valid += 1
                observed_wins += row["verified_win"] is True
            elif row.get("verified_win") is not False:
                return False
            if admission == "INVALID_TRIAL_RECEIPT":
                if type(row.get("trial")) is not dict:
                    return False
                valid, _ = _trial_reverified(
                    row["trial"], config.case_seeds[index],
                    config.generation_seeds[index], spec,
                    config.max_calls_per_case, config.call_timeout_seconds,
                    config.max_output_bytes)
                if valid:
                    return False
            if admission == "RUNNER_ERROR" and row.get("trial") is not None:
                return False
            if type(row.get("trial")) is dict:
                turns = row["trial"].get("turns")
                if type(turns) is list:
                    executed_calls += len(turns)
        return (aggregate.get("precommitted_case_count") == len(rows) and
                aggregate.get("source_verified_wins") == observed_wins and
                aggregate.get("receipts_valid") == observed_valid and
                aggregate.get("skipped_or_invalid") == len(rows) - observed_valid and
                aggregate.get("calls_reserved") == reserved_calls and
                aggregate.get("calls_executed") == executed_calls and
                aggregate.get("seconds_reserved") == round(reserved_seconds, 6) and
                aggregate.get("verified_fraction_planned") == {
                    "numerator": observed_wins, "denominator": len(rows)})
    except (ValueError, TypeError, KeyError, AttributeError, IndexError,
            OverflowError, RecursionError):
        return False


def _public_summary(report: dict[str, object], *, verified: bool,
                    private_report_saved: bool = False,
                    private_report_save_failed: bool = False) -> dict[str, object]:
    """Export only verified, aggregate source-public counts, never case/transcript data.

    The summary is deliberately NOT a substitute for the privately retained
    replayable batch receipt.
    """
    source = report.get("aggregate") if verified else None
    aggregate = source if type(source) is dict else {}
    allowed = ("precommitted_case_count", "source_verified_wins",
               "receipts_valid", "skipped_or_invalid")
    return {
        "schema": PUBLIC_SUMMARY_SCHEMA,
        "evidence_class": "SOURCE_PUBLIC_UNQUALIFIED_SUMMARY",
        "private_report_included": False,
        "private_report_saved": private_report_saved,
        "private_report_save_failed": private_report_save_failed,
        "full_receipt_replayable_from_stdout": False,
        "batch_receipt_sha256": report.get("batch_receipt_sha256") if verified else None,
        "batch_source_replay_valid": verified,
        "aggregate": {key: aggregate.get(key) for key in allowed},
        "neural_learning_verified": False,
        "private_heldout_qualified": False,
    }


def _write_private_report_exclusive(report: dict[str, object], path: str) -> None:
    """Publish a complete file atomically, mode 0600, never replacing a path.

    Creating a same-directory temporary file then linking the finished inode
    to the final path fails if the target already exists (including a symlink).
    This is not a same-UID adversary sandbox or a guarantee about parent dirs.
    """
    if type(path) is not str or not Path(path).is_absolute():
        raise ValueError("private report target must be an absolute path")
    destination = Path(path)
    data = _canonical(report) + b"\n"
    descriptor, staged_path = tempfile.mkstemp(
        prefix=".minotaur-private-", suffix=".tmp", dir=str(destination.parent))
    try:
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(staged_path, destination)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        os.unlink(staged_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--generation-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--mode", choices=("fixed", "command", "ollama"), default="fixed")
    parser.add_argument("--fixed-actions", nargs="*", default=[])
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--authorize-local", action="store_true")
    parser.add_argument("--max-calls-per-case", type=int, default=3)
    parser.add_argument("--max-total-calls", type=int, default=16)
    parser.add_argument("--call-timeout", type=float, default=1.0)
    parser.add_argument("--max-total-seconds", type=float, default=40.)
    parser.add_argument("--max-output-bytes", type=int, default=8192)
    export = parser.add_mutually_exclusive_group()
    export.add_argument("--private-report-file", metavar="ABSOLUTE_PATH",
                        help="write full host-private receipt exclusively as a 0600 file")
    export.add_argument("--emit-private-json-stdout", action="store_true",
                        help="UNSAFE opt-in: print full private receipt on stdout")
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    if args.private_report_file is not None and not Path(
            args.private_report_file).is_absolute():
        parser.error("--private-report-file requires an absolute path")
    seeds = tuple(args.seeds)
    generation = tuple(args.generation_seeds) if args.generation_seeds is not None else tuple(
        index for index in range(len(seeds)))
    config = BatchConfig(seeds, generation,
                         args.max_calls_per_case, args.max_total_calls,
                         args.call_timeout, args.max_total_seconds,
                         args.max_output_bytes)
    spec = AdapterSpec(mode=args.mode, fixed_actions=tuple(args.fixed_actions),
                       command_argv=tuple(args.command), ollama_model=args.model,
                       model_artifact_path=args.model_file,
                       explicit_authorization=args.authorize_local)
    result = run_batch(config, spec)
    verified = verify_batch_receipt(result)
    if args.emit_private_json_stdout:
        print(_canonical(result).decode("utf-8"))
    elif args.private_report_file is not None:
        try:
            _write_private_report_exclusive(result, args.private_report_file)
        except (OSError, ValueError, TypeError):
            print(_canonical(_public_summary(
                result, verified=verified,
                private_report_save_failed=True)).decode("utf-8"))
            return 2
        print(_canonical(_public_summary(
            result, verified=verified,
            private_report_saved=True)).decode("utf-8"))
    else:
        print(_canonical(_public_summary(result, verified=verified)).decode("utf-8"))
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
