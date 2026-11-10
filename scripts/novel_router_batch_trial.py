"""Bounded, precommitted Novel Router multi-case source-public process trial.

No model or service starts at import. CLI requires explicit seeds and defaults
to a zero-update Python fixture and redacted summary. External commands require
explicit authorization and an absolute executable. The private full receipt
contains case seeds, raw model stdout and argv: never auto-publish it.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any

from labyrinth import switchback_novel_transfer as novel
from scripts import novel_router_model_trial as single


SCHEMA = "labyrinth.novel-router-precommitted-batch.v1"
MAX_CASES = 32
MAX_BATCH_CALLS = MAX_CASES * (novel.MAX_PROBES + 1)
ADMISSION_OVERHEAD_SECONDS = 1.0
SOURCE_LABEL = "SOURCE_PUBLIC_MULTI_CASE_PROCESS_UNATTESTED"
PRIVATE_LABEL = "HOST_PRIVATE_CASE_SEEDS_AND_RAW_MODEL_OUTPUT"
MANIFEST_KEYS = frozenset(("schema", "config", "adapter",
                           "seed_order_committed", "source_public",
                           "candidate_factory_allowed"))
ROW_KEYS = frozenset(("index", "case_seed", "generation_seed", "admission",
                      "trial", "trial_receipt_valid", "verified_win",
                      "reserved_calls", "reserved_seconds"))
SUMMARY_KEYS = frozenset(("precommitted_case_count", "source_verified_wins",
                          "verified_fraction_planned", "receipts_valid",
                          "skipped_or_invalid", "calls_reserved",
                          "calls_executed", "seconds_reserved", "wall_elapsed_ms"))
REPORT_KEYS = frozenset(("schema", "evidence_class", "report_classification",
                         "source_commit", "manifest", "manifest_sha256",
                         "private_heldout_qualified", "neural_learning_verified",
                         "model_weight_update_verified", "model_identity_qualified",
                         "generalization_qualified", "generation_seed_effect_verified",
                         "comparison_qualified", "global_budget_scope", "cases",
                         "aggregate", "batch_receipt_sha256"))


def _json(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha(data: object) -> str:
    return sha256(_json(data)).hexdigest()


def _exact_number(actual: object, expected: int | float) -> bool:
    return type(actual) is type(expected) and actual == expected


def _plain_receipt_tree(root: object) -> bool:
    """Reject live Python callbacks before digesting or comparing a receipt.

    In-memory source reports use tuples that JSON serializes as arrays. Other
    container/scalar subclasses can redefine equality, and a rehashed lying
    string has previously made the verifier accept the wrong schema. This is
    a Python API trust boundary, not independent provenance authentication.
    """
    pending = [(root, 0, False)]
    active: set[int] = set()
    inspected = 0
    while pending:
        value, depth, leaving = pending.pop()
        if leaving:
            active.remove(id(value))
            continue
        inspected += 1
        if inspected > 250_000 or depth > 96:
            return False
        kind = type(value)
        if value is None or kind in (str, int, float, bool):
            continue
        if kind not in (dict, list, tuple):
            return False
        identity = id(value)
        if identity in active:
            return False
        active.add(identity)
        pending.append((value, depth, True))
        if kind is dict:
            for key, child in value.items():
                if type(key) is not str:
                    return False
                pending.append((child, depth + 1, False))
        else:
            pending.extend((child, depth + 1, False) for child in value)
    return True


@dataclass(frozen=True)
class BatchConfig:
    case_seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    max_calls_per_case: int = 2
    max_total_calls: int = 8
    call_timeout_seconds: float = 2.0
    max_total_seconds: float = 30.0
    max_output_bytes: int = 4096

    def __post_init__(self) -> None:
        if (type(self.case_seeds) is not tuple or
                not 1 <= len(self.case_seeds) <= MAX_CASES or
                any(type(seed) is not int or not 0 <= seed <= novel.MAX_SEED
                    for seed in self.case_seeds) or
                len(set(self.case_seeds)) != len(self.case_seeds)):
            raise ValueError("precommitted 1..32 distinct exact case seeds required")
        if (type(self.generation_seeds) is not tuple or
                len(self.generation_seeds) != len(self.case_seeds) or
                any(type(seed) is not int or not 0 <= seed <= novel.MAX_SEED
                    for seed in self.generation_seeds)):
            raise ValueError("one bounded public generation seed per case required")
        if (type(self.max_calls_per_case) is not int or
                not 1 <= self.max_calls_per_case <= novel.MAX_PROBES + 1 or
                type(self.max_total_calls) is not int or
                not 1 <= self.max_total_calls <= MAX_BATCH_CALLS):
            raise ValueError("invalid per-case/global charged call bound")
        if (type(self.call_timeout_seconds) not in (int, float) or
                not math.isfinite(self.call_timeout_seconds) or
                not .02 <= self.call_timeout_seconds <= 10 or
                type(self.max_total_seconds) not in (int, float) or
                not math.isfinite(self.max_total_seconds) or
                not .02 <= self.max_total_seconds <= 3600 or
                type(self.max_output_bytes) is not int or
                not 128 <= self.max_output_bytes <= single.MAX_OUTPUT):
            raise ValueError("invalid bounded child resources")
        if (self.max_calls_per_case * self.call_timeout_seconds +
                ADMISSION_OVERHEAD_SECONDS > 220):
            raise ValueError("per-case reservation exceeds single-trial horizon")


def _reservation(config: BatchConfig) -> float:
    return (config.max_calls_per_case * config.call_timeout_seconds +
            ADMISSION_OVERHEAD_SECONDS)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate candidate JSON key")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite candidate number")
    return number


def _reject_constant(_: str) -> Any:
    raise ValueError("candidate NaN/Infinity rejected")


def _raw_candidate_decision(raw: bytes) -> dict[str, object]:
    """Independent source-specific JSON grammar, not single's parser helper."""
    value = json.loads(raw.decode("utf-8", "strict"),
                       object_pairs_hook=_strict_pairs,
                       parse_constant=_reject_constant,
                       parse_float=_finite_float)
    if type(value) is not dict or type(value.get("kind")) is not str:
        raise ValueError("candidate did not return an exact JSON decision")
    kind = value["kind"]
    if kind == "ACTION" and set(value) == {"kind", "action"}:
        if type(value["action"]) is str and value["action"] in novel.PROBE_ACTIONS:
            return value
    elif kind == "PLAN" and set(value) == {"kind", "actions"}:
        actions = value["actions"]
        if (type(actions) is list and 1 <= len(actions) <= novel.MAX_MOVES and
                all(type(action) is str and action in novel.PLAN_ACTIONS
                    for action in actions)):
            return value
    elif kind in ("CLAIM_UNSAT", "ABSTAIN") and set(value) == {"kind"}:
        return value
    elif kind == "NOVEL_RULE" and set(value) == {"kind", "description"}:
        if (type(value["description"]) is str and
                0 < len(value["description"]) <= 2000):
            return value
    raise ValueError("unknown/invalid candidate action or terminal")


def _trial_reverified(raw: object, case_seed: int, generation_seed: int,
                      config: BatchConfig, spec: single.AdapterSpec
                      ) -> tuple[bool, str]:
    """Bind source receipt to this manifest entry and actual bounded stdout.

    This is same-interpreter source replay, not an independently signed host.
    """
    if type(raw) is not dict or raw.get("schema") != single.SCHEMA:
        return False, "NOT_A_SOURCE_PROCESS_TRIAL"
    expected = {"case_seed": case_seed, "generation_seed": generation_seed,
                "max_calls": config.max_calls_per_case,
                "call_timeout_seconds": config.call_timeout_seconds,
                "total_timeout_seconds": _reservation(config),
                "max_output_bytes": config.max_output_bytes}
    fields = raw.get("configuration")
    if (type(fields) is not dict or set(fields) != set(expected) or
            any(not _exact_number(fields[key], value)
                for key, value in expected.items()) or
            raw.get("configuration_sha256") != _sha(fields) or
            type(raw.get("adapter")) is not dict or
            _json(raw["adapter"]) != _json(asdict(spec)) or
            raw.get("adapter_sha256") != _sha(raw["adapter"])):
        return False, "WRONG_CASE_GENERATION_OR_ADAPTER_CONFIG"
    if (raw.get("private_custody") is not False or
            raw.get("neural_learning_verified") is not False or
            raw.get("generalization_qualified") is not False or
            raw.get("graduation_eligible") is not False or
            raw.get("model_weight_update_verified") is not False or
            raw.get("model_identity_qualified") is not False or
            raw.get("generation_seed_effect_verified") is not False or
            raw.get("source_receipt_replay_valid") is not True or
            raw.get("evidence_class") != "SOURCE_PUBLIC_LOCAL_PROCESS_UNATTESTED"):
        return False, "UNATTESTED_OR_FORGED_SOURCE_QUALIFICATION"
    try:
        if not single.verify_trial_receipt(raw):
            return False, "SINGLE_TRIAL_SOURCE_REPLAY_FAILS"
        source = raw.get("source_receipt")
        if (type(source) is not dict or
                source.get("schema") != "labyrinth.novel-router-session.v1" or
                type(source.get("adjudication")) is not dict or
                type(source["adjudication"].get("seed")) is not int or
                source["adjudication"]["seed"] != case_seed or
                source["adjudication"].get("case_id") !=
                novel.make_case(case_seed).case_id or
                raw.get("source_verdict") != source.get("verdict") or
                type(raw.get("turns")) is not list or
                len(raw["turns"]) > config.max_calls_per_case):
            return False, "SOURCE_CASE_OR_FINITE_WITNESS_MISMATCH"
        actions: list[str] = []
        terminal: dict[str, object] | None = None
        for index, turn in enumerate(raw["turns"]):
            if (type(turn) is not dict or type(turn.get("turn")) is not int or
                    turn["turn"] != index or type(turn.get("child")) is not dict):
                return False, "MALFORMED_CHARGED_PROCESS_TURN"
            child = turn["child"]
            if (type(child.get("raw_stdout_base64")) is not str or
                    type(child.get("response_bytes")) is not int):
                return False, "MISSING_CHILD_WIRE_BYTES"
            wire = base64.b64decode(child["raw_stdout_base64"], validate=True)
            if (len(wire) > config.max_output_bytes or
                    child["response_bytes"] != len(wire) or
                    child.get("response_sha256") !=
                    (sha256(wire).hexdigest() if wire else None)):
                return False, "CHILD_BYTES_DIGEST_OR_BOUND_DIVERGENCE"
            if spec.mode == "fixture":
                model_text = (spec.fixture_responses[index] if
                              index < len(spec.fixture_responses) else
                              '{"kind":"ABSTAIN"}')
                # A source-owned immutable fixture cannot secretly emit a
                # different answer and acquire neural-model credit.
                if child.get("status") == "OK" and wire != model_text.encode("utf-8"):
                    return False, "ZERO_UPDATE_FIXTURE_OUTPUT_WAS_REWRITTEN"
            if child.get("status") == "OK":
                try:
                    decision = _raw_candidate_decision(wire)
                except (ValueError, UnicodeError, TypeError, OverflowError,
                        RecursionError):
                    if (turn.get("decision") is not None or
                            turn.get("status") != "INVALID_MODEL_OUTPUT"):
                        return False, "INVALID_OUTPUT_WAS_RELABELED"
                    continue
                if _json(turn.get("decision")) != _json(decision):
                    return False, "CANDIDATE_RAW_WIRE_DECISION_MISMATCH"
                if decision["kind"] == "ACTION":
                    if terminal is not None:
                        return False, "ACTION_AFTER_TERMINAL_CLAIM"
                    actions.append(decision["action"])
                else:
                    if terminal is not None or index != len(raw["turns"]) - 1:
                        return False, "DUPLICATE_OR_MIDSTREAM_TERMINAL"
                    terminal = decision
            elif turn.get("decision") is not None:
                return False, "FAILED_CHILD_CANNOT_SUBMIT_PROPOSAL"
        if (source.get("probe_count") != len(actions) or
                type(source.get("probe_trace")) is not list or
                [item.get("action") for item in source["probe_trace"]
                 if type(item) is dict] != actions):
            return False, "CHARGED_PROBE_TRACE_DOES_NOT_MATCH_CHILD_ACTIONS"
        proposal = terminal if terminal is not None else {"kind": "ABSTAIN"}
        if _json(source["adjudication"].get("proposal")) != _json(proposal):
            return False, "MODEL_TERMINAL_PROPOSAL_MISMATCH"
        if raw.get("verified_success") is True:
            if (terminal is None or terminal.get("kind") != "PLAN" or
                    raw.get("terminal_reason") != "SUBMITTED" or
                    source.get("verified_success") is not True or
                    novel._independent_plan_verdict(novel.make_case(case_seed),
                                                    terminal["actions"]) !=
                    "VERIFIED_WIN"):
                return False, "FALSE_SOURCE_PROCESS_VICTORY"
        return True, "SOURCE_PUBLIC_CHILD_WIRE_AND_ORACLE_REPLAY_VALID"
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError,
            RecursionError, base64.binascii.Error):
        return False, "EXCEPTION_DURING_INDEPENDENT_BATCH_REPLAY"


def run_batch(config: BatchConfig, spec: single.AdapterSpec) -> dict[str, object]:
    if type(config) is not BatchConfig or type(spec) is not single.AdapterSpec:
        raise TypeError("only exact immutable BatchConfig and built-in AdapterSpec")
    manifest = {"schema": SCHEMA, "config": asdict(config),
                "adapter": asdict(spec), "seed_order_committed": True,
                "source_public": True, "candidate_factory_allowed": False}
    beginning = time.monotonic()
    remaining_calls = config.max_total_calls
    remaining_seconds = float(config.max_total_seconds)
    rows: list[dict[str, object]] = []
    for index, (seed, generation_seed) in enumerate(zip(
            config.case_seeds, config.generation_seeds)):
        row: dict[str, object] = {
            "index": index, "case_seed": seed, "generation_seed": generation_seed,
            "admission": "SKIPPED", "trial": None, "trial_receipt_valid": False,
            "verified_win": False, "reserved_calls": 0, "reserved_seconds": 0.0,
        }
        if time.monotonic() - beginning >= config.max_total_seconds:
            row["admission"] = "SKIPPED_GLOBAL_WALL_TIMEOUT"
        elif remaining_calls < config.max_calls_per_case:
            row["admission"] = "SKIPPED_GLOBAL_CALL_BUDGET"
        elif remaining_seconds < _reservation(config):
            row["admission"] = "SKIPPED_GLOBAL_TIME_RESERVATION"
        else:
            # Preallocate against immutable manifest, never recycle unused
            # early-win tokens to pick later cases based on their outcomes.
            remaining_calls -= config.max_calls_per_case
            remaining_seconds -= _reservation(config)
            row["admission"] = "ADMITTED"
            row["reserved_calls"] = config.max_calls_per_case
            row["reserved_seconds"] = _reservation(config)
            try:
                single_config = single.TrialConfig(
                    seed, generation_seed, config.max_calls_per_case,
                    config.call_timeout_seconds, _reservation(config),
                    config.max_output_bytes)
                trial = single.run_trial(single_config, spec)
                valid, reason = _trial_reverified(
                    trial, seed, generation_seed, config, spec)
                row["trial"] = trial
                row["trial_receipt_valid"] = valid
                row["verified_win"] = valid and trial["verified_success"] is True
                row["admission"] = "COMPLETED" if valid else "INVALID_TRIAL_RECEIPT"
                row["review_reason"] = reason
            except Exception as error:
                row["admission"] = "RUNNER_ERROR"
                row["review_reason"] = "BOUNDED_SOURCE_RUNNER_EXCEPTION"
                row["exception_class"] = type(error).__name__
        rows.append(row)
    wins = sum(row["verified_win"] is True for row in rows)
    valid_count = sum(row["trial_receipt_valid"] is True for row in rows)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "evidence_class": SOURCE_LABEL,
        "report_classification": PRIVATE_LABEL,
        "source_commit": single._git_receipt(),
        "manifest": manifest,
        "manifest_sha256": _sha(manifest),
        "private_heldout_qualified": False,
        "neural_learning_verified": False,
        "model_weight_update_verified": False,
        "model_identity_qualified": False,
        "generalization_qualified": False,
        "generation_seed_effect_verified": False,
        "comparison_qualified": False,
        "global_budget_scope": "PRE_ADMISSION_NOT_HOST_KERNEL_QUOTA",
        "cases": rows,
        "aggregate": {
            "precommitted_case_count": len(rows),
            "source_verified_wins": wins,
            "verified_fraction_planned": {"numerator": wins,
                                          "denominator": len(rows)},
            "receipts_valid": valid_count,
            "skipped_or_invalid": len(rows) - valid_count,
            "calls_reserved": config.max_total_calls - remaining_calls,
            "calls_executed": sum(len(row["trial"].get("turns", []))
                                  for row in rows if type(row["trial"]) is dict),
            "seconds_reserved": round(config.max_total_seconds - remaining_seconds, 6),
            "wall_elapsed_ms": round((time.monotonic() - beginning) * 1000),
        },
    }
    report["batch_receipt_sha256"] = _sha(report)
    return report


def verify_batch_receipt(report: object) -> bool:
    if (not _plain_receipt_tree(report) or type(report) is not dict or
            report.get("schema") != SCHEMA or
            set(report) != REPORT_KEYS or
            type(report.get("manifest")) is not dict or
            type(report.get("cases")) is not list or
            type(report.get("aggregate")) is not dict or
            type(report.get("batch_receipt_sha256")) is not str):
        return False
    try:
        if report["batch_receipt_sha256"] != _sha({
                key: value for key, value in report.items()
                if key != "batch_receipt_sha256"}):
            return False
        if (report.get("evidence_class") != SOURCE_LABEL or
                report.get("report_classification") != PRIVATE_LABEL or
                report.get("global_budget_scope") !=
                "PRE_ADMISSION_NOT_HOST_KERNEL_QUOTA" or
                any(report.get(key) is not False for key in (
                    "private_heldout_qualified", "neural_learning_verified",
                    "model_weight_update_verified", "model_identity_qualified",
                    "generalization_qualified", "generation_seed_effect_verified",
                    "comparison_qualified"))):
            return False
        manifest = report["manifest"]
        if (set(manifest) != MANIFEST_KEYS or
                report.get("manifest_sha256") != _sha(manifest) or
                manifest.get("schema") != SCHEMA or
                manifest.get("seed_order_committed") is not True or
                manifest.get("source_public") is not True or
                manifest.get("candidate_factory_allowed") is not False or
                type(manifest.get("config")) is not dict or
                type(manifest.get("adapter")) is not dict):
            return False
        config_data = deepcopy(manifest["config"])
        for key in ("case_seeds", "generation_seeds"):
            config_data[key] = tuple(config_data[key])
        config = BatchConfig(**config_data)
        adapter_data = deepcopy(manifest["adapter"])
        for key in ("fixture_responses", "command_argv"):
            adapter_data[key] = tuple(adapter_data[key])
        spec = single.AdapterSpec(**adapter_data)
        rows = report["cases"]
        if len(rows) != len(config.case_seeds):
            return False
        summary = report["aggregate"]
        if (set(summary) != SUMMARY_KEYS or
                any(type(summary.get(field)) is not int for field in (
                "precommitted_case_count", "source_verified_wins",
                "receipts_valid", "skipped_or_invalid", "calls_reserved",
                "calls_executed", "wall_elapsed_ms")) or
                type(summary.get("seconds_reserved")) is not float or
                type(summary.get("verified_fraction_planned")) is not dict or
                any(type(summary["verified_fraction_planned"].get(field)) is not int
                    for field in ("numerator", "denominator")) or
                set(summary["verified_fraction_planned"]) !=
                {"numerator", "denominator"} or
                summary["wall_elapsed_ms"] < 0):
            return False
        source_commit = report.get("source_commit")
        if (type(source_commit) is not dict or
                set(source_commit) != {"sha", "dirty"} or
                not (source_commit["sha"] is None or single._hex_digest(
                    source_commit["sha"], 40)) or
                (source_commit["dirty"] is not None and
                 type(source_commit["dirty"]) is not bool) or
                (source_commit["sha"] is None) !=
                (source_commit["dirty"] is None)):
            return False
        charged_calls = 0
        charged_seconds = 0.0
        executed_calls = 0
        valid_count = 0
        wins = 0
        wall_exhausted = False
        reserve = _reservation(config)
        for index, row in enumerate(rows):
            if (type(row) is not dict or type(row.get("index")) is not int or
                    row["index"] != index or
                    type(row.get("case_seed")) is not int or
                    row["case_seed"] != config.case_seeds[index] or
                    type(row.get("generation_seed")) is not int or
                    row["generation_seed"] != config.generation_seeds[index] or
                    type(row.get("admission")) is not str or
                    type(row.get("trial_receipt_valid")) is not bool or
                    type(row.get("verified_win")) is not bool or
                    type(row.get("reserved_calls")) is not int or
                    type(row.get("reserved_seconds")) is not float):
                return False
            status = row["admission"]
            if wall_exhausted and status != "SKIPPED_GLOBAL_WALL_TIMEOUT":
                return False
            if status == "SKIPPED_GLOBAL_WALL_TIMEOUT":
                wall_exhausted = True
            admitted = status in ("COMPLETED", "INVALID_TRIAL_RECEIPT",
                                  "RUNNER_ERROR")
            extra_keys = ({"review_reason", "exception_class"} if
                          status == "RUNNER_ERROR" else
                          {"review_reason"} if status in (
                              "COMPLETED", "INVALID_TRIAL_RECEIPT") else set())
            if set(row) != ROW_KEYS | extra_keys:
                return False
            if admitted:
                if (row["reserved_calls"] != config.max_calls_per_case or
                        row["reserved_seconds"] != reserve or
                        charged_calls + config.max_calls_per_case > config.max_total_calls or
                        charged_seconds + reserve > config.max_total_seconds + 1e-8):
                    return False
                charged_calls += config.max_calls_per_case
                charged_seconds += reserve
            else:
                if (status not in ("SKIPPED_GLOBAL_WALL_TIMEOUT",
                                   "SKIPPED_GLOBAL_CALL_BUDGET",
                                   "SKIPPED_GLOBAL_TIME_RESERVATION") or
                        row["trial"] is not None or
                        row["trial_receipt_valid"] is not False or
                        row["verified_win"] is not False or
                        row["reserved_calls"] != 0 or
                        row["reserved_seconds"] != 0.0):
                    return False
                if (status == "SKIPPED_GLOBAL_CALL_BUDGET" and
                        charged_calls + config.max_calls_per_case <= config.max_total_calls):
                    return False
                if (status == "SKIPPED_GLOBAL_TIME_RESERVATION" and
                        (charged_calls + config.max_calls_per_case >
                         config.max_total_calls or
                         charged_seconds + reserve <= config.max_total_seconds)):
                    return False
            if status == "COMPLETED":
                if row["trial_receipt_valid"] is not True:
                    return False
                valid, reason = _trial_reverified(row["trial"],
                                             config.case_seeds[index],
                                             config.generation_seeds[index],
                                             config, spec)
                if not valid or row.get("review_reason") != reason or \
                        row["verified_win"] is not (
                        row["trial"].get("verified_success") is True):
                    return False
                valid_count += 1
                wins += row["verified_win"] is True
            elif status == "INVALID_TRIAL_RECEIPT":
                if (row["trial_receipt_valid"] is not False or
                        row["verified_win"] is not False or
                        type(row["trial"]) is not dict):
                    return False
                valid, reason = _trial_reverified(row["trial"],
                                             config.case_seeds[index],
                                             config.generation_seeds[index],
                                             config, spec)
                if valid or row.get("review_reason") != reason:
                    return False
            elif status == "RUNNER_ERROR":
                if (row["trial"] is not None or
                        row["trial_receipt_valid"] is not False or
                        row["verified_win"] is not False or
                        type(row.get("exception_class")) is not str or
                        row.get("review_reason") !=
                        "BOUNDED_SOURCE_RUNNER_EXCEPTION"):
                    return False
            if type(row["trial"]) is dict:
                turns = row["trial"].get("turns")
                if type(turns) is list:
                    if len(turns) > config.max_calls_per_case:
                        return False
                    executed_calls += len(turns)
        if (wall_exhausted and summary["wall_elapsed_ms"] + 1 <
                config.max_total_seconds * 1000):
            return False
        return (summary["precommitted_case_count"] == len(rows) and
                summary["source_verified_wins"] == wins and
                summary["receipts_valid"] == valid_count and
                summary["skipped_or_invalid"] == len(rows) - valid_count and
                summary["calls_reserved"] == charged_calls and
                summary["calls_executed"] == executed_calls and
                summary["seconds_reserved"] == round(charged_seconds, 6) and
                summary["verified_fraction_planned"] == {
                    "numerator": wins, "denominator": len(rows)})
    except (ValueError, TypeError, KeyError, AttributeError, IndexError,
            OverflowError, RecursionError, base64.binascii.Error):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--generation-seeds", nargs="+", type=int)
    parser.add_argument("--max-calls-per-case", type=int, default=2)
    parser.add_argument("--max-total-calls", type=int, default=8)
    parser.add_argument("--call-timeout", type=float, default=2.)
    parser.add_argument("--max-total-seconds", type=float, default=30.)
    parser.add_argument("--max-output-bytes", type=int, default=4096)
    parser.add_argument("--fixture-json", nargs="*", default=[])
    parser.add_argument("--authorize-command", action="store_true")
    parser.add_argument("--model-artifact")
    parser.add_argument("--emit-private-json-stdout", action="store_true")
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    seeds = tuple(args.seeds)
    generation = (tuple(args.generation_seeds)
                  if args.generation_seeds is not None else tuple(range(len(seeds))))
    config = BatchConfig(seeds, generation, args.max_calls_per_case,
                         args.max_total_calls, args.call_timeout,
                         args.max_total_seconds, args.max_output_bytes)
    if args.command:
        spec = single.AdapterSpec(mode="command",
                                  command_argv=tuple(args.command),
                                  authorize_command=args.authorize_command,
                                  model_artifact_path=args.model_artifact)
    else:
        if args.authorize_command or args.model_artifact:
            parser.error("fixture cannot claim model/command authorization")
        spec = single.AdapterSpec(mode="fixture", fixture_responses=(
            tuple(args.fixture_json) if args.fixture_json else
            ('{"kind":"ABSTAIN"}',)))
    result = run_batch(config, spec)
    if not verify_batch_receipt(result):
        # The CLI is a scoring/publication boundary. Never print an
        # unverified apparent 100% or successful exit, even if the caller
        # explicitly opted into private receipt output.
        print(_json({
            "schema": "labyrinth.novel-router-batch-public-summary.v1",
            "evidence_class": SOURCE_LABEL,
            "private_receipt_omitted": True,
            "status": "EVALUATOR_DISAGREEMENT",
            "verified_success": False,
            "source_score_withheld": True,
        }).decode("utf-8"))
        return 2
    if args.emit_private_json_stdout:
        printed = result
    else:
        aggregate = result["aggregate"]
        printed = {
            "schema": "labyrinth.novel-router-batch-public-summary.v1",
            "evidence_class": SOURCE_LABEL,
            "private_receipt_omitted": True,
            "private_receipt_sha256": result["batch_receipt_sha256"],
            "precommitted_case_count": aggregate["precommitted_case_count"],
            "source_verified_wins": aggregate["source_verified_wins"],
            "receipts_valid": aggregate["receipts_valid"],
            "skipped_or_invalid": aggregate["skipped_or_invalid"],
            "verified_fraction_planned": aggregate["verified_fraction_planned"],
            "model_weight_update_verified": False,
            "generalization_qualified": False,
            "comparison_qualified": False,
        }
    print(_json(printed).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
