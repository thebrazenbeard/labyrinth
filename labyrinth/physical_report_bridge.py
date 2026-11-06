"""Byte-readback validation for *public* MiniLM/LoRA development reports.

This is deliberately separate from ``evaluation_provenance_bridge`` v1. That
protocol records one training case per compute step; MiniLM performs batches of
four cases per optimizer step. Encoding four case visits as one, or claiming
four optimizer steps, would falsify causal exposure or matched compute.

The input remains trainer-origin self-reporting even when its on-disk bytes,
batch schedule and per-case development predictions are internally coherent.
This module does not attest process identity, model inference, private exam
custody, control-key custody, or transfer/generalization. It never graduates.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

from .control_provenance import ARMS


_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_SCHEMA = "labyrinth.minilm-cpu-development.v2"
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_MAX_CYCLES = 16
_MAX_BATCHES = 4096
_MAX_BATCH_SIZE = 64
_MAX_PREDICTIONS = 4096


@dataclass(frozen=True)
class PublicMiniLMReadback:
    """Validated source-level evidence only; never a graduation verdict."""

    status: str
    report_sha256: str
    training_optimizer_steps: int
    training_example_visits: int
    prediction_case_counts: tuple[tuple[str, int], ...]
    reasons: tuple[str, ...]
    physical_model_verified: bool = False
    fresh_process_verified: bool = False
    evaluator_custody_verified: bool = False
    formal_v1_compatible: bool = False
    graduation_authorized: bool = False


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")
    return value


def _id(value: Any, label: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")
    return value


def _int(value: Any, label: str, *, maximum: int = 100000) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"invalid {label}")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise ValueError(f"invalid {label}")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON report field")
        result[name] = value
    return result


def _array(value: Any, label: str, *, maximum: int) -> list[Any]:
    if type(value) is not list or len(value) > maximum:
        raise ValueError(f"invalid {label}")
    return value


def _check_public_comparison(value: Any, *, accuracy: dict[str, float], split: str) -> None:
    summary = _object(value, f"{split} comparison")
    expected_keys = {"correct_minus_baseline", "correct_minus_zero",
                     "correct_minus_corrupted", "correct_outperforms_all_controls"}
    if set(summary) != expected_keys:
        raise ValueError(f"invalid {split} comparison fields")
    candidate = accuracy["correct_labels"]
    expectations = {
        "correct_minus_baseline": candidate - accuracy["baseline"],
        "correct_minus_zero": candidate - accuracy["zero_update"],
        "correct_minus_corrupted": candidate - accuracy["corrupted_labels"],
    }
    for label, expected in expectations.items():
        observed = summary[label]
        if (type(observed) not in (int, float) or not math.isfinite(observed)
                or not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-9)):
            raise ValueError(f"inconsistent {split} comparison summary")
    stronger = all(candidate > accuracy[name] for name in ARMS
                   if name != "correct_labels")
    if type(summary["correct_outperforms_all_controls"]) is not bool or summary[
        "correct_outperforms_all_controls"] != stronger:
        raise ValueError(f"inconsistent {split} comparison summary")


def _prediction_rows(value: Any, split: str) -> tuple[tuple[str, str, int, int], ...]:
    result: list[tuple[str, str, int, int]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for item in _array(value, f"{split} predictions", maximum=_MAX_PREDICTIONS):
        row = _object(item, f"{split} prediction")
        case_id = _id(row.get("case_id"), "prediction case ID")
        prompt_sha = _sha(row.get("prompt_sha256"), "prediction prompt")
        predicted = row.get("predicted_label")
        expected = row.get("expected_label")
        if (type(predicted) is not int or predicted not in (0, 1)
                or type(expected) is not int or expected not in (0, 1)
                or row.get("response") != str(predicted)):
            raise ValueError("prediction response does not match predicted label")
        if case_id in seen_ids or prompt_sha in seen_prompts:
            raise ValueError("duplicate prediction case or prompt")
        seen_ids.add(case_id)
        seen_prompts.add(prompt_sha)
        result.append((case_id, prompt_sha, expected, predicted))
    if not result:
        raise ValueError(f"missing {split} per-case predictions")
    return tuple(result)


def _batch_schedule(cycle: dict[str, Any]) -> tuple[tuple[tuple[str, str, str, int], ...], ...]:
    batches = _array(cycle.get("training_batches"), "training batches", maximum=_MAX_BATCHES)
    schedule = []
    identity_by_case: dict[str, tuple[str, str, int]] = {}
    for index, item in enumerate(batches):
        batch = _object(item, "training batch")
        if batch.get("step") != index or type(batch.get("step")) is not int:
            raise ValueError("training batch step sequence mismatch")
        examples = _array(batch.get("examples"), "training batch examples",
                          maximum=_MAX_BATCH_SIZE)
        if not examples:
            raise ValueError("empty training batch")
        records = []
        for item in examples:
            example = _object(item, "training example")
            target = example.get("target_label")
            if type(target) is not int or target not in (0, 1):
                raise ValueError("invalid training target label")
            record = (
                _id(example.get("case_id"), "training case ID"),
                _sha(example.get("prompt_sha256"), "training prompt"),
                _id(example.get("structure_key"), "training structure key"),
                target,
            )
            previous = identity_by_case.setdefault(record[0], record[1:])
            if previous != record[1:]:
                raise ValueError("training case identity/label changed within cycle")
            records.append(record)
        schedule.append(tuple(records))
    return tuple(schedule)


def _check_cycle(cycle: dict[str, Any], *, arm: str, index: int,
                 expected_before: tuple[str, str], batch_size: int
                 ) -> tuple[tuple[tuple[tuple[str, str, str, int], ...], ...],
                            tuple[str, str], tuple[tuple[str, str, int, int], ...],
                            tuple[tuple[str, str, int, int], ...]]:
    if type(cycle.get("cycle")) is not int or cycle["cycle"] != index:
        raise ValueError("nonsequential training cycle")
    before = (_sha(cycle.get("trainable_sha256_before"), "trainable before"),
              _sha(cycle.get("lora_sha256_before"), "LoRA before"))
    after = (_sha(cycle.get("trainable_sha256_after"), "trainable after"),
             _sha(cycle.get("lora_sha256_after"), "LoRA after"))
    if before != expected_before:
        raise ValueError("checkpoint continuity differs from preceding cycle")
    schedule = _batch_schedule(cycle)
    count = len(schedule)
    calls = _int(cycle.get("optimizer_step_calls"), "optimizer step calls")
    updates = _int(cycle.get("optimizer_updates"), "optimizer updates")
    forward = _int(cycle.get("forward_steps"), "forward steps")
    backward = _int(cycle.get("backward_steps"), "backward steps")
    examples = _int(cycle.get("train_example_visits"), "training example visits")
    if examples != sum(len(batch) for batch in schedule):
        raise ValueError("training example visits do not match batch exposure")
    claimed_visits = _array(cycle.get("training_prompt_sha256"),
                            "training prompt SHA sequence", maximum=_MAX_BATCHES * _MAX_BATCH_SIZE)
    actual_visits = [row[1] for batch in schedule for row in batch]
    if claimed_visits != actual_visits:
        raise ValueError("training prompt SHA sequence differs from batch exposure")
    if (_int(cycle.get("compute_steps"), "reported compute steps") != count
            or _int(cycle.get("training_forward_steps"), "reported training forward steps") != count
            or _int(cycle.get("training_backward_steps"), "reported training backward steps") != count):
        raise ValueError("redundant compute counter disagrees with batch exposure")
    if arm == "baseline":
        if (count or calls or updates or forward or backward or examples or after != before):
            raise ValueError("baseline must not train or change weights")
    else:
        if (count < 1 or calls != count or forward != count or backward != count
                or any(len(batch) != batch_size for batch in schedule)):
            raise ValueError("training steps or batch size mismatch")
        if arm == "zero_update":
            if updates or after != before:
                raise ValueError("zero-update arm must remain frozen")
        elif updates != count or after == before or after[1] == before[1]:
            raise ValueError("trained arm must report real LoRA optimizer updates")
    heldout = _prediction_rows(cycle.get("heldout_predictions"), "heldout")
    transfer = _prediction_rows(cycle.get("transfer_predictions"), "transfer")
    if ({x[0] for x in heldout} & {x[0] for x in transfer}
            or {x[1] for x in heldout} & {x[1] for x in transfer}):
        raise ValueError("duplicate heldout/transfer prediction")
    for split, rows in (("heldout", heldout), ("transfer", transfer)):
        accuracy = cycle.get(f"{split}_accuracy")
        if (type(accuracy) not in (int, float) or not math.isfinite(accuracy)
                or not 0 <= accuracy <= 1):
            raise ValueError("invalid reported public development accuracy")
        # Public expected labels in reports are not an independent gold source.
        predictions = cycle[f"{split}_predictions"]
        observed = sum(row["predicted_label"] == row["expected_label"]
                       for row in predictions) / len(predictions)
        if not math.isclose(accuracy, observed, abs_tol=1e-9):
            raise ValueError("aggregate accuracy differs from per-case predictions")
    return schedule, after, heldout, transfer


def inspect_public_minilm_report(
    report_path: str | Path, *, expected_report_sha256: str,
    expected_harness_sha256: str, expected_run_id: str, expected_model_id: str,
) -> PublicMiniLMReadback:
    """Read and validate public pilot bytes against separately supplied digests.

    The expected digest/harness/run identities MUST be externally precommitted
    for this to be more than trainer-supplied coherence checking. Even then,
    ``PROVISIONAL`` does *not* mean fresh inference, custody or model validation.
    """
    _sha(expected_report_sha256, "expected report digest")
    _sha(expected_harness_sha256, "expected harness digest")
    _id(expected_run_id, "expected run ID")
    if type(expected_model_id) is not str or not 1 <= len(expected_model_id) <= 256:
        raise ValueError("invalid expected model identity")
    path = Path(report_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("report must be an existing regular file, not symlink")
    with path.open("rb") as handle:
        content = handle.read(_MAX_REPORT_BYTES + 1)
    if len(content) > _MAX_REPORT_BYTES:
        raise ValueError("report exceeds byte budget")
    observed_sha = sha256(content).hexdigest()
    if observed_sha != expected_report_sha256:
        raise ValueError("report digest mismatch from on-disk byte readback")
    try:
        report = json.loads(content.decode("utf-8"),
                            object_pairs_hook=_unique_json_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                ValueError(f"nonfinite JSON value {value}")))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON report") from exc
    report = _object(report, "public MiniLM report")
    if report.get("schema") != _SCHEMA:
        raise ValueError("unknown public MiniLM report schema")
    if report.get("qualification") != "PUBLIC_DEVELOPMENT_ONLY":
        raise ValueError("not a public development report")
    if report.get("private_final_exam") != "NOT_RUN":
        raise ValueError("public report cannot claim private exam")
    if report.get("graduation_authorized") is not False:
        raise ValueError("public report cannot claim graduation")
    if report.get("physical_model_byte_custody") != "SELF_REPORTED":
        raise ValueError("report cannot self-assert independent byte custody")
    if report.get("harness_sha256") != expected_harness_sha256:
        raise ValueError("harness identity mismatch")
    if report.get("run_id") != expected_run_id:
        raise ValueError("run identity mismatch")
    if report.get("model_id") != expected_model_id:
        raise ValueError("model identity mismatch")
    batch_size = _int(report.get("physical_training_batch_size"),
                      "physical batch size", maximum=_MAX_BATCH_SIZE)
    if batch_size < 2:
        raise ValueError("batch-aware report requires a real multi-example batch")
    expected_steps = _int(report.get("steps_per_cycle"), "steps per cycle", maximum=32)
    if expected_steps < 1:
        raise ValueError("invalid declared steps per cycle")
    fixture_sizes = _object(report.get("fixture_sizes"), "fixture sizes")
    for split in ("heldout", "transfer"):
        if _int(fixture_sizes.get(split), f"{split} fixture size", maximum=_MAX_PREDICTIONS) < 1:
            raise ValueError("empty public prediction fixture")
    arms = _object(report.get("arms"), "control arms")
    if set(arms) != set(ARMS):
        raise ValueError("four arms required for development report")
    starts = set()
    every: dict[str, tuple[Any, ...]] = {}
    for arm in ARMS:
        arm_record = _object(arms[arm], "control arm")
        initial = (_sha(arm_record.get("initial_trainable_sha256"), "initial trainable"),
                   _sha(arm_record.get("initial_lora_sha256"), "initial LoRA"))
        starts.add(initial)
        cycles = _array(arm_record.get("cycles"), "training cycles", maximum=_MAX_CYCLES)
        if not cycles:
            raise ValueError("no training cycle report")
        prev = initial
        collected = []
        known_cases: dict[str, tuple[str, str, int]] = {}
        known_prompts: dict[str, str] = {}
        for index, entry in enumerate(cycles):
            cycle = _object(entry, "training cycle")
            schedule, prev, heldout, transfer = _check_cycle(
                cycle, arm=arm, index=index, expected_before=prev, batch_size=batch_size,
            )
            declared_size = _int(fixture_sizes.get(f"cycle{index}"),
                                 "declared cycle fixture size", maximum=100000)
            if declared_size < len({row[0] for batch in schedule for row in batch}):
                raise ValueError("training fixture size understates observed case IDs")
            if arm != "baseline" and len(schedule) != expected_steps:
                raise ValueError("training step count differs from declared steps per cycle")
            if len(heldout) != fixture_sizes["heldout"] or len(transfer) != fixture_sizes["transfer"]:
                raise ValueError("prediction coverage differs from declared fixture size")
            for batch in schedule:
                for case_id, prompt_sha, structure_key, label in batch:
                    identity = (prompt_sha, structure_key, label)
                    previous = known_cases.setdefault(case_id, identity)
                    if previous != identity:
                        raise ValueError("training case identity changed between cycles")
                    previous_case = known_prompts.setdefault(prompt_sha, case_id)
                    if previous_case != case_id:
                        raise ValueError("training prompt reused with a different case ID")
            collected.append((schedule, prev, heldout, transfer))
        every[arm] = tuple(collected)
    if len(starts) != 1:
        raise ValueError("control arms do not share starting parameters")
    num_cycles = len(every["correct_labels"])
    if any(len(every[arm]) != num_cycles for arm in ARMS):
        raise ValueError("control arm cycles not matched")
    count_steps = 0
    count_examples = 0
    first_predictions = None
    for index in range(num_cycles):
        baseline = every["baseline"][index]
        sham = every["zero_update"][index]
        correct = every["correct_labels"][index]
        corrupt = every["corrupted_labels"][index]
        schedules = (sham[0], correct[0], corrupt[0])
        identities = [tuple(tuple(row[:3] for row in batch) for batch in schedule)
                      for schedule in schedules]
        if not (identities[0] == identities[1] == identities[2]):
            raise ValueError("control arms lack matched batches/exposures")
        for b_sham, b_correct, b_corrupt in zip(*schedules):
            for row_sham, row_correct, row_corrupt in zip(b_sham, b_correct, b_corrupt):
                if row_sham[3] != row_correct[3] or row_corrupt[3] != 1 - row_correct[3]:
                    raise ValueError("sham/correct or corrupted labels do not match contract")
        if correct[1] == corrupt[1]:
            raise ValueError("correct and corrupted parameters cannot be identical")
        for split_index in (2, 3):
            all_rows = [every[arm][index][split_index] for arm in ARMS]
            identities = [tuple(row[:3] for row in rows) for rows in all_rows]
            if not all(rows == identities[0] for rows in identities[1:]):
                raise ValueError("development prediction fixture identities differ across arms")
        if baseline[2:] != sham[2:]:
            raise ValueError("frozen baseline and sham predictions differ")
        if index > 0 and baseline[2:] != every["baseline"][0][2:]:
            raise ValueError("frozen baseline predictions changed between cycles")
        if index > 0 and sham[2:] != every["zero_update"][0][2:]:
            raise ValueError("frozen sham predictions changed between cycles")
        predictions = (tuple(row[:3] for row in correct[2]),
                       tuple(row[:3] for row in correct[3]))
        if first_predictions is None:
            first_predictions = predictions
        elif predictions != first_predictions:
            raise ValueError("development prediction fixture changed across cycles")
        train_identifiers = {(row[0], row[1]) for batch in correct[0] for row in batch}
        trained_ids = {item[0] for item in train_identifiers}
        trained_hashes = {item[1] for item in train_identifiers}
        if any(pred[0] in trained_ids or pred[1] in trained_hashes
               for pred in correct[2] + correct[3]):
            raise ValueError("training/prediction prompt overlap")
        count_steps += len(correct[0])
        count_examples += sum(len(batch) for batch in correct[0])
    for split in ("heldout", "transfer"):
        accuracy = {arm: arms[arm]["cycles"][-1][f"{split}_accuracy"]
                    for arm in ARMS}
        _check_public_comparison(report.get(split), accuracy=accuracy, split=split)
    retention = report.get("retention")
    if num_cycles > 1:
        retention = _object(retention, "retention summary")
        if set(retention) != set(ARMS):
            raise ValueError("retention comparison arms incomplete")
        for arm in ARMS:
            observed = retention[arm]
            expected = (arms[arm]["cycles"][1]["heldout_accuracy"]
                        - arms[arm]["cycles"][0]["heldout_accuracy"])
            if (type(observed) not in (int, float) or not math.isfinite(observed)
                    or not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-9)):
                raise ValueError("inconsistent retention summary")
    elif retention is not None:
        raise ValueError("retention requires at least two reported cycles")
    return PublicMiniLMReadback(
        status="PROVISIONAL",
        report_sha256=observed_sha,
        training_optimizer_steps=count_steps,
        training_example_visits=count_examples,
        prediction_case_counts=(("heldout", len(first_predictions[0])),
                                ("transfer", len(first_predictions[1]))),
        reasons=("trainer-origin public development report: contents are internally coherent",
                 "independent inference/checkpoint bytes and examiner custody unverified",
                 "batch exposure cannot truthfully satisfy the one-case-per-step formal-v1 contract"),
    )
