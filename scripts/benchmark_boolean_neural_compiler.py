#!/usr/bin/env python3
"""Stdlib-only CPU benchmark: compiled ReLU weights vs a Python full-adder.

Measures the *same exhaustive eight binary inputs* for each implementation.
The compiled evaluator is the Labyrinth Boolean neural pilot. This is a tiny
reference implementation,
not a hardware neural accelerator or a claim about model generalization.

Usage:
  python -m scripts.benchmark_boolean_neural_compiler
  python -m scripts.benchmark_boolean_neural_compiler --repeats 5000 --samples 7
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
import os
import platform
import statistics
from time import perf_counter_ns, process_time_ns
import tracemalloc


def full_adder_reference(bits: tuple[int, ...]) -> tuple[int, int]:
    """Independent direct-Python oracle for the circuit's 3-bit input ABI."""
    total = bits[0] + bits[1] + bits[2]
    return total & 1, total >> 1


def _load_circuit():
    from labyrinth.boolean_neural_specs import build

    _parity, _majority, linked = build()
    return linked


def _all_outputs(cases: tuple[tuple[int, ...], ...], fn):
    return tuple(fn(bits) for bits in cases)


def _loop(cases: tuple[tuple[int, ...], ...], fn, repeats: int) -> int:
    """Consume results from each call to prevent benchmarking an unused path."""
    checksum = 0
    for _ in range(repeats):
        for bits in cases:
            summation_bit, carry_bit = fn(bits)
            checksum += summation_bit + 2 * carry_bit
    return checksum


def _measure_ns(cases: tuple[tuple[int, ...], ...], fn, repeats: int):
    start_wall = perf_counter_ns()
    start_cpu = process_time_ns()
    checksum = _loop(cases, fn, repeats)
    cpu_ns = process_time_ns() - start_cpu
    wall_ns = perf_counter_ns() - start_wall
    return (wall_ns, cpu_ns), checksum


def _trace_peak_bytes(cases: tuple[tuple[int, ...], ...], fn, repeats: int):
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        _loop(cases, fn, repeats)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"current_bytes": current, "peak_bytes": peak}


def _machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os_reported_logical_cpu_count": os.cpu_count(),
        "physical_ram_bytes_os_reported": None,
        "gpu": "not_detected_by_stdlib; no GPU inference measured",
    }
    try:
        info["physical_ram_bytes_os_reported"] = (
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        )
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return info


def _tensor_size(linked) -> dict:
    matrices = (linked.w1, linked.w2)
    vectors = (linked.b1, linked.b2)
    scalar_weights = [v for mat in matrices for row in mat for v in row]
    scalar_biases = [v for vec in vectors for v in vec]
    # JSON is a valid on-disk serialization size reference for this actual
    # artifact. It is *not* evidence for GGUF/BF16/quantized binary size.
    serialized = linked.manifest()
    serialized["sha256"] = linked.sha256()
    artifact_bytes = (json.dumps(serialized, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return {
        "layer_dimensions": [len(linked.input_names), len(linked.w1), len(linked.output_names)],
        "scalar_parameters_total": len(scalar_weights) + len(scalar_biases),
        "scalar_parameters_nonzero": sum(v != 0 for v in scalar_weights + scalar_biases),
        "numeric_representation": "exact Python integers, not FP tensors",
        "pretty_json_artifact_bytes": len(artifact_bytes),
        "pretty_json_artifact_sha256": sha256(artifact_bytes).hexdigest(),
        "manifest_semantic_sha256": linked.sha256(),
        "source_package_bytes_excluded": True,
    }


def _summary(samples_ns: list[int], cases_per_sample: int) -> dict:
    return {
        "samples_ns": samples_ns,
        "median_ns": int(statistics.median(samples_ns)),
        "min_ns": min(samples_ns),
        "max_ns": max(samples_ns),
        "median_ns_per_evaluation": statistics.median(samples_ns) / cases_per_sample,
    }


def benchmark(*, repeats: int, samples: int, warmup: int) -> dict:
    if repeats < 1 or samples < 1 or warmup < 0:
        raise ValueError("repeats and samples must be positive; warmup must be nonnegative")
    linked = _load_circuit()
    cases = tuple(product((0, 1), repeat=3))
    if tuple(linked.input_names) != ("a", "b", "carry_in"):
        raise RuntimeError("unexpected input ABI: benchmark must be reviewed for the new contract")
    if tuple(linked.output_names) != ("sum_bit", "carry_out"):
        raise RuntimeError("unexpected output ABI: benchmark must be reviewed for the new contract")
    neural_fn = linked.evaluate
    expected, actual = _all_outputs(cases, full_adder_reference), _all_outputs(cases, neural_fn)
    if expected != actual:
        raise AssertionError(f"exhaustive oracle disagreement: expected {expected}, received {actual}")
    # Equal warmup counts on exact same input sequence; never mix traced and
    # untraced timing measurements.
    _loop(cases, full_adder_reference, warmup)
    _loop(cases, neural_fn, warmup)
    elapsed = {"direct_python": [], "compiled_relu_weights": []}
    cpu_elapsed = {"direct_python": [], "compiled_relu_weights": []}
    checksums = {"direct_python": [], "compiled_relu_weights": []}
    implementations = (
        ("direct_python", full_adder_reference),
        ("compiled_relu_weights", neural_fn),
    )
    for idx in range(samples):
        # Alternate order to reduce one-sided scheduling/thermal bias.
        for name, fn in (implementations if idx % 2 == 0 else reversed(implementations)):
            (wall_ns, cpu_ns), check = _measure_ns(cases, fn, repeats)
            elapsed[name].append(wall_ns)
            cpu_elapsed[name].append(cpu_ns)
            checksums[name].append(check)
    if len(set(checksums["direct_python"] + checksums["compiled_relu_weights"])) != 1:
        raise AssertionError("timed execution result checksums disagree")
    timed_cases = repeats * len(cases)
    # Separate tracing pass: tracemalloc changes timing significantly.
    mem_repeats = min(repeats, 250)
    traced = {
        "direct_python": _trace_peak_bytes(cases, full_adder_reference, mem_repeats),
        "compiled_relu_weights": _trace_peak_bytes(cases, neural_fn, mem_repeats),
    }
    direct_med = statistics.median(elapsed["direct_python"])
    neural_med = statistics.median(elapsed["compiled_relu_weights"])
    cpu_direct_med = statistics.median(cpu_elapsed["direct_python"])
    cpu_neural_med = statistics.median(cpu_elapsed["compiled_relu_weights"])
    return {
        "benchmark": "boolean_relu_v1_exhaustive_cpu_python",
        "evidence_class": "LOCAL_PROCESS_BENCHMARK_ONLY",
        "machine": _machine_info(),
        "artifact": _tensor_size(linked),
        "domain": {
            "input_abi": list(linked.input_names),
            "output_abi": list(linked.output_names),
            "exhaustive_inputs": len(cases),
            "agreement": True,
            "training_steps": 0,
            "gradients_used": False,
        },
        "protocol": {
            "warmup_full_domain_passes_each": warmup,
            "timed_full_domain_passes_per_sample": repeats,
            "timed_evaluations_per_sample": timed_cases,
            "timing_samples_each": samples,
            "interleaved_alternating_order": True,
            "tracing_full_domain_passes_each_separate_from_timing": mem_repeats,
        },
        "timing": {
            "direct_python": _summary(elapsed["direct_python"], timed_cases),
            "compiled_relu_weights": _summary(elapsed["compiled_relu_weights"], timed_cases),
            "wall_clock": "time.perf_counter_ns",
            "median_compiled_over_direct_ratio": neural_med / direct_med,
            "result_checksum_each_timed_sample": checksums["direct_python"][0],
        },
        "process_cpu_time": {
            "direct_python": _summary(cpu_elapsed["direct_python"], timed_cases),
            "compiled_relu_weights": _summary(cpu_elapsed["compiled_relu_weights"], timed_cases),
            "clock": "time.process_time_ns",
            "median_compiled_over_direct_ratio": cpu_neural_med / cpu_direct_med,
        },
        "tracemalloc": {
            "observations": traced,
            "caveat": "Tracks Python allocations, NOT total process RSS, native library/GPU allocations, interpreter baseline, or maximum model-serving footprint. Tracing was run separately from timing.",
        },
        "interpretation_limits": [
            "Pure-Python exact-integer ReLU interpretation, not GPU-optimized tensor inference.",
            "Timing includes Python ABI validation/loops; tiny benchmark is overhead-sensitive.",
            "8 test inputs prove only exact behavior on this finite 3-bit domain.",
            "Larger Boolean domains have exponentially growing truth-table compiler output.",
            "No inference of LLM capabilities, GGUF compatibility or production speed from these data.",
            "Host-reported physical RAM may refer to host rather than container allocation.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2500)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        report = benchmark(repeats=args.repeats,
                           samples=args.samples, warmup=args.warmup)
    except (ValueError, AssertionError, RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
