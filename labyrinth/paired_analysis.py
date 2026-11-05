"""Independent, read-only paired analysis for real-neural Labyrinth trials.

This module NEVER invokes a model or a trainer. The evaluator keeps the gold
manifest private and consumes fixed, model-only trial records. Its output
contains aggregate scores, not answers or raw completions.

Trial and checkpoint hashes are caller assertions, not authenticated proof of
model origin or optimizer execution. Validate those separately from logs and
artifacts before qualifying neural training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any


MANIFEST_SCHEMA = "labyrinth.private-eval.v1"
TRIALS_SCHEMA = "labyrinth.neural-trials.v1"
REPORT_SCHEMA = "labyrinth.paired-analysis.v1"
REQUIRED_ARMS = ("baseline", "trained", "sham", "corrupted")
VALID_SLICES = frozenset({"within_template", "structural", "length_shift", "representation_shift", "external"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
_ANSWER = re.compile(r"[ \t]*ANSWER:[ \t]*([+-]?[0-9]{1,12})[ \t]*(?:\r?\n)?\Z", re.ASCII | re.IGNORECASE)


def _fail(message: str) -> None:
    raise ValueError(message)


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _private_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        _fail("unsupported private evaluation manifest")
    if manifest.get("split") not in ("dev", "final"):
        _fail("evaluation split must be dev or final")
    if not isinstance(manifest.get("source_commit"), str) or re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]) is None:
        _fail("private manifest needs an exact source commit")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 4096:
        _fail("evaluation requires 1..4096 cases")
    ids, prompts = set(), set()
    training_ids = manifest.get("train_case_ids")
    training_prompts = manifest.get("train_prompt_sha256")
    training_structures = manifest.get("train_structure_fingerprints")
    if not all(isinstance(x, list) for x in (training_ids, training_prompts, training_structures)):
        _fail("training split commitments must be lists")
    if any(not isinstance(x, str) or not x for x in training_ids):
        _fail("invalid training case IDs")
    if any(not _is_hex64(x) for x in training_prompts + training_structures):
        _fail("invalid training prompt or structure fingerprint")
    if any(isinstance(case, dict) and case.get("slice") in ("structural", "length_shift") for case in cases) and not training_structures:
        _fail("structural holdout requires committed training structure fingerprints")
    for case in cases:
        if not isinstance(case, dict):
            _fail("invalid evaluation case")
        case_id = case.get("case_id")
        prompt = case.get("prompt_sha256")
        template = case.get("template_id")
        slice_name = case.get("slice")
        gold = case.get("answer")
        structure = case.get("structure_fingerprint")
        if not isinstance(case_id, str) or not case_id or len(case_id) > 160:
            _fail("invalid evaluation case ID")
        if not _is_hex64(prompt) or not isinstance(template, str) or _OPAQUE_ID.fullmatch(template) is None:
            _fail("invalid prompt digest or template ID")
        if slice_name not in VALID_SLICES:
            _fail("invalid evaluation slice")
        if not isinstance(gold, str) or re.fullmatch(r"[+-]?[0-9]{1,12}", gold, re.ASCII) is None:
            _fail("gold answer must be a bounded decimal string")
        if case_id in ids or prompt in prompts:
            _fail("duplicate evaluation case ID or prompt digest")
        ids.add(case_id)
        prompts.add(prompt)
        if structure is not None and not _is_hex64(structure):
            _fail("invalid structure fingerprint")
        if slice_name in ("structural", "length_shift"):
            if not _is_hex64(structure):
                _fail("structural holdout needs a structure fingerprint")
            if structure in training_structures:
                _fail("structural holdout overlaps training structures")
        if slice_name == "within_template" and structure is not None and structure not in training_structures:
            _fail("within-template case uses an uncommitted training structure")
    if ids.intersection(training_ids) or prompts.intersection(training_prompts):
        _fail("train/evaluation ID or prompt overlap")
    return cases


def _parse_response(response: object) -> tuple[bool, str | None]:
    if not isinstance(response, str) or len(response) > 8192:
        return False, None
    match = _ANSWER.fullmatch(response)
    return (False, None) if match is None else (True, str(int(match[1])))


def _wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    fraction = correct / total
    denom = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denom
    half_width = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _mcnemar_exact(n01: int, n10: int) -> float:
    """Two-sided exact sign/binomial test on paired discordances."""
    n = n01 + n10
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _cluster_ci(cases: list[dict[str, Any]], baseline: list[bool], candidate: list[bool],
                seed: int, resamples: int = 2000) -> list[float] | None:
    """Exploratory 95% cluster bootstrap; not persuasive with very few templates."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, case in enumerate(cases):
        groups[case["template_id"]].append(i)
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    keys = sorted(groups)
    deltas = []
    for _ in range(resamples):
        drawn = [index for key in rng.choices(keys, k=len(keys)) for index in groups[key]]
        deltas.append(sum(int(candidate[i]) - int(baseline[i]) for i in drawn) / len(drawn))
    deltas.sort()
    return [deltas[int(0.025 * resamples)], deltas[int(0.975 * resamples) - 1]]


def _summary(flags: list[bool], formatted: list[bool]) -> dict[str, Any]:
    n = len(flags)
    correct = sum(flags)
    valid = sum(formatted)
    return {"correct": correct, "total": n, "accuracy": correct / n,
            "wilson_95": _wilson(correct, n),
            "format_valid": valid, "format_adherence": valid / n}


def analyze(manifest: dict[str, Any], trials: dict[str, Any]) -> dict[str, Any]:
    cases = _private_cases(manifest)
    if not isinstance(trials, dict) or trials.get("schema") != TRIALS_SCHEMA:
        _fail("unsupported model trial schema")
    if not isinstance(trials.get("source_commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", trials["source_commit"]):
        _fail("trial source commit must be a full Git SHA")
    if not _is_hex64(trials.get("base_model_sha256")):
        _fail("missing base model digest")
    if not _is_hex64(trials.get("base_tokenizer_sha256")) or not _is_hex64(trials.get("generation_config_sha256")):
        _fail("missing tokenizer or decoding-configuration digest")
    if trials["source_commit"] != manifest["source_commit"]:
        _fail("training/evaluation source commits differ")
    commitment = _hash(manifest)
    if trials.get("eval_manifest_sha256") != commitment:
        _fail("trial manifest commitment mismatch")
    arms = trials.get("arms")
    if not isinstance(arms, dict) or any(name not in arms for name in REQUIRED_ARMS):
        _fail("missing required neural control arms")
    if set(arms).difference({*REQUIRED_ARMS, "uniform", "adaptive"}):
        _fail("unrecognized neural control arm")
    if ("uniform" in arms) != ("adaptive" in arms):
        _fail("uniform/adaptive curriculum arms must be compared together")
    expected_ids = [case["case_id"] for case in cases]
    expected_hashes = [case["prompt_sha256"] for case in cases]
    scores: dict[str, list[bool]] = {}
    formats: dict[str, list[bool]] = {}
    expected_seeds: list[int] | None = None
    results: dict[str, Any] = {}
    for arm_name, arm in arms.items():
        if not isinstance(arm_name, str) or not isinstance(arm, dict):
            _fail("invalid trial arm")
        if not _is_hex64(arm.get("checkpoint_sha256")) or not _is_hex64(arm.get("parameter_sha256")):
            _fail(f"{arm_name}: missing checkpoint or parameter digest")
        if type(arm.get("optimizer_steps")) is not int or arm["optimizer_steps"] < 0:
            _fail(f"{arm_name}: invalid optimizer step count")
        if type(arm.get("train_case_visits")) is not int or arm["train_case_visits"] < 0:
            _fail(f"{arm_name}: invalid training case visits")
        entries = arm.get("trials")
        if not isinstance(entries, list) or len(entries) != len(cases):
            _fail(f"{arm_name}: trial count differs from manifest")
        if [item.get("case_id") for item in entries if isinstance(item, dict)] != expected_ids:
            _fail(f"{arm_name}: trial case IDs or order differ")
        if [item.get("prompt_sha256") for item in entries] != expected_hashes:
            _fail(f"{arm_name}: evaluated different prompts")
        seeds = [item.get("generation_seed") for item in entries]
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            _fail(f"{arm_name}: invalid generation seeds")
        if expected_seeds is not None and seeds != expected_seeds:
            _fail(f"{arm_name}: unequal generation seeds")
        expected_seeds = seeds
        parsed = [_parse_response(item.get("response")) for item in entries]
        formats[arm_name] = [valid for valid, _ in parsed]
        scores[arm_name] = [valid and pred == str(int(case["answer"]))
                            for case, (valid, pred) in zip(cases, parsed)]
        results[arm_name] = _summary(scores[arm_name], formats[arm_name])
    if arms["baseline"]["parameter_sha256"] != arms["sham"]["parameter_sha256"]:
        _fail("sham parameter digest differs from frozen baseline")
    if arms["trained"]["parameter_sha256"] == arms["baseline"]["parameter_sha256"]:
        _fail("claimed trained arm has unchanged parameter digest")
    budget = {(arms[name]["optimizer_steps"], arms[name]["train_case_visits"])
              for name in ("trained", "sham", "corrupted")}
    if len(budget) != 1 or next(iter(budget))[0] == 0:
        _fail("trained/sham/corrupted optimizer and case budgets differ or are zero")
    if "uniform" in arms:
        if (arms["uniform"]["optimizer_steps"], arms["uniform"]["train_case_visits"]) != (
            arms["adaptive"]["optimizer_steps"], arms["adaptive"]["train_case_visits"]
        ):
            _fail("uniform/adaptive training budgets differ")

    comparisons: dict[str, Any] = {}

    def paired(left_name: str, right_name: str) -> dict[str, Any]:
        baseline, candidate = scores[left_name], scores[right_name]
        n01 = sum(not left and right for left, right in zip(baseline, candidate))
        n10 = sum(left and not right for left, right in zip(baseline, candidate))
        return {
            "delta_accuracy": (n01 - n10) / len(cases),
            "baseline_wrong_candidate_right": n01,
            "baseline_right_candidate_wrong": n10,
            "exact_mcnemar_p_exploratory": _mcnemar_exact(n01, n10),
            "cluster_bootstrap_delta_95_exploratory": _cluster_ci(cases, baseline, candidate, seed=20261026),
        }
    for name in sorted(set(scores) - {"baseline"}):
        comparisons[name] = paired("baseline", name)
    adaptive_vs_uniform = paired("uniform", "adaptive") if "uniform" in arms else None
    # Never echo manifest-supplied template labels: they may encode evaluator
    # gold even when the characters satisfy the opaque-ID contract.
    slices: dict[str, dict[str, Any]] = {"slice": {}, "template_group": {}}
    for group in sorted({case["slice"] for case in cases}):
        ix = [i for i, case in enumerate(cases) if case["slice"] == group]
        slices["slice"][group] = {
            name: _summary([scores[name][i] for i in ix], [formats[name][i] for i in ix])
            for name in sorted(scores)
        }
    for ordinal, group in enumerate(sorted({case["template_id"] for case in cases}), 1):
        ix = [i for i, case in enumerate(cases) if case["template_id"] == group]
        slices["template_group"][f"group_{ordinal:04d}"] = {
            name: _summary([scores[name][i] for i in ix], [formats[name][i] for i in ix])
            for name in sorted(scores)
        }

    return {
        "schema": REPORT_SCHEMA,
        "source_commit": trials["source_commit"],
        "base_model_sha256": trials["base_model_sha256"],
        "base_tokenizer_sha256": trials["base_tokenizer_sha256"],
        "generation_config_sha256": trials["generation_config_sha256"],
        "eval_split": manifest["split"],
        "eval_manifest_sha256": commitment,
        "case_count": len(cases),
        "template_count": len({case["template_id"] for case in cases}),
        "arm_summary": results,
        "paired_vs_baseline": comparisons,
        "adaptive_vs_uniform": adaptive_vs_uniform,
        "by": slices,
        "limitations": [
            "Artifact digests and optimizer counts are caller assertions; verify actual files and step logs independently.",
            "Wilson intervals treat cases as independent and overstate precision with repeated templates.",
            "Cluster-bootstrap intervals are exploratory and unreliable with only a few structural templates.",
            "Exact McNemar p-values ignore template clustering and multiple comparisons.",
            "Any final-split use for tuning contaminates that split; this tool cannot detect prior unseen exposures.",
            "Same seeds and cases do not prove determinism or identical runtime state across inference arms.",
            "The independent correctness of private gold and semantic equivalence of representation rewrites require separate oracle checks.",
        ],
    }


def _read_json(path: str) -> Any:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > 16 * 1024 * 1024:
        _fail("input must be a regular JSON file no larger than 16 MiB")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj = {}
        for key, value in pairs:
            if key in obj:
                _fail("duplicate JSON object key")
            obj[key] = value
        return obj

    return json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_keys,
                      parse_constant=lambda token: _fail(f"invalid JSON constant {token}"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze sealed-gold neural Labyrinth paired trials")
    parser.add_argument("--private-manifest", required=True)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--report", required=True, help="New output path; never overwrite")
    args = parser.parse_args(argv)
    try:
        report = analyze(_read_json(args.private_manifest), _read_json(args.trials))
        path = Path(args.report)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"paired analysis error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
