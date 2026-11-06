"""Matched W0/Wt public development probes with fresh-process file reload.

Runs one new Python process per checkpoint. Every worker re-hashes the FULL
base model, tokenizer/config and adapter file manifest before/after inference;
the parent independently rechecks them too. A bytes-only hash delta is not
skill improvement. Even an observed matched public gain is not evidence of
training causality, independence, private graduation or production fitness.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit

from .entrance_state_v1 import (
    SCHEMA as ENTRANCE_SCHEMA, _inventory, _state_digest, verify_entrance_state,
)


SCHEMA = "labyrinth.checkpoint-predictor-compare.v1"
_ID = re.compile(r"devprobe-[A-Za-z0-9_.:-]{1,100}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_OUTPUT = re.compile(r"\s*ANSWER:\s*([+-]?\d+)\s*\Z", re.ASCII)
_INT = re.compile(r"[+-]?(?:0|[1-9][0-9]{0,15})\Z", re.ASCII)
_MAX_PROBES = 32
_PREDICTORS = {
    "mock-local-file-arithmetic", "mock-bytes-digest-only",
    "hf-local-file", "localhost-self-attested",
}


def validate_local_predictor_endpoint(endpoint: Any) -> str:
    """Never allow remote URLs, userinfo, redirects, proxies or file URLs."""
    if type(endpoint) is not str or len(endpoint) > 160:
        raise ValueError("bounded localhost-only predictor URL required")
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1")
            or parsed.username is not None or parsed.password is not None
            or parsed.port is None or not 1 <= parsed.port <= 65535
            or parsed.query or parsed.fragment
            or parsed.path not in ("", "/infer")):
        raise ValueError("predictor endpoint must be exact plaintext loopback only")
    return endpoint


def _probes(value: Any) -> list[dict[str, str]]:
    if type(value) is not list or not 1 <= len(value) <= _MAX_PROBES:
        raise ValueError("bounded public development probe cohort required")
    items = []
    ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for item in value:
        if (type(item) is not dict or set(item) != {"case_id", "skill", "prompt", "expected"}
                or any(type(item[key]) is not str for key in item)
                or _ID.fullmatch(item["case_id"]) is None
                or not 1 <= len(item["prompt"].encode("utf-8")) <= 512
                or not 1 <= len(item["skill"]) <= 64
                or _INT.fullmatch(item["expected"]) is None):
            raise ValueError("only exact development-only arithmetic probe cases accepted")
        digest = hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest()
        if item["case_id"] in ids or digest in prompt_hashes:
            raise ValueError("repeated public development prompt/case is prohibited")
        ids.add(item["case_id"])
        prompt_hashes.add(digest)
        items.append({**item, "prompt_sha256": digest})
    return items


def _run_worker(payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    env = os.environ.copy()
    for name in list(env):
        if name.upper().endswith("_PROXY"):
            del env[name]
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "NO_PROXY": "*", "no_proxy": "*"})
    code_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "labyrinth.checkpoint_predictor_worker"],
        input=json.dumps(payload, separators=(",", ":"), allow_nan=False),
        text=True, capture_output=True, cwd=code_root, env=env,
        timeout=timeout_seconds, check=False,
    )
    if result.returncode != 0:
        raise ValueError("fresh checkpoint predictor worker refused local load or inference: "
                         + result.stderr[-500:])
    if len(result.stdout.encode("utf-8")) > 64 * 1024:
        raise ValueError("fresh predictor response exceeds 64KiB")
    output = json.loads(result.stdout)
    if type(output) is not dict:
        raise ValueError("fresh predictor returned malformed JSON")
    return output


def _validate_worker(
    output: dict[str, Any], *, payload: dict[str, Any],
    independent_pid: int | None, scoring_probes: list[dict[str, str]],
) -> dict[str, Any]:
    # The subprocess is an inference surface, never a holder of the scorer's
    # expected values. Retain the answer key in this parent process only.
    if payload["public_probes"] != [
        {key: probe[key] for key in ("case_id", "skill", "prompt", "prompt_sha256")}
        for probe in scoring_probes
    ]:
        raise ValueError("predictor payload differs from prompt-only score cohort")
    if (set(output) != {"schema", "nonce", "pid", "loaded_state_sha256",
                        "worker_source_sha256", "runtime_binding", "predictions"}
            or output["schema"] != "labyrinth.checkpoint-predictor-worker.v1"
            or output["nonce"] != payload["nonce"]):
        raise ValueError("fresh predictor nonce/schema response is stale or forged")
    if (type(output["pid"]) is not int or output["pid"] == os.getpid()
            or output["pid"] == independent_pid):
        raise ValueError("predictor did not use an independent fresh process")
    if output["loaded_state_sha256"] != payload["loaded_state_sha256"]:
        raise ValueError("worker loaded a stale or substituted checkpoint state digest")
    if output["worker_source_sha256"] != payload["worker_source_sha256"]:
        raise ValueError("predictor worker source is stale or substituted")
    if (type(output["predictions"]) is not list
            or len(output["predictions"]) != len(payload["public_probes"])
            or any(type(answer) is not str or len(answer) > 512
                   for answer in output["predictions"])):
        raise ValueError("fresh predictor emitted malformed or missing public answers")
    if output["runtime_binding"] not in {
        "LOCAL_MODEL_BYTES_READ_BY_WORKER", "LOCAL_MODEL_BYTES_READ_BY_WORKER_NO_SKILL",
        "LOCAL_HF_WEIGHTS_LOADED_BY_WORKER_UNATTESTED",
        "UNVERIFIED_LOCALHOST_SERVICE_HASH_ECHO_ONLY",
    }:
        raise ValueError("worker reported unapproved runtime binding mode")
    expected_binding = {
        "mock-local-file-arithmetic": "LOCAL_MODEL_BYTES_READ_BY_WORKER",
        "mock-bytes-digest-only": "LOCAL_MODEL_BYTES_READ_BY_WORKER_NO_SKILL",
        "hf-local-file": "LOCAL_HF_WEIGHTS_LOADED_BY_WORKER_UNATTESTED",
        "localhost-self-attested": "UNVERIFIED_LOCALHOST_SERVICE_HASH_ECHO_ONLY",
    }[payload["predictor"]]
    if output["runtime_binding"] != expected_binding:
        raise ValueError("worker mode and actual runtime binding disagree")
    correct = 0
    details = []
    for probe, prediction in zip(scoring_probes, output["predictions"], strict=True):
        match = _OUTPUT.fullmatch(prediction)
        scored = match is not None and match[1] == probe["expected"]
        correct += int(scored)
        details.append({"case_id": probe["case_id"],
                        "prompt_sha256": probe["prompt_sha256"],
                        "expected": probe["expected"], "prediction": prediction,
                        "correct": scored})
    return {"pid": output["pid"], "loaded_state_sha256": output["loaded_state_sha256"],
            "runtime_binding": output["runtime_binding"], "correct": correct,
            "count": len(details), "cases": details}


def compare_fresh_process_checkpoints(
    *, entrance_path: Path, successor_adapter_dir: Path,
    public_probes: list[dict], expected_successor_sha256: str,
    predictor: str = "hf-local-file",
    localhost_endpoint: str | None = None,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    """Two independent fresh-process public probes; protected promotion excluded.

    The caller must pin W0 and Wt outside this function to defeat an attacker
    replacing both files and the whole manifest. The actual **full** base
    shard/tokenizer/config inventory is re-hashed, not a single model file.
    """
    if predictor not in _PREDICTORS or type(predictor) is not str:
        raise ValueError("unapproved checkpoint predictor")
    if type(timeout_seconds) not in (int, float) or not 1 <= timeout_seconds <= 180:
        raise ValueError("fresh-process timeout must be bounded")
    if predictor == "localhost-self-attested":
        validate_local_predictor_endpoint(localhost_endpoint)
    elif localhost_endpoint is not None:
        raise ValueError("file predictor cannot use localhost service endpoint")
    checked = _probes(public_probes)
    prompt_only_probes = [
        {key: probe[key] for key in ("case_id", "skill", "prompt", "prompt_sha256")}
        for probe in checked
    ]
    verify_entrance_state(entrance_path)
    entrance = json.loads(entrance_path.read_bytes())
    if entrance.get("schema") != ENTRANCE_SCHEMA or entrance.get("adapter") is None:
        raise ValueError("fresh W0 requires an actual pinned entrance adapter")
    base = _inventory(Path(entrance["base"]["root"]), kind="base")
    w0_adapter = _inventory(Path(entrance["adapter"]["root"]), kind="adapter")
    wt_adapter = _inventory(successor_adapter_dir, kind="adapter")
    if base != entrance["base"] or w0_adapter != entrance["adapter"]:
        raise ValueError("W0 entrance source model identity changed")
    w0_sha = _state_digest(base, w0_adapter)
    wt_sha = _state_digest(base, wt_adapter)
    if type(expected_successor_sha256) is not str or _SHA.fullmatch(expected_successor_sha256) is None:
        raise ValueError("successor digest must be exact SHA-256")
    if wt_sha != expected_successor_sha256:
        raise ValueError("successor checkpoint state digest does not match caller pin")
    if w0_sha != entrance["w0_state_sha256"]:
        raise ValueError("entrance W0 checkpoint digest changed")
    worker_file = Path(__file__).resolve().with_name("checkpoint_predictor_worker.py")
    worker_source_sha256 = hashlib.sha256(worker_file.read_bytes()).hexdigest()
    if len(json.dumps(checked, separators=(",", ":")).encode()) > 32768:
        raise ValueError("public development probe cohort is too large")

    observed: list[dict[str, Any]] = []
    for adapter, state_sha in ((w0_adapter, w0_sha), (wt_adapter, wt_sha)):
        nonce = secrets.token_hex(32)
        payload = {
            "schema": "labyrinth.checkpoint-predictor-request.v1",
            "nonce": nonce, "predictor": predictor, "base": base,
            "adapter": adapter, "loaded_state_sha256": state_sha,
            "public_probes": prompt_only_probes, "localhost_endpoint": localhost_endpoint,
            "worker_source_sha256": worker_source_sha256,
        }
        raw_result = _run_worker(payload, timeout_seconds=float(timeout_seconds))
        # A worker response alone is not proof: rehash complete source bytes.
        if (_inventory(Path(base["root"]), kind="base") != base
                or _inventory(Path(adapter["root"]), kind="adapter") != adapter):
            raise ValueError("checkpoint bytes changed around fresh-process inference")
        observed.append(_validate_worker(
            raw_result, payload=payload,
            independent_pid=observed[0]["pid"] if observed else None,
            scoring_probes=checked,
        ))
    w0, wt = observed
    status = ("UNQUALIFIED_LOCALHOST_SERVICE_SELF_REPORT"
              if predictor == "localhost-self-attested" else
              "PUBLIC_PROBE_GAIN_OBSERVED" if wt["correct"] > w0["correct"]
              else "NO_PUBLIC_PROBE_GAIN")
    evidence = ("LOCAL_TEST_PREDICTOR_NOT_REAL_LLM"
                if predictor.startswith("mock-") else
                "LOCALHOST_SELF_ATTESTATION_ONLY"
                if predictor == "localhost-self-attested" else
                "FRESH_PROCESS_LOCAL_HF_SOURCE_OBSERVATION")
    return {
        "schema": SCHEMA, "development_status": status,
        "evidence_kind": evidence, "probe_partition": "PUBLIC_DEVELOPMENT_ONLY",
        "entrance_manifest_sha256": entrance["manifest_sha256"],
        "source_tree_sha256": entrance["source_tree_sha256"],
        "predictor_worker_source_sha256": worker_source_sha256,
        "w0": w0, "wt": wt, "checkpoint_bytes_differ": w0_sha != wt_sha,
        "causal_training_effect_established": False,
        "graduation_authorized": False,
        "limitations": [
            "Matched public development gains are not independently blinded transfer evidence.",
            "File checksums verify exact bytes, not whether gradients caused the update.",
            "No protected exams, Q5-Q9, private data or promotion permissions were used.",
            "Localhost service reports its own model identity; this cannot prove which weights it truly serves."
            if predictor == "localhost-self-attested" else
            "Local load/inference is observed only under the selected predictor implementation.",
        ],
    }
