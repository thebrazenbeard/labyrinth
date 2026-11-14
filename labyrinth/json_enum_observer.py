"""Bounded source-public model experiment with syntax-only JSON enumeration.

The allowed outputs contain EVERY action trace through the bounded horizon
plus UNSAT. They encode no hidden oracle or winning action. The raw model
JSON is retained alongside the extracted candidate answer, permitting
external exact-file replay without pretending the model emitted plain text.

This is a local host observation, NOT an attestation of model weight bytes,
independent examiner custody, or learning/generalization.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import time
from urllib import request

from .blind_structural_exam import RESPONSE_SCHEMA
from .local_neural_observer import (_declared_digest, _loopback_endpoint,
                                    _public_trials, MODEL)
from .live_model_probe import (_NoRedirect, _canonical, _file_bytes, _http_json,
                               _receipt, _write_once, SHA, MAX_CASE_BYTES)

OBSERVATION_SCHEMA = "labyrinth.blind-json-enum-local-http.v1"
EVIDENCE = "LOCAL_HTTP_JSON_ENUM_SELF_REPORTED_NOT_ATTESTED"
STATUS = "LOCAL_JSON_ENUM_LOOPBACK_HTTP_ONLY"
POLICY = "PUBLIC_ALL_PLANS_UP_TO_THREE_PLUS_UNSAT"
GENERATION_POLICY = "CPU_TEMPERATURE_ZERO_SEED17_JSON_ENUM"
MAX_RAW_JSON = 512


def _sha(raw):
    return sha256(raw).hexdigest()


def _syntactic_schema(world):
    """Build a complete set of legal answer strings, not winning labels."""
    bound = world["max_moves"]
    if type(bound) is not int or not 1 <= bound <= 3:
        raise ValueError("JSON enum experiment only supports max_moves 1..3")
    choices = ["UNSAT"]
    for length in range(1, bound + 1):
        choices.extend("PLAN: " + " ".join(actions)
                       for actions in product(("A", "B", "C"), repeat=length))
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": choices}},
        "required": ["answer"],
        "additionalProperties": False,
    }


def _no_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate model JSON property")
        value[key] = item
    return value


def _parse_model_json(raw, schema):
    if type(raw) is not str or not 0 < len(raw) <= MAX_RAW_JSON:
        raise ValueError("invalid or oversized JSON model response")
    try:
        response = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, TypeError) as exc:
        raise ValueError("model response is not strict JSON") from exc
    if (type(response) is not dict or set(response) != {"answer"}
            or type(response["answer"]) is not str
            or response["answer"] not in schema["properties"]["answer"]["enum"]):
        raise ValueError("model JSON has extra fields or invalid answer")
    return response["answer"]


def observe_json_enum_exam(public_path, responses_out, observation_out,
                           endpoint, model, *, max_tokens=128, timeout=60):
    """Call Ollama using exact public prompts and syntax-only JSON format."""
    inputs = [Path(public_path), Path(responses_out), Path(observation_out)]
    if (len({str(p.absolute()) for p in inputs}) != 3
            or any(p.exists() for p in inputs[1:])):
        raise ValueError("observer requires three distinct paths and fresh outputs")
    base = _loopback_endpoint(endpoint)
    if (type(model) is not str or MODEL.fullmatch(model) is None
            or type(max_tokens) is not int or not 1 <= max_tokens <= 256
            or type(timeout) not in (int, float) or not 1 <= timeout <= 120):
        raise ValueError("invalid model or bounded generation settings")
    public_raw = _file_bytes(public_path, MAX_CASE_BYTES)
    bundle = _public_trials(public_raw)
    formats = [_syntactic_schema(t["world"]) for t in bundle["trials"]]
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
    declared = _declared_digest(opener, base, model, timeout)
    tag_checks = 1
    rows = []
    observations = []
    for trial, fmt in zip(bundle["trials"], formats):
        if _declared_digest(opener, base, model, timeout) != declared:
            raise ValueError("advertised model digest changed before trial")
        tag_checks += 1
        started = time.monotonic()
        result = _http_json(
            opener, base + "/api/generate",
            {
                "model": model, "prompt": trial["prompt"], "format": fmt,
                "stream": False, "think": False, "keep_alive": "0s",
                "options": {
                    "num_predict": max_tokens, "num_ctx": 2048, "num_thread": 2,
                    "num_gpu": 0, "temperature": 0, "seed": 17,
                },
            }, timeout=timeout)
        if (type(result) is not dict or result.get("model") != model
                or result.get("done") is not True
                or result.get("done_reason") != "stop"
                or type(result.get("eval_count")) is not int
                or not 0 <= result["eval_count"] <= 100000):
            raise ValueError("invalid or incomplete JSON model reply")
        raw = result.get("response")
        answer = _parse_model_json(raw, fmt)
        if _declared_digest(opener, base, model, timeout) != declared:
            raise ValueError("advertised model digest changed after trial")
        tag_checks += 1
        rows.append({"trial_id": trial["trial_id"], "response": answer})
        observations.append({
            "trial_id": trial["trial_id"],
            "prompt_sha256": _sha(trial["prompt"].encode("utf-8")),
            "format_sha256": _sha(_canonical(fmt)),
            "raw_json_response": raw,
            "raw_json_sha256": _sha(raw.encode("utf-8")),
            "answer_sha256": _sha(answer.encode("utf-8")),
            "done_reason": result["done_reason"],
            "eval_count": result["eval_count"],
            "elapsed_ms": max(0, round((time.monotonic() - started) * 1000)),
        })
    responses = {
        "schema": RESPONSE_SCHEMA, "exam_id": bundle["exam_id"],
        "responses": rows,
    }
    original_responses = _canonical(responses) + b"\n"
    receipt = _receipt({
        "schema": OBSERVATION_SCHEMA, "status": STATUS,
        "exam_id": bundle["exam_id"], "trial_count": len(rows),
        "public_file_sha256": _sha(public_raw),
        "responses_file_sha256": _sha(original_responses),
        "model_tag": model, "declared_model_digest": declared,
        "model_tag_check_policy": "INITIAL_AND_PER_TRIAL_BEFORE_AFTER",
        "model_tag_check_count": tag_checks,
        "loopback_endpoint": base,
        "constraint_policy": POLICY,
        "generation_policy": GENERATION_POLICY,
        "max_tokens": max_tokens,
        "trials": observations,
        "served_weight_bytes_attested": False,
        "independent_custody_verified": False,
        "neural_learning_verified": False,
        "limitations": [
            "All enum choices are syntactic, not a winning oracle.",
            "The response JSON was generated by a host-controlled HTTP endpoint.",
            "The model tag digest is advertised by that same untrusted host.",
            "No trained checkpoint or independent examiner is verified.",
        ],
    })
    _write_once(responses_out, responses)
    _write_once(observation_out, receipt)
    return receipt


def verify_json_enum_observation(observation_path, expected_observation_sha256,
                                 public_raw, responses_raw):
    """Replay raw model JSON against exact original held source-byte hashes."""
    if (type(expected_observation_sha256) is not str
            or SHA.fullmatch(expected_observation_sha256) is None
            or type(public_raw) is not bytes or len(public_raw) > MAX_CASE_BYTES
            or type(responses_raw) is not bytes
            or len(responses_raw) > MAX_CASE_BYTES):
        raise ValueError("invalid original JSON observation pin or source bytes")
    bundle = _public_trials(public_raw)
    from .live_model_probe import _load
    responses = _load(responses_raw)
    if (type(responses) is not dict
            or set(responses) != {"schema", "exam_id", "responses"}
            or responses["schema"] != RESPONSE_SCHEMA
            or responses["exam_id"] != bundle["exam_id"]
            or type(responses["responses"]) is not list
            or len(responses["responses"]) != len(bundle["trials"])):
        raise ValueError("invalid JSON-constrained candidate response roster")
    raw_receipt = _file_bytes(observation_path, MAX_CASE_BYTES)
    try:
        doc = json.loads(raw_receipt, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid original constrained HTTP observation") from exc
    required = {
        "schema", "status", "exam_id", "trial_count", "public_file_sha256",
        "responses_file_sha256", "model_tag", "declared_model_digest",
        "model_tag_check_policy", "model_tag_check_count",
        "loopback_endpoint", "constraint_policy", "generation_policy",
        "max_tokens", "trials", "served_weight_bytes_attested",
        "independent_custody_verified", "neural_learning_verified",
        "limitations", "receipt_sha256",
    }
    if (type(doc) is not dict or set(doc) != required
            or doc["schema"] != OBSERVATION_SCHEMA
            or doc["status"] != STATUS
            or type(doc["receipt_sha256"]) is not str
            or doc["receipt_sha256"] != expected_observation_sha256
            or _sha(_canonical({key: value for key, value in doc.items()
                                if key != "receipt_sha256"}))
               != doc["receipt_sha256"]
            or doc["exam_id"] != bundle["exam_id"]
            or type(doc["trial_count"]) is not int
            or doc["trial_count"] != len(bundle["trials"])
            or doc["public_file_sha256"] != _sha(public_raw)
            or doc["responses_file_sha256"] != _sha(responses_raw)
            or type(doc["model_tag"]) is not str
            or MODEL.fullmatch(doc["model_tag"]) is None
            or type(doc["declared_model_digest"]) is not str
            or SHA.fullmatch(doc["declared_model_digest"]) is None
            or doc["model_tag_check_policy"] != "INITIAL_AND_PER_TRIAL_BEFORE_AFTER"
            or type(doc["model_tag_check_count"]) is not int
            or doc["model_tag_check_count"] != 1 + 2 * len(bundle["trials"])
            or doc["constraint_policy"] != POLICY
            or doc["generation_policy"] != GENERATION_POLICY
            or type(doc["max_tokens"]) is not int
            or not 1 <= doc["max_tokens"] <= 256
            or type(doc["loopback_endpoint"]) is not str
            or doc["loopback_endpoint"] != _loopback_endpoint(doc["loopback_endpoint"])
            or doc["served_weight_bytes_attested"] is not False
            or doc["independent_custody_verified"] is not False
            or doc["neural_learning_verified"] is not False
            or type(doc["trials"]) is not list
            or len(doc["trials"]) != len(bundle["trials"])
            or type(doc["limitations"]) is not list):
        raise ValueError("original constrained HTTP observation or pin mismatch")
    trial_fields = {
        "trial_id", "prompt_sha256", "format_sha256", "raw_json_response",
        "raw_json_sha256", "answer_sha256", "done_reason", "eval_count",
        "elapsed_ms",
    }
    for trial, answer, observed in zip(
            bundle["trials"], responses["responses"], doc["trials"]):
        fmt = _syntactic_schema(trial["world"])
        if (type(answer) is not dict
                or set(answer) != {"trial_id", "response"}
                or answer["trial_id"] != trial["trial_id"]
                or type(answer["response"]) is not str
                or type(observed) is not dict
                or set(observed) != trial_fields
                or observed["trial_id"] != trial["trial_id"]
                or observed["prompt_sha256"] !=
                   _sha(trial["prompt"].encode("utf-8"))
                or observed["format_sha256"] != _sha(_canonical(fmt))
                or type(observed["raw_json_response"]) is not str
                or observed["raw_json_sha256"] !=
                   _sha(observed["raw_json_response"].encode("utf-8"))
                or observed["answer_sha256"] !=
                   _sha(answer["response"].encode("utf-8"))
                or observed["done_reason"] != "stop"
                or type(observed["eval_count"]) is not int
                or not 0 <= observed["eval_count"] <= 100000
                or type(observed["elapsed_ms"]) is not int
                or not 0 <= observed["elapsed_ms"] <= 120000):
            raise ValueError("constrained raw JSON observation disagrees with source")
        if _parse_model_json(observed["raw_json_response"], fmt) != answer["response"]:
            raise ValueError("candidate answer diverged from original raw JSON model text")
    return {
        "evidence": EVIDENCE,
        "declared_model_digest": doc["declared_model_digest"],
        "model_tag": doc["model_tag"],
        "served_weight_bytes_attested": False,
        "receipt_sha256": doc["receipt_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True)
    parser.add_argument("--responses-out", required=True)
    parser.add_argument("--observation-out", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    receipt = observe_json_enum_exam(
        args.public, args.responses_out, args.observation_out,
        args.endpoint, args.model, max_tokens=args.max_tokens,
        timeout=args.timeout)
    print(json.dumps({
        "evidence": EVIDENCE, "model": receipt["model_tag"],
        "trials": receipt["trial_count"],
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
