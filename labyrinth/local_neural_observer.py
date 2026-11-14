"""Bounded, source-public model-output observation over a loopback Ollama API.

The real HTTP requests provide an execution *observation*, not an attestation
of which weight bytes were loaded, external signer custody or neural learning.
The model process and all hashes remain under caller/host control.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib import parse, request
import re
import time

from .blind_structural_exam import PUBLIC_SCHEMA, RESPONSE_SCHEMA, _public_trial, _world
from .live_model_probe import (_NoRedirect, _canonical, _file_bytes, _http_json,
                               _load, _receipt, _write_once, ID, SHA, MAX_CASE_BYTES)

OBSERVATION_SCHEMA = "labyrinth.blind-model-local-http.v2"
LEGACY_OBSERVATION_SCHEMA = "labyrinth.blind-model-local-http.v1"
MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _loopback_endpoint(endpoint):
    if type(endpoint) is not str:
        raise ValueError("endpoint must be explicit literal 127.0.0.1")
    p = parse.urlsplit(endpoint)
    try:
        port = p.port
    except ValueError as exc:
        raise ValueError("invalid loopback endpoint port") from exc
    if (p.scheme != "http" or p.hostname != "127.0.0.1"
            or p.username is not None or p.password is not None
            or p.path not in ("", "/") or p.query or p.fragment
            or port is None or not 1 <= port <= 65535):
        raise ValueError("only explicit http://127.0.0.1:PORT loopback endpoint allowed")
    return f"http://127.0.0.1:{port}"


def _public_trials(raw):
    data = _load(raw)
    if (type(data) is not dict or set(data) != {"schema", "exam_id", "trials"}
            or data["schema"] != PUBLIC_SCHEMA
            or type(data["exam_id"]) is not str
            or ID.fullmatch(data["exam_id"]) is None
            or type(data["trials"]) is not list
            or not 1 <= len(data["trials"]) <= 16):
        raise ValueError("invalid public trial bundle schema")
    seen = set()
    for row in data["trials"]:
        if (type(row) is not dict or
                set(row) != {"trial_id", "world", "prompt"} or
                type(row["trial_id"]) is not str or
                ID.fullmatch(row["trial_id"]) is None or
                row["trial_id"] in seen or
                type(row["prompt"]) is not str):
            raise ValueError("public trial roster invalid or private answer leaked")
        seen.add(row["trial_id"])
        world = _world(row["world"])
        if row != _public_trial({"trial_id": row["trial_id"], "world": world}):
            raise ValueError("public trial differs from source-defined mechanics prompt")
    return data


def _declared_digest(opener, endpoint, model, timeout):
    result = _http_json(opener, endpoint + "/api/tags", timeout=timeout)
    models = result.get("models") if type(result) is dict else None
    if type(models) is not list or len(models) > 4096:
        raise ValueError("malformed server model tags")
    matching = [entry for entry in models if type(entry) is dict
                and entry.get("name") == model]
    if (len(matching) != 1
            or type(matching[0].get("digest")) is not str
            or SHA.fullmatch(matching[0]["digest"]) is None):
        raise ValueError("server model digest tag missing or ambiguous")
    return matching[0]["digest"]


def observe_public_exam(public_path, responses_out, observation_out, endpoint, model, *,
                        max_tokens=96, timeout=30):
    """Observe public questions through a bounded existing local Ollama server.

    Source does not read private answer files, run remote endpoints, start
    servers or modify model weights. This is not a trusted execution witness.
    """
    locations = [Path(public_path), Path(responses_out), Path(observation_out)]
    if (len({str(p.absolute()) for p in locations}) != 3 or
            any(p.exists() for p in locations[1:])):
        raise ValueError("observer requires distinct fresh output locations")
    base = _loopback_endpoint(endpoint)
    if (type(model) is not str or MODEL.fullmatch(model) is None or
            type(max_tokens) is not int or not 1 <= max_tokens <= 256 or
            type(timeout) not in (int, float) or not 1 <= timeout <= 120):
        raise ValueError("invalid bounded local model-generation settings")
    public_raw = _file_bytes(public_path, MAX_CASE_BYTES)
    bundle = _public_trials(public_raw)
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
    before_digest = _declared_digest(opener, base, model, timeout)
    tag_checks = 1
    rows = []
    observed_trials = []
    for trial in bundle["trials"]:
        # The original before/after-only check missed a model tag switched
        # for one case and restored before the final check. These additional
        # snapshots narrow that gap; they still cannot attest served weights.
        if _declared_digest(opener, base, model, timeout) != before_digest:
            raise ValueError("server advertised model digest changed before trial")
        tag_checks += 1
        start = time.monotonic()
        result = _http_json(opener, base + "/api/generate",
                            {"model": model, "prompt": trial["prompt"],
                             "stream": False, "keep_alive": "0s", "think": False,
                             "options": {"num_predict": max_tokens, "num_ctx": 2048,
                                         "num_thread": 2, "num_gpu": 0,
                                         "temperature": 0, "seed": 17}},
                            timeout=timeout)
        if (type(result) is not dict or result.get("model") != model
                or result.get("done") is not True
                or type(result.get("response")) is not str
                or len(result["response"]) > 128
                or type(result.get("done_reason")) is not str
                or result["done_reason"] != "stop"
                or type(result.get("eval_count")) is not int
                or not 0 <= result["eval_count"] <= 100000):
            raise ValueError("invalid local model reply or model identity")
        text = result["response"]
        if _declared_digest(opener, base, model, timeout) != before_digest:
            raise ValueError("server advertised model digest changed after trial")
        tag_checks += 1
        rows.append({"trial_id": trial["trial_id"], "response": text})
        observed_trials.append({
            "trial_id": trial["trial_id"],
            "prompt_sha256": _sha(trial["prompt"].encode("utf-8")),
            "response_sha256": _sha(text.encode("utf-8")),
            "done_reason": result["done_reason"],
            "eval_count": result["eval_count"],
            "elapsed_ms": max(0, round((time.monotonic() - start) * 1000)),
        })
    responses = {"schema": RESPONSE_SCHEMA, "exam_id": bundle["exam_id"],
                 "responses": rows}
    responses_raw = _canonical(responses) + b"\n"
    witness = _receipt({
        "schema": OBSERVATION_SCHEMA,
        "status": "LOCAL_LOOPBACK_HTTP_ONLY",
        "exam_id": bundle["exam_id"], "trial_count": len(rows),
        "public_file_sha256": _sha(public_raw),
        "responses_file_sha256": _sha(responses_raw),
        "model_tag": model, "declared_model_digest": before_digest,
        "model_tag_check_policy": "INITIAL_AND_PER_TRIAL_BEFORE_AFTER",
        "model_tag_check_count": tag_checks,
        "loopback_endpoint": base, "trials": observed_trials,
        "served_weight_bytes_attested": False,
        "independent_custody_verified": False,
        "neural_learning_verified": False,
        "limitations": [
            "HTTP responses and declared model tags are controlled by the local host.",
            "The tags pre/post snapshots cannot prove loaded intermediate weight bytes.",
            "Response receipt and externally supplied digest are not independent testimony.",
            "No training update, independent key custodian or protected evaluator is established.",
        ],
    })
    _write_once(responses_out, responses)
    _write_once(observation_out, witness)
    return witness


def verify_local_observation(observation_path, expected_observation_sha256,
                             public_raw, responses_raw):
    """Recheck original pinned source, exact response bytes and trial digests.

    An attacker controlling both the caller-supplied pin and all files can
    still forge this receipt; valid here means locally self-consistent only.
    """
    if (type(expected_observation_sha256) is not str
            or SHA.fullmatch(expected_observation_sha256) is None
            or type(public_raw) is not bytes or len(public_raw) > MAX_CASE_BYTES
            or type(responses_raw) is not bytes or len(responses_raw) > MAX_CASE_BYTES):
        raise ValueError("invalid original local observation pin or source bytes")
    bundle = _public_trials(public_raw)
    responses = _load(responses_raw)
    if (type(responses) is not dict
            or set(responses) != {"schema", "exam_id", "responses"}
            or responses["schema"] != RESPONSE_SCHEMA
            or responses["exam_id"] != bundle["exam_id"]
            or type(responses["responses"]) is not list
            or len(responses["responses"]) != len(bundle["trials"])):
        raise ValueError("invalid candidate response file")
    doc = _load(_file_bytes(observation_path, MAX_CASE_BYTES))
    common_fields = {"schema", "status", "exam_id", "trial_count",
                     "public_file_sha256", "responses_file_sha256", "model_tag",
                     "declared_model_digest", "loopback_endpoint", "trials",
                     "served_weight_bytes_attested", "independent_custody_verified",
                     "neural_learning_verified", "limitations", "receipt_sha256"}
    is_v2 = type(doc) is dict and doc.get("schema") == OBSERVATION_SCHEMA
    is_v1 = type(doc) is dict and doc.get("schema") == LEGACY_OBSERVATION_SCHEMA
    expected = common_fields | ({"model_tag_check_policy",
                                 "model_tag_check_count"} if is_v2 else set())
    if (type(doc) is not dict or not (is_v1 or is_v2)
            or set(doc) != expected
            or doc["status"] != "LOCAL_LOOPBACK_HTTP_ONLY"
            or type(doc["receipt_sha256"]) is not str
            or doc["receipt_sha256"] != expected_observation_sha256
            or _sha(_canonical({k: v for k, v in doc.items()
                                if k != "receipt_sha256"})) != doc["receipt_sha256"]
            or doc["exam_id"] != bundle["exam_id"]
            or type(doc["trial_count"]) is not int
            or doc["trial_count"] != len(bundle["trials"])
            or doc["public_file_sha256"] != _sha(public_raw)
            or doc["responses_file_sha256"] != _sha(responses_raw)
            or type(doc["model_tag"]) is not str
            or MODEL.fullmatch(doc["model_tag"]) is None
            or type(doc["declared_model_digest"]) is not str
            or SHA.fullmatch(doc["declared_model_digest"]) is None
            or (is_v2 and (
                doc["model_tag_check_policy"] != "INITIAL_AND_PER_TRIAL_BEFORE_AFTER"
                or type(doc["model_tag_check_count"]) is not int
                or doc["model_tag_check_count"] != 1 + 2 * len(bundle["trials"])))
            or doc["served_weight_bytes_attested"] is not False
            or doc["independent_custody_verified"] is not False
            or doc["neural_learning_verified"] is not False
            or type(doc["limitations"]) is not list
            or type(doc["trials"]) is not list
            or len(doc["trials"]) != len(bundle["trials"])):
        raise ValueError("local observation receipt pin or evidence ceiling mismatch")
    _loopback_endpoint(doc["loopback_endpoint"])
    for question, answer, observed in zip(
            bundle["trials"], responses["responses"], doc["trials"]):
        if (type(answer) is not dict or
                set(answer) != {"trial_id", "response"} or
                answer["trial_id"] != question["trial_id"] or
                type(answer["response"]) is not str or
                len(answer["response"]) > 128 or
                type(observed) is not dict or
                set(observed) != {"trial_id", "prompt_sha256", "response_sha256",
                                  "done_reason", "eval_count", "elapsed_ms"} or
                observed["trial_id"] != question["trial_id"] or
                observed["prompt_sha256"] !=
                _sha(question["prompt"].encode("utf-8")) or
                observed["response_sha256"] !=
                _sha(answer["response"].encode("utf-8")) or
                observed["done_reason"] not in (
                    ("stop",) if is_v2 else ("stop", "length")) or
                type(observed["eval_count"]) is not int or
                not 0 <= observed["eval_count"] <= 100000 or
                type(observed["elapsed_ms"]) is not int or
                not 0 <= observed["elapsed_ms"] <= 120000):
            raise ValueError("response bytes disagree with original local observation")
    # Historical V1 evidence is deliberately weaker. Older reports remain
    # auditable but never gain checks that were not executed at collection.
    evidence = ("LOCAL_HTTP_SELF_REPORTED_NOT_ATTESTED" if is_v2
                else "LOCAL_HTTP_V1_FINAL_ONLY_NOT_ATTESTED")
    return {"evidence": evidence,
            "declared_model_digest": doc["declared_model_digest"],
            "model_tag": doc["model_tag"],
            "served_weight_bytes_attested": False,
            "receipt_sha256": doc["receipt_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True)
    parser.add_argument("--responses-out", required=True)
    parser.add_argument("--observation-out", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args(argv)
    result = observe_public_exam(
        args.public, args.responses_out, args.observation_out,
        args.endpoint, args.model,
        max_tokens=args.max_tokens, timeout=args.timeout)
    print(json.dumps({"evidence": "LOCAL_HTTP_SELF_REPORTED_NOT_ATTESTED",
                      "receipt_sha256": result["receipt_sha256"],
                      "model": result["model_tag"], "trials": result["trial_count"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
