"""Bounded, write-once observations of local llama.cpp model responses.

This can observe a model, not certify learned reasoning, checkpoint training,
private evaluator custody, or that a process loaded the advertised weight bytes.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import time
from urllib import parse, request

CASE_SCHEMA = "labyrinth.live-probe-cases.v1"
COMMIT_SCHEMA = "labyrinth.live-probe-precommit.v1"
REPORT_SCHEMA = "labyrinth.live-probe-observation.v1"
COMPARE_SCHEMA = "labyrinth.live-probe-comparison.v1"
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z")
NUM = re.compile(r"-?[0-9]{1,8}\Z")
ANSWER = re.compile(r"ANSWER: (-?[0-9]{1,8})\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_CASE_BYTES = 128 * 1024
MAX_HTTP_BYTES = 128 * 1024


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _invalid_constant(value):
    raise ValueError("nonfinite JSON value")


def _load(raw: bytes):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                      parse_constant=_invalid_constant)


def _file_bytes(path: str | Path, limit: int) -> bytes:
    file = Path(path)
    before = file.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("input must be regular nonsymlink file")
    with file.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode) or
                (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)):
            raise ValueError("input identity changed before read")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (len(raw) > limit or (opened.st_dev, opened.st_ino) !=
            (after.st_dev, after.st_ino) or
            (opened.st_size, opened.st_mtime_ns) !=
            (after.st_size, after.st_mtime_ns)):
        raise ValueError("input exceeded limit or changed during read")
    return raw


def _cases(path: str | Path):
    raw = _file_bytes(path, MAX_CASE_BYTES)
    data = _load(raw)
    if type(data) is not dict or set(data) != {"schema", "cases"} or data["schema"] != CASE_SCHEMA:
        raise ValueError("unsupported probe case schema")
    rows = data["cases"]
    if type(rows) is not list or not 1 <= len(rows) <= 16:
        raise ValueError("probe requires 1..16 cases")
    ids = set()
    for row in rows:
        if (type(row) is not dict or set(row) != {"case_id", "prompt", "answer"}
                or type(row["case_id"]) is not str
                or ID.fullmatch(row["case_id"]) is None
                or row["case_id"] in ids
                or type(row["prompt"]) is not str
                or not 1 <= len(row["prompt"]) <= 2048
                or type(row["answer"]) is not str
                or NUM.fullmatch(row["answer"]) is None):
            raise ValueError("invalid or duplicate probe case")
        ids.add(row["case_id"])
    return raw, rows


def _write_once(path: str | Path, obj: dict) -> dict:
    payload = _canonical(obj)
    with Path(path).open("xb") as stream:
        stream.write(payload + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return obj


def commit_cases(cases_path, out_path):
    raw, rows = _cases(cases_path)
    payload = {
        "schema": COMMIT_SCHEMA,
        "cases_file_sha256": _sha(raw),
        "cases_canonical_sha256": _sha(_canonical(rows)),
        "case_count": len(rows),
        "status": "NOT_EXECUTED",
        "custody_class": "LOCAL_DEVELOPMENT_NOT_INDEPENDENT",
    }
    payload["receipt_sha256"] = _sha(_canonical(payload))
    return _write_once(out_path, payload)


def _read_commit(path):
    data = _load(_file_bytes(path, MAX_CASE_BYTES))
    if (type(data) is not dict or data.get("schema") != COMMIT_SCHEMA
            or type(data.get("receipt_sha256")) is not str):
        raise ValueError("invalid precommit receipt")
    expected = {k: v for k, v in data.items() if k != "receipt_sha256"}
    if _sha(_canonical(expected)) != data["receipt_sha256"]:
        raise ValueError("precommit receipt digest mismatch")
    return data


def _weight_sha(path: str | Path):
    model = Path(path)
    before = model.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise ValueError("model must be a nonempty regular nonsymlink file")
    digest = sha256()
    with model.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("model identity changed before hash")
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("model bytes changed during hash")
    return digest.hexdigest()


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _endpoint(url: str) -> str:
    if type(url) is not str:
        raise ValueError("endpoint must be string")
    parts = parse.urlsplit(url)
    if (parts.scheme != "http" or parts.username is not None
            or parts.password is not None or parts.path not in ("", "/")
            or parts.query or parts.fragment or not parts.hostname):
        raise ValueError("only plain loopback HTTP endpoint allowed")
    try:
        host = ipaddress.ip_address(parts.hostname)
        is_loopback = host.is_loopback
    except ValueError:
        is_loopback = parts.hostname.lower() == "localhost"
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("invalid endpoint port") from exc
    if not is_loopback or port is None:
        raise ValueError("only loopback endpoint with explicit port allowed")
    return url.rstrip("/")


def _http_json(opener, url, payload=None, timeout=30):
    data = _canonical(payload) if payload is not None else None
    req = request.Request(url, data=data,
                          headers={"Content-Type": "application/json"}
                          if data is not None else {},
                          method="POST" if data is not None else "GET")
    with opener.open(req, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("model HTTP request failed")
        raw = response.read(MAX_HTTP_BYTES + 1)
    if len(raw) > MAX_HTTP_BYTES:
        raise ValueError("model HTTP response too large")
    return _load(raw)


def _receipt(payload: dict) -> dict:
    payload["receipt_sha256"] = _sha(_canonical(payload))
    return payload


def observe_cases(cases_path, commitment_path, model_path, endpoint, out_path, *,
                  max_tokens=48, timeout=30):
    if Path(out_path).exists():
        raise FileExistsError("write-once observation exists")
    if (type(max_tokens) is not int or not 1 <= max_tokens <= 256 or
            type(timeout) not in (int, float) or not 1 <= timeout <= 120):
        raise ValueError("invalid bounded generation settings")
    base = _endpoint(endpoint)
    raw, rows = _cases(cases_path)
    plan = _read_commit(commitment_path)
    if (plan["cases_file_sha256"] != _sha(raw)
            or plan["cases_canonical_sha256"] != _sha(_canonical(rows))
            or plan["case_count"] != len(rows)):
        raise ValueError("case file differs from precommit commitment")
    model_hash = _weight_sha(model_path)
    model_id = str(Path(model_path).absolute())
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
    models = _http_json(opener, base + "/v1/models", timeout=timeout)
    advertised = models.get("data") if type(models) is dict else None
    if (type(advertised) is not list or
            not any(type(item) is dict and item.get("id") == model_id
                    for item in advertised)):
        raise ValueError("server advertised model does not match verified file path")

    trials = []
    for row in rows:
        start = time.monotonic()
        result = _http_json(
            opener, base + "/v1/chat/completions",
            {"model": model_id,
             "messages": [{"role": "user", "content": row["prompt"]}],
             "temperature": 0, "seed": 17, "max_tokens": max_tokens,
             "stream": False},
            timeout=timeout)
        choices = result.get("choices") if type(result) is dict else None
        if (type(choices) is not list or len(choices) < 1
                or type(choices[0]) is not dict or
                type(choices[0].get("message")) is not dict or
                type(choices[0]["message"].get("content")) is not str):
            raise ValueError("malformed model response")
        text = choices[0]["message"]["content"]
        if len(text) > 4096:
            raise ValueError("model response text exceeds limit")
        found = ANSWER.fullmatch(text.strip())
        trials.append({
            "case_id": row["case_id"],
            "prompt_sha256": _sha(row["prompt"].encode("utf-8")),
            "response": text,
            "response_sha256": _sha(text.encode("utf-8")),
            "format_valid": found is not None,
            "correct": bool(found and found.group(1) == row["answer"]),
            "elapsed_ms": round((time.monotonic() - start) * 1000),
        })
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "OBSERVED_SINGLE_MODEL_ONLY",
        "commitment_sha256": plan["receipt_sha256"],
        "cases_file_sha256": plan["cases_file_sha256"],
        "model": {"path": model_id, "sha256": model_hash,
                  "server_advertised_path_match": True,
                  "server_loaded_weight_bytes_independently_attested": False},
        "endpoint": base, "case_count": len(rows),
        "correct": sum(row["correct"] for row in trials),
        "format_valid": sum(row["format_valid"] for row in trials),
        "trials": trials,
        "neural_learning_verified": False,
        "independent_custody_verified": False,
        "limitations": [
            "Source data and server identity remain controlled by the caller.",
            "A server path announcement cannot prove which model bytes were loaded.",
            "A single observed model is not a before-after training comparison.",
            "Exact-answer scoring does not demonstrate general reasoning.",
        ],
    }
    return _write_once(out_path, _receipt(payload))


def _read_observation(path):
    data = _load(_file_bytes(path, MAX_CASE_BYTES))
    if type(data) is not dict or data.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported observation")
    given = data.get("receipt_sha256")
    if type(given) is not str or SHA.fullmatch(given) is None:
        raise ValueError("observation receipt digest missing")
    if _sha(_canonical({k: v for k, v in data.items() if k != "receipt_sha256"})) != given:
        raise ValueError("observation receipt digest mismatch")
    if (data.get("status") != "OBSERVED_SINGLE_MODEL_ONLY"
            or type(data.get("trials")) is not list
            or len(data["trials"]) != data.get("case_count")):
        raise ValueError("observation is not a complete live probe")
    # An unkeyed digest protects against accidental edits, not a caller who
    # can recompute it. Still enforce all internally checkable evidence before
    # producing a numerical comparison. Actual correctness requires the
    # independently held case answers and is NOT attested by these flags.
    count = data.get("case_count")
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError("invalid observation case count")
    if (type(data.get("correct")) is not int or
            type(data.get("format_valid")) is not int):
        raise ValueError("invalid observation aggregate types")
    if (type(data.get("commitment_sha256")) is not str or
            SHA.fullmatch(data["commitment_sha256"]) is None or
            type(data.get("cases_file_sha256")) is not str or
            SHA.fullmatch(data["cases_file_sha256"]) is None):
        raise ValueError("invalid observation case binding")
    model = data.get("model")
    if (type(model) is not dict or type(model.get("path")) is not str
            or not model["path"] or type(model.get("sha256")) is not str
            or SHA.fullmatch(model["sha256"]) is None
            or model.get("server_advertised_path_match") is not True
            or model.get("server_loaded_weight_bytes_independently_attested") is not False
            or data.get("neural_learning_verified") is not False
            or data.get("independent_custody_verified") is not False):
        raise ValueError("invalid observation model evidence ceiling")
    seen = set()
    observed_correct = 0
    observed_format = 0
    required = {"case_id", "prompt_sha256", "response", "response_sha256",
                "format_valid", "correct", "elapsed_ms"}
    for trial in data["trials"]:
        if (type(trial) is not dict or set(trial) != required
                or type(trial["case_id"]) is not str
                or ID.fullmatch(trial["case_id"]) is None
                or trial["case_id"] in seen):
            raise ValueError("invalid observation duplicate or malformed trial")
        seen.add(trial["case_id"])
        response = trial["response"]
        if (type(response) is not str or len(response) > 4096
                or type(trial["prompt_sha256"]) is not str
                or SHA.fullmatch(trial["prompt_sha256"]) is None
                or type(trial["response_sha256"]) is not str
                or SHA.fullmatch(trial["response_sha256"]) is None
                or _sha(response.encode("utf-8")) != trial["response_sha256"]):
            raise ValueError("invalid observation response hash or payload")
        if (type(trial["format_valid"]) is not bool or
                type(trial["correct"]) is not bool or
                type(trial["elapsed_ms"]) is not int or
                trial["elapsed_ms"] < 0):
            raise ValueError("invalid observation trial types")
        valid_format = ANSWER.fullmatch(response.strip()) is not None
        if (trial["format_valid"] is not valid_format or
                (trial["correct"] and not valid_format)):
            raise ValueError("invalid observation format or correctness flags")
        observed_format += int(valid_format)
        observed_correct += int(trial["correct"])
    if (observed_correct != data["correct"] or
            observed_format != data["format_valid"]):
        raise ValueError("observation aggregate score disagrees with trials")
    return data


def compare_reports(before_path, after_path, out_path):
    if Path(out_path).exists():
        raise FileExistsError("write-once comparison exists")
    before, after = _read_observation(before_path), _read_observation(after_path)
    if (before["commitment_sha256"] != after["commitment_sha256"]
            or before["cases_file_sha256"] != after["cases_file_sha256"]
            or [t["case_id"] for t in before["trials"]] !=
            [t["case_id"] for t in after["trials"]]):
        raise ValueError("different committed case rosters cannot be compared")
    same = before["model"]["sha256"] == after["model"]["sha256"]
    payload = {
        "schema": COMPARE_SCHEMA,
        "status": ("SAME_CHECKPOINT_NO_WEIGHT_UPDATE" if same
                   else "PAIRED_BEHAVIOR_NOT_TRAINING_ATTESTED"),
        "before_receipt_sha256": before["receipt_sha256"],
        "after_receipt_sha256": after["receipt_sha256"],
        "before_model_sha256": before["model"]["sha256"],
        "after_model_sha256": after["model"]["sha256"],
        "case_count": before["case_count"],
        "before_correct": before["correct"],
        "after_correct": after["correct"],
        "paired_candidate_only_wins": sum(
            not a["correct"] and b["correct"]
            for a, b in zip(before["trials"], after["trials"])),
        "paired_candidate_only_losses": sum(
            a["correct"] and not b["correct"]
            for a, b in zip(before["trials"], after["trials"])),
        "neural_learning_verified": False,
        "independent_custody_verified": False,
        "limitations": [
            "Distinct model hashes do not establish training lineage.",
            "Neither observation authenticates model serving-process weights.",
            "Evaluator and private-case independence are not attested.",
        ],
    }
    return _write_once(out_path, _receipt(payload))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    commit = sub.add_parser("commit", help="write-once case answer-key commitment")
    commit.add_argument("--cases", required=True)
    commit.add_argument("--out", required=True)
    run = sub.add_parser("observe", help="observe existing localhost llama.cpp server")
    run.add_argument("--cases", required=True)
    run.add_argument("--commitment", required=True)
    run.add_argument("--model-file", required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--max-tokens", type=int, default=48)
    run.add_argument("--timeout", type=int, default=30)
    compare = sub.add_parser("compare", help="compare two already observed models")
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "commit":
        result = commit_cases(args.cases, args.out)
    elif args.cmd == "observe":
        result = observe_cases(args.cases, args.commitment, args.model_file,
                               args.endpoint, args.out, max_tokens=args.max_tokens,
                               timeout=args.timeout)
    else:
        result = compare_reports(args.before, args.after, args.out)
    print(json.dumps({"status": result["status"], "receipt_sha256":
                      result["receipt_sha256"], "out": args.out},
                     sort_keys=True))


if __name__ == "__main__":
    main()
