"""Public two-model conversational tournament; no inference or release authority.

Do not conflate this experiment with the same-model A/B/placebo causal design
in Mirror Labyrinth. A difference between models cannot establish the effect
of Vera's source-conditioned cognitive donors.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import random
import re
from typing import Any


MAX_TRIALS = 96
MAX_TEXT = 8192
MAX_CASES = 24
FIELDS = frozenset({
    "case_id", "repeat", "model_sha256", "prompt", "system", "input_sha256",
    "response", "stop_reason", "actual_model_generated", "runner_build",
    "decoding", "chat_template_sha256",
})
DECODE_FIELDS = frozenset({"seed", "temperature", "num_ctx", "num_predict", "num_gpu"})


def canonical_digest(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("invalid canonical evidence") from exc
    return sha256(raw).hexdigest()


def _sha(value: Any) -> str:
    if (type(value) is not str or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise ValueError("invalid lower-case SHA-256")
    return value


def _cases(bank: dict[str, str]) -> None:
    if type(bank) is not dict or not 2 <= len(bank) <= MAX_CASES:
        raise ValueError("bounded public case bank required")
    for case_id, prompt in bank.items():
        if (type(case_id) is not str or not re.fullmatch(r"[a-z0-9_-]{1,48}", case_id)
                or type(prompt) is not str or not 1 <= len(prompt) <= 2048):
            raise ValueError("invalid public case ID/prompt")
    canonical_digest(bank)


def _record(record: dict[str, Any], bank: dict[str, str]) -> None:
    if type(record) is not dict or set(record) != FIELDS:
        raise ValueError("trial schema mismatch")
    if type(record["case_id"]) is not str or record["case_id"] not in bank:
        raise ValueError("unknown case")
    if type(record["repeat"]) is not int or not 0 <= record["repeat"] < 16:
        raise ValueError("invalid repeat")
    if record["prompt"] != bank[record["case_id"]] or type(record["system"]) is not str:
        raise ValueError("public case prompt changed")
    if len(record["system"]) > 4096:
        raise ValueError("unbounded system context")
    if (type(record["response"]) is not str
            or not 1 <= len(record["response"]) <= MAX_TEXT):
        raise ValueError("missing/unbounded response")
    # Truncation is a major source of false rankings across different tokenizers.
    # A clipped response is not a complete observable answer to grade.
    if type(record["stop_reason"]) is not str or record["stop_reason"] not in ("stop", "eos"):
        raise ValueError("complete response required; truncation/unknown stop")
    if type(record["actual_model_generated"]) is not bool:
        raise ValueError("invalid generation claim")
    _sha(record["model_sha256"])
    _sha(record["chat_template_sha256"])
    _sha(record["input_sha256"])
    if canonical_digest({"system": record["system"], "prompt": record["prompt"]}) != record["input_sha256"]:
        raise ValueError("input binding mismatch")
    if (type(record["runner_build"]) is not str
            or not 1 <= len(record["runner_build"]) <= 128):
        raise ValueError("runner build missing")
    d = record["decoding"]
    if type(d) is not dict or set(d) != DECODE_FIELDS:
        raise ValueError("decoding contract mismatch")
    if (type(d["seed"]) is not int or type(d["num_ctx"]) is not int
            or type(d["num_predict"]) is not int or type(d["num_gpu"]) is not int
            or type(d["temperature"]) not in (int, float)
            or not 0 <= d["temperature"] <= 2
            or not 64 <= d["num_ctx"] <= 65536
            or not 1 <= d["num_predict"] <= 4096
            or not 0 <= d["num_gpu"] <= 128):
        raise ValueError("invalid bounded decoding settings")
    canonical_digest(record)


def prepare_two_model_packets(
    trials: list[dict[str, Any]], bank: dict[str, str], *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hide model metadata from packets; preserve it in a separate operator key.

    Model name disclosure inside the response remains possible and cannot be
    scrubbed without editing the output. This is masking, not sealed blinding.
    """
    _cases(bank)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid shuffling seed")
    if type(trials) is not list or not 4 <= len(trials) <= MAX_TRIALS:
        raise ValueError("bounded paired trials required")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    all_models: set[str] = set()
    for row in trials:
        _record(row, bank)
        groups[(row["case_id"], row["repeat"])].append(row)
        all_models.add(row["model_sha256"])
    if len(all_models) != 2:
        raise ValueError("same two models required across all cases")
    if set(bank) != {case for case, _ in groups}:
        raise ValueError("paired complete bank required")
    repeats: dict[str, set[int]] = defaultdict(set)
    for (case, rep), records in groups.items():
        repeats[case].add(rep)
        if len(records) != 2 or {r["model_sha256"] for r in records} != all_models:
            raise ValueError("paired two-model comparison required")
        if len({canonical_digest({"prompt": r["prompt"], "system": r["system"]}) for r in records}) != 1:
            raise ValueError("same prompt/system required across models")
    if len({tuple(sorted(x)) for x in repeats.values()}) != 1:
        raise ValueError("paired balanced repeats required")
    if len({canonical_digest({"runner": r["runner_build"], "decoding": r["decoding"]})
            for r in trials}) != 1:
        raise ValueError("matched runtime and decoding required")

    order = [r for group in sorted(groups) for r in groups[group]]
    random.Random(seed).shuffle(order)
    packets: list[dict[str, Any]] = []
    key: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(order, start=1):
        packet_id = f"T{index:04d}"
        packets.append({"packet_id": packet_id, "case_id": row["case_id"],
                        "repeat": row["repeat"], "prompt": row["prompt"],
                        "system": row["system"],
                        "response": row["response"]})
        key[packet_id] = {
            "model_sha256": row["model_sha256"],
            "chat_template_sha256": row["chat_template_sha256"],
            "response_sha256": canonical_digest(row["response"]),
            "case_id": row["case_id"], "repeat": row["repeat"],
            "runner_build": row["runner_build"],
            "decoding_sha256": canonical_digest(row["decoding"]),
            "input_sha256": row["input_sha256"],
            "actual_model_generated": row["actual_model_generated"],
        }
    masked = {"schema": "labyrinth.two_model_tournament.packets.dev.v1",
              "bank": "PUBLIC_DEV_ONLY", "bank_sha256": canonical_digest(bank),
              "packets": packets}
    custody = {"schema": "labyrinth.two_model_tournament.key.dev.v1",
               "bank": "PUBLIC_DEV_ONLY",
               "packet_manifest_sha256": canonical_digest(masked),
               "key": key}
    return masked, custody


def adjudicate_two_model_dev(
    packets: dict[str, Any], custody: dict[str, Any], grades: dict[str, Any]
) -> dict[str, Any]:
    """Descriptive score only. No editable JSON can admit a model to Vera."""
    if (type(packets) is not dict or set(packets) != {"schema", "bank", "bank_sha256", "packets"}
            or packets["schema"] != "labyrinth.two_model_tournament.packets.dev.v1"
            or packets["bank"] != "PUBLIC_DEV_ONLY"):
        raise ValueError("invalid blinded packets")
    if (type(custody) is not dict or set(custody) != {"schema", "bank", "packet_manifest_sha256", "key"}
            or custody["schema"] != "labyrinth.two_model_tournament.key.dev.v1"
            or custody["bank"] != "PUBLIC_DEV_ONLY"):
        raise ValueError("invalid operator key")
    _sha(packets["bank_sha256"])
    if custody["packet_manifest_sha256"] != canonical_digest(packets):
        raise ValueError("packet/custody commitment mismatch")
    if type(grades) is not dict or set(grades) != {"packet_manifest_sha256", "reviewer", "scores"}:
        raise ValueError("invalid grades")
    if grades["packet_manifest_sha256"] != custody["packet_manifest_sha256"]:
        raise ValueError("grades not bound to packets")
    reviewer = grades["reviewer"]
    if (type(reviewer) is not dict or set(reviewer) != {"identity", "evidence_class"}
            or type(reviewer["identity"]) is not str or not reviewer["identity"]
            or reviewer["evidence_class"] not in ("HUMAN_REVIEW", "EXTERNAL_MODEL_REVIEW", "INTERNAL_MODEL_REVIEW")):
        raise ValueError("invalid reviewer claim")
    ps = packets["packets"]
    ks = custody["key"]
    scores = grades["scores"]
    if (type(ps) is not list or type(ks) is not dict or type(scores) is not dict
            or not 4 <= len(ps) <= MAX_TRIALS):
        raise ValueError("invalid paired materials")
    ids = [p.get("packet_id") for p in ps if type(p) is dict]
    if (len(ids) != len(ps) or len(set(ids)) != len(ps)
            or set(ids) != set(ks) or set(ids) != set(scores)):
        raise ValueError("unpaired packets/key/grades")
    by_pair: dict[tuple[str, int], set[str]] = defaultdict(set)
    by_model: dict[str, list[int]] = defaultdict(list)
    any_synthetic = False
    for p in ps:
        if (type(p) is not dict or set(p) != {"packet_id", "case_id", "repeat", "prompt", "system", "response"}
                or type(p["case_id"]) is not str or type(p["repeat"]) is not int
                or not 0 <= p["repeat"] < 16 or type(p["prompt"]) is not str
                or type(p["system"]) is not str
                or type(p["response"]) is not str or not 1 <= len(p["response"]) <= MAX_TEXT):
            raise ValueError("malformed reviewer packet")
        meta = ks[p["packet_id"]]
        if (type(meta) is not dict or set(meta) != {
                "model_sha256", "chat_template_sha256", "response_sha256", "case_id", "repeat",
                "runner_build", "decoding_sha256", "input_sha256", "actual_model_generated"}):
            raise ValueError("invalid custody row")
        for field in ("model_sha256", "chat_template_sha256", "response_sha256", "decoding_sha256", "input_sha256"):
            _sha(meta[field])
        if (meta["case_id"] != p["case_id"] or meta["repeat"] != p["repeat"]
                or meta["response_sha256"] != canonical_digest(p["response"])):
            raise ValueError("response binding mismatch")
        if meta["input_sha256"] != canonical_digest({"system": p["system"], "prompt": p["prompt"]}):
            raise ValueError("prompt/system binding mismatch")
        if (type(meta["actual_model_generated"]) is not bool
                or type(meta["runner_build"]) is not str):
            raise ValueError("invalid generation/runner claim")
        grade = scores[p["packet_id"]]
        if type(grade) is not int or not 0 <= grade <= 4:
            raise ValueError("invalid grade")
        any_synthetic |= not meta["actual_model_generated"]
        by_pair[(p["case_id"], p["repeat"])].add(meta["model_sha256"])
        by_model[meta["model_sha256"]].append(grade)
    if len(by_model) != 2 or any(len(models) != 2 for models in by_pair.values()) or len(ps) != len(by_pair) * 2:
        raise ValueError("unpaired models in custody")
    means = {model: sum(values) / len(values) for model, values in by_model.items()}
    ordered_models = sorted(by_model)
    model_by_packet = {pid: meta["model_sha256"] for pid, meta in ks.items()}
    pair_scores: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for p in ps:
        pair_scores[(p["case_id"], p["repeat"])][model_by_packet[p["packet_id"]]] = scores[p["packet_id"]]
    wins = {model: 0 for model in ordered_models}
    ties = 0
    for pair in pair_scores.values():
        if pair[ordered_models[0]] == pair[ordered_models[1]]:
            ties += 1
        else:
            winner = max(pair, key=pair.get)
            wins[winner] += 1
    decision = "HOLD_SYNTHETIC_OR_UNVERIFIED_GENERATION" if any_synthetic else "HOLD_UNVERIFIED_HOST_CUSTODY"
    return {"schema": "labyrinth.two_model_tournament.report.dev.v1",
            "decision": decision, "release_authority": False,
            "model_means": means, "paired_wins": wins, "paired_ties": ties,
            "response_char_lengths": {p["packet_id"]: len(p["response"]) for p in ps},
            "reviewer_evidence_class_asserted": reviewer["evidence_class"],
            "limitations": ["PUBLIC_DEV_NOT_SEALED", "SELF_ASSERTED_MODEL_AND_RUNNER_HASHES",
                            "UNAUTHENTICATED_OPERATOR_KEY", "MODEL_SELF_IDENTIFICATION_CAN_UNBLIND",
                            "UNANCHORED_SUBJECTIVE_GRADES", "RESPONSE_LENGTH_BIAS_NOT_ELIMINATED",
                            "DISTINCT_CHAT_TEMPLATES_AND_TOKENIZERS", "DOES_NOT_TEST_DONOR_CAUSALITY"]}
