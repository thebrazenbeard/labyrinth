"""Single-use public-development predictor worker, spawned per checkpoint.

Only local file-model loading, a deliberately toy file fixture, or an
explicit loopback-only service can be selected. No hosted inference, remote
URLs, external proxies, model downloads or private exam material. A model
service can echo a hash dishonestly: its own attestation is NOT proof it
loaded a particular checkpoint. The file-based worker rehashes before/after.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .entrance_state_v1 import _inventory, _state_digest


_MATH = re.compile(r"Calculate (\d+) ([+-]) (\d+)\.")


def _toy_responses(probes: list[dict], adapter: Path, *, digest_only: bool) -> list[str]:
    """Test double actually reads *checkpoint bytes*, never a caller digest."""
    raw = (adapter / "adapter_model.safetensors").read_bytes()
    if digest_only:
        return ["DIGEST:" + hashlib.sha256(raw).hexdigest() for _ in probes]
    if raw not in (b"TOY_WRONG_RESPONSE", b"TOY_CORRECT_RESPONSE"):
        raise ValueError("toy predictor cannot load unknown adapter bytes")
    result = []
    for probe in probes:
        matched = _MATH.fullmatch(probe["prompt"])
        if matched is None:
            raise ValueError("toy arithmetic predictor requires canonical public prompt")
        left, operation, right = matched.groups()
        actual = int(left) + int(right) if operation == "+" else int(left) - int(right)
        result.append(f"ANSWER: {actual if raw == b'TOY_CORRECT_RESPONSE' else -9999}")
    return result


def _local_hf_responses(probes: list[dict], base: Path, adapter: Path) -> list[str]:
    """Optional real CPU checkpoint restore, local_files_only and no trust-remote-code."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer  # optional
    from peft import PeftModel  # optional
    import torch  # optional

    tokenizer = AutoTokenizer.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False,
        torch_dtype="auto", device_map="cpu",
    )
    model = PeftModel.from_pretrained(
        model, str(adapter), is_trainable=False, local_files_only=True,
    )
    model.eval()
    outputs: list[str] = []
    with torch.inference_mode():
        for probe in probes:
            encoded = tokenizer(probe["prompt"], return_tensors="pt", truncation=True,
                                max_length=512)
            generated = model.generate(
                **encoded, do_sample=False, max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[0][encoded["input_ids"].shape[-1]:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True)[:512])
    return outputs


class _RefuseRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _localhost_responses(probes: list[dict], url: str, expected: str) -> list[str]:
    # Caller separately validates loopback-only URL; validate inside worker too.
    from .checkpoint_predictor_verification import validate_local_predictor_endpoint
    validate_local_predictor_endpoint(url)
    opener = build_opener(ProxyHandler({}), _RefuseRedirect())
    result = []
    for probe in probes:
        payload = json.dumps({"checkpoint_state_sha256": expected,
                              "prompt": probe["prompt"]}).encode("utf-8")
        request = Request(url, data=payload, method="POST",
                          headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=10) as response:
            data = response.read(4097)
        if len(data) > 4096:
            raise ValueError("localhost predictor response exceeds bound")
        reply = json.loads(data)
        if (type(reply) is not dict or set(reply) != {"checkpoint_state_sha256", "answer"}
                or reply["checkpoint_state_sha256"] != expected
                or type(reply["answer"]) is not str or len(reply["answer"]) > 512):
            raise ValueError("localhost predictor reports stale or unpinned state")
        result.append(reply["answer"])
    return result


def process(payload: dict[str, Any]) -> dict[str, Any]:
    expected = {"schema", "nonce", "predictor", "base", "adapter",
                "loaded_state_sha256", "public_probes", "localhost_endpoint",
                "worker_source_sha256"}
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("worker input is not an exact development-only predictor request")
    code_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if payload["worker_source_sha256"] != code_digest:
        raise ValueError("worker runtime source differs from caller-pinned exact implementation")
    base_expected, adapter_expected = payload["base"], payload["adapter"]
    base = Path(base_expected["root"])
    adapter = Path(adapter_expected["root"])
    current_base = _inventory(base, kind="base")
    current_adapter = _inventory(adapter, kind="adapter")
    state_sha = _state_digest(current_base, current_adapter)
    if (base_expected != current_base or adapter_expected != current_adapter
            or state_sha != payload["loaded_state_sha256"]):
        raise ValueError("fresh predictor worker found forged or stale checkpoint bytes")
    if payload["schema"] != "labyrinth.checkpoint-predictor-request.v1":
        raise ValueError("unsupported worker request version")
    predictor = payload["predictor"]
    if predictor == "mock-local-file-arithmetic":
        if payload["localhost_endpoint"] is not None:
            raise ValueError("file predictor cannot accept a service URL")
        predictions = _toy_responses(payload["public_probes"], adapter, digest_only=False)
        runtime_binding = "LOCAL_MODEL_BYTES_READ_BY_WORKER"
    elif predictor == "mock-bytes-digest-only":
        if payload["localhost_endpoint"] is not None:
            raise ValueError("file predictor cannot accept a service URL")
        predictions = _toy_responses(payload["public_probes"], adapter, digest_only=True)
        runtime_binding = "LOCAL_MODEL_BYTES_READ_BY_WORKER_NO_SKILL"
    elif predictor == "hf-local-file":
        if payload["localhost_endpoint"] is not None:
            raise ValueError("file predictor cannot accept a service URL")
        predictions = _local_hf_responses(payload["public_probes"], base, adapter)
        runtime_binding = "LOCAL_HF_WEIGHTS_LOADED_BY_WORKER_UNATTESTED"
    elif predictor == "localhost-self-attested":
        predictions = _localhost_responses(
            payload["public_probes"], payload["localhost_endpoint"], state_sha,
        )
        runtime_binding = "UNVERIFIED_LOCALHOST_SERVICE_HASH_ECHO_ONLY"
    else:
        raise ValueError("unapproved predictor mode")
    if (_inventory(base, kind="base") != current_base
            or _inventory(adapter, kind="adapter") != current_adapter
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != code_digest):
        raise ValueError("model bytes changed during predictor inference")
    return {"schema": "labyrinth.checkpoint-predictor-worker.v1",
            "nonce": payload["nonce"], "pid": os.getpid(),
            "loaded_state_sha256": state_sha,
            "worker_source_sha256": code_digest,
            "runtime_binding": runtime_binding,
            "predictions": predictions}


def main() -> int:
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536:
        raise ValueError("worker request exceeds 64KiB")
    result = process(json.loads(raw))
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
