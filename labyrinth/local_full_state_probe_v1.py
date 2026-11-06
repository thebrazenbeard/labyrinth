"""$0 loopback-only public inference probes for full-state Labyrinth populations.

Full physical files are rehashed; only public prompt-only input crosses the
transport. A loopback server's response does NOT prove the loaded model bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .campaign_full_state_v1 import verify_campaign_state
from .evolutionary_population_v1 import Population, prepare_population


_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_BACKEND_PATH = {
    "koboldcpp": "/api/v1/generate",
    "llamacpp": "/completion",
    "openai_local": "/v1/completions",
}
_EVIDENCE = "PUBLIC_LOCALHOST_OUTPUT_UNVERIFIED_LOADED_MODEL"
_SCHEMA = "labyrinth.local-full-state-public-probe.v1"


class LocalFullStateProbeError(ValueError):
    """Public probe provenance, configuration, or response failed a boundary."""


@dataclass(frozen=True)
class PublicPrompt:
    case_id: str
    prompt: str


@dataclass(frozen=True)
class DeterministicProbeConfig:
    """Requested deterministic decoding; hardware/backend may still differ."""

    max_tokens: int = 64
    seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class LocalProbeTarget:
    subject_id: str
    endpoint: str
    backend: str
    model_alias: str = "local"


@dataclass(frozen=True)
class PublicProbeActivation:
    """Owner-controlled subject switch; this is not a signed load attestation."""

    subject_id: str
    full_state_manifest_path: Path
    full_state_sha256: str
    endpoint: str
    backend: str


@dataclass(frozen=True)
class LocalSubjectProbe:
    subject_id: str
    full_state_sha256: str
    full_state_manifest_sha256: str
    output_texts: tuple[str, ...]
    response_sha256: str
    source_state_unchanged: bool
    model_bytes_loaded_proven: bool = False

    def __post_init__(self) -> None:
        if self.model_bytes_loaded_proven is not False or self.source_state_unchanged is not True:
            raise LocalFullStateProbeError("HTTP outputs cannot prove model load or changed files")


@dataclass(frozen=True)
class LocalPublicProbeReceipt:
    public_suite_sha256: str
    probe_source_sha256: str
    config_sha256: str
    subjects: tuple[LocalSubjectProbe, ...]
    classification: str = _EVIDENCE
    neural_improvement_proven: bool = False

    def __post_init__(self) -> None:
        if self.classification != _EVIDENCE or self.neural_improvement_proven is not False:
            raise LocalFullStateProbeError("public loopback evidence cannot assert neural gain")


class _RefuseRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _validated_public_inputs(
    probes: tuple[PublicPrompt, ...], config: DeterministicProbeConfig,
) -> tuple[str, str]:
    if (type(probes) is not tuple or not 1 <= len(probes) <= 32
            or any(type(p) is not PublicPrompt for p in probes)
            or any(type(p.case_id) is not str or _ID.fullmatch(p.case_id) is None
                   or type(p.prompt) is not str
                   or not 1 <= len(p.prompt.encode("utf-8")) <= 4096
                   or "\x00" in p.prompt for p in probes)
            or len({p.case_id for p in probes}) != len(probes)
            or len({_sha(p.prompt.encode("utf-8")) for p in probes}) != len(probes)):
        raise LocalFullStateProbeError("only bounded unique public prompts, without answers, allowed")
    if (type(config) is not DeterministicProbeConfig
            or type(config.max_tokens) is not int or not 1 <= config.max_tokens <= 256
            or type(config.seed) is not int or not 0 <= config.seed <= 2**31 - 1
            or type(config.temperature) not in (int, float)
            or type(config.top_p) not in (int, float)
            or not math.isfinite(config.temperature) or config.temperature != 0
            or not math.isfinite(config.top_p) or config.top_p != 1):
        raise LocalFullStateProbeError("deterministic bounded generation settings required")
    payload = {
        "schema": _SCHEMA, "partition": "PUBLIC_DEVELOPMENT_ONLY",
        "prompts": [{"case_id": p.case_id, "prompt_sha256": _sha(p.prompt.encode("utf-8")),
                     "prompt": p.prompt} for p in probes],
    }
    cfg = {"max_tokens": config.max_tokens, "seed": config.seed,
           "temperature": config.temperature, "top_p": config.top_p}
    return _sha(b"labyrinth.public-probe-cohort.v1\0" + _canonical(payload)), _sha(
        b"labyrinth.local-probe-config.v1\0" + _canonical(cfg))


def public_probe_suite_sha256(probes: tuple[PublicPrompt, ...]) -> str:
    """Precommit exactly what the model will see; never include score keys."""
    return _validated_public_inputs(probes, DeterministicProbeConfig())[0]


def _validate_target(target: LocalProbeTarget) -> None:
    if (type(target) is not LocalProbeTarget or type(target.subject_id) is not str
            or _ID.fullmatch(target.subject_id) is None
            or target.backend not in _BACKEND_PATH
            or type(target.model_alias) is not str
            or not 1 <= len(target.model_alias) <= 128
            or any(c in target.model_alias for c in "\x00\r\n")):
        raise LocalFullStateProbeError("invalid local target subject, backend or model alias")
    if type(target.endpoint) is not str or not 1 <= len(target.endpoint) <= 160:
        raise LocalFullStateProbeError("bounded IPv4 loopback HTTP endpoint required")
    try:
        uri = urlsplit(target.endpoint)
        port = uri.port
    except ValueError as exc:
        raise LocalFullStateProbeError("malformed endpoint port") from exc
    if (uri.scheme != "http" or uri.hostname != "127.0.0.1"
            or uri.netloc != f"127.0.0.1:{port}" or port is None or not 1 <= port <= 65535
            or uri.path != _BACKEND_PATH[target.backend]
            or uri.query or uri.fragment or uri.username or uri.password):
        raise LocalFullStateProbeError("remote URL, userinfo, redirect alias or wrong API refused")


def _local_http_transport(endpoint: str, payload: dict[str, Any], timeout_seconds: float
                          ) -> dict[str, Any]:
    # Recheck at the network boundary; never use system proxies or redirects.
    backend = next((backend for backend, path in _BACKEND_PATH.items()
                    if urlsplit(endpoint).path == path), None)
    _validate_target(LocalProbeTarget("probe", endpoint, backend or "invalid"))
    request = Request(endpoint, data=_canonical(payload), method="POST",
                      headers={"Content-Type": "application/json"})
    opener = build_opener(ProxyHandler({}), _RefuseRedirect())
    with opener.open(request, timeout=timeout_seconds) as response:
        raw = response.read(16 * 1024 + 1)
    if len(raw) > 16 * 1024:
        raise LocalFullStateProbeError("local API output exceeds strict size cap")
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise LocalFullStateProbeError("malformed local model API JSON") from exc
    if type(result) is not dict:
        raise LocalFullStateProbeError("model API response must be a JSON object")
    return result


def _generation_payload(target: LocalProbeTarget, prompt: str,
                        config: DeterministicProbeConfig) -> dict[str, Any]:
    if target.backend == "koboldcpp":
        return {"prompt": prompt, "max_length": config.max_tokens,
                "temperature": config.temperature, "top_p": config.top_p,
                "seed": config.seed}
    if target.backend == "llamacpp":
        return {"prompt": prompt, "n_predict": config.max_tokens,
                "temperature": config.temperature, "top_p": config.top_p,
                "seed": config.seed}
    return {"model": target.model_alias, "prompt": prompt,
            "max_tokens": config.max_tokens, "temperature": config.temperature,
            "top_p": config.top_p, "seed": config.seed}


def _parse_response(target: LocalProbeTarget, response: Any) -> str:
    if type(response) is not dict:
        raise LocalFullStateProbeError("local runtime output must be JSON object")
    if target.backend == "llamacpp":
        value = response.get("content")
    elif target.backend == "koboldcpp":
        results = response.get("results")
        value = (results[0].get("text") if type(results) is list and len(results) == 1
                 and type(results[0]) is dict else None)
    else:
        choices = response.get("choices")
        value = (choices[0].get("text") if type(choices) is list and len(choices) == 1
                 and type(choices[0]) is dict else None)
    if (type(value) is not str or len(value.encode("utf-8")) > 4096
            or "\x00" in value):
        raise LocalFullStateProbeError("local model API yielded no bounded textual generation")
    return value


def probe_local_full_state_population(
    *, population: Population,
    targets: tuple[LocalProbeTarget, ...],
    public_probes: tuple[PublicPrompt, ...],
    expected_public_suite_sha256: str,
    config: DeterministicProbeConfig = DeterministicProbeConfig(),
    timeout_seconds: float = 20.0,
    transport: Callable[[str, dict[str, Any], float], dict[str, Any]] = _local_http_transport,
    activate_target: Callable[[PublicProbeActivation], str] | None = None,
) -> LocalPublicProbeReceipt:
    """Send *identical* public prompts to parent, control and all W1 siblings.

    ``transport`` and ``activate_target`` are OPERATOR-INSTALLED Python code,
    never executed model-written source. A provider's own model label or
    claimed SHA is not an independent proof of which weights it served.
    """
    if (type(population) is not Population or type(targets) is not tuple
            or any(type(target) is not LocalProbeTarget for target in targets)
            or type(expected_public_suite_sha256) is not str
            or _SHA.fullmatch(expected_public_suite_sha256) is None
            or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 60
            or not callable(transport)
            or (activate_target is not None and not callable(activate_target))):
        raise LocalFullStateProbeError("exact source-only population, local transport and pins required")
    suite_sha, config_sha = _validated_public_inputs(public_probes, config)
    if suite_sha != expected_public_suite_sha256:
        raise LocalFullStateProbeError("public prompt cohort differs from precommitted suite")
    expected_paths = {"parent": population.parent_manifest_path,
                      "control": population.control_manifest_path,
                      **{c.candidate_id: c.manifest_path for c in population.candidates}}
    if (len(targets) != len(expected_paths)
            or {target.subject_id for target in targets} != set(expected_paths)
            or len({target.subject_id for target in targets}) != len(targets)):
        raise LocalFullStateProbeError("all physical parent/control/siblings need one endpoint target")
    for target in targets:
        _validate_target(target)
    if activate_target is None and len({target.endpoint for target in targets}) != len(targets):
        raise LocalFullStateProbeError("shared endpoint requires an explicit owner reload callback")
    # Existing Population holds no immutable authority of its own: rehash the
    # whole population and require exact same content before any inference.
    refreshed = prepare_population(
        development_generation=population.development_generation,
        origin_manifest_path=population.origin_manifest_path,
        parent_manifest_path=population.parent_manifest_path,
        control_manifest_path=population.control_manifest_path,
        candidates=population.candidates, plan=population.plan,
    )
    if refreshed != population:
        raise LocalFullStateProbeError("population changed before public inference")
    source_file = Path(__file__).resolve()
    source_digest = _sha(source_file.read_bytes())
    results: list[LocalSubjectProbe] = []
    # Subject order is fixed; targets cannot reorder the public comparison.
    by_id = {target.subject_id: target for target in targets}
    for subject_id, manifest_path in expected_paths.items():
        target = by_id[subject_id]
        verified = verify_campaign_state(manifest_path, fresh_process=True)
        if activate_target is not None:
            activated = activate_target(PublicProbeActivation(
                subject_id, manifest_path, verified["state_sha256"],
                target.endpoint, target.backend,
            ))
            if activated != "OPERATOR_REPORTS_ACTIVATED":
                raise LocalFullStateProbeError("operator could not report a local model activation")
        generated: list[str] = []
        for probe in public_probes:
            # Payload contains ONLY the public prompt and fixed generation
            # settings. No private exam plan, score, answer or manifest bytes.
            answer = transport(target.endpoint, _generation_payload(target, probe.prompt, config),
                               float(timeout_seconds))
            generated.append(_parse_response(target, answer))
        after = verify_campaign_state(manifest_path, fresh_process=True)
        # Each genuine fresh-process verifier uses a different PID.
        if {key: value for key, value in after.items() if key != "pid"} != {
            key: value for key, value in verified.items() if key != "pid"
        }:
            raise LocalFullStateProbeError("physical source model changed during public inference")
        results.append(LocalSubjectProbe(
            subject_id, verified["state_sha256"], verified["manifest_sha256"],
            tuple(generated), _sha(b"labyrinth.public-inference-output.v1\0" +
                                   _canonical({"subject_id": subject_id,
                                               "answers": generated})), True,
        ))
    if _sha(source_file.read_bytes()) != source_digest:
        raise LocalFullStateProbeError("public probe implementation changed during run")
    if prepare_population(
        development_generation=population.development_generation,
        origin_manifest_path=population.origin_manifest_path,
        parent_manifest_path=population.parent_manifest_path,
        control_manifest_path=population.control_manifest_path,
        candidates=population.candidates, plan=population.plan,
    ) != population:
        raise LocalFullStateProbeError("genome population changed by end of local probes")
    return LocalPublicProbeReceipt(suite_sha, source_digest, config_sha, tuple(results))


__all__ = [
    "LocalFullStateProbeError", "PublicPrompt", "DeterministicProbeConfig",
    "LocalProbeTarget", "PublicProbeActivation", "LocalSubjectProbe",
    "LocalPublicProbeReceipt", "public_probe_suite_sha256",
    "probe_local_full_state_population",
]
