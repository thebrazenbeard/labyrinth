"""Model-agnostic text-generation adapters for reasoning-maze experiments.

Adapters return *text*, never executable instructions.  The replay adapter is
hermetic; the Ollama adapter is opt-in and loopback-only by default.  A fixed
generation seed is a request to the runtime, not a cross-device determinism
guarantee.
"""

from __future__ import annotations

import ipaddress
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable
from urllib import error, parse, request


class AdapterError(RuntimeError):
    """A model adapter could not produce a valid text response."""


@runtime_checkable
class ModelAdapter(Protocol):
    """Common interface; callers may use any adapter satisfying this shape."""

    @property
    def model_id(self) -> str: ...

    def generate(self, prompt: str, *, seed: int | None = None) -> str: ...


@dataclass(frozen=True)
class ReplayAdapter:
    """Exact-prompt fixture adapter for fully reproducible offline tests.

    A missing prompt returns ``default`` if supplied, otherwise fails loudly.
    The seed is accepted to keep the same call signature across adapters, but
    does not change a replay's fixed answer.
    """

    responses: Mapping[str, str]
    default: str | None = None
    model_id: str = "replay"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be nonempty")
        if not isinstance(self.responses, Mapping):
            raise TypeError("responses must map prompts to text")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.responses.items()):
            raise TypeError("responses must map strings to strings")
        if self.default is not None and not isinstance(self.default, str):
            raise TypeError("default must be text or None")
        object.__setattr__(self, "responses", MappingProxyType(dict(self.responses)))

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        _validate_prompt_seed(prompt, seed)
        if prompt in self.responses:
            return self.responses[prompt]
        if self.default is not None:
            return self.default
        raise AdapterError("replay fixture has no answer for the supplied prompt")


ScriptedAdapter = ReplayAdapter


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class OllamaAdapter:
    """Call a user-operated Ollama runtime through its non-streaming API.

    By default, only literal loopback addresses and localhost are accepted.
    Set ``allow_remote=True`` deliberately for a trusted LAN runtime; never
    expose that mode to untrusted experiment configuration. Redirects and
    environment-configured HTTP proxies are disabled.
    """

    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 30.0
    allow_remote: bool = False
    max_response_bytes: int = 1_000_000
    max_output_tokens: int = 256
    _url: str = field(init=False, repr=False)
    _opener: request.OpenerDirector = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty name")
        if not isinstance(self.base_url, str):
            raise ValueError("base_url must be a URL string")
        parsed = parse.urlsplit(self.base_url)
        if (parsed.scheme != "http" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("base_url must be a plain http://host:port URL")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("invalid Ollama port") from exc
        if port is None:
            raise ValueError("Ollama URL requires an explicit port")
        if not self.allow_remote and not _is_loopback_host(parsed.hostname):
            raise ValueError("non-loopback Ollama endpoint requires allow_remote=True")
        if (not isinstance(self.timeout_seconds, (int, float))
                or isinstance(self.timeout_seconds, bool)
                or not math.isfinite(self.timeout_seconds)
                or not 0 < self.timeout_seconds <= 300):
            raise ValueError("timeout_seconds must be between 0 and 300")
        if not isinstance(self.max_response_bytes, int) or isinstance(self.max_response_bytes, bool) or not 1 <= self.max_response_bytes <= 10_000_000:
            raise ValueError("max_response_bytes outside safe range")
        if not isinstance(self.max_output_tokens, int) or isinstance(self.max_output_tokens, bool) or not 1 <= self.max_output_tokens <= 8192:
            raise ValueError("max_output_tokens outside safe range")
        # No redirects. No HTTP_PROXY/HTTPS_PROXY inherited from the host.
        object.__setattr__(self, "_opener", request.build_opener(request.ProxyHandler({}), _NoRedirect()))
        object.__setattr__(self, "_url", self.base_url.rstrip("/") + "/api/generate")

    @property
    def model_id(self) -> str:
        return "ollama:" + self.model

    def generate(self, prompt: str, *, seed: int | None = None) -> str:
        _validate_prompt_seed(prompt, seed)
        options: dict[str, object] = {"temperature": 0, "num_predict": self.max_output_tokens}
        if seed is not None:
            options["seed"] = seed
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }, ensure_ascii=False).encode("utf-8")
        req = request.Request(self._url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self._opener.open(req, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    raise AdapterError("Ollama returned non-success HTTP status")
                raw = response.read(self.max_response_bytes + 1)
        except (error.URLError, TimeoutError, OSError) as exc:
            raise AdapterError("Ollama request failed") from exc
        if len(raw) > self.max_response_bytes:
            raise AdapterError("Ollama response exceeded configured size limit")
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("response"), str) or data.get("done") is not True:
            raise AdapterError("Ollama returned an incomplete or malformed generation")
        return data["response"]


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_prompt_seed(prompt: str, seed: int | None) -> None:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be text")
    if len(prompt) > 64_000:
        raise ValueError("prompt exceeds 64000 characters")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**31 - 1):
        raise ValueError("seed must be an integer from 0 to 2^31-1")


def load_adapter(kind: str, **config: object) -> ModelAdapter:
    """Create an adapter from trusted configuration, with no dynamic imports."""

    if kind in ("replay", "scripted"):
        return ReplayAdapter(**config)  # type: ignore[arg-type]
    if kind == "ollama":
        return OllamaAdapter(**config)  # type: ignore[arg-type]
    raise ValueError(f"Unknown model adapter kind: {kind!r}")
