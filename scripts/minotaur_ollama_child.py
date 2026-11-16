"""Unattested, loopback-only Ollama process child for Minotaur's public protocol.

Use as the command of SubprocessCandidateAdapter; never import in the
examiner process. Only bounded public prompts cross stdin. This does not
provide a distinct OS principal, hidden-case custody or model attestation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# The process boundary starts from a temporary working directory with an
# intentionally empty PYTHONPATH. Import the pinned checkout explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labyrinth.adapters import OllamaAdapter  # noqa: E402

_MAX_INPUT_BYTES = 256_000
_MAX_OUTPUT_CHARS = 4096


def main(arguments: list[str]) -> int:
    if len(arguments) != 2:
        return 2
    model_name, endpoint = arguments
    try:
        raw = sys.stdin.buffer.read(_MAX_INPUT_BYTES + 1)
        if not raw or len(raw) > _MAX_INPUT_BYTES:
            return 2
        candidate_request = json.loads(raw)
        if (type(candidate_request) is not dict
                or set(candidate_request) != {"prompt", "seed"}
                or type(candidate_request["prompt"]) is not dict
                or set(candidate_request["prompt"])
                   != {"response_instructions", "public_start", "history", "turn"}
                or type(candidate_request["seed"]) is not int
                or not 0 <= candidate_request["seed"] < 2**31):
            return 2
        prompt = json.dumps(candidate_request["prompt"], sort_keys=True,
                            ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        # OllamaAdapter refuses non-loopback endpoints, redirects, proxy
        # inheritance and non-text/incomplete responses. No remote override.
        adapter = OllamaAdapter(
            model=model_name, base_url=endpoint,
            timeout_seconds=8.0, max_output_tokens=128,
            max_response_bytes=128_000)
        response = adapter.generate(prompt, seed=candidate_request["seed"])
        if type(response) is not str:
            return 3
        # The runner can retain one overflow character to prove a malformed
        # model answer was too long; never stream unbounded model text.
        if len(response) > _MAX_OUTPUT_CHARS:
            response = "X" * (_MAX_OUTPUT_CHARS + 1)
        sys.stdout.buffer.write(response.encode("utf-8"))
        sys.stdout.buffer.flush()
        return 0
    except (ValueError, TypeError, UnicodeError, OSError, RuntimeError):
        # Never print failed request data, response bodies or stack traces.
        # The parent records MODEL_CALL_ERROR; it cannot replay call failure.
        return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
