"""Run one bounded *public* Labyrinth world through an opt-in local model.

This runner sends observations to an existing loopback OpenAI-compatible
completion service and applies only validated actions through ``LivingWorld``.
It does not create, edit, load, attest, evaluate or promote neural weights.

Example on a trusted operator-owned machine with an already-running endpoint::

    python -m labyrinth.local_model_world_runner_v1 \
        --allow-local-inference --model your-local-model --port 11434

No network activity occurs on import, --help, or an invocation without the
explicit opt-in flag. The CLI writes a bounded JSON trace only to stdout.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import sys
from typing import Callable

from .living_world_v1 import LivingWorld, LivingWorldRecipe
from .local_model_action_adapter_v1 import (
    JSONChatTransport, LocalModelActionConfig, LocalModelWorldActor,
    LoopbackHTTPTransport, ModelActionBoundaryError,
)


SCHEMA = "labyrinth.local-model-public-world-run.v1"
EVIDENCE_CLASS = "LOCAL_MODEL_PUBLIC_ACTION_TRACE_NOT_WEIGHT_TRAINING_OR_GRADUATION"


def _trace_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(b"labyrinth.local-world-trace.v1\0" + encoded).hexdigest()


def run_local_model_world(
    config: LocalModelActionConfig,
    transport: JSONChatTransport,
    *,
    public_seed: int = 71,
    max_actions: int = 48,
    difficulty: int = 1,
) -> dict:
    """Run a bounded public episode with exactly one source-owned outcome per move.

    ``transport`` may be a fake test transport. A *live* caller must construct
    ``LoopbackHTTPTransport(allow_http=True)`` separately. Any model-proposed
    grades, hidden checks or unauthorized actions are refused by the adapter.
    No file-system mutation or checkpoint authority exists in this function.
    """
    if type(config) is not LocalModelActionConfig:
        raise ValueError("exact typed local model config required")
    LocalModelActionConfig.__post_init__(config)
    if type(max_actions) is not int or not 1 <= max_actions <= 128:
        raise ValueError("max_actions must be an integer in [1,128]")
    if type(difficulty) is not int or not 1 <= difficulty <= 5:
        raise ValueError("difficulty must be an integer in [1,5]")
    if type(public_seed) is not int or not 0 <= public_seed < 2 ** 64:
        raise ValueError("public_seed must be a 64-bit nonnegative integer")

    world = LivingWorld(LivingWorldRecipe(
        "saga_continuity", ("continuity", "navigation"), ("miren",),
        public_seed=public_seed, difficulty=difficulty,
        max_steps=max(32, max_actions),
    ))
    actor = LocalModelWorldActor(config, transport)
    turns: list[dict] = []
    stop_reason = "ACTION_BUDGET_EXHAUSTED"
    rejected_reason: str | None = None
    for _ in range(max_actions):
        observation = world.observe()
        before_sha256 = world.state_sha256
        try:
            event = world.step_from_actor(actor)
        except ModelActionBoundaryError as exc:
            # A rejected proposal is not a world event. Preserve the exact
            # unchanged state, completed prior actions and validation reason.
            if world.state_sha256 != before_sha256:
                raise AssertionError("rejected model action mutated source world") from exc
            stop_reason = "MODEL_ACTION_REJECTED"
            rejected_reason = str(exc)
            break
        turns.append({
            "observation": asdict(observation),
            "event": asdict(event),
        })
        if world.state.completed:
            stop_reason = "PUBLIC_QUEST_COMPLETED"
            break

    artifact: dict = {
        "schema": SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "model_request_name": config.model,
        "source": "operator-configured-loopback-or-test-transport",
        "world_sha256": world.definition.world_sha256,
        "public_seed": public_seed,
        "difficulty": difficulty,
        "max_actions": max_actions,
        "outcome": stop_reason,
        "failed_action_reason": rejected_reason,
        "model_completion_requests": actor.attempts,
        "rejected_action_reasons": list(actor.rejections),
        "world_final_state_sha256": world.state_sha256,
        "world_final_state": asdict(world.state),
        "turns": turns,
        "claims": {
            "actual_model_endpoint_responded": "NOT_ATTESTED_BY_SOURCE_ALONE",
            "weights_changed": False,
            "neural_training_performed": False,
            "minotaur_exam_passed": False,
        },
    }
    artifact["trace_sha256"] = _trace_digest(artifact)
    return artifact


def main(
    argv: list[str] | None = None,
    *,
    transport_factory: Callable[[], JSONChatTransport] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="existing model name in local server")
    parser.add_argument("--port", type=int, required=True, help="local server TCP port")
    parser.add_argument("--allow-local-inference", action="store_true",
                        help="explicitly opt in to local inference; no model is installed")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--difficulty", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--max-actions", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--reasoning-effort", choices=(
        "none", "minimal", "low", "medium", "high"),
        help="optional completion-server reasoning control (for example none)")
    parser.add_argument("--json-mode", action="store_true",
                        help="request response_format json_object if supported")
    parser.add_argument("--max-completion-tokens", type=int, default=64,
                        help="completion token budget (32..256)")
    options = parser.parse_args(argv)
    if not options.allow_local_inference:
        parser.error("--allow-local-inference required for local model calls")
    try:
        config = LocalModelActionConfig(
            port=options.port, model=options.model,
            timeout_seconds=options.timeout, retries=options.retries,
            reasoning_effort=options.reasoning_effort,
            json_mode=options.json_mode,
            max_tokens=options.max_completion_tokens,
        )
        # Tests can inject a fake; a CLI invocation's default is only the
        # operator-enabled loopback transport, with proxies/redirects off.
        transport = (transport_factory() if transport_factory is not None
                     else LoopbackHTTPTransport(allow_http=True))
        receipt = run_local_model_world(
            config, transport, public_seed=options.seed,
            difficulty=options.difficulty, max_actions=options.max_actions,
        )
    except (ModelActionBoundaryError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    if receipt["outcome"] == "MODEL_ACTION_REJECTED":
        return 2
    if receipt["outcome"] == "ACTION_BUDGET_EXHAUSTED":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
