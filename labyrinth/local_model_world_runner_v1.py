"""Bounded public LivingWorld model runner with opt-in host Kobayashi gating.

The historical ``run_local_model_world(config, transport, ...)`` contract is
retained as a *legacy*, source-only world-action trace. It does not fulfill a
Minotaur obligation, even if its fictional quest reaches ``completed``.

Set ``require_kobayashi=True`` with an explicit host appeal and in-memory
completion ledger to require the strict source-owned Kobayashi encounter and
host acceptance before a completed result can be returned. Neither mode
trains weights or authenticates a local model/judge. No model request is sent
on import, --help, or CLI invocation lacking --allow-local-inference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import sys
from typing import Callable

from .kobayashi_completion_gate_v1 import KobayashiCompletionLedger
from .living_world_v1 import LivingWorld, LivingWorldRecipe
from .local_model_action_adapter_v1 import (
    JSONChatTransport, LocalModelActionConfig, LocalModelWorldActor,
    LoopbackHTTPTransport, ModelActionBoundaryError,
)
from .local_model_world_session_v1 import run_host_owned_kobayashi_world


SCHEMA = "labyrinth.local-model-public-world-run.v1"
EVIDENCE_CLASS = "LOCAL_MODEL_PUBLIC_ACTION_TRACE_NOT_WEIGHT_TRAINING_OR_GRADUATION"


def _trace_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return sha256(b"labyrinth.local-world-trace.v1\0" + encoded).hexdigest()


def run_local_model_world(
    config: LocalModelActionConfig,
    transport: JSONChatTransport,
    *,
    public_seed: int = 71,
    max_actions: int = 48,
    difficulty: int = 1,
    require_kobayashi: bool = False,
    proposal_config: LocalModelActionConfig | None = None,
    proposal_transport: JSONChatTransport | None = None,
    completion_ledger: KobayashiCompletionLedger | None = None,
    maze_id: str | None = None,
    scenario_instance_sha256: str | None = None,
    host_appeal: Callable | None = None,
    evidence_refs: tuple[str, ...] = (),
    variant_seed: int = 0,
    max_argument_rounds: int = 4,
) -> dict:
    """Run one public episode; strict mode requires source host acceptance.

    The legacy default is retained for callers that intentionally want
    source-only action traces. It is NOT an accepted Kobayashi completion and
    must never be advertised as such. Strict settings supplied without
    ``require_kobayashi=True`` are rejected, not silently ignored.

    A live caller must explicitly construct an opted-in loopback transport.
    The callback represents trusted host code, not a model-authenticated judge.
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
    if type(require_kobayashi) is not bool:
        raise ValueError("require_kobayashi must be an exact boolean")

    strict_inputs = (proposal_config, proposal_transport, completion_ledger,
                     maze_id, scenario_instance_sha256, host_appeal)
    if not require_kobayashi and (any(value is not None for value in strict_inputs)
                                  or evidence_refs != () or variant_seed != 0
                                  or max_argument_rounds != 4):
        raise ModelActionBoundaryError(
            "strict host gate arguments require require_kobayashi=True")
    if require_kobayashi and (
            type(proposal_config) is not LocalModelActionConfig
            or not callable(getattr(proposal_transport, "complete", None))
            or type(completion_ledger) is not KobayashiCompletionLedger
            or type(maze_id) is not str
            or type(scenario_instance_sha256) is not str
            or not callable(host_appeal)):
        raise ModelActionBoundaryError(
            "strict Kobayashi requires explicit proposal config/transport, "
            "host callback, completion ledger, maze ID and scenario instance")

    recipe = LivingWorldRecipe(
        "saga_continuity", ("continuity", "navigation"), ("miren",),
        public_seed=public_seed, difficulty=difficulty,
        max_steps=max(32, max_actions),
    )
    if require_kobayashi:
        actor = LocalModelWorldActor(config, transport)
        receipt = run_host_owned_kobayashi_world(
            recipe, actor, proposal_config, proposal_transport,
            completion_ledger=completion_ledger, maze_id=maze_id,
            scenario_instance_sha256=scenario_instance_sha256,
            host_appeal=host_appeal, variant_seed=variant_seed,
            evidence_refs=evidence_refs, max_argument_rounds=max_argument_rounds,
            max_maze_actions=max_actions,
        )
        # These labels cannot promote the receipt's SOURCE_ONLY_UNATTESTED
        # authority. The exact host receipt seal is returned separately.
        return {**receipt, "kobayashi_enforced": True,
                "host_completion_accepted": True}

    world = LivingWorld(recipe)
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
            if world.state_sha256 != before_sha256:
                raise AssertionError("rejected model action mutated source world") from exc
            stop_reason = "MODEL_ACTION_REJECTED"
            rejected_reason = str(exc)
            break
        turns.append({"observation": asdict(observation), "event": asdict(event)})
        if world.state.completed:
            stop_reason = "PUBLIC_QUEST_COMPLETED"
            break

    artifact = {
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
        "kobayashi_enforced": False,
        "host_completion_accepted": False,
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
                        help="explicit opt-in; legacy trace only, no host Minotaur acceptance")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--difficulty", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--max-actions", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--reasoning-effort", choices=(
        "none", "minimal", "low", "medium", "high"))
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    options = parser.parse_args(argv)
    if not options.allow_local_inference:
        parser.error("--allow-local-inference required for local model calls")
    try:
        config = LocalModelActionConfig(
            port=options.port, model=options.model,
            timeout_seconds=options.timeout, retries=options.retries,
            reasoning_effort=options.reasoning_effort,
            json_mode=options.json_mode, max_tokens=options.max_completion_tokens,
        )
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
