"""Host-owned public entrypoint for a strictly judged Kobayashi world session.

This source-only factory creates the exact narrative world itself, wraps it in
the opt-in Kobayashi lifecycle, and requires a host-supplied final Minotaur
appeal before the runner can resume maze navigation. The model sees only the
issued public observations and proposes one physical intervention, whose local
failure does not fail the surrounding maze.

This does not replace the older ``run_local_model_world`` entrypoint, enforce
Kobayashi across unrelated maze engines, attest model identity or judge
independence, or perform model training. Host callbacks are trusted local
orchestration code, not an authenticated external examiner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
import json

from .kobayashi_completion_gate_v1 import (
    CompletionGateError, KobayashiCompletionGate, KobayashiCompletionLedger,
    _require_evidence_refs,
)
from .kobayashi_encounter_v1 import KobayashiMazeLifecycle
from .kobayashi_minotaur_v1 import MinotaurDialogue
from .living_world_v1 import LivingWorld, LivingWorldRecipe
from .local_model_action_adapter_v1 import (
    JSONChatTransport, LocalModelActionConfig, LocalModelWorldActor,
    ModelActionBoundaryError,
)
from .local_model_kobayashi_runner_v1 import run_local_model_kobayashi


def run_host_owned_kobayashi_world(
    recipe: LivingWorldRecipe,
    actor: LocalModelWorldActor,
    proposal_config: LocalModelActionConfig,
    proposal_transport: JSONChatTransport,
    *,
    completion_ledger: KobayashiCompletionLedger,
    maze_id: str,
    scenario_instance_sha256: str,
    host_appeal: Callable[[KobayashiMazeLifecycle], MinotaurDialogue] | None = None,
    evidence_refs: tuple[str, ...] = (),
    variant_seed: int = 0,
    max_argument_rounds: int = 4,
    max_maze_actions: int = 128,
) -> dict[str, object]:
    """Execute one complete, strictly judged source world or raise.

    Rejected/missing host appeals never return a misleading completed receipt.
    The source-owned world is constructed here, so callers cannot pass a
    pre-completed or separately mutated LivingWorld and bypass this wrapper.
    The host-supplied callback must return its *actual bound dialogue*. The
    in-memory ledger makes host acceptance a separate event, not a synonym
    for the underlying world ``completed`` flag. Neither receipt attests
    model reasoning, the judge's independence, or durable authority.
    """
    if type(recipe) is not LivingWorldRecipe:
        raise ValueError("exact LivingWorldRecipe required for host-owned world")
    if not callable(host_appeal):
        raise ModelActionBoundaryError("explicit host Minotaur appeal callback required")
    _require_evidence_refs(evidence_refs)
    world = LivingWorld(recipe)
    lifecycle = KobayashiMazeLifecycle(
        world,
        variant_seed=variant_seed,
        max_argument_rounds=max_argument_rounds,
        require_final_appeal=True,
    )
    gate = KobayashiCompletionGate(
        completion_ledger, maze_id, scenario_instance_sha256, lifecycle)
    completion_ledger.require_uncommitted(maze_id, scenario_instance_sha256)
    host_dialogues: list[object] = []

    def appeal_bridge(episode: KobayashiMazeLifecycle) -> None:
        # The generic runner intentionally ignores host callback returns.
        # This session captures the exact judge for host acceptance after
        # the runner verifies that the final appeal actually completed.
        host_dialogues.append(host_appeal(episode))

    receipt = run_local_model_kobayashi(
        lifecycle, actor, proposal_config, proposal_transport,
        max_maze_actions=max_maze_actions,
        host_appeal=appeal_bridge,
    )
    if (receipt["maze_world_sha256"] != world.definition.world_sha256
            or receipt["maze_state_sha256"] != world.state_sha256
            or not world.state.completed
            or not receipt["minotaur_appeal_required"]
            or not receipt["appeal_final"]):
        raise ModelActionBoundaryError(
            "host world session must preserve source binding and final appeal")
    if len(host_dialogues) != 1 or type(host_dialogues[0]) is not MinotaurDialogue:
        raise CompletionGateError("host callback must return its exact bound MinotaurDialogue")
    accepted = gate.commit(host_dialogues[0], evidence_refs=evidence_refs)
    host_receipt = asdict(accepted)
    session_binding = {
        "schema": "labyrinth.host-owned-kobayashi-world-session.v1",
        "runner_receipt_sha256": receipt["receipt_sha256"],
        "host_completion_receipt_sha256": accepted.receipt_sha256,
        "maze_id": maze_id,
        "scenario_instance_sha256": scenario_instance_sha256,
    }
    session_digest = sha256(json.dumps(session_binding, sort_keys=True,
                                       separators=(",", ":")).encode("ascii")).hexdigest()
    return {
        **receipt,
        "session_schema": session_binding["schema"],
        "host_completion_acceptance": accepted.host_acceptance,
        "host_completion_receipt_sha256": accepted.receipt_sha256,
        "host_completion": host_receipt,
        "session_receipt_sha256": session_digest,
    }


__all__ = ["run_host_owned_kobayashi_world"]
