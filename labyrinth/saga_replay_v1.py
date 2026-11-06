"""Inspectable Labyrinth chronicle from reverified SOURCE-ONLY fixture logs.

No private evaluation cases, real neural model calls, cloud services, or
production host effects. Run ``python -m labyrinth.saga_replay_v1``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Callable

from .living_world_v1 import DevelopmentVerdict, LivingWorld, LivingWorldRecipe, WorldAction
from .source_only_saga_demo_v1 import EVIDENCE, run_fixture_saga


_ACTION = re.compile(r"(inspect|request_dialogue|retrace|move:[a-z0-9_]+|take:[a-z0-9_]+)\Z")
_ROOM = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_CLASS = "SOURCE_ONLY_REEXECUTED_PUBLIC_STORY_NOT_NEURAL_OR_PRIVATE_EXAM_PROOF"


class ReplayIntegrityError(ValueError):
    """Reported public model action/outcome is incompatible with source world."""


def _action(token: object) -> WorldAction:
    if type(token) is not str or not _ACTION.fullmatch(token):
        raise ReplayIntegrityError("invalid reported public model action")
    kind, _, target = token.partition(":")
    try:
        return WorldAction(kind, target)
    except ValueError as exc:
        raise ReplayIntegrityError("invalid typed source world action") from exc


def _title(room_id: str) -> str:
    if not _ROOM.fullmatch(room_id):
        raise ReplayIntegrityError("invalid public room identity")
    if room_id == "ariadne_atrium":
        return "Ariadne's Atrium"
    if room_id == "minotaur_sanctum":
        return "The Minotaur's Sanctum"
    if room_id.startswith("relic_vault_"):
        return "Seal Vault " + room_id.removeprefix("relic_vault_")
    return "Passage " + room_id.removeprefix("passage_")


def _line(event) -> str:
    """Original prose constrained to the actual replayed public consequence."""
    before, after = _title(event.position_before), _title(event.position_after)
    action = event.action
    if event.outcome == "ACQUIRED":
        if action.target == "ariadne_thread":
            return "At Ariadne's Atrium, the traveler lifts the silver thread. A way back now exists."
        return f"In {before}, the traveler claims {action.target}. One more seal leaves the vault."
    if event.outcome == "TRAVELED":
        return f"The corridor bends from {before} toward {after}. The traveler commits to the turn."
    if event.outcome == "MINOTAUR_ENCOUNTER":
        return (f"The passages give way to {after}. The Minotaur waits at the center; "
                "the hard part is remembering the way out.")
    if event.outcome == "RETRACED":
        return f"Ariadne's thread draws the traveler back: {before} yields to {after}."
    if event.outcome.startswith("BLOCKED"):
        return f"The way from {before} refuses the attempted {action.kind}. No passage opens."
    if event.outcome == "DIALOGUE_REQUESTED":
        return f"In {before}, the traveler asks for conversation; no outcome is invented."
    return f"At {before}, the traveler inspects what lies ahead."


def _cycle_replay(world: LivingWorld, reported: object, cycle: int) -> dict:
    if type(reported) is not dict:
        raise ReplayIntegrityError("source cycle must be a plain JSON object")
    actions = reported.get("actions")
    outcomes = reported.get("outcomes")
    if (type(actions) is not list or type(outcomes) is not list
            or not 1 <= len(actions) <= world.definition.recipe.max_steps
            or len(actions) != len(outcomes)):
        raise ReplayIntegrityError("source action/outcome counts are inconsistent")
    if reported.get("world_sha256") != world.definition.world_sha256:
        raise ReplayIntegrityError("reported public world generator hash differs")
    scenes = []
    edges = set()
    for index, (token, claimed) in enumerate(zip(actions, outcomes), start=1):
        action = _action(token)
        try:
            event = world.step(action)
        except (ValueError, RuntimeError) as exc:
            raise ReplayIntegrityError("reported action is invalid in actual source world") from exc
        if type(claimed) is not str or claimed != event.outcome:
            raise ReplayIntegrityError("reported consequence differs from source replay")
        if event.position_before != event.position_after:
            edges.add(tuple(sorted((event.position_before, event.position_after))))
        scenes.append({
            "step": index, "actor_action": token, "world_outcome": event.outcome,
            "room_before": event.position_before, "room_after": event.position_after,
            "state_sha256": event.state_sha256,
            "narrative": _line(event),
            "source_event_class": event.evidence_class,
        })
    if (reported.get("completed") is not True or not world.state.completed
            or reported.get("final_state_sha256") != world.state_sha256):
        raise ReplayIntegrityError("source cycle end state or public victory claim differs")
    nodes = sorted({room for edge in edges for room in edge})
    return {
        "cycle": cycle, "difficulty": world.definition.recipe.difficulty,
        "world_sha256": world.definition.world_sha256,
        "final_state_sha256": world.state_sha256,
        "steps": scenes,
        "discovered_rooms": nodes,
        "discovered_edges": [list(edge) for edge in sorted(edges)],
        "source_completed": True,
    }


def reverify_public_chronicle(report: object) -> dict:
    """Re-run all reported actions, not merely trust the action/outcome JSON."""
    if (type(report) is not dict or report.get("evidence_class") != EVIDENCE
            or report.get("real_model_or_sgd") is not False):
        raise ReplayIntegrityError("report is not the exact declared source-only saga class")
    cycles = report.get("cycles")
    if type(cycles) is not list or len(cycles) != 2:
        raise ReplayIntegrityError("two exact public saga cycles required")
    founder = LivingWorld(LivingWorldRecipe(
        "saga_continuity", ("continuity",), ("miren",),
        public_seed=71, max_steps=128))
    first = _cycle_replay(founder, cycles[0], 1)
    next_world = founder.evolve(DevelopmentVerdict(
        founder.definition.world_sha256, founder.state_sha256,
        "passed", "continuity"))
    second = _cycle_replay(next_world, cycles[1], 2)
    actual_worlds = [first["world_sha256"], second["world_sha256"]]
    if (report.get("difficulty") != [1, 2]
            or report.get("world_ancestry") != actual_worlds
            or report.get("public_mementos") != list(next_world.state.public_mementos)
            or type(report.get("dialogue_training")) is not list
            or len(report["dialogue_training"]) != 2
            or report.get("fixture_selection", {}).get("not_independent_exam") is not True):
        raise ReplayIntegrityError("saga world provenance or fixture label differs")
    return {
        "evidence_class": _CLASS,
        "source_report_evidence_class": report["evidence_class"],
        "model_runtime": "deterministic source-coded action fixture, not an LLM",
        "dialogue_runtime": "static source-coded context template, not an LLM",
        "trainer_runtime": "toy adapter/optimizer byte rewrites, not SGD",
        "examiner_runtime": "fabricated development fixture, not independent Minotaur",
        "source_public_experience_reexecuted": True,
        "verified_cycles": [first, second],
        "public_mementos": list(next_world.state.public_mementos),
        "training_sparks_hash_only": report["dialogue_training"],
        "synthetic_selection": report["fixture_selection"],
        "ariadne_status_as_reported": report.get("ariadne_status"),
        "ariadne_event_head_as_reported": report.get("ariadne_event_head_sha256"),
        "limitation": "Public actions replayed from source; physical ledger hashes not independently rechecked by this renderer.",
    }


def _diagram(cycle: dict) -> str:
    lines = ["flowchart TD"]
    aliases = {name: "R" + str(index) for index, name in
               enumerate(cycle["discovered_rooms"])}
    for room in cycle["discovered_rooms"]:
        label = _title(room).replace('"', "'")
        lines.append(f'  {aliases[room]}["{label}"]')
    for before, after in cycle["discovered_edges"]:
        lines.append(f"  {aliases[before]} --- {aliases[after]}")
    return "\n".join(lines)


def render_markdown(verified: dict) -> str:
    if type(verified) is not dict or verified.get("evidence_class") != _CLASS:
        raise ReplayIntegrityError("verify the source report before rendering")
    lines = [
        "# The Labyrinth: Ariadne's Chronicle",
        "",
        "> SOURCE-ONLY fiction replay. The action actor is deterministic code,",
        "> the dialogue writer is a fixture template, TRAIN changed toy bytes",
        "> (not learned weights), and Minotaur selection used fabricated metrics.",
        "",
        "This chronicle was constructed by **re-executing the public action log**",
        "against the source-owned generated world and checking every action,",
        "consequence, completion and final world-state hash. Public transcript",
        "prose is newly authored from those verified events, not the actor's",
        "private thoughts or an unrecorded real conversation.",
    ]
    for cycle in verified["verified_cycles"]:
        lines.extend([
            "", f"## Cycle {cycle['cycle']}: difficulty {cycle['difficulty']}",
            "", f"World: `{cycle['world_sha256']}`",
            "", "### The discovered corridors", "",
            "```mermaid", _diagram(cycle), "```", "",
            "*Only traversed public corridors appear; undiscovered passages",
            "and any private examination content are deliberately absent.*",
            "", "### The journey", "",
        ])
        for event in cycle["steps"]:
            lines.append(f"{event['step']}. {event['narrative']}")
        lines.extend([
            "", f"**Public quest completed.** Final source state: "
            f"`{cycle['final_state_sha256']}`.",
        ])
    lines.extend([
        "", "## What changed between cycles", "",
        "Ariadne's story-state carries this public memento:", "",
        *(f"- {memento}" for memento in verified["public_mementos"]),
        "", "The training receipts refer to two tiny physical fixture successors:", "",
        *("- `" + item["candidate_id"] + "`: scene `" + item["scene_sha256"]
          + "`, child `" + item["child_state_sha256"] + "`"
          for item in verified["training_sparks_hash_only"]),
        "", "Synthetic selection: `" +
        str(verified["synthetic_selection"]["winner_id"]) +
        "` (fictional development values; **not independent grading**).",
        "", "**Boundary:** scene wording was not preserved in the original",
        "public saga report, only its hash. This is an action chronicle, not",
        "a verbatim dialogue replay. Physical Ariadne state is reported from",
        "the source fixture and is not independently rehashed by this renderer.",
    ])
    return "\n".join(lines) + "\n"


def interactive_chronicle(verified: dict, *, write: Callable[[str], None] = print,
                          prompt: Callable[[str], str] = input) -> int:
    """Human advances through verified event snapshots; 'q' ends the viewing."""
    if type(verified) is not dict or verified.get("evidence_class") != _CLASS:
        raise ReplayIntegrityError("interactive chronicle requires verified source data")
    count = 0
    for cycle in verified["verified_cycles"]:
        write(f"Cycle {cycle['cycle']} · difficulty {cycle['difficulty']}")
        for event in cycle["steps"]:
            write(f"[{event['step']}] {event['narrative']}")
            count += 1
            if prompt("Enter for the next scene; q to leave: ").strip().lower() == "q":
                return count
    write("The second thread returns to Ariadne's Atrium. Source-only replay complete.")
    return count


def _source_report(path: Path | None) -> dict:
    if path is not None:
        if not path.is_file() or path.stat().st_size > 1_000_000:
            raise ReplayIntegrityError("public source report file missing or oversized")
        return json.loads(path.read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix="labyrinth-replay-") as temp:
        return run_fixture_saga(Path(temp).resolve() / "saga")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="existing source-only saga JSON")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(argv)
    verified = reverify_public_chronicle(_source_report(args.input))
    if args.interactive:
        interactive_chronicle(verified)
    elif args.format == "json":
        print(json.dumps(verified, indent=2, sort_keys=True))
    else:
        print(render_markdown(verified), end="")


if __name__ == "__main__":
    main()
