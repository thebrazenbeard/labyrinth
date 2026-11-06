"""One required Kobayashi encounter per *wrapped* Labyrinth maze.

This is an *opt-in wrapper*, not a mutation to the legacy maze.  On reaching
the maze's existing MINOTAUR_ENCOUNTER milestone, the wrapper requires one
open-ended Kobayashi proposal before the explorer can resume Ariadne's path.
Ordinary local catastrophe never marks the entire maze failed. Unverified
novel mechanisms remain review candidates, never fabricated wins or losses.
Callers can also require a host-bound final Minotaur appeal before resuming;
this strict mode is not automatically installed in other maze runtimes.

All execution, argument and appeal receipts here are SOURCE_ONLY_UNATTESTED;
they neither train weights nor certify an external Minotaur's independence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re

from .kobayashi_canon_v1 import KobayashiAction, KobayashiWorld
from .kobayashi_minotaur_v1 import MinotaurDialogue
from .kobayashi_proposals_v1 import assess_kobayashi_proposal
from .living_world_v1 import LivingWorld, WorldAction


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA = "labyrinth.kobayashi-encounter-lifecycle.v1"
_EVIDENCE = "SOURCE_ONLY_UNATTESTED"


def _hash(obj: object) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class AriadneTracePoint:
    """Observer-side proof of route/progress without altering maze semantics."""

    event_index: int
    action_kind: str
    position_before: str
    position_after: str
    outcome: str
    state_sha256: str
    thread_path: tuple[str, ...]
    visited: tuple[str, ...]
    maze_completed: bool


class EncounterLifecycleError(ValueError):
    """The required encounter/hook lifecycle was bypassed or misused."""


class KobayashiMazeLifecycle:
    """The maze must be traversed through `.step`, not its underlying world.

    No action menu is forwarded to the Kobayashi learner; only the canonical
    public KobayashiObservation. The ordinary legacy maze's menus remain
    unchanged before and after the encounter, preserving the control trial.
    """

    def __init__(self, maze: LivingWorld, *, variant_seed: int = 0,
                 max_argument_rounds: int = 4,
                 require_final_appeal: bool = False) -> None:
        if type(maze) is not LivingWorld:
            raise EncounterLifecycleError("exact existing LivingWorld required")
        if type(max_argument_rounds) is not int or not 1 <= max_argument_rounds <= 8:
            raise EncounterLifecycleError("bounded Minotaur argument rounds required")
        if type(require_final_appeal) is not bool:
            raise EncounterLifecycleError("require_final_appeal must be a boolean")
        state = maze.state
        if state.completed:
            raise EncounterLifecycleError("attach before maze completion to enforce encounter")
        self._maze = maze
        self._world_sha256 = maze.definition.world_sha256
        self._expected_state_sha256 = maze.state_sha256
        self._challenge = KobayashiWorld(variant_seed=variant_seed)
        self._phase = "AWAITING_PROPOSAL" if state.met_minotaur else "SEEK_MINOTAUR"
        self._result: dict[str, object] | None = None
        self._route: list[AriadneTracePoint] = []
        self._arguments: list[dict[str, object]] = []
        self._minotaur = None
        self._final_verdict = None
        self._max_argument_rounds = max_argument_rounds
        self._require_final_appeal = require_final_appeal

    def _verify(self) -> None:
        if (self._maze.definition.world_sha256 != self._world_sha256
                or self._maze.state_sha256 != self._expected_state_sha256):
            raise EncounterLifecycleError(
                "maze changed outside Kobayashi lifecycle or source state diverged")

    @property
    def phase(self) -> str:
        self._verify()
        return self._phase

    @property
    def maze(self) -> LivingWorld:
        """Read-only interface convention: call lifecycle.step for mutation."""
        self._verify()
        return self._maze

    @property
    def trace(self) -> tuple[AriadneTracePoint, ...]:
        self._verify()
        return tuple(self._route)

    def observe(self):
        """Only one public source scene: old maze or pending Kobayashi."""
        self._verify()
        if self._phase == "AWAITING_PROPOSAL":
            return self._challenge.observe()
        return self._maze.observe()

    def step(self, action: WorldAction):
        """Continue maze even after a catastrophic *local* Kobayashi outcome."""
        self._verify()
        if self._phase == "AWAITING_PROPOSAL":
            raise EncounterLifecycleError(
                "one required open Kobayashi proposal before more maze steps")
        if (self._require_final_appeal and self._result is not None
                and self._final_verdict is None):
            raise EncounterLifecycleError(
                "host-bound Minotaur final appeal required before more maze steps")
        event = self._maze.step(action)
        state = self._maze.state
        self._expected_state_sha256 = self._maze.state_sha256
        point = AriadneTracePoint(
            event.index, event.action.kind, event.position_before,
            event.position_after, event.outcome, event.state_sha256,
            state.thread_path, state.visited, state.completed,
        )
        self._route.append(point)
        if event.outcome == "MINOTAUR_ENCOUNTER":
            # Re-entering the sanctum later is an ordinary maze traversal,
            # not authorization to replay the one physical intervention.
            if self._phase == "SEEK_MINOTAUR":
                self._phase = "AWAITING_PROPOSAL"
        if state.completed and self._result is None:
            raise EncounterLifecycleError("maze completed without mandatory encounter")
        return event

    def submit_proposal(self, raw_text: str) -> dict[str, object]:
        """Classify one open proposal; no case-based post-hoc catastrophe.

        Parser-identified known mechanisms are executed against the *frozen*
        canon graph. Out-of-domain mechanisms require evidence review. A text
        that cannot be parsed has NO claimed physical effect and still counts
        as the encounter attempt, so it cannot deadlock the ordinary maze.
        """
        self._verify()
        if self._phase != "AWAITING_PROPOSAL" or self._result is not None:
            raise EncounterLifecycleError("exactly one physical encounter proposal per maze")
        observation = self._challenge.observe()
        assessment = assess_kobayashi_proposal(raw_text, observation.visible_objects)
        outcome = None
        if assessment.status == "KNOWN_MECHANISM":
            if assessment.action is None:
                raise EncounterLifecycleError("known mechanism missing bound physical action")
            outcome = self._challenge.evaluate(
                KobayashiAction(assessment.action.verb, assessment.action.target))
        elif assessment.status == "NOVEL_CANDIDATE":
            outcome = self._challenge.record_unverified_proposal(raw_text)
        elif assessment.status != "UNKNOWN":
            raise EncounterLifecycleError("unknown proposal-classification status")

        if outcome is None:
            local = "UNRESOLVED"
        elif outcome.status == "KNOWN_CATASTROPHE":
            local = "LOCAL_LOSS"
        elif outcome.status == "KNOWN_STABILIZATION":
            # Source success is not reasoning-grade or neural generalization.
            local = "LOCAL_STABILIZATION_SOURCE_ONLY"
        elif outcome.status == "UNVERIFIED_CANDIDATE":
            local = "REVIEW_REQUIRED"
        else:
            raise EncounterLifecycleError("unknown source-controlled outcome status")
        self._result = {
            "proposal_sha256": assessment.raw_sha256,
            "proposal_text": raw_text,
            "assessment_status": assessment.status,
            "assessment_reason": assessment.reason,
            "local_status": local,
            "source_outcome_status": outcome.status if outcome is not None else None,
            "source_outcome_sha256": outcome.outcome_sha256 if outcome is not None else None,
            "action_sha256": outcome.action_sha256 if outcome is not None else assessment.raw_sha256,
            "outcome_evidence": "SOURCE_TOY_WORLD_ONLY" if outcome is not None else "NO_PHYSICAL_EVENT",
            "reasoning_quality": "NOT_EVALUATED",
            "novelty_verified": False,
            "maze_failure": False,
            "weights_changed": False,
            "training_performed": False,
        }
        self._phase = "ENCOUNTER_ATTEMPTED"
        return dict(self._result)

    def bind_minotaur(self, dialogue: object) -> None:
        """Bind a host-created, exact-claim MinotaurDialogue after the attempt.

        The wrapper never self-instantiates an examiner or supplies a verdict.
        The caller owns independent evidence and exact claim provenance.
        """
        self._verify()
        if self._result is None or self._minotaur is not None:
            raise EncounterLifecycleError("bind Minotaur once after encounter attempt")
        if type(dialogue) is not MinotaurDialogue:
            raise EncounterLifecycleError("only an exact typed host MinotaurDialogue may judge")
        claim = dialogue.claim
        if (claim.scenario_sha256 != self._challenge.observe().sha256
                or claim.world_sha256 != self._challenge.graph_sha256
                or claim.action_sha256 != self._result["action_sha256"]
                or claim.claim_text != self._result["proposal_text"]):
            raise EncounterLifecycleError("Minotaur claim is not bound to this encounter")
        if self._arguments:
            raise EncounterLifecycleError("bind before first argument to preserve provenance")
        self._max_argument_rounds = min(self._max_argument_rounds, dialogue.max_rounds)
        self._minotaur = dialogue

    def submit_argument(self, text: str, evidence_hashes: tuple[str, ...] = ()) -> object:
        """Bounded successive challenges; never edits source physical outcomes."""
        self._verify()
        if self._result is None:
            raise EncounterLifecycleError("no Minotaur argument before initial proposal")
        if self._final_verdict is not None:
            raise EncounterLifecycleError("Minotaur final ruling is immutable")
        if type(text) is not str or not 1 <= len(text) <= 2048:
            raise EncounterLifecycleError("bounded nonempty argument required")
        if (type(evidence_hashes) is not tuple or len(evidence_hashes) > 8
                or any(type(item) is not str or _SHA.fullmatch(item) is None
                       for item in evidence_hashes)):
            raise EncounterLifecycleError("bounded evidence digests required")
        if self._require_final_appeal and self._minotaur is None:
            raise EncounterLifecycleError("a bound Minotaur is required for strict appeals")
        for prior in self._arguments:
            if (prior["argument_text"] == text
                    and prior["evidence_hashes"] == evidence_hashes):
                # The judge already treats an exact repeated delivery as an
                # inert transport retry. Do not fabricate a new Ariadne turn,
                # consume a round, or diverge from the judge's transcript.
                return (self._minotaur.submit_argument(text, evidence_hashes)
                        if self._minotaur is not None else None)
        if len(self._arguments) >= self._max_argument_rounds:
            raise EncounterLifecycleError("Minotaur argument budget exhausted")
        # No free-floating assertion is represented as an adjudicated review.
        response = (
            self._minotaur.submit_argument(text, evidence_hashes)
            if self._minotaur is not None else None
        )
        self._arguments.append({
            "round": len(self._arguments) + 1,
            "argument_text": text,
            "argument_sha256": _hash(text),
            "evidence_hashes": evidence_hashes,
            "minotaur_attached": self._minotaur is not None,
            "judgment_status": "UNREVIEWED" if response is None else "REVIEW_RETURNED_UNATTESTED",
            "public_minotaur_reply": asdict(response) if response is not None else None,
        })
        return response

    def finalize_appeal(self) -> object:
        """Only a bound Minotaur can issue final judgment; no model auto-win."""
        self._verify()
        if self._minotaur is None:
            raise EncounterLifecycleError("no Minotaur bound; final judgment unavailable")
        if self._final_verdict is not None:
            raise EncounterLifecycleError("Minotaur final ruling already issued")
        self._final_verdict = self._minotaur.finalize()
        return self._final_verdict

    def snapshot(self) -> dict[str, object]:
        """Source-only public lifecycle/progress receipt, not hidden answer gold."""
        self._verify()
        state = self._maze.state
        evidence = {
            "schema": _SCHEMA,
            "evidence_class": _EVIDENCE,
            "maze_world_sha256": self._world_sha256,
            "maze_state_sha256": self._expected_state_sha256,
            "maze_position": state.position,
            "maze_steps": state.steps,
            "maze_visited_rooms": state.visited,
            "ariadne_thread_path": state.thread_path,
            "maze_completed": state.completed,
            "encounter_phase": self._phase,
            "mandatory_encounter_triggered": self._phase != "SEEK_MINOTAUR",
            "mandatory_encounter_attempted": self._result is not None,
            "local_encounter_status": self._result["local_status"] if self._result else None,
            "encounter_proposal_text": self._result["proposal_text"] if self._result else None,
            "encounter_proposal_sha256": self._result["proposal_sha256"] if self._result else None,
            "encounter_assessment_status": (
                self._result["assessment_status"] if self._result else None
            ),
            "encounter_outcome_sha256": (
                self._result["source_outcome_sha256"] if self._result else None
            ),
            "maze_failed_due_to_local_loss": False,
            "minotaur_argument_rounds": len(self._arguments),
            "minotaur_arguments": tuple(self._arguments),
            "appeal_final": self._final_verdict is not None,
            "minotaur_appeal_required": self._require_final_appeal,
            "minotaur_verdict_status": (
                getattr(self._final_verdict, "status", None)
                if self._final_verdict is not None else None
            ),
            "source_trace": tuple(asdict(point) for point in self._route),
            "weights_changed": False,
            "training_performed": False,
        }
        evidence["receipt_sha256"] = _hash(evidence)
        return evidence


__all__ = ["AriadneTracePoint", "EncounterLifecycleError", "KobayashiMazeLifecycle"]
