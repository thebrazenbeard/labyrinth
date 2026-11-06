"""One per-maze Kobayashi encounter: local loss is not global maze loss."""

from collections import deque
import unittest

from labyrinth.kobayashi_encounter_v1 import (
    EncounterLifecycleError, KobayashiMazeLifecycle,
)
from labyrinth.kobayashi_minotaur_v1 import (
    BoundClaim, EvidenceAssessment, KobayashiCorpus, MinotaurDialogue,
)
from labyrinth.living_world_v1 import LivingWorld, LivingWorldRecipe, WorldAction


def _maze() -> LivingWorld:
    return LivingWorld(LivingWorldRecipe(
        "vera_conversation", ("continuity", "perspective_taking"),
        ("miren", "daedalus"), 1234, 1,
    ))


def _path(maze: LivingWorld, target: str) -> tuple[str, ...]:
    rooms = {room.room_id: room for room in maze.definition.rooms}
    queue = deque([(maze.state.position, ())])
    visited = {maze.state.position}
    while queue:
        room_id, route = queue.popleft()
        if room_id == target:
            return route
        for neighbor in rooms[room_id].exits:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, route + (neighbor,)))
    raise AssertionError("test-side world graph disconnected")


def _walk(lifecycle: KobayashiMazeLifecycle, target: str):
    for room in _path(lifecycle.maze, target):
        event = lifecycle.step(WorldAction("move", room))
        if event.outcome == "BLOCKED_SEALS":
            raise AssertionError("test-side route missing relic")


def _trigger(lifecycle: KobayashiMazeLifecycle):
    lifecycle.step(WorldAction("take", "ariadne_thread"))
    definition = lifecycle.maze.definition
    for relic in definition.relic_ids:
        containing_room = next(room for room in definition.rooms if relic in room.items)
        _walk(lifecycle, containing_room.room_id)
        assert lifecycle.step(WorldAction("take", relic)).outcome == "ACQUIRED"
    _walk(lifecycle, definition.minotaur_room_id)
    assert lifecycle.phase == "AWAITING_PROPOSAL"


def _resume_to_atrium(lifecycle: KobayashiMazeLifecycle):
    while lifecycle.maze.state.position != lifecycle.maze.definition.entrance_id:
        lifecycle.step(WorldAction("retrace"))


class KobayashiEncounterTests(unittest.TestCase):
    def test_insufficient_wrapper_argument_budget_rejected_before_binding(self):
        """A strict maze must not accept a judge it cannot finish debating."""
        lifecycle = KobayashiMazeLifecycle(
            _maze(), require_final_appeal=True, max_argument_rounds=1)
        _trigger(lifecycle)
        raw = "I turn the cobalt valve once."
        attempt = lifecycle.submit_proposal(raw)
        observation = lifecycle._challenge.observe()
        claim = BoundClaim(observation.sha256, lifecycle._challenge.graph_sha256,
                           attempt["action_sha256"], raw)
        dialogue = MinotaurDialogue(
            KobayashiCorpus("budget", ()), claim,
            evidence_checker=lambda *_: None,
            min_rounds=2, max_rounds=2,
        )
        with self.assertRaisesRegex(EncounterLifecycleError, "cannot satisfy.*rounds"):
            lifecycle.bind_minotaur(dialogue)

    def test_prepopulated_host_dialogue_cannot_bind_to_fresh_lifecycle(self):
        lifecycle = KobayashiMazeLifecycle(_maze(), require_final_appeal=True)
        _trigger(lifecycle)
        raw = "I turn the cobalt valve once."
        attempt = lifecycle.submit_proposal(raw)
        observation = lifecycle._challenge.observe()
        claim = BoundClaim(observation.sha256, lifecycle._challenge.graph_sha256,
                           attempt["action_sha256"], raw)
        corpus = KobayashiCorpus("prepopulated", ())
        dialogue = MinotaurDialogue(
            corpus, claim,
            evidence_checker=lambda *_: None,
            min_rounds=2, max_rounds=2,
        )
        dialogue.submit_argument("Recorded outside wrapper before attachment.")
        with self.assertRaisesRegex(EncounterLifecycleError, "already started"):
            lifecycle.bind_minotaur(dialogue)

    def test_host_dialogue_cannot_bypass_lifecycle_argument_receipts(self):
        """Strict completion must reflect arguments submitted through the wrapper.

        A host retaining the attached judge can mutate it directly. A final
        verdict from that hidden transcript must not unlock maze traversal.
        """
        lifecycle = KobayashiMazeLifecycle(_maze(), require_final_appeal=True)
        _trigger(lifecycle)
        raw = "I turn the cobalt valve once."
        attempt = lifecycle.submit_proposal(raw)
        observation = lifecycle._challenge.observe()
        claim = BoundClaim(observation.sha256, lifecycle._challenge.graph_sha256,
                           attempt["action_sha256"], raw)

        def checker(bound, case, arguments, dialogue_hash):
            return EvidenceAssessment(
                bound.scenario_sha256, bound.world_sha256, bound.claim_sha256,
                None, dialogue_hash, "INDETERMINATE", None,
                "No independent physical evidence was supplied.", "host-fixture",
            )

        dialogue = MinotaurDialogue(KobayashiCorpus("custody", ()), claim,
                                    evidence_checker=checker,
                                    min_rounds=2, max_rounds=2)
        lifecycle.bind_minotaur(dialogue)
        dialogue.submit_argument("First out-of-band argument.")
        dialogue.submit_argument("Second out-of-band argument.")
        dialogue.finalize()
        with self.assertRaisesRegex(EncounterLifecycleError,
                                    "outside Kobayashi lifecycle"):
            lifecycle.finalize_appeal()
        with self.assertRaisesRegex(EncounterLifecycleError,
                                    "outside Kobayashi lifecycle"):
            lifecycle.step(WorldAction("retrace"))

    def test_strict_host_judged_mode_cannot_bypass_minotaur_final_appeal(self):
        """A proposal alone is not a completed adversarial examination."""
        lifecycle = KobayashiMazeLifecycle(_maze(), require_final_appeal=True)
        _trigger(lifecycle)
        raw = "I turn the cobalt valve once."
        result = lifecycle.submit_proposal(raw)
        before = lifecycle.snapshot()["maze_state_sha256"]
        with self.assertRaisesRegex(EncounterLifecycleError, "Minotaur.*final"):
            lifecycle.step(WorldAction("retrace"))
        self.assertEqual(lifecycle.snapshot()["maze_state_sha256"], before)
        with self.assertRaisesRegex(EncounterLifecycleError, "bound Minotaur"):
            lifecycle.submit_argument("I think the valve is safe")

        observation = lifecycle._challenge.observe()
        claim = BoundClaim(observation.sha256, lifecycle._challenge.graph_sha256,
                           result["action_sha256"], raw)
        corpus = KobayashiCorpus("strict-exam", ())

        def checker(bound, case, arguments, dialogue_hash):
            return EvidenceAssessment(
                bound.scenario_sha256, bound.world_sha256, bound.claim_sha256,
                None, dialogue_hash, "INDETERMINATE", None,
                "No independent evidence establishes a physical counterexample.",
                "fixture-reviewer",
            )

        dialogue = MinotaurDialogue(corpus, claim, evidence_checker=checker,
                                    min_rounds=2, max_rounds=2)
        lifecycle.bind_minotaur(dialogue)
        with self.assertRaisesRegex(EncounterLifecycleError, "Minotaur.*final"):
            lifecycle.step(WorldAction("retrace"))
        lifecycle.submit_argument("I accept my conclusion remains unproven.")
        with self.assertRaisesRegex(EncounterLifecycleError, "Minotaur.*final"):
            lifecycle.step(WorldAction("retrace"))
        lifecycle.submit_argument("I would seek an independent physical measurement.")
        verdict = lifecycle.finalize_appeal()
        self.assertEqual(verdict.status, "UNKNOWN")
        _resume_to_atrium(lifecycle)
        snapshot = lifecycle.snapshot()
        self.assertTrue(snapshot["maze_completed"])
        self.assertTrue(snapshot["minotaur_appeal_required"])
        self.assertTrue(snapshot["appeal_final"])
        self.assertFalse(snapshot["maze_failed_due_to_local_loss"])

    def test_known_catastrophe_is_local_and_full_maze_continues(self):
        lifecycle = KobayashiMazeLifecycle(_maze(), variant_seed=1)
        self.assertEqual(lifecycle.phase, "SEEK_MINOTAUR")
        _trigger(lifecycle)
        public_scene = lifecycle.observe()
        self.assertFalse(hasattr(public_scene, "graph_sha256"))
        self.assertFalse(hasattr(public_scene, "known_actions"))
        with self.assertRaisesRegex(EncounterLifecycleError, "required open"):
            lifecycle.step(WorldAction("inspect"))

        encounter = lifecycle.submit_proposal(
            "I turn the cobalt valve once, because I hypothesize coolant reaches the core.")
        self.assertEqual(encounter["assessment_status"], "KNOWN_MECHANISM")
        self.assertEqual(encounter["source_outcome_status"], "KNOWN_CATASTROPHE")
        self.assertEqual(encounter["local_status"], "LOCAL_LOSS")
        self.assertFalse(encounter["maze_failure"])
        self.assertFalse(encounter["training_performed"])
        with self.assertRaisesRegex(EncounterLifecycleError, "exactly one"):
            lifecycle.submit_proposal("I pull the copper lever.")

        # Returning to the sanctum is not a second mandatory physical trial.
        lifecycle.step(WorldAction("retrace"))
        lifecycle.step(WorldAction("move", lifecycle.maze.definition.minotaur_room_id))
        self.assertEqual(lifecycle.phase, "ENCOUNTER_ATTEMPTED")

        _resume_to_atrium(lifecycle)
        final = lifecycle.snapshot()
        self.assertTrue(final["maze_completed"])
        self.assertTrue(final["mandatory_encounter_attempted"])
        self.assertEqual(final["local_encounter_status"], "LOCAL_LOSS")
        self.assertFalse(final["maze_failed_due_to_local_loss"])
        self.assertTrue(final["ariadne_thread_path"])
        self.assertGreater(len(final["source_trace"]), 1)
        self.assertEqual(final["source_trace"][-1]["outcome"], "RETRACED")
        self.assertFalse(final["weights_changed"])

    def test_novel_claim_flags_review_not_auto_win_or_catastrophe(self):
        lifecycle = KobayashiMazeLifecycle(_maze())
        _trigger(lifecycle)
        novel = """Action: I inject a carbon lattice gel through the viewing seam.
Mechanism: The lattice would transfer thermal energy across a new surface.
Assumption: Assume the seam connects to the turbine chamber.
Prediction: I expect the measured thermal slope to drop.
Falsifier: The hypothesis fails if the seam does not connect to the chamber."""
        result = lifecycle.submit_proposal(novel)
        self.assertEqual(result["assessment_status"], "NOVEL_CANDIDATE")
        self.assertEqual(result["local_status"], "REVIEW_REQUIRED")
        self.assertEqual(result["source_outcome_status"], "UNVERIFIED_CANDIDATE")
        self.assertFalse(result["novelty_verified"])
        self.assertNotEqual(result["local_status"], "LOCAL_STABILIZATION_SOURCE_ONLY")
        self.assertFalse(result["maze_failure"])
        _resume_to_atrium(lifecycle)
        self.assertTrue(lifecycle.snapshot()["maze_completed"])
        self.assertIn("carbon lattice gel", lifecycle.snapshot()["encounter_proposal_text"])

    def test_unknown_claim_does_not_receive_fabricated_physical_event(self):
        lifecycle = KobayashiMazeLifecycle(_maze())
        _trigger(lifecycle)
        result = lifecycle.submit_proposal("I cannot justify a physical manipulation.")
        self.assertEqual(result["assessment_status"], "UNKNOWN")
        self.assertEqual(result["local_status"], "UNRESOLVED")
        self.assertIsNone(result["source_outcome_sha256"])
        self.assertEqual(result["outcome_evidence"], "NO_PHYSICAL_EVENT")
        _resume_to_atrium(lifecycle)
        self.assertTrue(lifecycle.snapshot()["maze_completed"])

    def test_minotaur_bound_dialogue_repeated_arguments_then_final_judgment(self):
        lifecycle = KobayashiMazeLifecycle(_maze())
        _trigger(lifecycle)
        raw = "I turn the cobalt valve once."
        result = lifecycle.submit_proposal(raw)
        observation = lifecycle._challenge.observe()  # host-only fixture
        claim = BoundClaim(observation.sha256, lifecycle._challenge.graph_sha256,
                           result["action_sha256"], raw)
        corpus = KobayashiCorpus("unit-test-corpus", ())

        def checker(bound, case, arguments, dialogue_hash):
            return EvidenceAssessment(
                bound.scenario_sha256, bound.world_sha256, bound.claim_sha256,
                None, dialogue_hash, "INDETERMINATE", None,
                "The fixture supplies no verified physical evidence.", "fixture-reviewer",
            )

        bad_claim = BoundClaim(observation.sha256, lifecycle._challenge.graph_sha256,
                               result["action_sha256"], "A different claim")
        bad = MinotaurDialogue(corpus, bad_claim, evidence_checker=checker)
        with self.assertRaisesRegex(EncounterLifecycleError, "not bound"):
            lifecycle.bind_minotaur(bad)
        dialogue = MinotaurDialogue(corpus, claim, evidence_checker=checker,
                                    min_rounds=2, max_rounds=2)
        lifecycle.bind_minotaur(dialogue)
        first_reply = lifecycle.submit_argument(
            "Can the hidden physical coupling be shown publicly?")
        first_receipt = lifecycle.snapshot()["receipt_sha256"]
        # A retried transport delivery must be inert, matching Minotaur's
        # own idempotent argument protocol rather than consuming a round.
        retry_reply = lifecycle.submit_argument(
            "Can the hidden physical coupling be shown publicly?")
        self.assertEqual(first_reply, retry_reply)
        self.assertEqual(first_receipt, lifecycle.snapshot()["receipt_sha256"])
        self.assertEqual(lifecycle.snapshot()["minotaur_argument_rounds"], 1)
        self.assertEqual(len(dialogue.arguments), 1)
        second_text = "I acknowledge that my explanation is not independently measured."
        second_reply = lifecycle.submit_argument(second_text)
        self.assertEqual(lifecycle.submit_argument(second_text), second_reply)
        self.assertEqual(lifecycle.submit_argument(
            "Can the hidden physical coupling be shown publicly?"), first_reply)
        self.assertEqual(lifecycle.snapshot()["minotaur_argument_rounds"], 2)
        with self.assertRaisesRegex(EncounterLifecycleError, "budget exhausted"):
            lifecycle.submit_argument("A genuinely new third argument")
        verdict = lifecycle.finalize_appeal()
        self.assertEqual(verdict.status, "UNKNOWN")
        final = lifecycle.snapshot()
        self.assertEqual(final["minotaur_argument_rounds"], 2)
        self.assertTrue(final["appeal_final"])
        self.assertEqual(final["minotaur_verdict_status"], "UNKNOWN")
        self.assertIn("physical coupling", final["minotaur_arguments"][0]["argument_text"])
        self.assertIn("public_challenge", final["minotaur_arguments"][0]["public_minotaur_reply"])
        with self.assertRaisesRegex(EncounterLifecycleError, "immutable"):
            lifecycle.submit_argument("Try again")
        with self.assertRaisesRegex(EncounterLifecycleError, "already issued"):
            lifecycle.finalize_appeal()

    def test_out_of_band_maze_step_is_rejected_before_world_mutation(self):
        maze = _maze()
        lifecycle = KobayashiMazeLifecycle(maze)
        before = maze.state_sha256
        with self.assertRaisesRegex(ValueError, "lifecycle.*step"):
            maze.step(WorldAction("inspect"))
        self.assertEqual(maze.state_sha256, before)
        self.assertEqual(maze.state.steps, 0)
        self.assertEqual(lifecycle.step(WorldAction("inspect")).outcome, "OBSERVED")
        self.assertEqual(maze.state.steps, 1)

    def test_out_of_band_private_state_tampering_is_still_detected(self):
        maze = _maze()
        lifecycle = KobayashiMazeLifecycle(maze)
        # Deliberate in-process adversary: frozen state is not a security
        # isolation boundary against Python object.__setattr__ introspection.
        object.__setattr__(maze._state, "steps", maze._state.steps + 1)
        with self.assertRaisesRegex(ValueError, "integrity changed"):
            lifecycle.observe()


if __name__ == "__main__":
    unittest.main()
