"""Executable source-only Labyrinth saga: original world, two fake candidates.

Run ``python -m labyrinth.source_only_saga_demo_v1``. This is a bounded
development fixture, NOT Qwen inference, SGD, genuine examination or a host
training authorization. All side effects live under one temporary directory.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import shutil
import struct
from tempfile import TemporaryDirectory

from .ariadne_recovery_v1 import AriadneJournal
from .campaign_full_state_v1 import capture_campaign_state, save_campaign_state_once
from .campaign_orchestrator_v1 import _checkpoint_from_manifest
from .dynamic_scene_authorization_v1 import (
    DynamicSceneAuthority, DynamicTrainPolicy, run_one_approved_dynamic_train,
)
from .daedalus_social_director_v1 import SocialSceneContext
from .evolutionary_population_v1 import (
    Candidate, EvaluationPlan, Examination, Metrics, MutationKind,
    deterministic_seed, prepare_population, select_generation,
)
from .living_world_v1 import (
    DevelopmentVerdict, LivingSceneDirector, LivingWorld, LivingWorldRecipe,
    ModelSceneProposal, WorldAction,
)
from .model_saga_runtime import CheckpointPaths
from .social_experience_training_bridge_v1 import (
    OperatorDialogueAssessment, OperatorDialogueFeedback, SocialExperienceTrainingBridge,
)
from .social_world_consequence_v1 import CharacterRelationship, OperatorDialogueEffect
from .training_scope import TrainingScopeSpec


EVIDENCE = "SCRIPTED_FIXTURE_ONLY_NOT_LLM_OR_NEURAL_OR_INDEPENDENT_EXAM"


class FictionalActionModel:
    """Fixture actor that uses only observation + its own discovered map.

    It is an active planner, not a predetermined room-by-room script: it
    explores seeded branches by discovery, gathers seals, and retraces thread.
    """

    def __init__(self):
        self.visited: set[str] = set()
        self.parent: dict[str, str | None] = {}
        self.adjacency: dict[str, tuple[str, ...]] = {}
        self.met_minotaur = False
        self.choice_log: list[str] = []

    def choose_action(self, observation):
        room = observation.room_id
        if room not in self.visited:
            self.visited.add(room)
            self.parent[room] = self.parent.get(room)
        self.adjacency[room] = observation.visible_exits
        if room == "minotaur_sanctum":
            self.met_minotaur = True
        if self.met_minotaur:
            action = WorldAction("retrace")
        elif observation.visible_items:
            action = WorldAction("take", observation.visible_items[0])
        else:
            seals = sum(item.startswith("seal_") for item in observation.inventory)
            seal_goal = min(observation.difficulty, 3)
            # A seal found in a remote side-vault can reopen a gate in an
            # *already visited* passage. Navigate the learned public graph;
            # don't rely on a fixed full route or omniscient level layout.
            minotaur_neighbors = [
                known_room for known_room, neighbors in self.adjacency.items()
                if "minotaur_sanctum" in neighbors
            ]
            route = None
            if seals >= seal_goal and minotaur_neighbors:
                queue = deque([(room, ())])
                seen = {room}
                while queue:
                    current, steps = queue.popleft()
                    if current in minotaur_neighbors:
                        route = steps + ("minotaur_sanctum",)
                        break
                    for next_room in self.adjacency[current]:
                        if next_room in self.adjacency and next_room not in seen:
                            seen.add(next_room)
                            queue.append((next_room, steps + (next_room,)))
            if route:
                action = WorldAction("move", route[0])
                self.choice_log.append(action.kind + ":" + action.target)
                return action
            possible = (
                neighbor for neighbor in observation.visible_exits
                if neighbor not in self.visited
                and (neighbor != "minotaur_sanctum" or seals >= seal_goal)
            )
            target = next(possible, None)
            if target is not None:
                self.parent[target] = room
                action = WorldAction("move", target)
            elif self.parent[room] is not None:
                action = WorldAction("move", self.parent[room])
            elif "minotaur_sanctum" in observation.visible_exits:
                action = WorldAction("move", "minotaur_sanctum")
            else:
                # The fixture must not mask an unwinnable graph by spinning.
                raise RuntimeError("fictional actor exhausted known routes without Minotaur")
        self.choice_log.append(action.kind + (":" + action.target if action.target else ""))
        return action


class FictionalDialogueModel:
    """Scripted adapter: newly composes public dialogue, never grades itself."""

    def write_scene(self, context):
        observation = context.world_observation
        return ModelSceneProposal(
            context.character_ids[0], context.allowed_skills[0],
            "Miren studies the map in " + observation.room_title
            + ". 'If the corridors change, what promise will you remember?', she asks.",
        )


class FictionalResponder:
    def respond(self, observed):
        return "I will carry your promise, Miren, even when the walls change."


class FixtureDialogueOperator:
    def __init__(self, candidate_id):
        self.candidate_id = candidate_id

    def assess_train(self, beat, observation, response):
        return OperatorDialogueAssessment(
            OperatorDialogueEffect(1, 2, "Miren remembers a carefully kept promise."),
            OperatorDialogueFeedback(
                "fixture_feedback_" + self.candidate_id,
                "Keep my promise as the rooms change.",
                "Rehearse consistency and acknowledge uncertainty.",
            ),
        )


def _tiny_tensor(payload: bytes) -> bytes:
    header = json.dumps({"weight": {"dtype": "U8", "shape": [len(payload)],
                                    "data_offsets": [0, len(payload)]}}).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + payload


class FixturePhysicalTrainer:
    """Copies an existing toy genome and changes bytes. **Never SGD**."""

    def __init__(self, destination: Path, label: str):
        self.destination = destination
        self.label = label
        self.calls = 0

    def train_supervised(self, case, parent):
        if case.split != "train":
            raise ValueError("fixture trainer received a non-TRAIN case")
        self.calls += 1
        shutil.copytree(Path(parent.weights_path).parent.parent, self.destination)
        weights = self.destination / "adapter" / "adapter_model.safetensors"
        weights.write_bytes(_tiny_tensor((self.label + case.learner_response).encode()))
        optimizer = self.destination / "optimizer" / "state.pt"
        optimizer.write_bytes(("fixture-optimizer-" + self.label).encode())
        (self.destination / "cursor.json").write_text('{"step":1}', encoding="utf-8")
        return CheckpointPaths(str(weights), str(optimizer))


def _capture(root: Path, genome: Path, parent: dict | None, label: str) -> tuple[dict, Path]:
    manifest = capture_campaign_state(
        base_dir=genome / "base", adapter_dir=genome / "adapter",
        optimizer_dir=genome / "optimizer", scheduler_dir=genome / "scheduler",
        rng_dir=genome / "rng", data_cursor_path=genome / "cursor.json",
        family="hf_safetensors_lora", mutation_kind="lora_adapter",
        parent_manifest=parent,
    )
    manifest_path = root / (label + ".json")
    save_campaign_state_once(manifest, manifest_path)
    return manifest, manifest_path


def _founder(root: Path) -> tuple[dict, Path, Path]:
    genome = root / "w0"
    for part in ("base", "adapter", "optimizer", "scheduler", "rng"):
        (genome / part).mkdir(parents=True)
    (genome / "base" / "model.safetensors").write_bytes(_tiny_tensor(b"fixture-base"))
    (genome / "base" / "config.json").write_text('{"model_type":"toy"}')
    (genome / "base" / "tokenizer.json").write_text('{"tokens":["A"]}')
    (genome / "adapter" / "adapter_model.safetensors").write_bytes(_tiny_tensor(b"W0"))
    (genome / "adapter" / "adapter_config.json").write_text('{"r":4}')
    for part, filename in (("optimizer", "state.pt"), ("scheduler", "scheduler.pt"),
                           ("rng", "rng.pt")):
        (genome / part / filename).write_bytes(("fixture-" + part).encode())
    (genome / "cursor.json").write_text('{"step":0}')
    manifest, path = _capture(root, genome, None, "w0")
    return manifest, path, genome


def _play_maze(world: LivingWorld) -> dict:
    actor = FictionalActionModel()
    encounter_outcomes = []
    for _ in range(world.definition.recipe.max_steps):
        event = world.step_from_actor(actor)
        encounter_outcomes.append(event.outcome)
        if world.state.completed:
            return {"actions": actor.choice_log,
                    "outcomes": encounter_outcomes,
                    "world_sha256": world.definition.world_sha256,
                    "final_state_sha256": world.state_sha256,
                    "completed": True}
    raise RuntimeError("fictional model did not finish within source step limit")


def run_fixture_saga(root: Path) -> dict:
    """Run a two-cycle, full-state-backed, explicitly unqualified demo.

    Caller supplies a NEW empty directory; no network, real weights or host
    accounts are touched. Two sibling physical bytes candidates are compared
    under fabricated-for-demonstration development metrics only.
    """
    if not root.is_absolute() or root.exists() or not root.parent.is_dir():
        raise ValueError("saga root must be a new absolute fixture directory")
    root.mkdir(mode=0o700)
    scope = TrainingScopeSpec("saga_continuity", "conversation", ("continuity",),
                              specificity=2, scale=1, max_cycles=2, seed=71)
    world = LivingWorld(LivingWorldRecipe(scope.objective_id, scope.skills,
                                          ("miren",), public_seed=71,
                                          max_steps=128))
    founder, founder_path, founder_genome = _founder(root)
    journal = AriadneJournal.create(root / "ariadne")
    journal.initialize(
        w0_manifest_path=founder_path, world_id="world_1", version=1,
        curriculum={"objective_id": scope.objective_id, "train_only": True},
        world_state={"characters": [{"character_id": "miren", "trust": 0,
                                      "interactions": 0, "promises": []}]},
        idempotency_key="saga_founder",
    )
    policy = DynamicTrainPolicy(scope.objective_id, world.definition.world_sha256,
                                scope.skills, ("miren",), minimum_difficulty=1)
    authority = DynamicSceneAuthority(policy)
    director = LivingSceneDirector(world, FictionalDialogueModel())
    scene_context = SocialSceneContext(scope.objective_id, 0, 71,
                                       scope.skills, ("miren",), (), (), 1)
    children = []
    sparks = []
    for candidate_id in ("cand_ariadne", "cand_daedalus"):
        scene = authority.approve(
            world=world, director=director, context=scene_context,
            operator_approves=lambda beat, public: (
                beat.split == "train" and public.room_id == "ariadne_atrium"),
        )
        trainer = FixturePhysicalTrainer(root / candidate_id, candidate_id)
        bridge = SocialExperienceTrainingBridge(
            scope, characters=(CharacterRelationship("miren"),),
            checkpoint=_checkpoint_from_manifest(founder),
            responder=FictionalResponder(),
            operator=FixtureDialogueOperator(candidate_id), trainer=trainer,
            journal_directory=root / ("social_" + candidate_id),
        )

        def manifest_builder(receipt, parent, *, label=candidate_id):
            return _capture(root, root / label, parent, label)[1]

        trained = run_one_approved_dynamic_train(
            authority=authority, scene=scene, world=world, journal=journal,
            bridge=bridge, candidate_id=candidate_id,
            trainer_id="fixture_trainer", manifest_builder=manifest_builder,
        )
        if trainer.calls != 1:
            raise RuntimeError("source-only TRAIN fixture did not apply exactly one physical change")
        children.append(Candidate(
            candidate_id, trained.child_manifest_path, MutationKind.ADAPTER_DELTA,
            "fixture_trainer", trained.ariadne_commit_sha256,
            founder["state_sha256"], deterministic_seed(founder["state_sha256"],
                                                         1, candidate_id),
        ))
        sparks.append({"candidate_id": candidate_id,
                       "scene_sha256": trained.scene_sha256,
                       "child_state_sha256": trained.child_state_sha256,
                       "fixture_physical_changes": trainer.calls})

    # The actor must actually traverse Ariadne's changed world and confront
    # the Minotaur before the caller supplies any development-only decision.
    first_cycle = _play_maze(world)

    # A physically unchanged control, not a second invented baseline.
    control_genome = root / "unchanged_control"
    shutil.copytree(founder_genome, control_genome)
    _, control_path = _capture(root, control_genome, None, "unchanged_control")
    fixture_digest = lambda digit: digit * 64
    plan = EvaluationPlan("fixture_examiner", fixture_digest("a"),
                          fixture_digest("b"), fixture_digest("c"), fixture_digest("d"))
    population = prepare_population(
        development_generation=1, origin_manifest_path=founder_path,
        parent_manifest_path=founder_path, control_manifest_path=control_path,
        candidates=tuple(children), plan=plan,
    )
    fixture_metrics = {
        "parent": Metrics(.50, .50, .99, .99, .02),
        "control": Metrics(.50, .50, .99, .99, .02),
        "cand_ariadne": Metrics(.82, .83, .99, .99, .01),
        "cand_daedalus": Metrics(.70, .72, .99, .99, .01),
    }
    physical_states = {"parent": population.parent_state_sha256,
                       "control": population.parent_state_sha256,
                       **{name: state for name, state, _ in population.candidate_states}}
    exams = tuple(Examination(
        subject_id=name, state_sha256=state,
        examiner_id=plan.examiner_id, suite_sha256=plan.suite_sha256,
        sealed_case_commitment=plan.sealed_case_commitment,
        preregistration_sha256=plan.preregistration_sha256,
        private_cohort_commitment=plan.private_cohort_commitment,
        examiner_run_sha256=f"{index + 1:064x}",
        metrics=fixture_metrics[name],
    ) for index, (name, state) in enumerate(physical_states.items()))
    decision = select_generation(population, exams)
    for exam in exams:
        if exam.subject_id not in ("parent", "control"):
            journal.record_evaluation(
                candidate_id=exam.subject_id,
                evaluation={"metric_id": "fake_dev_metrics_no_private_gold",
                            "score": exam.metrics.holdout, "direction": "higher",
                            "dataset_ref": "synthetic_no_real_cohort",
                            "evidence": {"source_only": True}},
                idempotency_key="fixture_eval_" + exam.subject_id,
            )
    for rejected in decision.rejected_candidate_ids:
        journal.reject_candidate(candidate_id=rejected,
                                 reason="synthetic development fixture not selected",
                                 idempotency_key="fixture_reject_" + rejected)
    if decision.winner_id is None:
        raise RuntimeError("the deliberately positive fixture did not select any child")
    journal.select_candidate(
        candidate_id=decision.winner_id,
        selection_reason="source_fixture_comparison_not_real_minotaur",
        idempotency_key="fixture_select_" + decision.winner_id,
    )

    successor = world.evolve(DevelopmentVerdict(world.definition.world_sha256,
        world.state_sha256, "passed", "continuity"))
    journal.record_world(
        world_id="world_2", parent_world_id="world_1", version=2,
        curriculum={"objective_id": scope.objective_id, "train_only": True,
                    "living_world_sha256": successor.definition.world_sha256},
        world_state={"characters": [{"character_id": "miren", "trust": 1,
                                      "interactions": 1, "promises": []}],
                     "public_mementos": list(successor.state.public_mementos)},
        idempotency_key="saga_world_2",
    )
    second_cycle = _play_maze(successor)
    descriptor = journal.continuation_descriptor()
    if (descriptor["status"] != "READY_FOR_EXTERNAL_FULL_STATE_LOAD"
            or descriptor["selected_parent_state_sha256"] != decision.selected_state_sha256):
        raise RuntimeError("synthetic winner failed Ariadne full-state readback")
    return {
        "evidence_class": EVIDENCE,
        "story": "Ariadne's thread carried a fictional apprentice through the Minotaur's halls.",
        "dialogue_training": sparks,
        "fixture_selection": {"winner_id": decision.winner_id,
                              "selected_state_sha256": decision.selected_state_sha256,
                              "not_independent_exam": True},
        "cycles": [first_cycle, second_cycle],
        "difficulty": [world.definition.recipe.difficulty,
                       successor.definition.recipe.difficulty],
        "world_ancestry": [world.definition.world_sha256,
                           successor.definition.world_sha256],
        "public_mementos": list(successor.state.public_mementos),
        "ariadne_status": descriptor["status"],
        "ariadne_event_head_sha256": descriptor["event_head_sha256"],
        "real_model_or_sgd": False,
    }


def main():
    with TemporaryDirectory(prefix="labyrinth-fixture-saga-") as directory:
        result = run_fixture_saga(Path(directory).resolve() / "saga")
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
