"""Two-generation source-only social campaign, physical ancestry and WAL tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from labyrinth.ariadne_recovery_v1 import AriadneJournal
from labyrinth.campaign_full_state_v1 import capture_campaign_state, save_campaign_state_once
from labyrinth.campaign_orchestrator_v1 import (
    CampaignBoundaryError, CampaignOrchestrator, CandidateLaunch,
    CampaignGenerationResult, GenerationPendingExam,
)
from labyrinth.conversation_objective_v1 import build_synthetic_provider
from labyrinth.evolutionary_population_v1 import EvaluationPlan, Examination, Metrics
from labyrinth.model_saga_runtime import CheckpointPaths
from labyrinth.social_experience_training_bridge_v1 import (
    OperatorDialogueAssessment, OperatorDialogueFeedback,
)
from labyrinth.social_world_consequence_v1 import (
    CharacterRelationship, OperatorDialogueEffect,
)
from labyrinth.training_scope import TrainingScopeSpec


def _tensor(value: bytes) -> bytes:
    header = json.dumps({"weight": {"dtype": "U8", "shape": [len(value)],
                                     "data_offsets": [0, len(value)]}}).encode()
    return struct.pack("<Q", len(header)) + header + value


def _h(value: str) -> str:
    return value * 64


class FixtureDirector:
    """Scripted source fixture, never a verified model director."""

    def __init__(self, scenes):
        self.scenes = scenes
        self.contexts = []

    def propose_scene(self, context):
        self.contexts.append(context)
        return self.scenes[len(self.contexts) - 1]


class FixtureResponder:
    """Scripted mock dialogue, not Qwen inference."""

    def __init__(self):
        self.inputs = []

    def respond(self, observed):
        self.inputs.append(observed)
        return "I remember your correction and will act carefully."


class FixtureOperator:
    def __init__(self, candidate_id):
        self.candidate_id = candidate_id

    def assess_train(self, beat, observation, response):
        return OperatorDialogueAssessment(
            OperatorDialogueEffect(1, 2, "Miren hears and trusts a little more."),
            OperatorDialogueFeedback(
                "approved_" + self.candidate_id + "_" + beat.beat_id,
                "Acknowledge the changed detail.", "Remember only public facts.",
            ),
        )


class FixturePhysicalTrainer:
    """Actual file writes; NOT SGD, an HF loader or a trained model."""

    def __init__(self, root: Path, marker: str, *, crash=False):
        self.root = root
        self.marker = marker
        self.crash = crash
        self.calls = 0
        self.predecessors = []
        self.split_seen = []

    def train_supervised(self, case, parent):
        self.calls += 1
        self.predecessors.append(parent.weights_sha256)
        self.split_seen.append(case.split)
        if case.split != "train":
            raise AssertionError("private exam gold passed into source TRAIN")
        source = Path(parent.weights_path).parent.parent
        shutil.copytree(source, self.root)
        weights = self.root / "adapter" / "adapter_model.safetensors"
        weights.write_bytes(_tensor((self.marker + case.learner_response).encode()))
        optimizer = self.root / "optimizer" / "state.pt"
        optimizer.write_bytes(("mock-optimizer-" + self.marker).encode())
        (self.root / "cursor.json").write_text('{"step":1}')
        if self.crash:
            raise RuntimeError("injected crash after physical candidate bytes")
        return CheckpointPaths(str(weights), str(optimizer))


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.provider = build_synthetic_provider()
        self.scope = TrainingScopeSpec(
            self.provider.objective_id, "conversation", ("continuity",), 2, 1, 2, 123,
        )
        self.w0_root = self.root / "w0"
        self._make_w0()
        self.w0_manifest, self.w0_path = self._capture(self.w0_root, None, "w0")
        self.plan = EvaluationPlan("fixture_minotaur", _h("a"), _h("b"),
                                   _h("c"), _h("d"))
        self.journal = AriadneJournal.create(self.root / "ariadne")
        self.orchestrator = CampaignOrchestrator(
            journal=self.journal, origin_manifest_path=self.w0_path,
            scope=self.scope, provider=self.provider,
            characters=(CharacterRelationship("miren"),),
        )

    def _make_w0(self):
        for role in ("base", "adapter", "optimizer", "scheduler", "rng"):
            (self.w0_root / role).mkdir(parents=True)
        (self.w0_root / "base" / "config.json").write_text('{"model_type":"toy"}')
        (self.w0_root / "base" / "tokenizer.json").write_text('{"tokens":["A"]}')
        (self.w0_root / "base" / "model.safetensors").write_bytes(_tensor(b"base"))
        (self.w0_root / "adapter" / "adapter_config.json").write_text('{"r":4}')
        (self.w0_root / "adapter" / "adapter_model.safetensors").write_bytes(_tensor(b"W0"))
        (self.w0_root / "optimizer" / "state.pt").write_bytes(b"mock optimizer W0")
        (self.w0_root / "scheduler" / "scheduler.pt").write_bytes(b"mock scheduler W0")
        (self.w0_root / "rng" / "rng.pt").write_bytes(b"mock rng W0")
        (self.w0_root / "cursor.json").write_text('{"step":0}')

    def _capture(self, directory, parent, label):
        manifest = capture_campaign_state(
            base_dir=directory / "base", adapter_dir=directory / "adapter",
            optimizer_dir=directory / "optimizer", scheduler_dir=directory / "scheduler",
            rng_dir=directory / "rng", data_cursor_path=directory / "cursor.json",
            family="hf_safetensors_lora", mutation_kind="lora_adapter",
            parent_manifest=parent,
        )
        path = self.root / f"{label}.json"
        save_campaign_state_once(manifest, path)
        return manifest, path

    def _control(self, source_root: Path, generation: int):
        path = self.root / f"control_{generation}"
        shutil.copytree(source_root, path)
        _, manifest_path = self._capture(path, None, f"control_{generation}")
        return manifest_path

    def _launches(self, generation: int, *, crash_first=False):
        launched = []
        trainers = []
        responders = []
        for name in ("left", "right"):
            candidate_id = f"g{generation}_{name}"
            folder = self.root / f"gen{generation}_{name}"
            trainer = FixturePhysicalTrainer(folder, candidate_id,
                                             crash=crash_first and name == "left")
            responder = FixtureResponder()
            beats = self.provider.public_train_beats(
                "paper-starship", "miren", region_id=f"region_{generation}_{name}",
                chapter_id=f"chapter_{generation}_{name}")

            def build(receipt, parent, *, result_root=folder, ident=candidate_id):
                return self._capture(result_root, parent, ident)[1]

            launched.append(CandidateLaunch(
                candidate_id, "paper-starship", "miren",
                f"region_{generation}_{name}", f"chapter_{generation}_{name}",
                FixtureDirector(beats), responder, FixtureOperator(candidate_id),
                trainer, build, self.root / f"bridge_journal_{candidate_id}",
                "fixture_trainer", max_beats=1))
            trainers.append(trainer)
            responders.append(responder)
        return tuple(launched), trainers, responders

    def _external_fixture_exams(self, pop, generation):
        """Untrusted operator-supplied test values; NO independent exam run."""
        states = {"parent": pop.parent_state_sha256,
                  "control": pop.parent_state_sha256,
                  **{name: sha for name, sha, _ in pop.candidate_states}}
        metrics = {
            "parent": Metrics(.5, .5, .99, .99, .02),
            "control": Metrics(.5, .5, .99, .99, .02),
        }
        for name in states:
            if name not in metrics:
                stronger = name.endswith("right") if generation == 1 else name.endswith("left")
                metrics[name] = Metrics(.8 if stronger else .65,
                                        .8 if stronger else .65, .99, .99, .01)
        return tuple(Examination(
            subject_id=name, state_sha256=state,
            examiner_id=self.plan.examiner_id, suite_sha256=self.plan.suite_sha256,
            sealed_case_commitment=self.plan.sealed_case_commitment,
            preregistration_sha256=self.plan.preregistration_sha256,
            private_cohort_commitment=self.plan.private_cohort_commitment,
            examiner_run_sha256=f"{generation * 10 + index + 1:064x}",
            metrics=metrics[name],
        ) for index, (name, state) in enumerate(states.items()))

    def test_campaign_receipts_cannot_be_relabelled_as_graduation(self):
        pending = GenerationPendingExam(1, None, ())
        completed = CampaignGenerationResult(None, "unknown", ())
        for unearned in ("GRADUATED", "NEURAL_UPDATE_VERIFIED", "INDEPENDENT_REVIEW_PASS"):
            with self.subTest(unearned=unearned):
                with self.assertRaisesRegex(CampaignBoundaryError, "cannot attest"):
                    replace(pending, evidence_class=unearned)
                with self.assertRaisesRegex(CampaignBoundaryError, "cannot attest"):
                    replace(completed, evidence_class=unearned)

    def test_two_generations_carry_selected_social_world_and_full_model_parent(self):
        first_launches, first_trainers, first_responders = self._launches(1)
        pending1 = self.orchestrator.run_generation(
            launches=first_launches, control_manifest_path=self._control(self.w0_root, 1),
            evaluation_plan=self.plan)
        self.assertEqual(len(pending1.experiences), 2)
        self.assertTrue(all(experience.social_receipt.sparks
                            for experience in pending1.experiences))
        self.assertTrue(all(trainer.calls == 1 and trainer.split_seen == ["train"]
                            for trainer in first_trainers))
        self.assertTrue(all(obs.message_role == "user"
                            for responder in first_responders for obs in responder.inputs))
        self.assertTrue(all("reference_claims" not in obs.as_user_message()
                            for responder in first_responders for obs in responder.inputs))
        # No examiner was run or score invented during the candidate TRAIN stage.
        self.assertEqual(len(self.journal.continuation_descriptor()["selected_lineage"]), 0)
        result1 = self.orchestrator.finalize_generation(
            pending1, self._external_fixture_exams(pending1.population, 1))
        self.assertEqual(result1.decision.winner_id, "g1_right")
        self.assertEqual(result1.selected_characters[0].trust, 1)
        self.assertEqual(result1.selected_characters[0].last_public_reaction,
                         "Miren hears and trusts a little more.")
        self.assertFalse(self.journal.continuation_descriptor()["model_loaded"])
        worlds_after_first = self.journal.continuation_descriptor()["worlds"]
        # Pre-feature GENESIS shape is byte-compatible for absent reaction.
        self.assertNotIn("last_public_reaction",
                         worlds_after_first[0]["world_state"]["characters"][0])
        self.assertEqual(
            worlds_after_first[1]["world_state"]["characters"][0]["last_public_reaction"],
            "Miren hears and trusts a little more.",
        )
        self.assertNotIn("quality_score",
                         worlds_after_first[1]["world_state"]["characters"][0])

        selected_path = result1.decision.selected_manifest_path
        selected_root = Path(json.loads(selected_path.read_text())["adapter"]["root"]).parent
        second_launches, second_trainers, second_responders = self._launches(2)
        pending2 = self.orchestrator.run_generation(
            launches=second_launches,
            control_manifest_path=self._control(selected_root, 2),
            evaluation_plan=self.plan)
        self.assertEqual(pending2.population.parent_state_sha256,
                         result1.decision.selected_state_sha256)
        self.assertEqual(second_responders[0].inputs[0].public_observation.character_context.count(
            "tentatively willing to open up"), 1)
        self.assertIn("Miren hears and trusts a little more.",
                      second_responders[0].inputs[0].public_observation.character_context)
        result2 = self.orchestrator.finalize_generation(
            pending2, self._external_fixture_exams(pending2.population, 2))
        self.assertEqual(result2.decision.winner_id, "g2_left")
        self.assertEqual(result2.selected_characters[0].trust, 2)
        continuation = AriadneJournal.open(self.journal.root).continuation_descriptor(
            expected_head_sha256=result2.journal_head_sha256)
        self.assertEqual(continuation["next_generation_index"], 2)
        self.assertEqual(len(continuation["selected_lineage"]), 2)
        self.assertEqual(continuation["selected_parent_state_sha256"],
                         result2.decision.selected_state_sha256)
        self.assertEqual(continuation["status"], "READY_FOR_EXTERNAL_FULL_STATE_LOAD")
        self.assertFalse(continuation["neural_update_proven"])
        self.assertFalse(continuation["graduation_authorized"])

    def test_crash_after_actual_file_write_keeps_both_wals_and_no_replay(self):
        launches, trainers, _ = self._launches(1, crash_first=True)
        with self.assertRaisesRegex(RuntimeError, "injected crash"):
            self.orchestrator.run_generation(
                launches=launches, control_manifest_path=self._control(self.w0_root, 1),
                evaluation_plan=self.plan)
        self.assertEqual(trainers[0].calls, 1)
        self.assertEqual(trainers[1].calls, 0)
        self.assertTrue((self.root / "bridge_journal_g1_left" /
                         "beat-0000.intent.json").exists())
        self.assertFalse((self.root / "bridge_journal_g1_left" /
                          "beat-0000.commit.json").exists())
        self.assertEqual(self.journal.continuation_descriptor()["status"],
                         "INDETERMINATE_TRAINER_EFFECT")
        with self.assertRaisesRegex(CampaignBoundaryError, "pending or indeterminate"):
            self.orchestrator.run_generation(
                launches=launches, control_manifest_path=self.w0_path,
                evaluation_plan=self.plan)
        self.assertEqual(trainers[0].calls, 1)

    def test_forged_trainer_examination_stops_before_selection_and_replay(self):
        launches, _, _ = self._launches(1)
        pending = self.orchestrator.run_generation(
            launches=launches, control_manifest_path=self._control(self.w0_root, 1),
            evaluation_plan=self.plan)
        exams = self._external_fixture_exams(pending.population, 1)
        forged = exams[:2] + (replace(exams[2], producer_role="TRAINER"),) + exams[3:]
        with self.assertRaisesRegex(ValueError, "forged/self-graded"):
            self.orchestrator.finalize_generation(pending, forged)
        self.assertEqual(self.journal.continuation_descriptor()["next_generation_index"], 0)
        with self.assertRaisesRegex(CampaignBoundaryError, "pending or indeterminate"):
            self.orchestrator.run_generation(
                launches=launches, control_manifest_path=self.w0_path,
                evaluation_plan=self.plan)

    def test_no_fitness_winner_retains_exact_parent_and_no_false_generation(self):
        launches, _, _ = self._launches(1)
        pending = self.orchestrator.run_generation(
            launches=launches, control_manifest_path=self._control(self.w0_root, 1),
            evaluation_plan=self.plan)
        exams = self._external_fixture_exams(pending.population, 1)
        unchanged = tuple(replace(
            exam, metrics=Metrics(.50, .50, .99, .99, .02))
            if exam.subject_id not in ("parent", "control") else exam
            for exam in exams)
        result = self.orchestrator.finalize_generation(pending, unchanged)
        self.assertIsNone(result.decision.winner_id)
        self.assertEqual(result.decision.selected_manifest_path, self.w0_path)
        continuation = self.journal.continuation_descriptor(
            expected_head_sha256=result.journal_head_sha256)
        self.assertEqual(continuation["next_generation_index"], 0)
        self.assertEqual(continuation["selected_parent_state_sha256"],
                         self.w0_manifest["state_sha256"])
        self.assertEqual({c["fate"] for c in continuation["candidates"]}, {"REJECTED"})


if __name__ == "__main__":
    unittest.main()
