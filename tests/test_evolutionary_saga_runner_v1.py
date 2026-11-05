"""Mechanism-only two-generation saga: no neural learning is asserted."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.evolutionary_saga_runner_v1 import (
    CandidateSagaPlan, fresh_process_checkpoint_control,
    run_mechanism_campaign, run_saga_trial,
)
from labyrinth.model_saga_runtime import (
    CheckpointPaths, ExamCoveragePolicy, SagaSession,
)
from labyrinth.training_scope import TrainingScopeSpec
from tests.test_model_saga_runtime import (
    ConsequentialDirector, FileLearner, OperatorExaminer, PressureModel, _case,
)


class ScoreScriptedExaminer(OperatorExaminer):
    """Test-only receipt fabrication; never an authentic model examiner."""

    def __init__(self, should_pass):
        super().__init__()
        self.should_pass = should_pass

    def assess(self, request, predict):
        receipt = super().assess(request, predict)
        if self.should_pass:
            return receipt
        return replace(receipt, cases=tuple(replace(case, score=0.0)
                                            for case in receipt.cases))


def build_mock_session(root: Path, parent: CheckpointPaths, *, should_pass: bool):
    root.mkdir(parents=True)
    learner = FileLearner(root)
    Path(learner.initial.weights).write_bytes(Path(parent.weights).read_bytes())
    scope = TrainingScopeSpec(
        objective_id="evo-saga-mechanics", domain="arithmetic", skills=("arithmetic",),
        specificity=2, scale=1, max_cycles=2, seed=19,
    )
    return SagaSession(
        scope, learner=learner, checkpoint=learner.initial,
        baseline_train_cases=(_case(0), _case(1)),
        scene_director=ConsequentialDirector(), minotaur_director=PressureModel(),
        examiner=ScoreScriptedExaminer(should_pass),
        exam_coverage_policy=ExamCoveragePolicy(2),
    )


class EvolutionarySagaRunnerV1Tests(unittest.TestCase):
    def test_actual_session_navigation_and_source_train_instead_of_fake_evolution(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            w0 = root / "w0.bin"
            w0.write_bytes(b"mock frozen pre-entry bytes")
            session = build_mock_session(root / "entrant", CheckpointPaths(str(w0)),
                                         should_pass=True)
            result = run_saga_trial(session, choose_threshold_action=lambda state: "left")
            self.assertTrue(result.terminal)
            self.assertEqual(result.developmental_status, "PASS")
            self.assertEqual(result.generation_cycles, 2)
            self.assertEqual(result.train_spark_count, 4)
            self.assertEqual(result.traversed_branches, ("left",) * 4)
            self.assertTrue(all(tactic == "contrast" for tactic in result.training_tactics))
            self.assertEqual(result.classification, "SOURCE_MECHANISM_ONLY_UNQUALIFIED")
            self.assertNotIn("NEURAL_UPDATE_OBSERVED", repr(result))
            self.assertNotIn("GRADUATED", repr(result))
            self.assertNotEqual(result.entry_checkpoint_sha256, result.exit_checkpoint_sha256)
            self.assertFalse(any(spark.train_correct for spark in session.training_sparks))

    def test_two_generation_siblings_selected_successors_and_real_fresh_control(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            w0 = workspace / "original-w0.bin"
            w0.write_bytes(b"mock full pre-entry weights for this tiny mechanical test")
            parent_hashes_seen = {}

            def candidate(tag: str, generation: int, *, should_pass: bool, direction: str):
                def launch(parent):
                    parent_hashes_seen[tag] = Path(parent.weights).read_bytes()
                    return build_mock_session(
                        workspace / f"generation-{generation}" / tag, parent,
                        should_pass=should_pass,
                    )

                return CandidateSagaPlan(tag, launch, lambda state: direction)

            generations = (
                (candidate("g0_archive", 0, should_pass=True, direction="left"),
                 candidate("g0_forge", 0, should_pass=False, direction="right")),
                (candidate("g1_archive", 1, should_pass=False, direction="left"),
                 candidate("g1_forge", 1, should_pass=True, direction="right")),
            )

            def selector(gen, evidence):
                winners = [item.candidate_id for item in evidence
                           if item.trial.developmental_status == "PASS"]
                self.assertEqual(len(winners), 1)
                return winners[0]

            outcome = run_mechanism_campaign(
                CheckpointPaths(str(w0)), generations, select_winner=selector,
            )
            self.assertEqual(outcome.classification,
                             "MECHANISM_ONLY_NO_NEURAL_EVOLUTION_CLAIM")
            self.assertNotIn("NEURAL_UPDATE_OBSERVED", repr(outcome))
            self.assertNotIn("GRADUATED", repr(outcome))
            self.assertEqual(outcome.w0_control.classification,
                             "HASH_CONTROL_ONLY_NOT_MODEL_GENERALIZATION")
            self.assertNotEqual(outcome.w0_control.separate_pid, os.getpid())
            self.assertEqual(outcome.w0_control.weights_sha256, outcome.w0.weights_sha256)
            self.assertEqual(len(outcome.generations), 2)
            first, second = outcome.generations
            self.assertEqual(first.selected_candidate_id, "g0_archive")
            self.assertEqual(second.selected_candidate_id, "g1_forge")
            self.assertEqual(first.selection_classification,
                             "OPERATOR_SCRIPTED_MECHANISM_SELECTION_ONLY")
            self.assertEqual({item.candidate_id for item in first.candidates},
                             {"g0_archive", "g0_forge"})
            self.assertEqual({item.candidate_id for item in second.candidates},
                             {"g1_archive", "g1_forge"})
            self.assertEqual(next(item for item in first.candidates
                                  if item.candidate_id == "g0_forge").trial.developmental_status,
                             "FAIL")
            selected = next(item for item in first.candidates
                            if item.candidate_id == first.selected_candidate_id)
            self.assertEqual(second.parent_candidate_id, "g0_archive")
            self.assertEqual(second.parent_checkpoint_sha256,
                             selected.successor.weights_sha256)
            self.assertEqual(parent_hashes_seen["g0_archive"],
                             parent_hashes_seen["g0_forge"])
            self.assertEqual(parent_hashes_seen["g1_archive"],
                             parent_hashes_seen["g1_forge"])
            self.assertEqual(parent_hashes_seen["g1_archive"],
                             Path(selected.successor.weights_path).read_bytes())
            self.assertEqual(Path(w0).read_bytes(),
                             b"mock full pre-entry weights for this tiny mechanical test")

    def test_invalid_navigation_and_preexisting_trial_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            w0 = root / "w0.bin"
            w0.write_bytes(b"frozen")
            session = build_mock_session(root / "entrant", CheckpointPaths(str(w0)),
                                         should_pass=True)
            with self.assertRaisesRegex(ValueError, "authorized threshold branch"):
                run_saga_trial(session, choose_threshold_action=lambda state: "train")
            session.step("left")
            with self.assertRaisesRegex(ValueError, "fresh, unmodified"):
                run_saga_trial(session, choose_threshold_action=lambda state: "left")

    def test_campaign_refuses_parent_mismatch_or_failed_champion(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            w0 = workspace / "w0.bin"
            w0.write_bytes(b"immutable-parent")

            def corrupt_launch(parent):
                forged = workspace / "forged.bin"
                forged.write_bytes(b"not the parent's physical bytes")
                return build_mock_session(workspace / "forged", CheckpointPaths(str(forged)),
                                          should_pass=True)

            bad = CandidateSagaPlan("forgery", corrupt_launch, lambda state: "left")
            good = CandidateSagaPlan(
                "honest", lambda parent: build_mock_session(
                    workspace / "honest", parent, should_pass=False),
                lambda state: "right",
            )
            with self.assertRaisesRegex(ValueError, "unique physical clone"):
                run_mechanism_campaign(
                    CheckpointPaths(str(w0)), ((bad, good), (bad, good)),
                    select_winner=lambda gen, cases: "forgery",
                )
            second = CandidateSagaPlan(
                "other", lambda parent: build_mock_session(
                    workspace / "other", parent, should_pass=False),
                lambda state: "left",
            )
            with self.assertRaisesRegex(ValueError, "development-failed"):
                run_mechanism_campaign(
                    CheckpointPaths(str(w0)), ((good, second), (good, second)),
                    select_winner=lambda gen, cases: "honest",
                )

    def test_fresh_process_checkpoint_control_matches_physical_w0_only(self):
        with TemporaryDirectory() as tmp:
            w0 = Path(tmp) / "w0.bin"
            w0.write_bytes(b"never changed control")
            result = fresh_process_checkpoint_control(CheckpointPaths(str(w0)))
            self.assertNotEqual(result.separate_pid, os.getpid())
            self.assertEqual(result.classification,
                             "HASH_CONTROL_ONLY_NOT_MODEL_GENERALIZATION")


if __name__ == "__main__":
    unittest.main()
