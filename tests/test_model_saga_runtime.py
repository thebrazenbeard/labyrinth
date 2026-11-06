"""Original model-directed saga exercises real TRAIN and source-level exam gates."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import hashlib
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from labyrinth.maze import Challenge
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt, evaluate_minotaur
from labyrinth.model_saga_runtime import (
    CheckpointPaths, ExamPressure, ExamCoveragePolicy, ExaminerRequest,
    OperatorArithmeticDevelopmentExaminer, PedagogicalEncounter, SagaSession, SceneNode,
    WorldProposal, capture_checkpoint,
)
from labyrinth.saga_ariadne import (
    _is_local_absolute_checkpoint_path, _validate_summary, verify_saga_ariadne,
)
from labyrinth.runner import load_json
from labyrinth.training_scope import TrainingScopeSpec


def _case(index: int) -> Challenge:
    return Challenge(
        id=f"train-entrance-probe-{index}", kind="arithmetic", tier=1,
        split="train", prompt=f"Calculate {index + 2} + 3.\nRespond with exactly one line: ANSWER: <integer>.",
        answer=str(index + 5), metadata=(),
    )


class FileLearner:
    def __init__(self, root: Path):
        self.root = root
        self.index = 0
        initial = root / "weights-0.bin"
        initial.write_bytes(b"real baseline model bytes")
        self.initial = CheckpointPaths(str(initial))
        self.training_targets: list[str] = []

    def predict(self, prompt: str) -> str:
        return "ANSWER: 0"

    def train_supervised(self, case: Challenge) -> CheckpointPaths:
        if case.split != "train":
            raise AssertionError("private evaluation case leaked into TRAIN")
        self.training_targets.append(case.answer)
        self.index += 1
        path = self.root / f"weights-{self.index}.bin"
        with path.open("xb") as output:
            output.write(f"state-{self.index}-{case.answer}".encode())
        return CheckpointPaths(str(path))


class GeneratedDirector:
    def __init__(self):
        self.requests = []

    def propose_world(self, context):
        self.requests.append(context)
        # Model, not the game engine, decides scenes and topology; prior
        # weakness changes prose, and the cycle flips a real graph edge.
        return WorldProposal((
            SceneNode("threshold", f"Gate {context.cycle_index}; weak {context.weak_skills}",
                      (("left", "archive"), ("right", "forge"))),
            SceneNode("archive", f"Archive of {context.skill} {context.public_seed}",
                      (("forward", "chamber"),) +
                      (("back", "threshold"),) if context.cycle_index else (("forward", "chamber"),)),
            SceneNode("forge", f"Forge after {context.prior_training_correct}",
                      (("forward", "chamber"),) +
                      (("back", "threshold"),) if not context.cycle_index else (("forward", "chamber"),)),
            SceneNode("chamber", "A training chamber, not a private examination.", ()),
        ), encounters=(
            PedagogicalEncounter("archive", "contrast", 1),
            PedagogicalEncounter("forge", "transfer", -1),
        ))


class ConsequentialDirector(GeneratedDirector):
    """Model routes offer different, source-owned ways to learn the same skill."""

    def propose_world(self, context):
        base = super().propose_world(context)
        return replace(base, encounters=(
            PedagogicalEncounter("archive", "contrast", 1),
            PedagogicalEncounter("forge", "transfer", -1),
        ))


class PressureModel:
    def __init__(self):
        self.contexts = []

    def propose_pressure(self, context):
        self.contexts.append(context)
        return ExamPressure("arithmetic", min(5, context.cycle_index + 1), "mixed")


class OperatorExaminer:
    def __init__(self):
        self.requests = []

    def assess(self, request, predict):
        self.requests.append(request)
        assert callable(predict)
        # An operator-owned source controls exam cases, split and scores.
        score = float(request.cycle_index == 1)
        cases = tuple(
            ExamCaseResult(
                case_id=f"exam-dev-{request.cycle_index}-{split}-{i}",
                skill="arithmetic", split=split, score=score, baseline_score=0.0,
                prompt_sha256=hashlib.sha256(
                    f"sealed-evaluation-{request.fresh_seed}-{split}-{i}".encode()
                ).hexdigest(),
            )
            for split in ("heldout", "transfer") for i in range(4)
        )
        return MinotaurExamReceipt(
            exam_id=f"exam-{request.cycle_index}",
            objective_id=request.objective_id, cycle_index=request.cycle_index,
            checkpoint_sha256=request.checkpoint_sha256,
            baseline_sha256=request.baseline_sha256,
            evaluator_id="operator-owned-development", cases=cases,
        )


class SagaSessionTests(unittest.TestCase):
    def test_model_created_world_must_causally_shape_training_by_default(self):
        class CosmeticDirector(GeneratedDirector):
            def propose_world(self, context):
                return replace(super().propose_world(context), encounters=())

        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "requires consequential pedagogical"):
                self.make_session(Path(tmp), director=CosmeticDirector())
        # A caller deliberately invoking the obsolete decorative mode can
        # still reproduce historical sessions; it is not the living default.
        with TemporaryDirectory() as tmp:
            learner = FileLearner(Path(tmp))
            scope = TrainingScopeSpec("old-story", "arithmetic", ("arithmetic",), 2, 1, 2, 19)
            session = SagaSession(
                scope, learner=learner, checkpoint=learner.initial,
                baseline_train_cases=(_case(0), _case(1)),
                scene_director=CosmeticDirector(), minotaur_director=PressureModel(),
                examiner=OperatorExaminer(), exam_coverage_policy=ExamCoveragePolicy(2),
                require_pedagogical_encounters=False,
            )
            self.assertFalse(session.world.encounters)

    def test_pedagogical_path_feedback_reaches_daedalus_and_minotaur_without_gold(self):
        class ResponsivePressure(PressureModel):
            def propose_pressure(self, context):
                self.contexts.append(context)
                tactic = context.prior_training_outcomes[0].tactic
                return ExamPressure("arithmetic", 4 if tactic == "contrast" else 2, "mixed")

        results = {}
        for branch in ("left", "right"):
            with TemporaryDirectory() as tmp:
                director = ConsequentialDirector()
                session, _, _, examiner = self.make_session(Path(tmp), director=director)
                session.minotaur_director = ResponsivePressure()
                session.step(branch)
                session.step("forward")
                first_tactic = session.runner.training_registry[-1].tactic
                session.step("train")
                self.assertEqual(director.requests[-1].prior_training_tactics, (first_tactic,))
                session.step("right")
                session.step("forward")
                session.step("train")
                self.assertEqual(session.phase, "MINOTAUR")
                session.step("face")
                pressure_context = session.minotaur_director.contexts[-1]
                self.assertEqual(pressure_context.prior_training_outcomes[0].tactic,
                                 first_tactic)
                self.assertTrue(all(type(item.correct) is bool
                                    for item in pressure_context.prior_training_outcomes))
                self.assertNotIn("ANSWER:", repr(pressure_context))
                self.assertNotIn("case_id", repr(pressure_context))
                results[branch] = examiner.requests[-1].pressure.difficulty
        self.assertEqual(results, {"left": 4, "right": 2})

    def test_failed_pedagogical_task_materialization_cannot_be_retried(self):
        with TemporaryDirectory() as tmp:
            session, _, _, _ = self.make_session(Path(tmp), director=ConsequentialDirector())
            session.step("left")

            def partial_failure(*, route):
                raise RuntimeError("source TRAIN admission failed ambiguously")

            session.runner.next_training_prompt = partial_failure
            with self.assertRaisesRegex(RuntimeError, "admission failed"):
                session.step("forward")
            self.assertTrue(session.indeterminate)
            self.assertEqual(session.observe().available_actions, ())
            with self.assertRaisesRegex(ValueError, "indeterminate training"):
                session.step("forward")
            with self.assertRaisesRegex(ValueError, "indeterminate training"):
                session.save_receipt_new(Path(tmp) / "false-progress.json")

    def test_world_branch_choice_changes_actual_approved_train_challenge(self):
        with TemporaryDirectory() as left_tmp, TemporaryDirectory() as right_tmp:
            left, _, _, _ = self.make_session(Path(left_tmp), director=ConsequentialDirector())
            right, _, _, _ = self.make_session(Path(right_tmp), director=ConsequentialDirector())
            # The source-owned challenge must not exist before the learner
            # actually chooses a route in the model-created world.
            self.assertIsNone(left.view)
            self.assertIsNone(right.view)
            self.assertEqual(left.runner.training_registry, ())
            self.assertEqual(right.runner.training_registry, ())
            left.step("left")
            left.step("forward")
            right.step("right")
            right.step("forward")
            self.assertIn("Expression A", left.view.prompt)
            self.assertIn("workshop", right.view.prompt.lower())
            self.assertNotEqual(left.view.prompt_sha256, right.view.prompt_sha256)
            self.assertEqual(left.runner.training_registry[-1].tactic, "contrast")
            self.assertEqual(right.runner.training_registry[-1].tactic, "transfer")
            self.assertNotEqual(left.runner.training_registry[-1].difficulty,
                                right.runner.training_registry[-1].difficulty)
            self.assertEqual(left.runner.operator_pending_train_case(left.view).split, "train")
            self.assertEqual(right.runner.operator_pending_train_case(right.view).split, "train")
            left_result = left.step("train")
            right_result = right.step("train")
            self.assertEqual(left_result.event, "TRAIN_COMPLETED")
            self.assertEqual(right_result.event, "TRAIN_COMPLETED")
            for session, expected_branch, expected_tactic in (
                (left, "archive", "contrast"),
                (right, "forge", "transfer"),
            ):
                train_receipt = next(event["summary"] for event in session.ariadne_events
                                     if event["kind"] == "train")
                self.assertEqual(train_receipt["branch_node_id"], expected_branch)
                self.assertEqual(train_receipt["train_tactic"], expected_tactic)
                self.assertEqual(train_receipt["train_prompt_sha256"],
                                 session.runner.observations[0].prompt_sha256)
                self.assertEqual(train_receipt["train_difficulty"],
                                 session.runner.observations[0].difficulty)
            for session in (left, right):
                self.assertNotEqual(session.training_sparks[0].after.weights_sha256,
                                    session.entrance.checkpoint.weights_sha256)
            # No behavior-based improvement is claimed: the test learner
            # gives the same wrong answer in both counterfactuals.
            self.assertFalse(left.training_sparks[0].train_correct)
            self.assertFalse(right.training_sparks[0].train_correct)
            self.assertNotEqual(left.ariadne_events[-2]["summary"]["policy_after_sha256"],
                                right.ariadne_events[-2]["summary"]["policy_after_sha256"])

    def test_representation_transfer_is_real_modality_not_claimed_new_arithmetic(self):
        class RepresentationDirector(GeneratedDirector):
            def propose_world(self, context):
                return replace(super().propose_world(context), encounters=(
                    PedagogicalEncounter("archive", "practice", 0),
                    PedagogicalEncounter("forge", "transfer", 0),
                ))

        with TemporaryDirectory() as one, TemporaryDirectory() as two:
            left, _, _, _ = self.make_session(Path(one), director=RepresentationDirector())
            right, _, _, _ = self.make_session(Path(two), director=RepresentationDirector())
            for session, choice in ((left, "left"), (right, "right")):
                session.step(choice)
                session.step("forward")
            self.assertNotEqual(left.view.prompt_sha256, right.view.prompt_sha256)
            self.assertEqual(left.runner.training_registry[-1].tactic, "practice")
            self.assertEqual(right.runner.training_registry[-1].tactic, "transfer")
            # Presentation/modality differs, but the operator must not infer
            # a novel arithmetic computation or neural learning from that.
            self.assertIn("Calculate", left.view.prompt)
            self.assertIn("workshop", right.view.prompt.lower())

    def test_lived_train_outcomes_survive_failed_minotaur_into_regenerated_world(self):
        with TemporaryDirectory() as tmp:
            session, _, director, _ = self.make_session(Path(tmp), director=ConsequentialDirector())
            for choice in ("left", "right"):
                session.step(choice)
                session.step("forward")
                session.step("train")
            self.assertEqual(session.phase, "MINOTAUR")
            session.step("face")  # Synthetic examiner fails cycle 0.
            cycle_one_context = next(req for req in director.requests if req.cycle_index == 1)
            self.assertEqual(cycle_one_context.prior_training_tactics,
                             ("contrast", "transfer"))
            self.assertEqual(cycle_one_context.prior_training_correct, (False, False))
            self.assertEqual(len(cycle_one_context.prior_training_outcomes), 2)
            self.assertEqual([item.difficulty for item in cycle_one_context.prior_training_outcomes],
                             [3, 1])
            self.assertNotIn("ANSWER:", repr(cycle_one_context))
            self.assertNotIn("case_id", repr(cycle_one_context))

    def test_minotaur_cannot_lower_operator_owned_exam_difficulty_floor(self):
        with TemporaryDirectory() as tmp:
            session, _, _, examiner = self.make_session(
                Path(tmp), director=ConsequentialDirector(),
                specificity=1, exam_floor=5,
            )
            session.minotaur_director = PressureModel()  # Proposes difficulty 1.
            while session.phase != "MINOTAUR":
                session.step("left")
                session.step("forward")
                session.step("train")
            context = session.minotaur_director.contexts
            self.assertEqual(context, [])
            session.step("face")
            self.assertEqual(context[-1].minimum_difficulty, 5)
            self.assertEqual(examiner.requests[-1].minimum_difficulty, 5)
            self.assertEqual(examiner.requests[-1].pressure.difficulty, 5)
            verdict = next(event for event in session.ariadne_events if event["kind"] == "verdict")
            self.assertEqual(verdict["summary"]["pressure_difficulty"], 5)

    def test_world_rejects_unsafe_or_nonconsequential_pedagogical_proposals(self):
        class InvalidEncounterDirector(GeneratedDirector):
            def __init__(self, variant):
                super().__init__()
                self.variant = variant

            def propose_world(self, context):
                world = super().propose_world(context)
                good = PedagogicalEncounter("archive", "contrast", 1)
                second = PedagogicalEncounter("forge", "transfer", -1)
                variants = {
                    "unapproved_tactic": (good, replace(second, tactic="graduate")),
                    "gold_injection": (good, replace(second, tactic="ANSWER: 42")),
                    "unbounded_difficulty": (good, replace(second, difficulty_delta=9)),
                    "missing_branch": (good,),
                    "duplicate_branch": (good, replace(second, branch_node_id="archive")),
                    "cosmetic_only": (good, replace(second, tactic="contrast", difficulty_delta=1)),
                    "clamped_same_tactic": (
                        PedagogicalEncounter("archive", "practice", 0),
                        PedagogicalEncounter("forge", "practice", 1),
                    ),
                    "nonbranch": (good, replace(second, branch_node_id="chamber")),
                }
                return replace(world, encounters=variants[self.variant])

        for variant in ("unapproved_tactic", "gold_injection", "unbounded_difficulty",
                        "missing_branch", "duplicate_branch", "cosmetic_only",
                        "clamped_same_tactic", "nonbranch"):
            with self.subTest(variant=variant), TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    self.make_session(Path(tmp), director=InvalidEncounterDirector(variant))

    def test_ariadne_path_accepts_lappy_drive_but_not_relative_or_unc(self):
        self.assertTrue(_is_local_absolute_checkpoint_path(r"D:\VERA\models\w0.bin"))
        self.assertTrue(_is_local_absolute_checkpoint_path("/tmp/weights/w0.bin"))
        self.assertFalse(_is_local_absolute_checkpoint_path(r"D:VERA\models\w0.bin"))
        self.assertFalse(_is_local_absolute_checkpoint_path(r"\\TheSimsVault\models\w0.bin"))
        self.assertFalse(_is_local_absolute_checkpoint_path("relative/weights.bin"))
        valid_summary = {
            "checkpoint_path": r"D:\VERA\models\w0.bin",
            "optimizer_status": "WEIGHTS_ONLY_CONTINUITY",
            "optimizer_sha256": None,
            "baseline_count": 2,
            "baseline_correct": 0,
        }
        _validate_summary("entrance", valid_summary)
        for invalid in (r"\\TheSimsVault\models\w0.bin", "../w0.bin"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _validate_summary("entrance", {**valid_summary, "checkpoint_path": invalid})

    def make_session(self, root: Path, *, budget: int = 64, director=None, examiner=None,
                     specificity: int = 2, exam_floor: int = 2):
        learner = FileLearner(root)
        director = director or GeneratedDirector()
        examiner = examiner or OperatorExaminer()
        scope = TrainingScopeSpec(
            objective_id="model-saga", domain="arithmetic", skills=("arithmetic",),
            specificity=specificity, scale=1, max_cycles=2, seed=19,
        )
        session = SagaSession(
            scope, learner=learner, checkpoint=learner.initial,
            baseline_train_cases=(_case(0), _case(1)),
            scene_director=director, minotaur_director=PressureModel(),
            examiner=examiner, exam_coverage_policy=ExamCoveragePolicy(exam_floor),
            action_budget=budget,
        )
        return session, learner, director, examiner

    def test_real_navigation_training_checkpoints_regeneration_and_minotaur(self):
        with TemporaryDirectory() as tmp:
            session, learner, director, examiner = self.make_session(Path(tmp))
            baseline_digest = session.entrance.checkpoint.weights_sha256
            self.assertEqual(session.entrance.checkpoint.continuity, "WEIGHTS_ONLY_CONTINUITY")
            self.assertEqual(session.entrance.baseline.case_count, 2)
            self.assertEqual(session.entrance.baseline.correct_count, 0)
            self.assertEqual(session.observe().available_actions, ("look", "left", "right"))
            events = []
            for _ in range(16):
                if session.phase == "MINOTAUR":
                    result = session.step("face")
                else:
                    session.step("left")
                    session.step("forward")
                    result = session.step("train")
                events.append(result.event)
                if session.terminated:
                    break
            self.assertTrue(session.terminated)
            self.assertFalse(session.truncated)
            self.assertEqual(events.count("MINOTAUR_FAIL"), 1)
            self.assertEqual(events.count("MINOTAUR_PASS"), 1)
            self.assertEqual(len(learner.training_targets), 4)
            self.assertNotEqual(session.checkpoint.weights_sha256, baseline_digest)
            self.assertEqual(len(session.training_sparks), len(learner.training_targets))
            self.assertEqual(session.training_sparks[0].before_sha256, baseline_digest)
            self.assertEqual(session.training_sparks[-1].after.weights_sha256,
                             session.checkpoint.weights_sha256)
            self.assertTrue(all(spark.loss is None for spark in session.training_sparks))
            self.assertTrue(Path(session.checkpoint.weights_path).is_file())
            self.assertEqual(session.journey.phase, "development_passed_pending_private_qualification")
            self.assertIn("graduation remains unestablished", result.observation)
            self.assertEqual(len(examiner.requests), 2)
            self.assertNotEqual(examiner.requests[0].fresh_seed, examiner.requests[1].fresh_seed)
            self.assertEqual(examiner.requests[0].baseline_sha256, baseline_digest)
            self.assertEqual(examiner.requests[1].baseline_sha256, baseline_digest)
            self.assertGreater(len(director.requests), 2)
            cycle_one = [r for r in director.requests if r.cycle_index == 1]
            self.assertTrue(cycle_one)
            self.assertEqual(cycle_one[0].weak_skills, ("arithmetic",))
            self.assertNotIn("exam-dev-", repr([e for e in events]))
            self.assertNotIn("train-taskmaster-", result.observation)

    def test_deterministic_world_proposals_and_invalid_actions(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            a, _, da, _ = self.make_session(Path(first))
            b, _, db, _ = self.make_session(Path(second))
            self.assertEqual(a.observe().observation, b.observe().observation)
            self.assertEqual(da.requests[0], db.requests[0])
            with self.assertRaisesRegex(ValueError, "invalid action"):
                a.step("delete-everything")
            self.assertEqual(a.location, "threshold")
            self.assertEqual(a.action_count, 0)
            self.assertEqual(a.step("right").event, "MOVE")
            self.assertEqual(a.step("back").event, "MOVE")

    def test_action_budget_truncates_without_training(self):
        with TemporaryDirectory() as tmp:
            session, learner, _, _ = self.make_session(Path(tmp), budget=6)
            for _ in range(6):
                self.assertFalse(session.step("look").truncated)
            final = session.step("look")
            self.assertTrue(final.truncated)
            self.assertEqual(final.event, "ACTION_BUDGET_EXHAUSTED")
            self.assertEqual(learner.index, 0)
            self.assertEqual(session.ariadne_events[-1]["kind"], "exit")
            self.assertEqual(session.ariadne_events[-1]["summary"]["phase"], "ACTION_BUDGET_EXHAUSTED")
            verify_saga_ariadne(session.ariadne_events, require_exit=True)
            with self.assertRaisesRegex(ValueError, "episode has ended"):
                session.step("look")

    def test_rejects_model_leaking_answer_or_unreachable_world(self):
        class BadDirector:
            def __init__(self, answer):
                self.answer = answer

            def propose_world(self, context):
                world = GeneratedDirector().propose_world(context)
                nodes = list(world.nodes)
                if self.answer:
                    nodes[-1] = replace(nodes[-1], prose="ANSWER: 45")
                else:
                    nodes[0] = replace(nodes[0], exits=(("left", "archive"), ("right", "archive")))
                    nodes[2] = replace(nodes[2], exits=())
                return WorldProposal(tuple(nodes))

        for answer in (True, False):
            with self.subTest(answer=answer), TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    self.make_session(Path(tmp), director=BadDirector(answer))

    def test_malformed_untrusted_scene_fields_refuse_without_type_error(self):
        class MalformedDirector:
            def __init__(self, break_id):
                self.break_id = break_id

            def propose_world(self, context):
                nodes = list(GeneratedDirector().propose_world(context).nodes)
                if self.break_id:
                    nodes[0] = replace(nodes[0], node_id=["not-a-hashable-id"])
                else:
                    nodes[0] = replace(nodes[0], exits=(([], "archive"), ("right", "forge")))
                return WorldProposal(tuple(nodes))

        for malformed in (True, False):
            with self.subTest(malformed=malformed), TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    self.make_session(Path(tmp), director=MalformedDirector(malformed))

    def test_rejects_branch_that_is_reachable_but_cannot_reach_training(self):
        class DeadEndDirector:
            def propose_world(self, context):
                original = GeneratedDirector().propose_world(context)
                nodes = list(original.nodes)
                nodes[2] = replace(nodes[2], exits=())
                return replace(original, nodes=tuple(nodes))

        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "every narrative branch must have a route"):
                self.make_session(Path(tmp), director=DeadEndDirector())

    def test_ariadne_binds_entrance_sparks_verdict_and_write_once_export(self):
        with TemporaryDirectory() as tmp:
            session, learner, _, _ = self.make_session(Path(tmp))
            while not session.terminated:
                if session.phase == "MINOTAUR":
                    session.step("face")
                else:
                    session.step("right")
                    session.step("forward")
                    session.step("train")
            head = verify_saga_ariadne(session.ariadne_events, require_exit=True)
            kinds = [event["kind"] for event in session.ariadne_events]
            self.assertEqual(kinds[0], "entrance")
            self.assertEqual(kinds[-1], "exit")
            self.assertEqual(kinds.count("train"), len(learner.training_targets))
            train_summaries = [event["summary"] for event in session.ariadne_events
                               if event["kind"] == "train"]
            self.assertEqual({item["branch_node_id"] for item in train_summaries}, {"forge"})
            self.assertEqual({item["train_tactic"] for item in train_summaries}, {"transfer"})
            self.assertTrue(all(len(item["train_prompt_sha256"]) == 64
                                for item in train_summaries))
            self.assertEqual(kinds.count("verdict"), 2)
            self.assertEqual(kinds.count("regeneration"), 1)
            self.assertEqual(session.ariadne_events[0]["summary"]["baseline_correct"], 0)
            self.assertEqual(session.ariadne_events[0]["checkpoint_before_sha256"],
                             session.entrance.checkpoint.weights_sha256)
            self.assertEqual(session.ariadne_events[-1]["checkpoint_after_sha256"],
                             session.checkpoint.weights_sha256)
            for event in session.ariadne_events:
                self.assertNotIn("ANSWER:", repr(event))
                self.assertNotIn("case_id", repr(event))
            altered = deepcopy(session.ariadne_events)
            altered[2]["summary"]["train_correct"] = not altered[2]["summary"]["train_correct"]
            with self.assertRaisesRegex(ValueError, "chain/content mismatch"):
                verify_saga_ariadne(altered)
            altered = deepcopy(session.ariadne_events)
            actual_train = next(item for item in altered if item["kind"] == "train")
            actual_train["summary"]["branch_node_id"] = "archive"
            with self.assertRaisesRegex(ValueError, "chain/content mismatch"):
                verify_saga_ariadne(altered)
            altered = deepcopy(session.ariadne_events)
            actual_train = next(item for item in altered if item["kind"] == "train")
            del actual_train["summary"]["train_prompt_sha256"]
            with self.assertRaisesRegex(ValueError, "chain/content mismatch"):
                verify_saga_ariadne(altered)
            with self.assertRaisesRegex(ValueError, "invalid or incomplete"):
                _validate_summary("train", actual_train["summary"])
            altered = deepcopy(session.ariadne_events)
            altered[-1]["parent_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "chain/content mismatch"):
                verify_saga_ariadne(altered)

            report_path = Path(tmp) / "saga-report.json"
            self.assertEqual(session.save_receipt_new(report_path), head)
            saved = load_json(report_path, strict=True)
            self.assertEqual(saved["ariadne_head_sha256"], head)
            self.assertEqual(len(saved["checkpoint_manifest"]), len(session.checkpoint_refs))
            with self.assertRaises(FileExistsError):
                session.save_receipt_new(report_path)

    def test_old_v1_ariadne_train_summary_still_validates_without_route_fields(self):
        historical_summary = {
            "skill": "arithmetic", "room_index": 0, "train_correct": False,
            "policy_before_sha256": "0" * 64,
            "policy_after_sha256": "1" * 64,
            "checkpoint_path": "/tmp/weights-1.bin",
        }
        _validate_summary("train", historical_summary)
        with self.assertRaisesRegex(ValueError, "invalid or incomplete"):
            _validate_summary("train", {**historical_summary, "branch_node_id": "forge"})

    def test_export_rejects_later_external_checkpoint_mutation(self):
        with TemporaryDirectory() as tmp:
            session, learner, _, _ = self.make_session(Path(tmp))
            session.step("left")
            session.step("forward")
            session.step("train")
            Path(learner.initial.weights).write_bytes(b"externally altered old checkpoint")
            with self.assertRaisesRegex(ValueError, "retained checkpoint artifact changed"):
                session.save_receipt_new(Path(tmp) / "untrusted-report.json")

    def test_partial_training_hook_failure_quarantines_session_without_retry(self):
        with TemporaryDirectory() as tmp:
            session, learner, _, _ = self.make_session(Path(tmp))
            session.step("left")
            session.step("forward")
            original = learner.train_supervised
            physical_updates = []

            def crash_after_effect(case):
                written = original(case)
                physical_updates.append(written)
                raise RuntimeError("simulated process crash after physical checkpoint write")

            learner.train_supervised = crash_after_effect
            with self.assertRaisesRegex(RuntimeError, "after physical checkpoint write"):
                session.step("train")
            self.assertEqual(len(physical_updates), 1)
            self.assertEqual(len(session.training_sparks), 0)
            self.assertTrue(session.indeterminate)
            self.assertEqual(session.observe().available_actions, ())
            self.assertEqual(session.observe().development_status, "INDETERMINATE")
            learner.train_supervised = original
            with self.assertRaisesRegex(ValueError, "indeterminate training state"):
                session.step("train")
            self.assertEqual(len(physical_updates), 1)
            with self.assertRaisesRegex(ValueError, "indeterminate training"):
                session.save_receipt_new(Path(tmp) / "false-success.json")

    def test_in_progress_export_is_explicitly_not_terminated(self):
        with TemporaryDirectory() as tmp:
            session, _, _, _ = self.make_session(Path(tmp))
            report = Path(tmp) / "development-in-progress.json"
            session.save_receipt_new(report)
            value = load_json(report, strict=True)
            self.assertEqual(value["session_status"], "IN_PROGRESS")
            self.assertEqual(value["qualification"], "NOT_ESTABLISHED")

    def test_requires_real_checkpoint_and_rejects_claimed_update_without_new_bytes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.bin"
            with self.assertRaises(OSError):
                capture_checkpoint(CheckpointPaths(str(path)))
            session, learner, _, _ = self.make_session(Path(tmp))
            def unchanged(case):
                return learner.initial
            learner.train_supervised = unchanged
            session.step("left")
            session.step("forward")
            with self.assertRaisesRegex(ValueError, "new write-once checkpoint"):
                session.step("train")

    def test_rejects_trainer_that_mutates_entrance_artifact_in_place(self):
        with TemporaryDirectory() as tmp:
            session, learner, _, _ = self.make_session(Path(tmp))

            def bad_train(case):
                Path(learner.initial.weights).write_bytes(b"illicit in-place overwrite")
                successor = Path(tmp) / "weights-1.bin"
                successor.write_bytes(b"some different bytes")
                return CheckpointPaths(str(successor))

            learner.train_supervised = bad_train
            session.step("left")
            session.step("forward")
            with self.assertRaisesRegex(ValueError, "overwrote a retained checkpoint"):
                session.step("train")

    def test_later_training_cannot_overwrite_w0_while_preserving_immediate_predecessor(self):
        with TemporaryDirectory() as tmp:
            session, learner, _, _ = self.make_session(Path(tmp))
            session.step("left")
            session.step("forward")
            session.step("train")
            self.assertEqual(learner.index, 1)
            original_train = learner.train_supervised

            def overwrite_baseline(case):
                Path(learner.initial.weights).write_bytes(b"malicious rewrite of old W0")
                return original_train(case)

            learner.train_supervised = overwrite_baseline
            session.step("left")
            session.step("forward")
            with self.assertRaisesRegex(ValueError, "overwrote a retained checkpoint"):
                session.step("train")

    def test_operator_math_exam_uses_model_pressure_but_owns_gold_and_scores(self):
        def exact_arithmetic(prompt: str) -> str:
            symbolic = re.match(r"Compute (\d+) ([+-]) (\d+)\.", prompt)
            if symbolic:
                first, op, second = symbolic.groups()
                answer = int(first) + int(second) if op == "+" else int(first) - int(second)
            else:
                story = re.match(r"A vessel carries (\d+) units, then (receives|spends) (\d+) units", prompt)
                self.assertIsNotNone(story)
                first, verb, second = story.groups()
                answer = int(first) + int(second) if verb == "receives" else int(first) - int(second)
            return f"ANSWER: {answer}"

        examiner = OperatorArithmeticDevelopmentExaminer(lambda prompt: "ANSWER: invalid")
        base = ExaminerRequest(
            "math-saga", "a" * 64, 0, "b" * 64, "c" * 64,
            ExamPressure("arithmetic", 1, "mixed"), 12345,
        )
        low = examiner.assess(base, exact_arithmetic)
        high = examiner.assess(replace(base, pressure=ExamPressure("arithmetic", 5, "mixed")),
                               exact_arithmetic)
        self.assertEqual(len(low.cases), 8)
        self.assertTrue(all(case.score == 1 and case.baseline_score == 0 for case in low.cases))
        self.assertNotEqual(
            {case.prompt_sha256 for case in low.cases},
            {case.prompt_sha256 for case in high.cases},
        )
        self.assertEqual({case.split for case in low.cases}, {"heldout", "transfer"})
        self.assertTrue(all(case.prompt_sha256 and not hasattr(case, "answer") for case in low.cases))
        with self.assertRaisesRegex(ValueError, "unsupported arithmetic development pressure"):
            examiner.assess(replace(base, minimum_difficulty=5), exact_arithmetic)

    def test_easy_pressure_would_false_pass_weak_model_but_coverage_policy_prevents_it(self):
        def weak_single_digit_predict(prompt: str) -> str:
            match = re.match(r"Compute (\d+) ([+-]) (\d+)\.", prompt)
            if match:
                a, operation, b = int(match[1]), match[2], int(match[3])
            else:
                match = re.match(
                    r"A vessel carries (\d+) units, then (receives|spends) (\d+) units",
                    prompt,
                )
                if not match:
                    return "ANSWER: invalid"
                a, operation, b = int(match[1]), match[2], int(match[3])
            if a > 9 or b > 9:
                return "ANSWER: invalid"
            result = a + b if operation in ("+", "receives") else a - b
            return f"ANSWER: {result}"

        scope = TrainingScopeSpec("weak-exam", "arithmetic", ("arithmetic",), 1, 1, 2, 61)
        examiner = OperatorArithmeticDevelopmentExaminer(lambda prompt: "ANSWER: invalid")
        request = ExaminerRequest(
            scope.objective_id, scope.fingerprint, 0, "b" * 64, "a" * 64,
            ExamPressure("arithmetic", 1, "mixed"), 781,
        )
        weak_exposure = examiner.assess(request, weak_single_digit_predict)
        self.assertEqual(evaluate_minotaur(scope, weak_exposure).status, "PASS")
        # This is why the Minotaur may not unilaterally set exam difficulty.
        # A separately declared operator coverage floor of 5 prevents the
        # same weak predictor from obtaining this misleading easy-exam PASS.
        with self.assertRaisesRegex(ValueError, "unsupported arithmetic development pressure"):
            examiner.assess(replace(request, minimum_difficulty=5), weak_single_digit_predict)
        mandatory = examiner.assess(replace(
            request, pressure=ExamPressure("arithmetic", 5, "mixed"),
            minimum_difficulty=5,
        ), weak_single_digit_predict)
        self.assertEqual(evaluate_minotaur(scope, mandatory).status, "FAIL")


if __name__ == "__main__":
    unittest.main()
