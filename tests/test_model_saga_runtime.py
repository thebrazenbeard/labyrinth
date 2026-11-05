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
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt
from labyrinth.model_saga_runtime import (
    CheckpointPaths, ExamPressure, ExaminerRequest,
    OperatorArithmeticDevelopmentExaminer, SagaSession, SceneNode,
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

    def make_session(self, root: Path, *, budget: int = 64, director=None, examiner=None):
        learner = FileLearner(root)
        director = director or GeneratedDirector()
        examiner = examiner or OperatorExaminer()
        scope = TrainingScopeSpec(
            objective_id="model-saga", domain="arithmetic", skills=("arithmetic",),
            specificity=2, scale=1, max_cycles=2, seed=19,
        )
        session = SagaSession(
            scope, learner=learner, checkpoint=learner.initial,
            baseline_train_cases=(_case(0), _case(1)),
            scene_director=director, minotaur_director=PressureModel(),
            examiner=examiner, action_budget=budget,
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
                nodes = list(GeneratedDirector().propose_world(context).nodes)
                nodes[2] = replace(nodes[2], exits=())
                return WorldProposal(tuple(nodes))

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

    def test_export_rejects_later_external_checkpoint_mutation(self):
        with TemporaryDirectory() as tmp:
            session, learner, _, _ = self.make_session(Path(tmp))
            session.step("left")
            session.step("forward")
            session.step("train")
            Path(learner.initial.weights).write_bytes(b"externally altered old checkpoint")
            with self.assertRaisesRegex(ValueError, "retained checkpoint artifact changed"):
                session.save_receipt_new(Path(tmp) / "untrusted-report.json")

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


if __name__ == "__main__":
    unittest.main()
