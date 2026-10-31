"""Real toy-weight and Journey composition across an interrupted maze cycle.

The checkpoint round trip models a process boundary, not pretrained-model
learning or independent examiner custody.  It deliberately exercises the
public Journey state transitions outside the convenience journey runner.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from labyrinth.evaluation import score_response
from labyrinth.journey import (
    AWAITING_EXAM, EXHAUSTED, TRAINING, TrainingCompletionReceipt,
    begin_journey, complete_maze, face_minotaur,
)
from labyrinth.journey_challenges import prepare_training_rooms
from labyrinth.journey_toy import _exam_cases
from labyrinth.minotaur_exam import ExamCaseResult, MinotaurExamReceipt, evaluate_minotaur
from labyrinth.toy_backend import ToyLinearAdapter
from labyrinth.training_scope import TrainingScopeSpec


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class RegenerativeCompositionTests(unittest.TestCase):
    def test_real_toy_parameters_survive_disk_reload_and_regenerated_maze(self):
        scope = TrainingScopeSpec(
            objective_id="composition-arithmetic", domain="arithmetic",
            skills=("arithmetic",), specificity=2, scale=4,
            max_cycles=2, seed=7,
        )
        learner = ToyLinearAdapter(seed=0)
        baseline = ToyLinearAdapter(seed=0)
        original_hash = learner.model_hash
        state = begin_journey(scope, checkpoint_sha256=original_hash)
        earlier_train: set[str] = set()
        earlier_exams: set[str] = set()

        with tempfile.TemporaryDirectory(prefix="labyrinth-composition-") as temp:
            checkpoint_dir = Path(temp)
            for cycle in range(2):
                self.assertEqual(state.phase, TRAINING)
                self.assertEqual(state.cycle_index, cycle)
                self.assertEqual(learner.model_hash, state.checkpoint_sha256)
                prepared = prepare_training_rooms(state.maze)
                training_cases = tuple(entry.challenge for entry in prepared)
                self.assertFalse({case.id for case in training_cases} & earlier_train)
                self.assertFalse({case.id for case in training_cases} & earlier_exams)

                if cycle == 1:
                    # A development case cannot be recycled as a TRAIN visit,
                    # even if an untrusted host invents a changed digest.
                    recycled = next(iter(earlier_exams))
                    room_ids = tuple(room.room_id for room in state.maze.rooms)
                    with self.assertRaisesRegex(ValueError, "training case was reused or came from an exam"):
                        complete_maze(state, TrainingCompletionReceipt(
                            objective_id=scope.objective_id, cycle_index=cycle,
                            checkpoint_before_sha256=state.checkpoint_sha256,
                            checkpoint_after_sha256="f" * 64,
                            room_case_ids=tuple(
                                (rid, recycled if index == 0 else f"novel-{index}",
                                 f"{index + 80:064x}")
                                for index, rid in enumerate(room_ids)
                            ),
                        ))
                    self.assertEqual(learner.model_hash, state.checkpoint_sha256)

                before = learner.model_hash
                trained = learner.train(training_cases, epochs=1, learning_rate=0.5)
                self.assertGreater(trained["parameter_updates"], 0)
                self.assertNotEqual(learner.model_hash, before)
                checkpoint_path = checkpoint_dir / f"cycle-{cycle}.json"
                self.assertEqual(learner.save_checkpoint(checkpoint_path), learner.model_hash)

                # Rebuild a fresh learner from persisted bytes, rather than
                # retaining the in-memory object between maze cycles.
                learner = ToyLinearAdapter.load_checkpoint(checkpoint_path)
                self.assertEqual(learner.model_hash, trained["after_model_hash"])
                state = complete_maze(state, TrainingCompletionReceipt(
                    objective_id=scope.objective_id, cycle_index=cycle,
                    checkpoint_before_sha256=before,
                    checkpoint_after_sha256=learner.model_hash,
                    room_case_ids=tuple(
                        (entry.room_id, entry.challenge.id,
                         _prompt_sha(entry.challenge.prompt)) for entry in prepared
                    ),
                ))
                self.assertEqual(state.phase, AWAITING_EXAM)

                cases = _exam_cases(scope, cycle, 8)
                observations = tuple(
                    ExamCaseResult(
                        case_id=case.id, skill="arithmetic", split=section,
                        score=score_response(case, learner.generate(case.prompt)).points,
                        baseline_score=score_response(case, baseline.generate(case.prompt)).points,
                        prompt_sha256=_prompt_sha(case.prompt),
                    )
                    for section, case in cases
                )
                exam = MinotaurExamReceipt(
                    exam_id=f"composition-development-{cycle}",
                    objective_id=scope.objective_id, cycle_index=cycle,
                    checkpoint_sha256=learner.model_hash,
                    baseline_sha256=original_hash,
                    evaluator_id="local-development-only", cases=observations,
                )
                decision = evaluate_minotaur(scope, exam)
                self.assertEqual(decision.status, "FAIL")
                self.assertEqual(decision.weak_skills, ("arithmetic",))
                state = face_minotaur(state, exam)
                self.assertEqual(state.checkpoint_sha256, learner.model_hash)
                earlier_train.update(case.id for case in training_cases)
                earlier_exams.update(case.case_id for case in observations)

                if cycle == 0:
                    self.assertEqual(state.phase, TRAINING)
                    self.assertEqual(state.cycle_index, 1)
                    self.assertEqual(state.attempts[0].weak_skills, ("arithmetic",))
                    self.assertNotEqual(state.checkpoint_sha256, original_hash)
                else:
                    self.assertEqual(state.phase, EXHAUSTED)
                    self.assertEqual(len(state.attempts), 2)
                    self.assertNotEqual(state.checkpoint_sha256, original_hash)
                    self.assertNotEqual(state.attempts[0].checkpoint_after_sha256,
                                        state.attempts[1].checkpoint_after_sha256)
                    self.assertFalse(hasattr(state, "graduation_authorized"))


if __name__ == "__main__":
    unittest.main()
