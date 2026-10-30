"""Train-only routing-policy adaptation; no exam or neural claim."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from labyrinth.minotaur_model import RoomContext, plan_training_room
from labyrinth.training_scope import TrainingScopeSpec
from labyrinth.minotaur_policy_learning import (
    MinotaurTrainingPolicy, TrainRoomOutcome, TrainCaseApproval, load_policy,
)


def approval(index: int, *, tactic: str = "practice", difficulty: int = 3) -> TrainCaseApproval:
    return TrainCaseApproval(
        room_id=f"room-{index}", case_id=f"case-{index}",
        skill="arithmetic", tactic=tactic, difficulty=difficulty,
        prompt_sha256=hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
    )


APPROVED = tuple(approval(i) for i in range(16))
SCOPE = TrainingScopeSpec(
    objective_id="objective-one", domain="arithmetic", skills=("arithmetic",),
    specificity=3, scale=2, max_cycles=4, seed=7,
)


def model() -> MinotaurTrainingPolicy:
    return MinotaurTrainingPolicy(
        scope=SCOPE, approved_training_cases=APPROVED,
    )


def context() -> RoomContext:
    return RoomContext(
        objective_id="objective-one", domain="arithmetic", skill="arithmetic",
        room_id="room-0", cycle_index=0, case_seed=24, specificity=3, scale=2,
    )


def outcome(index: int, *, correct: bool, tactic: str = "practice", difficulty: int = 3,
            split: str = "train", skill: str = "arithmetic") -> TrainRoomOutcome:
    return TrainRoomOutcome(
        objective_id="objective-one", room_id=f"room-{index}",
        case_id=f"case-{index}", skill=skill, tactic=tactic, difficulty=difficulty,
        prompt_sha256=hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
        was_correct=correct, split=split,
    )


class MinotaurPolicyLearningTests(unittest.TestCase):
    def test_training_outcomes_change_real_taskmaster_proposals(self):
        learner = model()
        initial = plan_training_room(context(), taskmaster=learner)
        self.assertEqual((initial.tactic, initial.difficulty), ("practice", 3))
        before = learner.model_hash
        for i in range(3):
            receipt = learner.observe(outcome(i, correct=True))
            self.assertTrue(receipt["updated"])
        after = plan_training_room(context(), taskmaster=learner)
        self.assertNotEqual(before, learner.model_hash)
        self.assertEqual((after.tactic, after.difficulty), ("contrast", 4))
        self.assertEqual(after.source, "untrusted_model_suggestion")

    def test_failures_lower_difficulty_without_changing_scope(self):
        learner = model()
        for i in range(3):
            learner.observe(outcome(i, correct=False))
        adjusted = learner.propose_room(context().public_metadata())
        self.assertEqual(adjusted, {"difficulty": 1, "tactic": "practice"})

    def test_current_tactic_failures_are_not_hidden_by_old_practice_success(self):
        # Practice mastery authorizes an attempt at contrast, but repeated
        # contrast failures are evidence against keeping that stage difficult.
        # Historical practice wins must not drown out current-stage failures.
        cases = tuple(
            approval(i, tactic="practice" if i < 8 else "contrast")
            for i in range(12)
        )
        learner = MinotaurTrainingPolicy(scope=SCOPE, approved_training_cases=cases)
        for i in range(8):
            learner.observe(outcome(i, correct=True))
        self.assertEqual(learner.propose_room(context().public_metadata())["tactic"], "contrast")
        for i in range(8, 12):
            learner.observe(outcome(i, tactic="contrast", correct=False))
        self.assertEqual(
            learner.propose_room(context().public_metadata()),
            {"difficulty": 1, "tactic": "contrast"},
        )

    def test_zero_update_and_corrupt_label_controls_are_distinguishable(self):
        correct, sham, corrupt = model(), model(), model()
        initial = correct.model_hash
        for i in range(3):
            case = outcome(i, correct=True)
            correct.observe(case)
            sham.observe(case, mode="zero_update")
            corrupt.observe(case, mode="corrupt_label")
        self.assertNotEqual(correct.model_hash, initial)
        self.assertEqual(sham.model_hash, initial)
        self.assertNotEqual(corrupt.model_hash, initial)
        self.assertNotEqual(correct.model_hash, corrupt.model_hash)
        self.assertEqual(correct.propose_room(context().public_metadata())["tactic"], "contrast")
        self.assertEqual(corrupt.propose_room(context().public_metadata())["difficulty"], 1)

    def test_eval_or_private_outcomes_rejected_before_any_update(self):
        learner = model()
        baseline = learner.model_hash
        for split in ("eval", "heldout", "transfer", "private_final", "test"):
            with self.assertRaises(ValueError):
                learner.observe(outcome(1, correct=True, split=split))
        self.assertEqual(learner.model_hash, baseline)
        with self.assertRaises(ValueError):
            learner.observe(outcome(1, correct=True, skill="cognitive_empathy"))
        self.assertEqual(learner.model_hash, baseline)

    def test_relabelled_private_final_cannot_be_observed_as_train(self):
        learner = model()
        initial = learner.model_hash
        forged = TrainRoomOutcome(
            objective_id="objective-one", room_id="room-private",
            case_id="private-final-123", skill="arithmetic", tactic="practice",
            difficulty=3, prompt_sha256=hashlib.sha256(b"private-final").hexdigest(),
            was_correct=True, split="train",
        )
        with self.assertRaises(ValueError):
            learner.observe(forged)
        self.assertEqual(learner.model_hash, initial)

        mismatched_prompt = TrainRoomOutcome(
            objective_id="objective-one", room_id="room-1",
            case_id="case-1", skill="arithmetic", tactic="practice", difficulty=3,
            prompt_sha256=hashlib.sha256(b"forged-answer-key").hexdigest(),
            was_correct=True, split="train",
        )
        with self.assertRaises(ValueError):
            learner.observe(mismatched_prompt)
        self.assertEqual(learner.model_hash, initial)

    def test_only_predeclared_train_manifest_and_exact_case_identity_admitted(self):
        with self.assertRaises(ValueError):
            MinotaurTrainingPolicy(
                scope=SCOPE, approved_training_cases=(),
            )
        with self.assertRaises(ValueError):
            TrainCaseApproval(
                room_id="room-7", case_id="private-final-7",
                skill="arithmetic", tactic="practice", difficulty=3,
                prompt_sha256=hashlib.sha256(b"exam").hexdigest(),
                split="private_final",
            )

        learner = model()
        before = learner.model_hash
        altered = (
            {"room_id": "room-else"}, {"tactic": "contrast"},
            {"difficulty": 5}, {"skill": "unregistered"},
        )
        for change in altered:
            fields = dict(vars(outcome(1, correct=True)))
            fields.update(change)
            with self.assertRaises(ValueError):
                learner.observe(TrainRoomOutcome(**fields))
        self.assertEqual(learner.model_hash, before)

    def test_explicit_new_cycle_registry_retains_learned_state(self):
        learner = model()
        learner.observe(outcome(0, correct=True))
        before = learner.propose_room(context().public_metadata())
        next_cycle_case = approval(20)
        learner.extend_training_registry((next_cycle_case,))
        self.assertEqual(learner.propose_room(context().public_metadata()), before)
        learner.observe(outcome(20, correct=True))
        self.assertEqual(learner.propose_room(context().public_metadata())["tactic"], "contrast")
        with self.assertRaises(ValueError):
            learner.extend_training_registry((next_cycle_case,))

    def test_policy_cannot_silently_transfer_between_declared_training_scopes(self):
        learner = model()
        baseline = learner.model_hash
        # Identical objective/domain/skill labels but different declared
        # specificity or scale must not inherit same-policy lineage.
        for changes in ({"specificity": 1}, {"specificity": 5}, {"scale": 5},
                        {"max_cycles": 3}, {"seed": 8}):
            other_scope = replace(SCOPE, **changes)
            self.assertNotEqual(other_scope.fingerprint, SCOPE.fingerprint)
            other_policy = MinotaurTrainingPolicy(
                scope=other_scope, approved_training_cases=APPROVED,
            )
            self.assertNotEqual(other_policy.model_hash, baseline)
            with tempfile.TemporaryDirectory() as tmp:
                location = Path(tmp) / "original-scope.json"
                learner.save_policy(location)
                with self.assertRaises(ValueError):
                    load_policy(location, scope=other_scope,
                                approved_training_cases=APPROVED)

        for changes in ({"specificity": 1}, {"specificity": 5},
                        {"scale": 5}, {"cycle_index": SCOPE.max_cycles}):
            altered = dict(context().public_metadata())
            altered.update(changes)
            with self.assertRaises(ValueError):
                learner.propose_room(altered)
        self.assertEqual(learner.model_hash, baseline)

    def test_no_replayed_case_or_prompt_evidence(self):
        learner = model()
        learner.observe(outcome(1, correct=True))
        snapshot = learner.model_hash
        with self.assertRaises(ValueError):
            learner.observe(outcome(1, correct=False))
        duplicate_prompt = TrainRoomOutcome(
            objective_id="objective-one", room_id="room-new", case_id="case-new",
            skill="arithmetic", tactic="practice", difficulty=3,
            prompt_sha256=outcome(1, correct=True).prompt_sha256,
            was_correct=True,
        )
        with self.assertRaises(ValueError):
            learner.observe(duplicate_prompt)
        self.assertEqual(learner.model_hash, snapshot)

    def test_save_reload_in_new_process_preserves_policy_and_nonoverwrite(self):
        learner = model()
        for i in range(3):
            learner.observe(outcome(i, correct=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "taskmasker.json"
            stored = learner.save_policy(path)
            self.assertEqual(stored, learner.model_hash)
            self.assertEqual(load_policy(path, scope=SCOPE, approved_training_cases=APPROVED).model_hash, stored)
            with self.assertRaises(ValueError):
                load_policy(path, scope=SCOPE, approved_training_cases=APPROVED[:-1])
            with self.assertRaises(TypeError):
                load_policy(path)
            with self.assertRaises(FileExistsError):
                learner.save_policy(path)
            code = (
                "from labyrinth.minotaur_policy_learning import load_policy,TrainCaseApproval; "
                "from labyrinth.training_scope import TrainingScopeSpec; "
                "import hashlib; "
                "scope=TrainingScopeSpec(objective_id='objective-one',domain='arithmetic',"
                "skills=('arithmetic',),specificity=3,scale=2,max_cycles=4,seed=7); "
                "cases=tuple(TrainCaseApproval(room_id=f'room-{i}',case_id=f'case-{i}',"
                "skill='arithmetic',tactic='practice',difficulty=3,"
                "prompt_sha256=hashlib.sha256(f'prompt-{i}'.encode()).hexdigest())"
                " for i in range(16)); "
                f"p=load_policy({str(path)!r},scope=scope,approved_training_cases=cases); "
                "print(p.model_hash); "
                "print(p.propose_room({'objective_id':'objective-one','domain':'arithmetic',"
                "'skill':'arithmetic','room_id':'room-0','cycle_index':0,"
                "'case_seed':24,'specificity':3,'scale':2}))"
            )
            run = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                 text=True, check=True)
            self.assertEqual(run.stdout.splitlines()[0], stored)
            self.assertIn("'contrast'", run.stdout)

    def test_failed_checkpoint_sync_never_publishes_partial_final_file(self):
        learner = model()
        learner.observe(outcome(0, correct=True))
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "training-state.json"
            with patch("labyrinth.minotaur_policy_learning.os.fsync",
                       side_effect=OSError("disk sync failed")):
                with self.assertRaisesRegex(OSError, "disk sync failed"):
                    learner.save_policy(destination)
            self.assertFalse(destination.exists(), "failed sync must not publish checkpoint")
            self.assertEqual(list(Path(tmp).iterdir()), [], "failed save must remove staging files")
            # A subsequent safe retry must be possible using the same exact name.
            self.assertEqual(learner.save_policy(destination), learner.model_hash)
            self.assertEqual(load_policy(destination, scope=SCOPE,
                                         approved_training_cases=APPROVED).model_hash,
                             learner.model_hash)

    def test_checkpoint_publishing_failure_preserves_existing_target(self):
        learner = model()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "training-state.json"
            destination.write_bytes(b"preexisting checkpoint")
            with patch("labyrinth.minotaur_policy_learning.os.link",
                       side_effect=FileExistsError("publish refused")):
                with self.assertRaises(FileExistsError):
                    learner.save_policy(destination)
            self.assertEqual(destination.read_bytes(), b"preexisting checkpoint")
            self.assertEqual(list(Path(tmp).iterdir()), [destination])

    def test_checkpoint_tamper_and_untrusted_code_rejected(self):
        learner = model()
        learner.observe(outcome(0, correct=True))
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.json"
            learner.save_policy(good)
            data = json.loads(good.read_text(encoding="utf-8"))
            data["counts"]["arithmetic"]["practice"]["correct"] = 99
            altered = Path(tmp) / "altered.json"
            altered.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy(altered, scope=SCOPE, approved_training_cases=APPROVED)
            dup = Path(tmp) / "duplicate.json"
            dup.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy(dup, scope=SCOPE, approved_training_cases=APPROVED)

            alias = Path(tmp) / "alias.json"
            try:
                alias.symlink_to(good)
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaises(ValueError):
                    load_policy(alias, scope=SCOPE, approved_training_cases=APPROVED)
            if hasattr(os, "mkfifo"):
                fifo = Path(tmp) / "checkpoint.fifo"
                os.mkfifo(fifo)
                with self.assertRaises(ValueError):
                    load_policy(fifo, scope=SCOPE, approved_training_cases=APPROVED)

    def test_invalid_untrusted_room_context_fails_before_proposal(self):
        learner = model()
        public = dict(context().public_metadata())
        public["objective_id"] = "different-objective"
        with self.assertRaises(ValueError):
            learner.propose_room(public)

        with self.assertRaises(ValueError):
            MinotaurTrainingPolicy(
                scope="not a frozen training scope", approved_training_cases=APPROVED,
            )
        public = dict(context().public_metadata())
        public["answer"] = "ANSWER: 7"
        with self.assertRaises(ValueError):
            learner.propose_room(public)


if __name__ == "__main__":
    unittest.main()
