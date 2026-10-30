"""Evaluator-custodied development exam: separate gold, signed score and replay."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from labyrinth.blind_evaluation_custody import (
    BlindCustodyError, PrivateExamCase, initialize_assessment,
    public_exam_questions, score_assessment, verify_assessment,
)


EXAM_KEY = b"e" * 32
STATE_KEY = b"s" * 32


def _cases():
    return (
        PrivateExamCase("held-1", "sentiment", "heldout", "A warm, useful reply.",
                        "direct", "positive"),
        PrivateExamCase("held-2", "sentiment", "heldout", "A rude, broken reply.",
                        "direct", "negative"),
        PrivateExamCase("trans-1", "sentiment", "transfer", "The client left smiling.",
                        "narrative", "positive"),
        PrivateExamCase("trans-2", "sentiment", "transfer", "The client left furious.",
                        "narrative", "negative"),
    )


class BlindCustodyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "model.safetensors"
        self.checkpoint.write_bytes(b"precommitted adapter bytes (test only)")
        self.checkpoint_sha = sha256(self.checkpoint.read_bytes()).hexdigest()
        self.db = self.root / "evaluation.sqlite"
        self.answers = {case.case_id: case.gold_answer for case in _cases()}

    def _initialize(self):
        return initialize_assessment(
            self.db, _cases(), run_id="run-1", objective_id="sentiment-1",
            checkpoint_sha256=self.checkpoint_sha,
            evaluator_key=EXAM_KEY, state_key=STATE_KEY,
        )

    def _score(self, *, cases=None, answers=None, key=EXAM_KEY):
        return score_assessment(
            self.db, _cases() if cases is None else cases,
            self.answers if answers is None else answers,
            artifact_root=self.root, artifact_name="model.safetensors",
            evaluator_key=key, state_key=STATE_KEY,
        )

    def test_platform_native_path_subclass_accepted(self):
        # On POSIX this is a PosixPath, on Windows a WindowsPath; neither has
        # exact type Path, so exact-type checks would reject every exam.
        self.assertIsInstance(self.root, Path)
        self.assertIsNot(type(self.root), Path)
        self._initialize()
        self._score()

    def test_public_questions_never_contain_gold_and_receipt_aggregates_only(self):
        public = public_exam_questions(_cases())
        self.assertEqual(len(public), 4)
        self.assertTrue(all("gold_answer" not in item for item in public))
        self._initialize()
        signed = self._score()
        payload = json.loads(signed.receipt_bytes)
        self.assertEqual(payload["status"], "DEVELOPMENT_SCORED")
        self.assertEqual(payload["groups"], [
            {"correct": 2, "skill": "sentiment", "split": "heldout", "total": 2},
            {"correct": 2, "skill": "sentiment", "split": "transfer", "total": 2},
        ])
        self.assertFalse(payload["graduation_authorized"])
        self.assertFalse(payload["model_execution_authenticated"])
        for case in _cases():
            self.assertNotIn(case.prompt, signed.receipt_bytes.decode())
        self.assertNotIn("gold_answer", signed.receipt_bytes.decode())
        self.assertNotIn("positive", signed.receipt_bytes.decode())
        self.assertEqual(verify_assessment(
            signed.receipt_bytes, signed.signature_hex, evaluator_key=EXAM_KEY,
            run_id="run-1", checkpoint_sha256=self.checkpoint_sha,
        )["status"], "DEVELOPMENT_SCORED")

    def test_signed_development_exam_is_one_shot_across_reopen(self):
        self._initialize()
        self._score()
        with self.assertRaisesRegex(BlindCustodyError, "consumed"):
            self._score()
        with self.assertRaisesRegex(BlindCustodyError, "already exists"):
            self._initialize()

    def test_wrong_or_changed_checkpoint_bytes_rejected_before_consumption(self):
        self._initialize()
        self.checkpoint.write_bytes(b"changed after precommit")
        with self.assertRaisesRegex(BlindCustodyError, "checkpoint.*digest"):
            self._score()
        self.checkpoint.write_bytes(b"precommitted adapter bytes (test only)")
        self._score()  # invalid model bytes did not burn the legitimate exam

    def test_spoofed_public_exam_or_answer_coverage_rejected(self):
        self._initialize()
        cases = (replace(_cases()[0], prompt="altered question"),) + _cases()[1:]
        with self.assertRaisesRegex(BlindCustodyError, "exam.*commitment"):
            self._score(cases=cases)
        poisoned_gold = (replace(_cases()[0], gold_answer="negative"),) + _cases()[1:]
        with self.assertRaisesRegex(BlindCustodyError, "oracle seal mismatch"):
            self._score(cases=poisoned_gold)
        with self.assertRaisesRegex(ValueError, "exactly"):
            self._score(answers={"held-1": "positive"})
        self._score()

    def test_train_cases_cannot_be_placed_in_private_exam(self):
        train_case = (replace(_cases()[0], split="train"),) + _cases()[1:]
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            public_exam_questions(train_case)
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            initialize_assessment(self.db, train_case, run_id="run-1",
                                  objective_id="sentiment-1",
                                  checkpoint_sha256=self.checkpoint_sha,
                                  evaluator_key=EXAM_KEY, state_key=STATE_KEY)

    def test_duplicate_normalized_prompt_across_splits_rejected(self):
        cases = _cases()
        repeated = cases[:2] + (
            replace(cases[2], prompt="   A   WARM, useful reply.   "),
            cases[3],
        )
        with self.assertRaisesRegex(ValueError, "duplicate.*prompt"):
            public_exam_questions(repeated)
        with self.assertRaisesRegex(ValueError, "duplicate.*prompt"):
            initialize_assessment(self.db, repeated, run_id="run-1",
                                  objective_id="sentiment-1",
                                  checkpoint_sha256=self.checkpoint_sha,
                                  evaluator_key=EXAM_KEY, state_key=STATE_KEY)

    def test_structure_family_overlap_across_splits_rejected(self):
        cases = _cases()
        repeated_family = cases[:2] + (
            replace(cases[2], structure_key="direct"), cases[3],
        )
        with self.assertRaisesRegex(ValueError, "structure.*split"):
            public_exam_questions(repeated_family)

    def test_each_skill_requires_heldout_and_transfer_groups(self):
        cases = _cases()
        # Group totals and global split coverage are valid, but the second
        # skill has no transfer cases and the first no heldout cases.
        wrong_grouping = (
            replace(cases[0], skill="courtesy"),
            replace(cases[1], skill="courtesy"),
            cases[2], cases[3],
        )
        with self.assertRaisesRegex(ValueError, "each skill.*both splits"):
            public_exam_questions(wrong_grouping)

    def test_whitespace_only_prompt_rejected(self):
        cases = _cases()
        with self.assertRaisesRegex(ValueError, "blank.*prompt"):
            public_exam_questions((replace(cases[0], prompt=" \t \n"),) + cases[1:])

    def test_wrong_key_and_tampered_state_fail_closed(self):
        self._initialize()
        with self.assertRaisesRegex(BlindCustodyError, "state.*seal|oracle.*seal"):
            self._score(key=b"z" * 32)
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE assessment SET state='CONSUMED' WHERE id=1")
        with self.assertRaisesRegex(BlindCustodyError, "state.*seal"):
            self._score()

    def test_receipt_signature_and_identity_verifier_fail_closed(self):
        self._initialize()
        signed = self._score()
        for body, signature, run_id, digest in (
            (signed.receipt_bytes, "bad-signature", "run-1", self.checkpoint_sha),
            (signed.receipt_bytes + b" ", signed.signature_hex, "run-1", self.checkpoint_sha),
            (signed.receipt_bytes.decode(), signed.signature_hex, "run-1", self.checkpoint_sha),
            (signed.receipt_bytes, signed.signature_hex, "run-other", self.checkpoint_sha),
            (signed.receipt_bytes, signed.signature_hex, "run-1", "f" * 64),
        ):
            with self.subTest(run_id=run_id, signature=signature[:8]):
                with self.assertRaises(BlindCustodyError):
                    verify_assessment(body, signature, evaluator_key=EXAM_KEY,
                                      run_id=run_id, checkpoint_sha256=digest)

    def test_symlink_artifact_rejected(self):
        self._initialize()
        linked = self.root / "linked.safetensors"
        try:
            linked.symlink_to(self.checkpoint)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not permitted on this host")
        with self.assertRaisesRegex(BlindCustodyError, "symlink"):
            score_assessment(self.db, _cases(), self.answers,
                             artifact_root=self.root, artifact_name="linked.safetensors",
                             evaluator_key=EXAM_KEY, state_key=STATE_KEY)


if __name__ == "__main__":
    unittest.main()
