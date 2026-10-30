"""Public post-hoc counterfactual probe checks: no final-exam claims."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from scripts import qualify_taskmaster_counterfactual_cpu as eval_cpu
from scripts import qualify_minotaur_neural_taskmaster_cpu as trainer


class CounterfactualFixtureTests(unittest.TestCase):
    def test_four_distinct_structures_and_balanced_minimal_pairs(self) -> None:
        metadata = eval_cpu.validate_cases()
        self.assertEqual(metadata["count"], 16)
        self.assertEqual(metadata["family_count"], 4)
        self.assertEqual(metadata["tactic_counts"], [4, 4, 4, 4])
        self.assertRegex(metadata["fixture_commitment_sha256"], r"^[0-9a-f]{64}$")
        rows = eval_cpu.cases()
        for family in {case.family for case in rows}:
            siblings = [case for case in rows if case.family == family]
            self.assertEqual({x.gold for x in siblings}, {0, 1, 2, 3})
        structures = {re.sub(r"\b(?:10|[0-9])/10\b", "?/10", x.prompt)
                      for x in rows}
        self.assertEqual(len(structures), 4)

    def test_surface_keyword_baseline_cannot_classify_metrics(self) -> None:
        rows = eval_cpu.cases()
        predictions = [eval_cpu.keyword_baseline(x.prompt) for x in rows]
        self.assertEqual(set(predictions), {3})
        self.assertEqual(sum(p == x.gold for p, x in zip(predictions, rows)), 4)
        self.assertEqual(len({eval_cpu.normalize(x.prompt) for x in rows}), 16)
        original = {eval_cpu.normalize(text) for split in trainer.fixtures().values()
                    for text, _ in split}
        self.assertFalse(original.intersection({eval_cpu.normalize(x.prompt) for x in rows}))

    def test_numeric_rubric_priority_and_boundaries(self) -> None:
        self.assertEqual(eval_cpu.inferred_tactic(2, 9, 9, 9), 0)
        self.assertEqual(eval_cpu.inferred_tactic(9, 5, 9, 9), 1)
        self.assertEqual(eval_cpu.inferred_tactic(9, 9, 9, 9), 2)
        self.assertEqual(eval_cpu.inferred_tactic(2, 4, 5, 8), 3)
        self.assertEqual(eval_cpu.inferred_tactic(6, 7, 6, 8), 2)
        for bad in (-1, 11, 1.0, True):
            with self.assertRaises(ValueError):
                eval_cpu.inferred_tactic(bad, 9, 9, 9)

    def test_reject_duplicate_and_wrong_gold_and_nonminimal_pairs(self) -> None:
        rows = list(eval_cpu.cases())
        rows[1] = replace(rows[1], prompt=rows[0].prompt)
        with self.assertRaises(ValueError):
            eval_cpu.validate_cases(tuple(rows))
        rows = list(eval_cpu.cases())
        rows[1] = replace(rows[1], gold=2)
        with self.assertRaises(ValueError):
            eval_cpu.validate_cases(tuple(rows))
        rows = list(eval_cpu.cases())
        rows[1] = replace(rows[1], prompt=rows[1].prompt + " Shortcut!")
        with self.assertRaisesRegex(ValueError, "non-minimal"):
            eval_cpu.validate_cases(tuple(rows))

    def test_unicode_normalization_and_training_overlap_rejected(self) -> None:
        self.assertEqual(eval_cpu.normalize(" A\u00a0  B  "), eval_cpu.normalize("a b"))
        self.assertEqual(eval_cpu.normalize("\uff26OO"), "foo")
        rows = list(eval_cpu.cases())
        rows[0] = replace(rows[0], prompt=trainer.fixtures()["cycle0"][0][0])
        with self.assertRaises(ValueError):
            eval_cpu.validate_cases(tuple(rows))


class TrainingReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "training"
        self.source.mkdir()
        self.results = self.root / "evals"
        self.results.mkdir()
        self.checkpoint_bytes = {}
        arms = {}
        for arm in trainer.ARMS:
            arm_root = self.source / arm
            arm_root.mkdir()
            payload = f"synthetic public checkpoint-{arm}".encode()
            (arm_root / "cycle1.safetensors").write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.checkpoint_bytes[arm] = payload
            arms[arm] = {"cycles": [
                {"cycle": 0},
                {"cycle": 1, "after_sha256": "f" * 64, "checkpoint_sha256": digest},
            ]}
        self.report = {"schema": trainer.SCHEMA, "model_id": trainer.MODEL_ID,
                       "device": "cpu", "tactics": list(trainer.TACTICS),
                       "graduation_authorized": False, "independent_exam": "NOT_RUN",
                       "model_execution_authenticated": False,
                       "frozen_encoder_sha256": "a" * 64, "arms": arms}
        self.pin = self.save_receipt()

    def save_receipt(self) -> str:
        payload = json.dumps(self.report, sort_keys=True).encode("utf-8")
        (self.source / "report.json").write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def test_authenticated_preflight_is_no_write(self) -> None:
        observed = eval_cpu.plan(self.source, self.results, "public_dev01", self.pin)
        self.assertEqual(observed["report_sha256"], self.pin)
        self.assertEqual(observed["cases"]["count"], 16)
        self.assertEqual(list(self.results.iterdir()), [])

    def test_cli_defaults_to_no_write_and_never_loads_model(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            result = eval_cpu.main([
                "--training-run-dir", str(self.source), "--output-parent", str(self.results),
                "--run-name", "public_dev01", "--expected-report-sha256", self.pin,
            ])
        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["execute"])
        self.assertFalse(payload["graduation_authorized"])
        self.assertEqual(list(self.results.iterdir()), [])

    def test_receipt_pin_and_checkpoint_hash_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "digest mismatches"):
            eval_cpu.plan(self.source, self.results, "x", "0" * 64)
        with self.assertRaisesRegex(ValueError, "pin is required"):
            eval_cpu.plan(self.source, self.results, "x", "MISSING")
        (self.source / "correct" / "cycle1.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint digest mismatch"):
            eval_cpu.plan(self.source, self.results, "x", self.pin)

    def test_no_graduation_and_no_private_exam_escalation(self) -> None:
        self.report["graduation_authorized"] = True
        pin = self.save_receipt()
        with self.assertRaisesRegex(ValueError, "escalated"):
            eval_cpu.plan(self.source, self.results, "x", pin)
        self.report["graduation_authorized"] = False
        self.report["independent_exam"] = "PASS"
        pin = self.save_receipt()
        with self.assertRaisesRegex(ValueError, "escalated"):
            eval_cpu.plan(self.source, self.results, "x", pin)

    def test_output_isolated_and_preexisting_run_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "isolated"):
            eval_cpu.plan(self.source, self.source, "nested", self.pin)
        (self.results / "reserved").mkdir()
        with self.assertRaises(FileExistsError):
            eval_cpu.plan(self.source, self.results, "reserved", self.pin)

    def test_reject_linked_checkpoint_without_following(self) -> None:
        target = self.source / "correct" / "cycle1.safetensors"
        elsewhere = self.root / "elsewhere"
        target.rename(elsewhere)
        try:
            target.symlink_to(elsewhere)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "linked"):
            eval_cpu.plan(self.source, self.results, "x", self.pin)


if __name__ == "__main__":
    unittest.main()
