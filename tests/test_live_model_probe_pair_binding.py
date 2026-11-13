"""Adversarial paired-prompt binding checks for model observation comparison."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from labyrinth.live_model_probe import compare_reports
from test_live_model_probe_receipt_integrity import _observation, _receipt


class PairedPromptBindingTests(unittest.TestCase):
    def test_rehashed_changed_prompt_does_not_count_as_same_evaluation_case(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            before = folder / "before.json"
            after = folder / "after.json"
            original = _observation()
            changed = _observation()
            # Same nominal case ID and case-file commitment, but a different
            # prompt digest. Both observation digests are deliberately valid.
            changed["trials"][0]["prompt_sha256"] = "e" * 64
            before.write_text(json.dumps(_receipt(original)), encoding="utf-8")
            after.write_text(json.dumps(_receipt(changed)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt|different committed case"):
                compare_reports(before, after, folder / "comparison.json")

    def test_identical_prompt_digest_still_allows_unqualified_comparison(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            before = folder / "before.json"
            after = folder / "after.json"
            value = _observation()
            before.write_text(json.dumps(_receipt(value)), encoding="utf-8")
            after.write_text(json.dumps(_receipt(value)), encoding="utf-8")
            result = compare_reports(before, after, folder / "comparison.json")
            self.assertEqual(result["status"], "SAME_CHECKPOINT_NO_WEIGHT_UPDATE")
            self.assertFalse(result["independent_custody_verified"])


if __name__ == "__main__":
    unittest.main()
