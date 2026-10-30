"""Qualification-bundle provenance is broader than the runtime source digest."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from labyrinth.qualification_bundle import (
    capture_qualification_bundle, verify_qualification_bundle,
)


class QualificationBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = {
            "pyproject.toml": "[project]\nname = 'sample'\n",
            "scripts/qualify_hf_regenerative.py": "# two-cycle driver\n",
            "scripts/minotaur_ci.py": "# membership gate\n",
            "scripts/make_tiny_hf_fixture.py": "# fixture maker\n",
            ".github/workflows/ci.yml": "name: Python CI\n",
            ".github/workflows/minotaur-neural-cpu.yml": "name: CPU CI\n",
            "tests/minotaur_required_tests.txt": "test_example.Case.test_one\n",
            "tests/test_example.py": "def test_one(): pass\n",
            "labyrinth/runner.py": "# runtime\n",
            "labyrinth/hf_pilot.py": "# model adapter\n",
        }
        for name, contents in self.samples.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")

    def test_repeated_capture_is_stable_and_covers_drivers_tests_workflows(self):
        first = capture_qualification_bundle(self.root)
        self.assertEqual(first, capture_qualification_bundle(self.root))
        self.assertEqual(first["schema"], "labyrinth.qualification-bundle.v1")
        self.assertTrue(verify_qualification_bundle(first, self.root))
        names = [entry["path"] for entry in first["files"]]
        self.assertEqual(names, sorted(names))
        self.assertEqual(names, sorted(self.samples))
        driver = next(entry for entry in first["files"] if entry["path"] ==
                      "scripts/qualify_hf_regenerative.py")
        self.assertEqual(driver["sha256"], hashlib.sha256(
            self.samples[driver["path"]].encode()).hexdigest())
        self.assertEqual(driver["bytes"], len(self.samples[driver["path"]].encode()))
        self.assertEqual(len(first["bundle_sha256"]), 64)
        self.assertEqual(json.loads(json.dumps(first)), first)

    def test_driver_edit_and_workflow_edit_invalidate_previous_manifest(self):
        prior = capture_qualification_bundle(self.root)
        self._change("scripts/qualify_hf_regenerative.py")
        self.assertFalse(verify_qualification_bundle(prior, self.root))
        after_driver = capture_qualification_bundle(self.root)
        self.assertNotEqual(prior["bundle_sha256"], after_driver["bundle_sha256"])
        self._change(".github/workflows/minotaur-neural-cpu.yml")
        self.assertFalse(verify_qualification_bundle(after_driver, self.root))

    def test_test_disappearance_and_added_test_invalidate_previous_manifest(self):
        prior = capture_qualification_bundle(self.root)
        (self.root / "tests/test_example.py").unlink()
        self.assertFalse(verify_qualification_bundle(prior, self.root))
        (self.root / "tests/test_example.py").write_text(
            self.samples["tests/test_example.py"], encoding="utf-8")
        self.assertTrue(verify_qualification_bundle(prior, self.root))
        (self.root / "tests/test_added.py").write_text("# new test\n")
        self.assertFalse(verify_qualification_bundle(prior, self.root))

    def test_missing_required_driver_is_rejected(self):
        (self.root / "scripts/minotaur_ci.py").unlink()
        with self.assertRaisesRegex(ValueError, "missing qualification file"):
            capture_qualification_bundle(self.root)

    def test_modified_report_cannot_verify_without_matching_current_sources(self):
        snapshot = capture_qualification_bundle(self.root)
        snapshot["files"][0]["sha256"] = "0" * 64
        self.assertFalse(verify_qualification_bundle(snapshot, self.root))
        snapshot = capture_qualification_bundle(self.root)
        snapshot["bundle_sha256"] = "0" * 64
        self.assertFalse(verify_qualification_bundle(snapshot, self.root))

    def test_symlinked_driver_is_rejected(self):
        target = self.root / "scripts/qualify_hf_regenerative.py"
        target.rename(target.with_name("driver-backup.py"))
        try:
            target.symlink_to("driver-backup.py")
        except (OSError, NotImplementedError):
            self.skipTest("symlink privilege unavailable")
        with self.assertRaisesRegex(ValueError, "non-regular or symlink"):
            capture_qualification_bundle(self.root)

    def _change(self, name):
        path = self.root / name
        path.write_text(path.read_text() + "# edit\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
