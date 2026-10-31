"""Fail-closed report publication checks for the HF pilot's recovery path.

These tests intentionally describe the required atomic-publication behavior.
They do not claim that optimizer state or model training can already resume.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from labyrinth.hf_pilot import _save_new


class HFAtomicReportRecoveryTests(unittest.TestCase):
    @staticmethod
    def _stages(root: str) -> list[Path]:
        return list(Path(root).glob(".pilot_report.json.stage-*"))

    def test_failed_serialization_does_not_publish_partial_final_report(self):
        with TemporaryDirectory() as root:
            report = Path(root) / "pilot_report.json"
            with self.assertRaises(TypeError):
                _save_new(report, {"schema": "example", "unserializable": object()})

            self.assertFalse(
                report.exists(),
                "interrupted/invalid serialization must not publish an incomplete final report",
            )
            _save_new(report, {"schema": "example", "run_state": "COMPLETE"})
            self.assertEqual(
                json.loads(report.read_text(encoding="utf-8")),
                {"schema": "example", "run_state": "COMPLETE"},
            )
            self.assertEqual(self._stages(root), [])

    def test_interrupted_stream_write_cannot_expose_prefix_as_report(self):
        with TemporaryDirectory() as root:
            report = Path(root) / "pilot_report.json"

            def interrupted_dump(_value, handle, **_kwargs):
                handle.write('{"schema":')
                raise OSError("simulated interrupted write")

            with patch("labyrinth.hf_pilot.json.dump", side_effect=interrupted_dump):
                with self.assertRaisesRegex(OSError, "interrupted write"):
                    _save_new(report, {"schema": "example"})
            self.assertFalse(report.exists())
            self.assertEqual(self._stages(root), [])
            _save_new(report, {"schema": "retry", "run_state": "COMPLETE"})
            self.assertEqual(json.loads(report.read_text())["schema"], "retry")

    def test_unsupported_hard_link_fails_closed_without_orphan(self):
        with TemporaryDirectory() as root:
            report = Path(root) / "pilot_report.json"
            with patch("labyrinth.hf_pilot.os.link", side_effect=OSError("no hard links")):
                with self.assertRaisesRegex(OSError, "no hard links"):
                    _save_new(report, {"schema": "example"})
            self.assertFalse(report.exists())
            self.assertEqual(self._stages(root), [])

    def test_failed_file_sync_does_not_publish_report(self):
        with TemporaryDirectory() as root:
            report = Path(root) / "pilot_report.json"
            with patch("labyrinth.hf_pilot.os.fsync", side_effect=OSError("sync failed")):
                with self.assertRaisesRegex(OSError, "sync failed"):
                    _save_new(report, {"schema": "example"})
            self.assertFalse(report.exists())
            self.assertEqual(self._stages(root), [])

    def test_published_file_is_complete_at_atomic_link_boundary(self):
        with TemporaryDirectory() as root:
            report = Path(root) / "pilot_report.json"
            payload = {"schema": "example", "run_state": "COMPLETE"}
            actual_link = os.link

            def verify_then_link(staged: Path, destination: Path):
                self.assertEqual(json.loads(Path(staged).read_text()), payload)
                self.assertFalse(Path(destination).exists())
                actual_link(staged, destination)

            with patch("labyrinth.hf_pilot.os.link", side_effect=verify_then_link):
                _save_new(report, payload)
            self.assertEqual(json.loads(report.read_text()), payload)
            self.assertTrue(report.read_text().endswith("\n"))
            self.assertEqual(self._stages(root), [])

    def test_parallel_publication_has_one_winner_and_no_partial_report(self):
        with TemporaryDirectory() as root:
            report = Path(root) / "pilot_report.json"

            def publish(index: int):
                try:
                    _save_new(report, {"schema": "example", "index": index})
                    return "published"
                except FileExistsError:
                    return "refused"

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(publish, (1, 2)))
            self.assertCountEqual(outcomes, ("published", "refused"))
            self.assertIn(json.loads(report.read_text())["index"], (1, 2))
            self.assertEqual(self._stages(root), [])

    def test_existing_symlink_is_not_followed_or_replaced(self):
        with TemporaryDirectory() as root:
            report = Path(root) / "pilot_report.json"
            other = Path(root) / "valuable.json"
            other.write_text("valuable", encoding="utf-8")
            try:
                report.symlink_to(other)
            except (NotImplementedError, OSError):
                self.skipTest("symlinks unavailable on this platform")
            with self.assertRaises(FileExistsError):
                _save_new(report, {"schema": "replacement"})
            self.assertTrue(report.is_symlink())
            self.assertEqual(other.read_text(), "valuable")
            self.assertEqual(self._stages(root), [])

    def test_existing_complete_report_is_never_overwritten(self):
        with TemporaryDirectory() as root:
            report = Path(root) / "pilot_report.json"
            original = {"schema": "example", "run_state": "COMPLETE"}
            _save_new(report, original)
            previous = report.read_bytes()
            with self.assertRaises(FileExistsError):
                _save_new(report, {"schema": "replacement"})
            self.assertEqual(report.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
