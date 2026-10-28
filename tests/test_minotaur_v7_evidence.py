"""Negative-result V7 evidence must remain source-bound and fail closed."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from labyrinth.minotaur_v7_evidence import audit_v7_report

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"docs/evidence/MINOTAUR_V7_NUMERIC_2026-10-28.json"

class EvidenceTests(unittest.TestCase):
    def test_frozen_source_and_negative_results_verified(self):
        audit=audit_v7_report(REPORT,ROOT)
        self.assertEqual(audit["run_count"],2)
        self.assertEqual(audit["target_53_correct"],[0,0])
        self.assertTrue(audit["no_promotion"])
        self.assertEqual(len(audit["report_sha256"]),64)

    def test_missing_controls_are_refused(self):
        with TemporaryDirectory() as d:
            data=json.loads(REPORT.read_text(encoding="utf-8"))
            data["runs"][0]["controls"].pop("frozen_random")
            p=Path(d)/"bad.json";p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):audit_v7_report(p,ROOT)
    def test_positive_promotion_is_refused(self):
        with TemporaryDirectory() as d:
            data=json.loads(REPORT.read_text(encoding="utf-8"))
            data["promotion_recommendation"]="PROMOTE"
            p=Path(d)/"bad.json";p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):audit_v7_report(p,ROOT)
