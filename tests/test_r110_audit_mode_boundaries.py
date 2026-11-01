"""Independent R110 #94/#95 schema and source-mode boundary regressions.

These tests deliberately consume the public audit and pilot entrypoints.  A
passing source-only audit is not proof of authenticated neural execution.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from labyrinth.hf_adaptive_source_v1 import ADAPTIVE_SCHEMA
from labyrinth.hf_pilot import (
    PilotConfig, PRE_EXPOSURE_SCHEMA, SCHEMA, _training_visit_exposures,
    run_pilot, training_cases,
)
from labyrinth.hf_report_audit import EXPOSURE_MARKER, audit_hf_report
from tests.test_hf_report_audit import (
    adaptive_pilot_fixture, matched_v3_fixture,
)


def _strict_exposure_fixture() -> dict:
    """Promote the independently constructed historical fixture to v3-exposure."""
    report = matched_v3_fixture()
    report["schema"] = SCHEMA
    report["optimizer_exposure_recording"] = EXPOSURE_MARKER
    config = PilotConfig(
        model_path="unused", output_dir="unused", seed=13,
        train_count=2, eval_count=2, tiers=(1,), steps=2,
    )
    train, _ = training_cases(config)
    expected = [list(pair) for pair in _training_visit_exposures(
        train, seed=config.seed, steps=config.steps,
    )]
    for arm in ("sham", "candidate", "corrupted"):
        report["training"][arm]["visited_exposures"] = [list(x) for x in expected]
    return report


class ReconciledAuditModeBoundaries(unittest.TestCase):
    def test_strict_default_requires_all_three_exposure_ledgers_and_marker(self):
        report = _strict_exposure_fixture()
        checked = audit_hf_report(report)
        self.assertEqual(checked["status"], "PASS", checked)
        self.assertFalse(checked["authenticated_model_execution"])
        for arm in ("sham", "candidate", "corrupted"):
            altered = json.loads(json.dumps(report))
            altered["training"][arm].pop("visited_exposures")
            with self.subTest(missing_arm=arm):
                self.assertEqual(audit_hf_report(altered)["status"], "FAIL")
        no_marker = json.loads(json.dumps(report))
        no_marker.pop("optimizer_exposure_recording")
        self.assertEqual(audit_hf_report(no_marker)["status"], "FAIL")

    def test_caller_cannot_silently_downgrade_strict_exposure_to_historical_v3(self):
        report = _strict_exposure_fixture()
        report["schema"] = PRE_EXPOSURE_SCHEMA
        report.pop("optimizer_exposure_recording")
        for arm in ("sham", "candidate", "corrupted"):
            report["training"][arm].pop("visited_exposures")
        self.assertEqual(audit_hf_report(report)["status"], "FAIL")
        explicit = audit_hf_report(report, allow_legacy=True)
        self.assertEqual(explicit["status"], "PASS", explicit)
        self.assertEqual(explicit["evidence_contract"], "legacy-internal-consistency-only")
        self.assertFalse(explicit["authenticated_model_execution"])

    def test_adaptive_and_staged_manifest_fields_are_schema_exclusive(self):
        adaptive = adaptive_pilot_fixture()
        self.assertEqual(adaptive["schema"], ADAPTIVE_SCHEMA)
        checked = audit_hf_report(adaptive)
        self.assertEqual(checked["status"], "PASS", checked)
        self.assertFalse(checked["authenticated_model_execution"])

        wrong_adaptive = json.loads(json.dumps(adaptive))
        wrong_adaptive["minotaur_staged_train"] = {"schema": "untrusted"}
        self.assertEqual(audit_hf_report(wrong_adaptive)["status"], "FAIL")

        strict = _strict_exposure_fixture()
        strict["adaptive_train_source_v1"] = adaptive["adaptive_train_source_v1"]
        self.assertEqual(audit_hf_report(strict)["status"], "FAIL")

        historical = matched_v3_fixture()
        historical["adaptive_train_source_v1"] = adaptive["adaptive_train_source_v1"]
        self.assertEqual(audit_hf_report(historical, allow_legacy=True)["status"], "FAIL")

    def test_conflicting_roster_modes_are_rejected_before_any_dependency_or_io(self):
        invalid = PilotConfig(
            model_path="/deliberately/absent/base",
            output_dir="/deliberately/absent/output",
        )
        combinations = (
            {"minotaur_roster": object(), "adaptive_train_roster": object()},
            {"minotaur_roster": object(), "allow_unattested_adaptive_train": True},
            {"minotaur_roster": object(), "adaptive_train_roster": object(),
             "allow_unattested_adaptive_train": True},
        )
        with (patch("labyrinth.hf_pilot.validate_config",
                    side_effect=AssertionError("filesystem validation was reached")) as validate,
              patch("labyrinth.hf_pilot._deps",
                    side_effect=AssertionError("neural dependency import was reached")) as deps):
            for modes in combinations:
                with self.subTest(modes=tuple(modes)):
                    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                        run_pilot(invalid, **modes)
            validate.assert_not_called()
            deps.assert_not_called()


if __name__ == "__main__":
    unittest.main()
