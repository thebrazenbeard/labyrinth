"""Fail-closed legacy Python factory gate; typed summary anti-laundering."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from labyrinth.minotaur_cross_exam import ExamSpec, run_cross_exam
from labyrinth.minotaur_trusted_launcher import (
    ProcessCandidateConfig, TrustedCaseReceipt, candidate_public_summary,
    run_trusted_cross_exam,
)


class FactoryTrustGateTests(unittest.TestCase):
    def test_unsafe_factory_fails_before_private_case_construction(self):
        class HostileFactory:
            called = False

            def __call__(self, _public):
                self.called = True
                raise AssertionError("hostile Python ran inside examiner")

        factory = HostileFactory()
        with patch("labyrinth.minotaur_cross_exam.make_public_case") as build:
            with self.assertRaises(PermissionError):
                run_cross_exam((ExamSpec("oracle", 739, sealed=True),),
                               candidate_factory=factory, authorize_local_adapter=True)
            build.assert_not_called()
        self.assertFalse(factory.called)

    def test_explicitly_trusted_factory_still_works_but_is_unqualified(self):
        from labyrinth.minotaur_candidate_bridge import FixedActionAdapter
        called = []

        def trusted_host_factory(_public):
            called.append(True)
            return FixedActionAdapter(())

        receipts = run_cross_exam((ExamSpec("oracle", 739),),
                                  candidate_factory=trusted_host_factory,
                                  trust_host_python_factory=True)
        self.assertEqual(len(called), 1)
        self.assertEqual(len(receipts), 1)
        self.assertFalse(receipts[0].candidate.verified_escape)

    def test_incorrect_trust_indicator_is_rejected(self):
        with self.assertRaises(ValueError):
            run_cross_exam((ExamSpec("oracle", 0),),
                           trust_host_python_factory="true")

    def test_forged_summary_nested_records_and_fields_fail_closed(self):
        receipts = run_trusted_cross_exam((ExamSpec("timegate", 17),),
                                          ProcessCandidateConfig(mode="probe"))
        item = receipts[0]
        fake_case = SimpleNamespace(family="timegate", candidate=SimpleNamespace(
            final_verdict="VERIFIED_WIN", verified_escape=True),
            baseline_escape_verified=True)
        with self.assertRaises((TypeError, ValueError)):
            candidate_public_summary((replace(item, cross_case=fake_case),))
        with self.assertRaises((TypeError, ValueError)):
            candidate_public_summary((replace(item, candidate_kind={"seed": 17}),))
        fake_candidate = replace(item.cross_case.candidate, final_verdict="VERIFIED_WIN")
        with self.assertRaises((TypeError, ValueError)):
            candidate_public_summary((replace(item, cross_case=replace(
                item.cross_case, candidate=fake_candidate)),))
        summary = candidate_public_summary(receipts)[0]
        self.assertEqual(summary["evidence_status"], "UNATTESTED_HOST_SELF_HASH")


if __name__ == "__main__":
    unittest.main()
