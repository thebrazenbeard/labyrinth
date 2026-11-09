"""Behavioral protocol tests: public policy, private evaluator, frozen trials."""

import dataclasses
import json
import time
import unittest

from labyrinth.novel_router_behavioral_protocol import (
    Budget,
    CandidateChoice,
    evaluate,
    feedback_witness,
    fixed_witness,
    precommit,
    verify_local_record,
)

_POLICY_BLOCKED = []


def _hostile_public_port(port):
    for attr in ("seed", "case", "_case", "search", "make_case",
                 "submit", "__dict__"):
        try:
            getattr(port, attr)
        except AttributeError:
            _POLICY_BLOCKED.append(attr)
    public = port.view()
    if any(key in public for key in ("seed", "case_id", "before_rule", "after_rule")):
        raise AssertionError("private field leaked to candidate")
    return CandidateChoice("ABSTAIN", (), 0.5)


def _probe_budget_abuser(port):
    for _ in range(30):
        port.probe("TICK")
    return CandidateChoice("PLAN", ("COMMIT",), 1.0)


def _unsupported_unsat(port):
    return CandidateChoice("CLAIM_UNSAT", (), 0.0)


_MUTABLE_POLICY_DECISION = "ABSTAIN"


def _global_dependency_control(port):
    if _MUTABLE_POLICY_DECISION == "PLAN":
        return CandidateChoice("PLAN", ("COMMIT",), 1.0)
    return CandidateChoice("ABSTAIN", (), 0.5)


def _slow_hostile(port):
    time.sleep(.025)
    return CandidateChoice("PLAN", ("COMMIT",), 1.)


class BehavioralProtocolTests(unittest.TestCase):
    def test_freezes_identical_limits_before_policy_trials(self):
        manifest = precommit(((0, 100), (1, 101)),
                             (fixed_witness, feedback_witness),
                             Budget(max_probes=20, max_calls=90,
                                    max_seconds=1.0))
        self.assertEqual(len(manifest.pairs), 2)
        report = evaluate(manifest, (fixed_witness, feedback_witness))
        self.assertEqual(report.manifest_sha256, manifest.sha256)
        self.assertEqual(len(report.trials), 8)
        self.assertEqual(set(row.budget for row in report.trials), {manifest.budget})
        self.assertTrue(all(row.probes <= 20 and row.calls <= 90
                            and row.terminal in ("SUBMITTED", "BUDGET_EXHAUSTED",
                                                 "INVALID_DECISION", "POLICY_ERROR")
                            for row in report.trials))
        self.assertTrue(all(len(row.receipt_sha256) == 64 and
                            row.receipt_sha256 == __import__("hashlib").sha256(
                                row.receipt_json.encode("utf-8")).hexdigest()
                            for row in report.trials))
        self.assertTrue(all(row.source_verdict in (
            "VERIFIED_WIN", "VERIFIED_FAILURE", "UNKNOWN") for row in report.trials))
        self.assertTrue(all(verify_local_record(manifest, row) for row in report.trials))

    def test_source_public_witnesses_do_not_claim_false_wins(self):
        pairs = ((0, 100), (1, 101), (2, 102), (3, 103),
                 (4, 104), (5, 105), (6, 106), (7, 107))
        report = evaluate(precommit(pairs, (fixed_witness, feedback_witness)),
                          (fixed_witness, feedback_witness))
        self.assertEqual(report.totals["false_wins"], 0)
        self.assertEqual(report.totals["false_unsat_claims"], 0)
        self.assertEqual(report.totals["optimizer_updates"], 0)
        self.assertEqual(report.totals["count"], 32)
        self.assertEqual(sum(report.totals[k] for k in (
            "verified_wins", "failures", "abstentions", "other")), 32)
        self.assertFalse(report.generalization_qualified)
        self.assertFalse(report.neural_learning_verified)

    def test_policy_receives_only_public_view_and_probe(self):
        _POLICY_BLOCKED.clear()
        report = evaluate(precommit(((0, 100),), (_hostile_public_port,)),
                          (_hostile_public_port,))
        self.assertEqual(len(_POLICY_BLOCKED), 7 * 2)
        self.assertEqual(report.totals["abstentions"], 2)

    def test_policy_replacement_after_precommit_rejected(self):
        manifest = precommit(((0, 100),), (fixed_witness,))

        def replacement(port):
            return CandidateChoice("ABSTAIN", (), 0.5)

        with self.assertRaisesRegex(ValueError, "precommit"):
            evaluate(manifest, (replacement,))

    def test_invalid_policy_cannot_forge_a_win_and_is_penalized(self):
        def liar(port):
            return CandidateChoice("PLAN", ("COMMIT",), 1.0)

        report = evaluate(precommit(((0, 100),), (liar,)), (liar,))
        self.assertEqual(report.totals["verified_wins"], 0)
        self.assertEqual(report.totals["false_wins"], 2)
        self.assertGreater(report.totals["total_loss"], 0)

    def test_bad_rosters_and_unfrozen_mutation_fail_closed(self):
        with self.assertRaises(ValueError):
            precommit(((0, 0),), (fixed_witness,))
        with self.assertRaises(ValueError):
            precommit(((0, 100), (0, 101)), (fixed_witness,))
        with self.assertRaises(ValueError):
            precommit(((0, 100),), (fixed_witness, fixed_witness))
        with self.assertRaises(ValueError):
            Budget(max_probes=21)
        manifest = precommit(((0, 100),), (fixed_witness,))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            manifest.sha256 = "tampered"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            manifest.budget.max_calls = 999

    def test_penalty_and_calibration_disfavor_trivial_abstention(self):
        def always_abstain(port):
            return CandidateChoice("ABSTAIN", (), 0.5)

        report = evaluate(precommit(((0, 100),), (always_abstain,)),
                          (always_abstain,))
        self.assertEqual(report.totals["count"], 2)
        self.assertEqual(report.totals["abstentions"], 2)
        self.assertGreaterEqual(report.totals["total_loss"], 2 * 0.25)
        self.assertGreater(report.totals["abstention_penalty"], 0)
        self.assertTrue(all(0 <= row.confidence_brier <= 1 for row in report.trials))
        self.assertTrue(all(json.loads(row.receipt_json)["schema"] ==
                            "labyrinth.novel-router-session.v1"
                            for row in report.trials))

    def test_probe_budget_abuse_and_false_unsat_are_recorded(self):
        report = evaluate(precommit(((0, 100),),
                                    (_probe_budget_abuser, _unsupported_unsat),
                                    Budget(max_probes=3, max_calls=8)),
                          (_probe_budget_abuser, _unsupported_unsat))
        abusers = [r for r in report.trials if r.policy.endswith("_probe_budget_abuser")]
        self.assertEqual(len(abusers), 2)
        self.assertTrue(all(r.probes == 3 and
                            r.terminal == "BUDGET_EXHAUSTED" and
                            r.proposal_kind == "ABSTAIN" for r in abusers))
        self.assertGreaterEqual(report.totals["false_unsat_claims"], 1)
        self.assertEqual(report.totals["verified_wins"], 0)

    def test_code_hash_is_not_host_custody_or_frozen_mutable_globals(self):
        global _MUTABLE_POLICY_DECISION
        _MUTABLE_POLICY_DECISION = "ABSTAIN"
        manifest = precommit(((0, 100),), (_global_dependency_control,))
        try:
            _MUTABLE_POLICY_DECISION = "PLAN"
            report = evaluate(manifest, (_global_dependency_control,))
            self.assertEqual(report.totals["false_wins"], 2)
            self.assertFalse(report.private_custody)
            self.assertFalse(report.generalization_qualified)
        finally:
            _MUTABLE_POLICY_DECISION = "ABSTAIN"

    def test_case_bound_receipt_replay_falsifies_mutated_receipt(self):
        manifest = precommit(((0, 100),), (fixed_witness,))
        record = evaluate(manifest, (fixed_witness,)).trials[0]
        self.assertTrue(verify_local_record(manifest, record))
        original = json.loads(record.receipt_json)
        original["probe_trace"][0]["action"] = "ROUTE"
        tampered = json.dumps(original, sort_keys=True, separators=(",", ":"))
        from hashlib import sha256
        altered = dataclasses.replace(record, receipt_json=tampered,
                                      receipt_sha256=sha256(tampered.encode()).hexdigest())
        self.assertFalse(verify_local_record(manifest, altered))

    def test_post_hoc_time_violation_is_fail_closed_but_not_preemptive(self):
        manifest = precommit(((0, 100),), (_slow_hostile,),
                             Budget(max_seconds=.01))
        report = evaluate(manifest, (_slow_hostile,))
        self.assertTrue(all(row.terminal == "BUDGET_EXHAUSTED" and
                            row.proposal_kind == "ABSTAIN" for row in report.trials))


if __name__ == "__main__":
    unittest.main()
