"""Independent public-only Switchback revision-vs-stationarity controls."""

from collections import Counter
import unittest
from unittest import mock

from labyrinth.switchback_nonstationary_controls import (
    SOURCE_RULE_FUNCTIONS, falsely_stationary_feedback, open_loop_source_plan,
    revision_aware_feedback,
)
from labyrinth.switchback_circuit import MAX_PROBES, ProbeSession, make_case
from labyrinth.switchback_oracle import search, verify_receipt_replay


class _PublicOnly:
    __slots__ = ("view", "probe", "submit")

    def __init__(self, examiner):
        self.view = examiner.view
        self.probe = examiner.probe
        self.submit = examiner.submit


_POLICIES = (open_loop_source_plan, revision_aware_feedback,
             falsely_stationary_feedback)


def _run(seed, policy):
    examiner = ProbeSession(make_case(seed))
    return policy(_PublicOnly(examiner)), examiner


class SwitchbackNonstationaryControlsTests(unittest.TestCase):
    def test_source_rule_hypotheses_are_deduplicated_not_model_weights(self):
        self.assertEqual(SOURCE_RULE_FUNCTIONS, 378)

    def test_policy_reads_only_public_methods_no_private_oracle_search(self):
        for policy in _POLICIES:
            for seed in (0, 1, 13, 14, 44, 100, 117, 199):
                with self.subTest(policy=policy.__name__, seed=seed):
                    with mock.patch("labyrinth.switchback_oracle.search",
                                    side_effect=AssertionError("privileged oracle prohibited")):
                        result, session = _run(seed, policy)
                    self.assertTrue(verify_receipt_replay(result.outcome))
                    self.assertEqual(result.optimizer_updates, 0)
                    self.assertEqual(result.model_checkpoint, "NONE")
                    self.assertFalse(result.neural_learning_established)
                    self.assertFalse(result.independent_mechanic_families_verified)
                    self.assertEqual(result.graduation_authority, "none")
                    self.assertEqual(tuple(row["action"] for row in
                                           session.view()["observations"]),
                                     result.probes)
                    self.assertLessEqual(len(result.probes), MAX_PROBES)
                    self.assertLessEqual(len(result.plan or ()), 9)
                    self.assertIn(result.proposal, ("PLAN", "ABSTAIN"))
                    if result.proposal == "ABSTAIN":
                        self.assertEqual(result.verdict, "UNKNOWN")
                    elif policy is not open_loop_source_plan:
                        self.assertEqual(result.verdict, "VERIFIED_WIN")

    def test_open_loop_may_fail_but_feedback_policies_never_guess_unsat(self):
        for policy in _POLICIES:
            for seed in range(20):
                first, _ = _run(seed, policy)
                again, _ = _run(seed, policy)
                self.assertEqual((first.verdict, first.probes, first.plan),
                                 (again.verdict, again.probes, again.plan))
                if policy is open_loop_source_plan:
                    self.assertEqual(first.probes, ())
                else:
                    self.assertEqual(first.probes[:2], ("1", "RESET"))
                self.assertNotEqual(first.verdict, "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_train_heldout_paired_revision_and_stationarity_metrics(self):
        data = {}
        for start, label in ((0, "TRAIN"), (100, "HELDOUT")):
            outcomes = {}
            for policy in _POLICIES:
                verdicts = Counter()
                probes = Counter()
                contradictions = 0
                for seed in range(start, start + 100):
                    result, _ = _run(seed, policy)
                    self.assertTrue(verify_receipt_replay(result.outcome))
                    verdicts[result.verdict] += 1
                    probes[len(result.probes)] += 1
                    contradictions += result.contradicted_stationary_assumption
                    self.assertFalse(result.neural_learning_established)
                outcomes[policy.__name__] = (verdicts, probes, contradictions)
                self.assertEqual(sum(verdicts.values()), 100)
            winnable = sum(search(make_case(seed))["verdict"] == "VERIFIED_WIN"
                           for seed in range(start, start + 100))
            data[label] = (outcomes, winnable)
        self.assertEqual(data["TRAIN"][1], 67)
        self.assertEqual(data["HELDOUT"][1], 72)
        expected = {
            "TRAIN": {
                "open_loop_source_plan": ({"VERIFIED_WIN": 23,
                                           "VERIFIED_FAILURE": 77}, 0, 0),
                "revision_aware_feedback": ({"VERIFIED_WIN": 47,
                                             "UNKNOWN": 53}, 702, 0),
                "falsely_stationary_feedback": ({"VERIFIED_WIN": 19,
                                                "UNKNOWN": 81}, 602, 89),
            },
            "HELDOUT": {
                "open_loop_source_plan": ({"VERIFIED_WIN": 17,
                                           "VERIFIED_FAILURE": 83}, 0, 0),
                "revision_aware_feedback": ({"VERIFIED_WIN": 40,
                                             "UNKNOWN": 60}, 733, 0),
                "falsely_stationary_feedback": ({"VERIFIED_WIN": 14,
                                                "UNKNOWN": 86}, 630, 89),
            },
        }
        for split, measures in expected.items():
            for policy, (verdicts, total_probes, contradictions) in measures.items():
                observed, probe_counts, contradicted = data[split][0][policy]
                self.assertEqual(observed, verdicts)
                self.assertEqual(sum(count * frequency for count, frequency in
                                     probe_counts.items()), total_probes)
                self.assertEqual(contradicted, contradictions)
        # False stationarity is intentionally a known-bad source assumption,
        # not a neural model or novel withheld family. Public `switch_at`
        # supplies the correct change point to the revision-aware policy.

    def test_seed_split_does_not_create_independent_mechanic_families(self):
        train = [make_case(seed) for seed in range(100)]
        heldout = [make_case(seed) for seed in range(100, 200)]
        train_families = {(c.before.family, c.after.family) for c in train}
        heldout_families = {(c.before.family, c.after.family) for c in heldout}
        self.assertTrue(heldout_families <= train_families)
        self.assertEqual(len(heldout_families), 24)
        self.assertEqual(len({c.mechanic_signature for c in heldout}), 100)
        # Mechanic signatures vary, but all are generated from five published
        # function templates; independent novel-family effective n is UNKNOWN.


if __name__ == "__main__":
    unittest.main()
