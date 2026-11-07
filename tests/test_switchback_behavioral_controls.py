"""Exact, non-neural behavioral controls for future Switchback comparisons."""

from collections import Counter, defaultdict
import json
import unittest
from unittest import mock

from labyrinth.switchback_behavioral_controls import (
    blind_memorized_plan, deterministic_zero_witness, seeded_feedback_witness,
)
from labyrinth.switchback_circuit import MAX_PROBES, ProbeSession, make_case, public_view
from labyrinth.switchback_oracle import search, verify_receipt_replay


class _CandidatePort:
    """Only methods exposed to policy code; same-process is not host custody."""

    __slots__ = ("view", "probe", "submit")

    def __init__(self, session):
        self.view = session.view
        self.probe = session.probe
        self.submit = session.submit


def _run(seed, policy, *, policy_seed=17):
    session = ProbeSession(make_case(seed))
    candidate = _CandidatePort(session)
    if policy is seeded_feedback_witness:
        result = policy(candidate, policy_seed=policy_seed)
    else:
        result = policy(candidate)
    return result, session


class SwitchbackWave15BehavioralControls(unittest.TestCase):
    def test_no_hidden_search_or_rule_access_during_behavioral_policy(self):
        for policy in (blind_memorized_plan, deterministic_zero_witness,
                       seeded_feedback_witness):
            for seed in (0, 13, 14, 44, 100, 117, 199):
                with self.subTest(policy=policy.__name__, seed=seed):
                    with mock.patch("labyrinth.switchback_oracle.search",
                                    side_effect=AssertionError("privileged search forbidden")):
                        receipt, session = _run(seed, policy)
                    self.assertTrue(verify_receipt_replay(receipt.outcome))
                    self.assertEqual(receipt.optimizer_updates, 0)
                    self.assertEqual(receipt.model_checkpoint, "NONE")
                    self.assertFalse(receipt.neural_learning_established)
                    self.assertEqual(receipt.graduation_authority, "none")
                    self.assertIn(receipt.verdict,
                                  ("VERIFIED_WIN", "VERIFIED_FAILURE", "UNKNOWN"))
                    self.assertEqual(tuple(item["action"] for item in
                                           session.view()["observations"]),
                                     receipt.probe_actions)
                    self.assertLessEqual(len(receipt.probe_actions), MAX_PROBES)

    def test_public_feedback_and_fixed_are_reproducible_and_fail_closed(self):
        for policy in (deterministic_zero_witness, seeded_feedback_witness):
            for seed in range(20):
                first, _ = _run(seed, policy)
                second, _ = _run(seed, policy)
                self.assertEqual(first.verdict, second.verdict)
                self.assertEqual(first.probe_actions, second.probe_actions)
                self.assertEqual(first.submitted_plan, second.submitted_plan)
                self.assertIn(first.proposal, ("ABSTAIN", "PLAN"))
                self.assertNotEqual(first.verdict, "VERIFIED_FAILURE")
                if first.proposal == "PLAN":
                    self.assertEqual(first.verdict, "VERIFIED_WIN")
                    self.assertLessEqual(len(first.submitted_plan), 9)
                else:
                    self.assertEqual(first.verdict, "UNKNOWN")
        with self.assertRaises(ValueError):
            _run(0, seeded_feedback_witness, policy_seed=True)
        with self.assertRaises(ValueError):
            _run(0, seeded_feedback_witness, policy_seed=-1)

    def test_fixed_train_vs_disjoint_heldout_measured_denominators_and_budgets(self):
        results = {}
        for start, label in ((0, "TRAIN_0_TO_99"), (100, "HELDOUT_100_TO_199")):
            by_policy = {}
            for policy in (blind_memorized_plan, deterministic_zero_witness,
                           seeded_feedback_witness):
                outcomes = Counter()
                probe_count = Counter()
                action_budgets = []
                for seed in range(start, start + 100):
                    result, _ = _run(seed, policy)
                    self.assertTrue(verify_receipt_replay(result.outcome))
                    outcomes[result.verdict] += 1
                    probe_count[len(result.probe_actions)] += 1
                    action_budgets.append(len(result.submitted_plan or ()))
                    self.assertEqual(result.optimizer_updates, 0)
                    self.assertEqual(result.model_checkpoint, "NONE")
                self.assertEqual(sum(outcomes.values()), 100)
                self.assertTrue(all(count <= MAX_PROBES for count in probe_count))
                by_policy[policy.__name__] = (outcomes, probe_count,
                                             max(action_budgets))
            truth_winnable = sum(search(make_case(seed))["verdict"] == "VERIFIED_WIN"
                                 for seed in range(start, start + 100))
            results[label] = (by_policy, truth_winnable)
        # Freeze exact metrics after reproducible source-only execution.
        self.assertEqual(results["TRAIN_0_TO_99"][1], 67)
        self.assertEqual(results["HELDOUT_100_TO_199"][1], 72)
        self.assertEqual(results["TRAIN_0_TO_99"][0]["blind_memorized_plan"][0],
                         {"VERIFIED_WIN": 7, "VERIFIED_FAILURE": 93})
        self.assertEqual(results["TRAIN_0_TO_99"][0]["blind_memorized_plan"][1],
                         {0: 100})
        expected = {
            "TRAIN_0_TO_99": {
                "blind_memorized_plan": ({"VERIFIED_WIN": 7,
                                           "VERIFIED_FAILURE": 93}, 0, 0, 0),
                "deterministic_zero_witness": ({"VERIFIED_WIN": 31,
                                                "UNKNOWN": 69}, 515, 3, 8),
                "seeded_feedback_witness": ({"VERIFIED_WIN": 32,
                                             "UNKNOWN": 68}, 506, 3, 8),
            },
            "HELDOUT_100_TO_199": {
                "blind_memorized_plan": ({"VERIFIED_WIN": 1,
                                           "VERIFIED_FAILURE": 99}, 0, 0, 0),
                "deterministic_zero_witness": ({"VERIFIED_WIN": 18,
                                                "UNKNOWN": 82}, 547, 3, 8),
                "seeded_feedback_witness": ({"VERIFIED_WIN": 23,
                                             "UNKNOWN": 77}, 536, 3, 8),
            },
        }
        for split, values in expected.items():
            for name, (counts, total_probes, minimum, maximum) in values.items():
                observed_counts, probe_frequency, max_plan = results[split][0][name]
                self.assertEqual(observed_counts, counts)
                self.assertEqual(sum(n * frequency for n, frequency in
                                     probe_frequency.items()), total_probes)
                self.assertEqual(min(probe_frequency), minimum)
                self.assertEqual(max(probe_frequency), maximum)
                self.assertLessEqual(max_plan, 9)
        self.assertTrue(all(policy_outcomes[0]["VERIFIED_FAILURE"] == 0 for name,
                            policy_outcomes in results["HELDOUT_100_TO_199"][0].items()
                            if name != "blind_memorized_plan"))

    def test_published_heldout_seed_catalogue_is_contamination_not_holdout(self):
        # Test-only source-public leakage ceiling. Here the supposedly heldout
        # cases are expressly included in an offline catalog *before* test.
        entries = defaultdict(list)
        for seed in range(100, 200):
            case = make_case(seed)
            public = json.dumps(public_view(case), sort_keys=True)
            observed = ProbeSession(case).probe("1")["current_register"]
            hidden_search = search(case)  # Offline, never target runtime.
            proposal = ({"kind": "PLAN", "actions": hidden_search["witness"]}
                        if hidden_search["verdict"] == "VERIFIED_WIN"
                        else {"kind": "CLAIM_UNSAT"})
            entries[(public, observed)].append(proposal)
        self.assertEqual(len(entries), 100)
        successes = Counter()
        advertised_keys = ("prompt", "initial_register", "target_register",
                           "plan_actions", "probe_actions", "maximum_moves",
                           "switch_at", "maximum_pulses")
        for seed in range(100, 200):
            session = ProbeSession(make_case(seed))
            shown = session.view()
            initial_public = json.dumps({field: shown[field]
                                         for field in advertised_keys}, sort_keys=True)
            first_effect = session.probe("1")["current_register"]
            candidates = entries.get((initial_public, first_effect), [])
            self.assertEqual(len(candidates), 1)
            result = session.submit(candidates[0])
            self.assertTrue(verify_receipt_replay(result))
            successes[result["verdict"]] += 1
        self.assertEqual(successes, {"VERIFIED_WIN": 72,
                                    "PROVEN_UNSAT_UNDER_DECLARED_RULES": 28})
        # 100/100 catalogue reuse with one probe is NOT independent success.


if __name__ == "__main__":
    unittest.main()
