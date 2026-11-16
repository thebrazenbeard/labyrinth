"""Novel Router public-only no-weight baselines and negative controls."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import inspect
import json
import unittest
from unittest import mock

from labyrinth import novel_router_public_controls as dut
from labyrinth.switchback_novel_transfer import ProbeSession, make_case, search


class _PublicOnly:
    """Policy receives methods only, not examiner object or private Case."""
    __slots__ = ("view", "probe", "submit")

    def __init__(self, session):
        self.view = session.view
        self.probe = session.probe
        self.submit = session.submit


_POLICIES = (dut.blind_delayed_delivery, dut.deterministic_trial_witness,
             dut.seeded_feedback_witness)


def _run(seed, policy):
    session = ProbeSession(make_case(seed))
    return policy(_PublicOnly(session)), session


def _digest_ok(receipt):
    if type(receipt) is not dict or type(receipt.get("sha256")) is not str:
        return False
    unsigned = {key: val for key, val in receipt.items() if key != "sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, ensure_ascii=True,
                           allow_nan=False, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest() == receipt["sha256"]


class NovelRouterPublicOnlyControlsTests(unittest.TestCase):
    def test_policy_source_forbids_case_seed_or_privileged_solver(self):
        source = inspect.getsource(dut)
        for forbidden in ("make_case(", "search(", "adjudicate(", "._case",
                          ".before", ".after", "case_id", "mechanic_signature"):
            self.assertNotIn(forbidden, source)
        for policy in _POLICIES:
            for seed in (0, 1, 13, 44, 100, 117, 199):
                examiner = ProbeSession(make_case(seed))
                public = _PublicOnly(examiner)
                self.assertFalse(hasattr(public, "_case"))
                with mock.patch("labyrinth.switchback_novel_transfer.search",
                                side_effect=AssertionError("privileged search forbidden")):
                    receipt = policy(public)
                self.assertTrue(_digest_ok(receipt.outcome))
                self.assertEqual(receipt.optimizer_updates, 0)
                self.assertEqual(receipt.model_checkpoint, "NONE")
                self.assertFalse(receipt.neural_learning_established)
                self.assertFalse(receipt.independent_custody)
                self.assertEqual(receipt.graduation_authority, "none")
                self.assertEqual(tuple(item["action"] for item in
                                       examiner.view()["observations"]),
                                 receipt.probes)
                self.assertTrue(all(item["valid"] is True and item["applied"] is True
                                    for item in examiner.view()["observations"]))
                self.assertLessEqual(len(receipt.probes), 20)
                self.assertLessEqual(len(receipt.plan or ()), 11)
                self.assertIn(receipt.verdict,
                              ("VERIFIED_WIN", "VERIFIED_FAILURE", "UNKNOWN"))

    def test_feedback_only_commits_observed_witness_and_never_claims_unsat(self):
        for policy in (dut.deterministic_trial_witness,
                       dut.seeded_feedback_witness):
            for seed in range(25):
                first, _ = _run(seed, policy)
                second, _ = _run(seed, policy)
                self.assertEqual((first.verdict, first.probes, first.plan),
                                 (second.verdict, second.probes, second.plan))
                self.assertIn(first.proposal, ("PLAN", "ABSTAIN"))
                self.assertNotEqual(first.verdict, "PROVEN_UNSAT_UNDER_DECLARED_RULES")
                self.assertNotEqual(first.verdict, "NOVEL_RULESET_CHALLENGE")
                if first.proposal == "PLAN":
                    self.assertEqual(first.verdict, "VERIFIED_WIN")
                    self.assertEqual(first.plan[-1], "COMMIT")
                else:
                    self.assertEqual(first.verdict, "UNKNOWN")
        with self.assertRaises(ValueError):
            _run(0, lambda port: dut.seeded_feedback_witness(port, policy_seed=True))

    def test_fixed_train_and_disjoint_heldout_denominators(self):
        data = {}
        for start, label in ((0, "TRAIN"), (100, "HELDOUT")):
            outcome = {}
            for policy in _POLICIES:
                counts = Counter()
                probes = Counter()
                invalid = 0
                for seed in range(start, start + 100):
                    receipt, session = _run(seed, policy)
                    self.assertTrue(_digest_ok(receipt.outcome))
                    counts[receipt.verdict] += 1
                    probes[len(receipt.probes)] += 1
                    invalid += sum(not row["valid"] or not row["applied"]
                                   for row in session.view()["observations"])
                    self.assertFalse(receipt.neural_learning_established)
                self.assertEqual(sum(counts.values()), 100)
                outcome[policy.__name__] = (counts, probes, invalid)
            winnable = sum(search(make_case(seed))["verdict"] == "VERIFIED_WIN"
                           for seed in range(start, start + 100))
            data[label] = outcome, winnable
        self.assertEqual(sum(data[split][1] for split in data), 124)
        self.assertEqual(data["TRAIN"][1], 62)
        self.assertEqual(data["HELDOUT"][1], 62)
        expected = {
            "TRAIN": {
                "blind_delayed_delivery": ({"VERIFIED_WIN": 5,
                                            "VERIFIED_FAILURE": 95}, 0, 0, 0),
                "deterministic_trial_witness": ({"VERIFIED_WIN": 13,
                                                 "UNKNOWN": 87}, 1513, 4, 20),
                "seeded_feedback_witness": ({"VERIFIED_WIN": 12,
                                             "UNKNOWN": 88}, 749, 4, 10),
            },
            "HELDOUT": {
                "blind_delayed_delivery": ({"VERIFIED_WIN": 6,
                                            "VERIFIED_FAILURE": 94}, 0, 0, 0),
                "deterministic_trial_witness": ({"VERIFIED_WIN": 17,
                                                 "UNKNOWN": 83}, 1456, 4, 20),
                "seeded_feedback_witness": ({"VERIFIED_WIN": 13,
                                             "UNKNOWN": 87}, 720, 4, 10),
            },
        }
        for split, measured in expected.items():
            for policy, (counts, budget_total, minimum, maximum) in measured.items():
                observed, probes, invalid = data[split][0][policy]
                self.assertEqual(observed, counts)
                self.assertEqual(sum(n * frequency for n, frequency in
                                     probes.items()), budget_total)
                self.assertEqual((min(probes), max(probes)), (minimum, maximum))
                self.assertEqual(invalid, 0)
        # 200 different public-seed cases remain a SOURCE-PUBLIC synthetic
        # suite, not 200 independently invented mechanism families.

    def test_equal_resource_budget_pairing_removes_obvious_budget_confound(self):
        train, heldout = {}, {}
        for start, mapping in ((0, train), (100, heldout)):
            for seed in range(start, start + 100):
                c = make_case(seed)  # EVALUATOR selection, never policy input.
                key = (c.switch_at, c.max_loads, c.max_routes)
                mapping.setdefault(key, []).append(seed)
        pairs = [(x, y, budget) for budget in sorted(set(train) | set(heldout))
                 for x, y in zip(sorted(train.get(budget, ())),
                                 sorted(heldout.get(budget, ())))]
        self.assertEqual(len(pairs), 76)
        self.assertEqual(len({p[2] for p in pairs}), 18)
        expected = {
            "TRAIN": (52, {"blind_delayed_delivery": (5, 71, 0),
                           "deterministic_trial_witness": (13, 0, 1121),
                           "seeded_feedback_witness": (12, 0, 556)}),
            "HELDOUT": (47, {"blind_delayed_delivery": (6, 70, 0),
                             "deterministic_trial_witness": (13, 0, 1144),
                             "seeded_feedback_witness": (8, 0, 559)}),
        }
        for index, label in ((0, "TRAIN"), (1, "HELDOUT")):
            ids = [pair[index] for pair in pairs]
            true_win = sum(search(make_case(seed))["verdict"] == "VERIFIED_WIN"
                           for seed in ids)
            self.assertEqual(true_win, expected[label][0])
            for policy in _POLICIES:
                wins = failures = probes = 0
                for seed in ids:
                    receipt, _ = _run(seed, policy)
                    wins += receipt.verdict == "VERIFIED_WIN"
                    failures += receipt.verdict == "VERIFIED_FAILURE"
                    probes += len(receipt.probes)
                self.assertEqual((wins, failures, probes),
                                 expected[label][1][policy.__name__])

    def test_distinct_seed_signatures_do_not_create_novel_independent_families(self):
        observations = []
        for low in (0, 100):
            cases = [make_case(seed) for seed in range(low, low + 100)]
            observations.append((len({c.case_id for c in cases}),
                                 len({c.mechanic_signature for c in cases}),
                                 len({(c.before, c.after) for c in cases}),
                                 len({c.before for c in cases}),
                                 len({c.after for c in cases})))
        self.assertEqual(observations, [(100, 100, 88, 24, 23),
                                        (100, 100, 91, 24, 24)])
        # Source generator is one designed delayed-router class with 24 law
        # parameterizations per phase; distinct hashes are not n_eff=200.

    def test_invalid_probe_and_detached_final_receipt_do_not_fake_history(self):
        session = ProbeSession(make_case(0))
        invalid = session.probe("ILLEGAL")
        self.assertEqual(len(invalid["observations"]), 1)
        self.assertFalse(invalid["observations"][0]["valid"])
        self.assertFalse(invalid["observations"][0]["applied"])
        self.assertEqual(invalid["remaining_probes"], 19)
        receipt = session.submit({"kind": "ABSTAIN"})
        self.assertEqual(receipt["verdict"], "UNKNOWN")
        self.assertEqual(receipt["probe_count"], 1)
        self.assertTrue(_digest_ok(receipt))
        frozen = deepcopy(receipt)
        receipt["probe_trace"][0]["action"] = "LOAD"
        self.assertFalse(_digest_ok(receipt))
        self.assertEqual(session.submit({"kind": "ABSTAIN"}), frozen)
        self.assertTrue(_digest_ok(session.submit({"kind": "ABSTAIN"})))


if __name__ == "__main__":
    unittest.main()
