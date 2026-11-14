"""Opt-in fourth source-public Minotaur mechanic, without changing v1/v2."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from labyrinth import minotaur_public_exam as exam
from labyrinth import temporal_repair_tribunal as temporal


def invoke(*args: str, optimized: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if optimized:
        command.append("-O")
    return subprocess.run(
        [*command, "-B", "-m", "labyrinth", "minotaur-exam", *args],
        capture_output=True, text=True, check=False, timeout=30,
    )


class TemporalRepairCLIContractTests(unittest.TestCase):
    def test_temporal_repair_is_explicit_opt_in_fourth_mechanic(self):
        result = invoke("--seed", "17", "--variant", "2",
                        "--exact-cover", "--temporal-repair")
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["protocol"], "minotaur-source-public-v3")
        self.assertEqual(receipt["summary"]["families_run"], 4)
        self.assertEqual(receipt["summary"]["cases_run"], 16)
        self.assertTrue(receipt["summary"]["independent_checks_agree"])
        self.assertIn("temporal_repair", receipt["families"])
        self.assertEqual(len(receipt["families"]["temporal_repair"]["cases"]), 4)
        self.assertEqual(receipt["claim_scope"]["neural_weight_updates"], 0)
        self.assertFalse(receipt["claim_scope"]["structural_transfer_tested"])

    def test_temporal_repair_requires_explicit_prior_family_opt_in(self):
        result = invoke("--temporal-repair")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--exact-cover", result.stderr)

    def test_previous_v1_and_v2_protocols_unchanged(self):
        v1 = invoke("--seed", "17", "--variant", "2")
        v2 = invoke("--seed", "17", "--variant", "2", "--exact-cover")
        self.assertEqual(v1.returncode, 0, v1.stderr)
        self.assertEqual(v2.returncode, 0, v2.stderr)
        a, b = json.loads(v1.stdout), json.loads(v2.stdout)
        self.assertEqual((a["protocol"], a["summary"]["cases_run"]),
                         ("minotaur-source-public-v1", 8))
        self.assertEqual((b["protocol"], b["summary"]["cases_run"]),
                         ("minotaur-source-public-v2", 12))
        self.assertNotIn("temporal_repair", a["families"])
        self.assertNotIn("temporal_repair", b["families"])

    def test_temporal_paraphrases_preserve_rules_not_holdout_independence(self):
        first = invoke("--seed", "9", "--variant", "0",
                       "--exact-cover", "--temporal-repair")
        other = invoke("--seed", "9", "--variant", "7",
                       "--exact-cover", "--temporal-repair")
        optimized = invoke("--seed", "9", "--variant", "0",
                           "--exact-cover", "--temporal-repair", optimized=True)
        for result in (first, other, optimized):
            self.assertEqual(result.returncode, 0, result.stderr)
        a, b, c = [json.loads(result.stdout) for result in
                   (first, other, optimized)]
        self.assertEqual(a, c)
        for name, case in a["families"]["temporal_repair"]["cases"].items():
            changed = b["families"]["temporal_repair"]["cases"][name]
            self.assertEqual(case["rules_fingerprint"], changed["rules_fingerprint"])
            self.assertEqual(case["verdict"], changed["verdict"])
            self.assertNotEqual(case["public_prompt"], changed["public_prompt"])

    def test_false_solver_win_is_vetoed(self):
        s = temporal.families()["dependency_cycle"]
        with patch.object(temporal, "search_schedule", return_value={
            "verdict": "VERIFIED_WIN", "schedule": (0, 1, 2),
            "rules_fingerprint": temporal.fingerprint(s),
        }):
            result = exam._temporal_repair_case(s, variant=0)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["independent_oracle_agrees"])
        self.assertIsNone(result["symbolic_schedule"])

    def test_existence_plus_correlated_fake_replay_cannot_certify_bad_witness(self):
        s = temporal.families()["recharge_window"]
        bad = (0, 1, 2)  # Missing required WAIT/release constraints.
        self.assertEqual(temporal.verify_schedule(s, bad)["verdict"],
                         "VERIFIED_FAILURE")
        self.assertTrue(temporal.independent_reachability(s))
        with (patch.object(temporal, "search_schedule", return_value={
                  "verdict": "VERIFIED_WIN", "schedule": bad,
                  "rules_fingerprint": temporal.fingerprint(s),
              }),
              patch.object(temporal, "verify_schedule", return_value={
                  "verdict": "VERIFIED_WIN",
                  "rules_fingerprint": temporal.fingerprint(s),
              })):
            result = exam._temporal_repair_case(s, variant=0)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["independent_specific_witness_valid"])
        self.assertIsNone(result["symbolic_schedule"])

    def test_two_false_unsat_oracles_cannot_overrule_existing_win(self):
        s = temporal.families()["recharge_window"]
        self.assertTrue(temporal.independent_depth_enumeration(s))
        with (patch.object(temporal, "search_schedule", return_value={
                  "verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES",
                  "exhaustive": True, "rules_fingerprint": temporal.fingerprint(s),
              }),
              patch.object(temporal, "independent_reachability", return_value=False)):
            result = exam._temporal_repair_case(s, variant=0)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertTrue(result["independent_dfs_has_win_on_unsat_claim"])

    def test_exhausted_search_is_not_false_unsat(self):
        s = temporal.families()["dependency_cycle"]
        with patch.object(temporal, "search_schedule", return_value={
            "verdict": "NO_KNOWN_WIN", "exhaustive": False,
            "rules_fingerprint": temporal.fingerprint(s),
        }):
            result = exam._temporal_repair_case(s, variant=0)
        self.assertEqual(result["verdict"], "NO_KNOWN_WIN")
        self.assertFalse(result["independent_oracle_agrees"])

    def test_invalid_solver_status_does_not_program_verdict_comparison(self):
        class DeceptiveStatus(str):
            def __eq__(self, other):
                return other == "VERIFIED_WIN"

        s = temporal.families()["recharge_window"]
        with patch.object(temporal, "search_schedule", return_value={
            "verdict": DeceptiveStatus("VERIFIED_FAILURE"),
            "schedule": (0, -1, 1, 2),
            "rules_fingerprint": temporal.fingerprint(s),
        }):
            result = exam._temporal_repair_case(s, variant=0)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["solver_verdict"], "INVALID_SOLVER_VERDICT")
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
