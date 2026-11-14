"""Opt-in third public Minotaur family: the legacy v1 receipt remains stable."""

from __future__ import annotations

import json
from itertools import combinations
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from labyrinth import exact_cover_tribunal as cover
from labyrinth import minotaur_public_exam as exam


def invoke(*arguments: str, optimized: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if optimized:
        command.append("-O")
    return subprocess.run(
        [*command, "-m", "labyrinth", "minotaur-exam", *arguments],
        text=True, capture_output=True, check=False, timeout=30,
    )


class ExactCoverCLIContractTests(unittest.TestCase):
    def test_legacy_v1_default_is_unchanged(self):
        receipt = exam.run_public_exam(seed=17, variant=2)
        self.assertEqual(receipt["protocol"], "minotaur-source-public-v1")
        self.assertEqual(receipt["parameters"], {"seed": 17, "variant": 2})
        self.assertEqual(receipt["summary"]["families_run"], 2)
        self.assertEqual(receipt["summary"]["cases_run"], 8)
        self.assertNotIn("exact_cover", receipt["families"])

    def test_opt_in_third_mechanic_has_independent_controls(self):
        result = invoke("--seed", "17", "--variant", "2", "--exact-cover")
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["protocol"], "minotaur-source-public-v2")
        self.assertEqual(receipt["parameters"], {
            "seed": 17, "variant": 2, "include_exact_cover": True,
        })
        self.assertEqual(receipt["summary"]["families_run"], 3)
        self.assertEqual(receipt["summary"]["cases_run"], 12)
        self.assertTrue(receipt["summary"]["independent_checks_agree"])
        self.assertEqual(receipt["claim_scope"]["neural_weight_updates"], 0)
        self.assertFalse(receipt["claim_scope"]["structural_transfer_tested"])
        third = receipt["families"]["exact_cover"]
        self.assertEqual(third["challenge_mechanics"], "combinatorial_exact_cover_allocation")
        self.assertEqual(len(third["cases"]), 4)
        self.assertEqual(third["fixed_script_control"]["fixed_script_wins"], 0)
        self.assertEqual(third["fixed_script_control"]["symbolic_wins"], 2)
        self.assertTrue(all(case["independent_oracle_agrees"]
                            for case in third["cases"].values()))
        counts = receipt["summary"]["verdict_counts"]
        self.assertEqual(counts["VERIFIED_WIN"], 7)
        self.assertEqual(counts["PROVEN_UNSAT_UNDER_DECLARED_RULES"], 5)

    def test_opt_in_paraphrases_and_python_o_preserve_mechanics(self):
        first = invoke("--seed", "9", "--variant", "0", "--exact-cover")
        repeat = invoke("--seed", "9", "--variant", "0", "--exact-cover", optimized=True)
        varied = invoke("--seed", "9", "--variant", "7", "--exact-cover")
        for result in (first, repeat, varied):
            self.assertEqual(result.returncode, 0, result.stderr)
        a, b, c = (json.loads(result.stdout) for result in (first, repeat, varied))
        self.assertEqual(a, b)
        for name, case in a["families"]["exact_cover"]["cases"].items():
            other = c["families"]["exact_cover"]["cases"][name]
            self.assertEqual(case["rules_fingerprint"], other["rules_fingerprint"])
            self.assertEqual(case["verdict"], other["verdict"])
            self.assertNotEqual(case["public_prompt"], other["public_prompt"])

    def test_false_solver_win_and_false_unsat_are_vetoed(self):
        unsat = cover.families()["cyclic_deficiency"]
        with patch.object(cover, "search_allocation", return_value={
            "verdict": "VERIFIED_WIN", "allocation": (0, 1),
        }):
            result = exam._exact_cover_case(unsat, variant=0)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["independent_oracle_agrees"])

        solvable = cover.families()["repairable_by_declared_team"]
        with patch.object(cover, "search_allocation", return_value={
            "verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "exhaustive": True,
        }):
            result = exam._exact_cover_case(solvable, variant=0)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["independent_oracle_agrees"])

    def test_node_cap_exhaustion_is_not_relabelled_impossibility(self):
        scenario = cover.families()["cyclic_deficiency"]
        with patch.object(cover, "search_allocation", return_value={
            "verdict": "NO_KNOWN_WIN", "exhaustive": False,
        }):
            result = exam._exact_cover_case(scenario, variant=0)
        self.assertEqual(result["verdict"], "NO_KNOWN_WIN")
        self.assertFalse(result["independent_oracle_agrees"])

    def test_unserializable_fixed_script_metadata_does_not_crash_receipt(self):
        scenario = cover.families()["cyclic_deficiency"]
        with patch.object(cover, "verify_allocation", return_value={"verdict": object()}):
            result = exam._exact_cover_case(scenario, variant=0)
        self.assertEqual(result["fixed_script_verdict"], "INVALID_FIXED_SCRIPT_VERDICT")
        json.dumps(result)

    def test_existence_oracle_cannot_certify_false_specific_witness(self):
        # A true existential answer proves only SOME allocation exists, not
        # that a solver's particular allocation satisfies the declared rules.
        scenario = cover.families()["repairable_by_declared_team"]
        invalid = (0, 1)
        self.assertEqual(cover.verify_allocation(scenario, invalid)["verdict"],
                         "VERIFIED_FAILURE")
        self.assertTrue(cover.independent_subset_oracle(scenario))
        with (patch.object(cover, "search_allocation", return_value={
                "verdict": "VERIFIED_WIN", "allocation": invalid,
                "exhaustive": False, "searched_states": 1,
              }),
              patch.object(cover, "verify_allocation", return_value={
                  "verdict": "VERIFIED_WIN",
              })):
            result = exam._exact_cover_case(scenario, variant=0)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["independent_oracle_agrees"])
        self.assertIsNone(result["symbolic_allocation"])

    def test_two_correlated_false_unsat_claims_are_vetoed_by_third_dp(self):
        scenario = cover.families()["repairable_by_declared_team"]
        self.assertTrue(cover.independent_subset_oracle(scenario))
        with (patch.object(cover, "search_allocation", return_value={
                "verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "exhaustive": True,
              }),
              patch.object(cover, "independent_subset_oracle", return_value=False)):
            result = exam._exact_cover_case(scenario, variant=0)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["independent_oracle_agrees"])

    def test_independent_dp_matches_subsets_on_seventy_two_new_graphs(self):
        for site_count in (3, 6, 9, 12, 15, 18):
            sites = tuple(chr(ord("a") + i) for i in range(site_count))
            available = tuple(combinations(sites, 3))
            for seed in range(12):
                rng = random.Random(site_count * 1000 + seed)
                crews = tuple(rng.sample(available, k=min(len(available), 1 + seed)))
                scenario = cover.Scenario(sites, crews, min(6, 1 + seed // 2))
                self.assertEqual(exam._independent_cover_exists_dp(scenario),
                                 cover.independent_subset_oracle(scenario),
                                 (site_count, seed))

    def test_independent_dp_cap_is_unknown_not_proven_unsat(self):
        scenario = cover.families()["cyclic_deficiency"]
        with (patch.object(cover, "search_allocation", return_value={
                "verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "exhaustive": True,
              }),
              patch.object(exam, "_independent_cover_exists_dp", return_value=None)):
            result = exam._exact_cover_case(scenario, variant=0)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["independent_oracle_agrees"])


if __name__ == "__main__":
    unittest.main()
