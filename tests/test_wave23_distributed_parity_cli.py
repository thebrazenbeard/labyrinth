"""The fifth mechanic is opt-in and cannot change legacy Minotaur receipts."""

import json
import contextlib
import io
import subprocess
import sys
import unittest
from unittest.mock import patch

from labyrinth.minotaur_public_exam import run_public_exam
from labyrinth.cli import main


class DistributedParityCLIIntegrationTests(unittest.TestCase):
    def _cli(self, *options):
        return subprocess.run([sys.executable, "-m", "labyrinth", "minotaur-exam",
                               "--seed", "17", "--variant", "2", *options],
                              capture_output=True, text=True, timeout=30)

    def test_existing_three_versions_remain_byte_identical(self):
        for options, config in (
                ((), {}),
                (("--exact-cover",), {"include_exact_cover": True}),
                (("--exact-cover", "--temporal-repair"),
                 {"include_exact_cover": True, "include_temporal_repair": True})):
            with self.subTest(options=options):
                run = self._cli(*options)
                self.assertEqual(run.returncode, 0, run.stderr)
                expected = json.dumps(run_public_exam(seed=17, variant=2, **config),
                                      indent=2, ensure_ascii=False, sort_keys=True) + "\n"
                self.assertEqual(run.stdout, expected)

    def test_fifth_family_requires_explicit_previous_mechanics(self):
        for options in (("--distributed-parity",),
                        ("--exact-cover", "--distributed-parity"),
                        ("--temporal-repair", "--distributed-parity")):
            with self.subTest(options=options):
                run = self._cli(*options)
                self.assertEqual(run.returncode, 2)
                self.assertIn("requires", run.stderr)

    def test_opt_in_fifth_family_counts_and_no_model_promotion(self):
        run = self._cli("--exact-cover", "--temporal-repair", "--distributed-parity")
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report["protocol"], "minotaur-source-public-v4")
        self.assertEqual(report["summary"]["families_run"], 5)
        self.assertEqual(report["summary"]["cases_run"], 20)
        self.assertEqual(report["summary"]["verdict_counts"], {
            "PROVEN_UNSAT_UNDER_DECLARED_RULES": 9, "VERIFIED_WIN": 11})
        self.assertTrue(report["summary"]["independent_checks_agree"])
        self.assertFalse(report["claim_scope"]["neural_learning_tested"])
        self.assertFalse(report["claim_scope"]["structural_transfer_tested"])
        self.assertEqual(report["claim_scope"]["neural_weight_updates"], 0)
        family = report["families"]["distributed_parity"]
        self.assertEqual(len(family["cases"]), 4)
        self.assertEqual(family["fixed_script_control"]["symbolic_wins"], 2)
        self.assertEqual(family["fixed_script_control"]["fixed_script_wins"], 0)
        for case in family["cases"].values():
            self.assertTrue(case["independent_oracle_agrees"])

    def test_deceptive_source_win_is_demoted_without_pinned_witness_proof(self):
        from labyrinth.distributed_parity_tribunal import families, fingerprint
        from labyrinth.minotaur_public_v4 import parity_case

        scenario = families()["coordinated_parity"]
        forged = {"verdict": "VERIFIED_WIN", "policy": ((0, 0), (0, 0)),
                  "rules_fingerprint": fingerprint(scenario), "policies_examined": 1,
                  "independent_gf2": True, "independent_dfs": True}
        with patch("labyrinth.distributed_parity_tribunal.search_policy", return_value=forged), \
                patch("labyrinth.distributed_parity_tribunal.verify_policy",
                      return_value={"verdict": "VERIFIED_WIN", "rules_fingerprint": fingerprint(scenario)}), \
                patch("labyrinth.distributed_parity_tribunal.independent_witness_check",
                      return_value=True):
            result = parity_case(scenario, variant=2)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["independent_oracle_agrees"])

    def test_cli_exits_nonzero_if_symbolic_verification_disagrees(self):
        false_receipt = {"summary": {"independent_checks_agree": False}}
        with patch("labyrinth.minotaur_public_exam.run_public_exam", return_value=false_receipt):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["minotaur-exam"]), 2)
        with patch("labyrinth.minotaur_public_v4.run_public_exam_v4", return_value=false_receipt):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["minotaur-exam", "--exact-cover",
                                       "--temporal-repair", "--distributed-parity"]), 2)


if __name__ == "__main__":
    unittest.main()
