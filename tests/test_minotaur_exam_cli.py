"""Public CLI contracts: challenge execution, not neural learning claims."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from labyrinth import constant_exam
from labyrinth import minotaur_public_exam
from labyrinth import veiled_fork


def invoke(*arguments: str, optimized: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if optimized:
        command.append("-O")
    return subprocess.run(
        [*command, "-m", "labyrinth", "minotaur-exam", *arguments],
        check=False, capture_output=True, text=True, timeout=30,
    )


class MinotaurExamCLITests(unittest.TestCase):
    def test_public_control_runs_distinct_mechanics_with_independent_checks(self):
        result = invoke("--seed", "17", "--variant", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["protocol"], "minotaur-source-public-v1")
        self.assertEqual(receipt["parameters"], {"seed": 17, "variant": 2})
        self.assertEqual(receipt["claim_scope"]["neural_weight_updates"], 0)
        self.assertFalse(receipt["claim_scope"]["neural_learning_tested"])
        self.assertFalse(receipt["claim_scope"]["hidden_custody_tested"])
        self.assertFalse(receipt["claim_scope"]["structural_transfer_tested"])
        self.assertEqual(receipt["summary"]["families_run"], 2)
        self.assertEqual(receipt["summary"]["cases_run"], 8)
        self.assertTrue(receipt["summary"]["independent_checks_agree"])

        constant = receipt["families"]["constant_exam"]
        self.assertEqual(constant["challenge_mechanics"], "finite_navigation_and_offering")
        self.assertEqual(constant["cases"]["apparently_sealed"]["verdict"], "VERIFIED_WIN")
        self.assertEqual(constant["cases"]["apparently_sealed"]["independent_verdict"], "VERIFIED_WIN")
        self.assertTrue(constant["cases"]["apparently_sealed"]["witness_actions"])
        self.assertNotEqual(constant["cases"]["apparently_sealed"]["fixed_script_verdict"], "VERIFIED_WIN")
        self.assertEqual(constant["cases"]["sealed_control"]["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertTrue(constant["cases"]["sealed_control"]["independent_certificate_valid"])
        self.assertNotEqual(constant["cases"]["apparently_sealed"]["semantic_family_id"],
                            constant["cases"]["sealed_control"]["semantic_family_id"])
        self.assertEqual(len(constant["source_fingerprint"]), 64)

        fork = receipt["families"]["veiled_fork"]
        self.assertEqual(fork["challenge_mechanics"], "contingent_epistemic_planning")
        self.assertEqual(len(fork["cases"]), 6)
        self.assertEqual(fork["cases"]["identifiable_fork"]["verdict"], "VERIFIED_WIN")
        self.assertEqual(fork["cases"]["unobservable_fork"]["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertEqual(fork["cases"]["too_late_to_know"]["verdict"],
                         "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        self.assertTrue(all(item["independent_oracle_agrees"] for item in fork["cases"].values()))
        self.assertTrue(all(len(item["rules_fingerprint"]) == 64
                            for item in fork["cases"].values()))
        self.assertNotIn("world_traces", result.stdout)
        self.assertNotIn("world_exits", result.stdout)

    def test_reproducibility_and_honest_paraphrase_scope(self):
        first = invoke("--seed", "17", "--variant", "0")
        repeat = invoke("--seed", "17", "--variant", "0", optimized=True)
        varied = invoke("--seed", "17", "--variant", "3")
        for run in (first, repeat, varied):
            self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(first.stdout), json.loads(repeat.stdout))
        same = json.loads(first.stdout)
        paraphrase = json.loads(varied.stdout)
        for name in ("apparently_sealed", "sealed_control"):
            self.assertEqual(same["families"]["constant_exam"]["cases"][name]["semantic_family_id"],
                             paraphrase["families"]["constant_exam"]["cases"][name]["semantic_family_id"])
            self.assertEqual(same["families"]["constant_exam"]["cases"][name]["witness_actions"],
                             paraphrase["families"]["constant_exam"]["cases"][name]["witness_actions"])
        self.assertEqual(same["families"]["veiled_fork"], paraphrase["families"]["veiled_fork"])

    def test_inputs_are_bounded_and_fail_before_receipt(self):
        for arguments in (("--variant", "-1"), ("--variant", "100000"),
                          ("--seed", "not-an-int"), ("--unknown", "yes")):
            result = invoke(*arguments)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)

    def test_report_write_once_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "public.json"
            first = invoke("--seed", "9", "--report", str(target))
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(first.stdout, "")
            original = target.read_bytes()
            self.assertEqual(json.loads(original)["summary"]["cases_run"], 8)
            second = invoke("--seed", "11", "--report", str(target))
            self.assertEqual(second.returncode, 2)
            self.assertEqual(target.read_bytes(), original)

    def test_forged_solver_win_cannot_bypass_independent_witness(self):
        fake = constant_exam.Search("VERIFIED_WIN", ("north", "north", "leave"))
        with patch.object(constant_exam, "search", return_value=fake):
            result = minotaur_public_exam._constant_case(17, 0, "apparently_sealed")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["independent_verdict"], "NOT_WON")
        self.assertFalse(result["independent_check_agrees"])

    def test_forged_unsat_certificate_rejected(self):
        fake = constant_exam.Search(
            "UNSAT_IN_DECLARED_FINITE_MODEL", reachable=frozenset({constant_exam.State()}),
        )
        with patch.object(constant_exam, "search", return_value=fake):
            result = minotaur_public_exam._constant_case(17, 0, "sealed_control")
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["independent_verdict"], "CLOSURE_REJECTED")
        self.assertFalse(result["independent_certificate_valid"])

    def test_false_veiled_unsat_and_winning_policy_rejected_by_oracle(self):
        scenario = veiled_fork.families()["identifiable_fork"]
        fake_unsat = {"verdict": "PROVEN_UNSAT_UNDER_DECLARED_RULES", "exhaustive": True}
        with patch.object(veiled_fork, "find_guaranteed_policy", return_value=fake_unsat):
            result = minotaur_public_exam._fork_case(scenario)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertFalse(result["independent_oracle_agrees"])

        fake_win = {"verdict": "VERIFIED_WIN", "policy": {"kind": "commit", "exit": 0}}
        with patch.object(veiled_fork, "find_guaranteed_policy", return_value=fake_win):
            result = minotaur_public_exam._fork_case(scenario)
        self.assertEqual(result["verdict"], "UNKNOWN")
        self.assertEqual(result["independent_replay_verdict"], "VERIFIED_FAILURE")
        self.assertFalse(result["independent_oracle_agrees"])


if __name__ == "__main__":
    unittest.main()
