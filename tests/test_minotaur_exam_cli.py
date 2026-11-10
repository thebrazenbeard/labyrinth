"""Public CLI contracts: challenge execution, not neural learning claims."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from labyrinth import constant_exam
from labyrinth import constant_exam_verifier as constant_verifier
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
    def test_replay_mutation_cannot_swap_verified_public_policy(self):
        """Verification must bind to exactly the witness emitted in the receipt."""
        original_find = veiled_fork.find_guaranteed_policy
        original_replay = veiled_fork.replay_policy
        captured = {}

        def capture_solver(scenario):
            answer = original_find(scenario)
            captured["policy"] = answer["policy"]
            return answer

        def mutating_replay(scenario, policy):
            result = original_replay(scenario, policy)
            if policy is captured.get("policy"):
                policy.clear()
                policy.update({"kind": "commit", "exit": 999})
            return result

        scenario = veiled_fork.families()["identifiable_fork"]
        with patch.object(veiled_fork, "find_guaranteed_policy", side_effect=capture_solver), \
                patch.object(veiled_fork, "replay_policy", side_effect=mutating_replay):
            receipt = minotaur_public_exam._fork_case(scenario)
        serialized = json.loads(json.dumps(receipt))
        self.assertEqual(serialized["verdict"], "VERIFIED_WIN")
        self.assertTrue(serialized["independent_oracle_agrees"])
        self.assertEqual(
            original_replay(scenario, serialized["symbolic_policy"])["verdict"],
            "VERIFIED_WIN",
        )

    def test_unserializable_solver_metadata_does_not_break_fork_receipt(self):
        original = veiled_fork.find_guaranteed_policy

        def poisoned_solver(scenario):
            return {**original(scenario), "searched_belief_states": object()}

        with patch.object(veiled_fork, "find_guaranteed_policy", side_effect=poisoned_solver):
            receipt = minotaur_public_exam._fork_case(
                veiled_fork.families()["unobservable_fork"],
            )
        serialized = json.loads(json.dumps(receipt))
        self.assertIsNone(serialized["searched_belief_states"])
        self.assertEqual(serialized["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")

    def test_invalid_constant_witness_and_expansion_metadata_fail_closed(self):
        forged = constant_exam.Search(
            "UNSOLVED", witness=(object(),), expansions=object(),
        )
        with patch.object(constant_exam, "search", return_value=forged):
            receipt = minotaur_public_exam._constant_case(17, 0, "apparently_sealed")
        serialized = json.loads(json.dumps(receipt))
        self.assertEqual(serialized["verdict"], "UNKNOWN")
        self.assertEqual(serialized["solver_status"], "INVALID_SOLVER_WITNESS")
        self.assertEqual(serialized["witness_actions"], [])
        self.assertIsNone(serialized["finite_states_explored"])

    def test_deceptive_independent_trace_cannot_certify_fake_win(self):
        class FalseIndependent(str):
            def __eq__(self, other):
                return other == "VERIFIED_WIN"

        fake = type("Result", (), {"outcome": FalseIndependent("VERIFIED_FAILURE")})()
        with patch.object(constant_verifier, "verify_trace", return_value=fake):
            receipt = minotaur_public_exam._constant_case(17, 0, "apparently_sealed")
        serialized = json.loads(json.dumps(receipt))
        self.assertEqual(serialized["verdict"], "UNKNOWN")
        self.assertFalse(serialized["independent_check_agrees"])
        self.assertEqual(serialized["independent_verdict"], "INVALID_INDEPENDENT_VERDICT")
        self.assertEqual(serialized["fixed_script_verdict"], "INVALID_FIXED_SCRIPT_VERDICT")

    def test_nonboolean_unsat_certificate_cannot_claim_proof(self):
        with patch.object(constant_verifier, "verify_unsat_certificate", return_value=1):
            receipt = minotaur_public_exam._constant_case(17, 0, "sealed_control")
        serialized = json.loads(json.dumps(receipt))
        self.assertEqual(serialized["verdict"], "UNKNOWN")
        self.assertFalse(serialized["independent_certificate_valid"])
        self.assertFalse(serialized["independent_check_agrees"])

    def test_deceptive_fork_fixed_script_receipt_is_rejected(self):
        class FalseReplay(str):
            def __eq__(self, other):
                return other == "VERIFIED_WIN"

        truthful = veiled_fork.replay_policy

        def poisoned_replay(scenario, policy):
            result = truthful(scenario, policy)
            if policy == {"kind": "commit", "exit": 0}:
                return {"verdict": FalseReplay("VERIFIED_FAILURE")}
            return result

        with patch.object(veiled_fork, "replay_policy", side_effect=poisoned_replay):
            receipt = minotaur_public_exam._fork_case(
                veiled_fork.families()["unobservable_fork"],
            )
        serialized = json.loads(json.dumps(receipt))
        self.assertEqual(serialized["fixed_script_verdict"], "INVALID_FIXED_SCRIPT_VERDICT")

    def test_deceptive_constant_solver_status_cannot_claim_independent_agreement(self):
        class FalseStatus(str):
            def __eq__(self, other):
                return other == "VERIFIED_WIN"

        exam = constant_exam.make_exam(17, kind="apparently_sealed")
        truthful = constant_exam.search(exam)
        fake = constant_exam.Search(
            FalseStatus("UNSAT_IN_DECLARED_FINITE_MODEL"), truthful.witness,
            truthful.reachable, truthful.expansions,
        )
        with patch.object(constant_exam, "search", return_value=fake):
            result = minotaur_public_exam._constant_case(17, 0, "apparently_sealed")
        serialized = json.loads(json.dumps(result))
        self.assertEqual(serialized["solver_status"], "INVALID_SOLVER_STATUS")
        self.assertEqual(serialized["verdict"], "UNKNOWN")
        self.assertFalse(serialized["independent_check_agrees"])

    def test_deceptive_solver_verdict_cannot_turn_unsat_into_verified_win(self):
        """A Python str subclass must not disagree with its JSON receipt."""

        class FalseWin(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                return other == "PROVEN_UNSAT_UNDER_DECLARED_RULES"

        scenario = veiled_fork.families()["unobservable_fork"]
        with patch.object(veiled_fork, "find_guaranteed_policy", return_value={
                "verdict": FalseWin("VERIFIED_WIN"), "exhaustive": True,
        }):
            result = minotaur_public_exam._fork_case(scenario)
        serialized = json.loads(json.dumps(result))
        self.assertFalse(serialized["independent_oracle_exists"])
        self.assertFalse(serialized["independent_oracle_agrees"])
        self.assertEqual(serialized["verdict"], "UNKNOWN")

    def test_deceptive_replay_verdict_cannot_certify_losing_policy(self):
        class LyingReplay(str):
            def __eq__(self, other):
                return other == "VERIFIED_WIN"

        scenario = veiled_fork.families()["identifiable_fork"]
        fake_solver = {"verdict": "VERIFIED_WIN",
                       "policy": {"kind": "commit", "exit": 0}}
        fake_replay = {"verdict": LyingReplay("VERIFIED_FAILURE")}
        with patch.object(veiled_fork, "find_guaranteed_policy", return_value=fake_solver), \
                patch.object(veiled_fork, "replay_policy", return_value=fake_replay):
            result = minotaur_public_exam._fork_case(scenario)
        serialized = json.loads(json.dumps(result))
        self.assertFalse(serialized["independent_oracle_agrees"])
        self.assertEqual(serialized["verdict"], "UNKNOWN")
        self.assertEqual(serialized["independent_replay_verdict"], "INVALID_REPLAY_VERDICT")

    def test_nonboolean_oracle_result_cannot_claim_independent_agreement(self):
        class LyingOracle(int):
            def __bool__(self):
                return True

        scenario = veiled_fork.families()["identifiable_fork"]
        with patch.object(veiled_fork, "brute_force_oracle", return_value=LyingOracle(0)):
            result = minotaur_public_exam._fork_case(scenario)
        serialized = json.loads(json.dumps(result))
        self.assertFalse(serialized["independent_oracle_exists"])
        self.assertFalse(serialized["independent_oracle_agrees"])
        self.assertEqual(serialized["verdict"], "UNKNOWN")

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
