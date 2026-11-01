"""Offline cognitive TRAIN demo regressions; no model/host authority is used."""

from contextlib import redirect_stdout
from io import StringIO
import json
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.experiments.cognitive_train_demo import DISCLAIMER, run_demo


class CognitiveTrainDemoTests(unittest.TestCase):
    def test_two_cycle_prompt_walkthrough_is_public_only(self):
        output = StringIO()
        with redirect_stdout(output):
            result = run_demo(seed=107, policy="rotate-abc", show_prompts=True)
        shown = output.getvalue()
        self.assertIn(DISCLAIMER, shown)
        self.assertEqual(shown.count("Respond with only A, B, or C."), 30)
        self.assertEqual(shown.count("[Cycle 1/2, room "), 15)
        self.assertEqual(shown.count("[Cycle 2/2, room "), 15)
        self.assertIn("Fabricated DEV failure only", shown)
        self.assertNotIn("ANSWER:", shown)
        self.assertNotIn("correct_choice", shown)
        self.assertNotIn("gold_answer", shown)
        self.assertNotIn("private_exam_answer", shown)
        self.assertEqual(result["distinct_training_case_count"], 30)
        self.assertEqual(result["distinct_training_prompt_count"], 30)
        self.assertEqual([c["rooms_exposed"] for c in result["cycles"]], [15, 15])
        self.assertFalse(result["q6_host_authorized"])
        self.assertFalse(result["physical_training_performed"])
        self.assertFalse(result["real_model_checkpoint_observed"])
        self.assertFalse(result["independent_exam_performed"])
        self.assertFalse(result["mastery_or_graduation_authorized"])

    def test_source_receipt_is_deterministic_and_does_not_emit_gold(self):
        first = run_demo(seed=107, policy="all-a", show_prompts=False)
        repeat = run_demo(seed=107, policy="all-a", show_prompts=False)
        self.assertEqual(first, repeat)
        serialized = json.dumps(first, sort_keys=True)
        for prohibited in ('"answer"', '"gold"', '"answer_key"',
                           '"correct_option"', '"correct_choice"'):
            self.assertNotIn(prohibited, serialized)
        identities = [tuple(case) for cycle in first["cycles"] for case in
                      cycle["source_only_self_reported_receipt"]
                      ["room_case_ids_and_prompt_commitments"]]
        self.assertEqual(len(identities), 30)
        self.assertEqual(len(set(case_id for _room, case_id, _prompt in identities)), 30)
        self.assertEqual(len(set(prompt for _room, _case, prompt in identities)), 30)
        self.assertTrue(all(
            cycle["source_only_self_reported_receipt"]["is_real_checkpoint"] is False
            and cycle["source_only_self_reported_receipt"]["is_independently_attested"] is False
            for cycle in first["cycles"]
        ))

    def test_run_demo_requires_no_socket_or_http_endpoint(self):
        with patch.object(socket, "socket", side_effect=AssertionError("no sockets")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("no HTTP")):
            result = run_demo(seed=202, policy="all-b", show_prompts=False)
        self.assertTrue(result["offline"])
        self.assertTrue(result["source_only"])
        self.assertEqual(sum(c["rooms_exposed"] for c in result["cycles"]), 30)

    def test_module_cli_json_only_smoke(self):
        run = subprocess.run(
            [sys.executable, "-m", "scripts.experiments.cognitive_train_demo",
             "--seed", "107", "--mock-policy", "rotate-abc", "--json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=15, check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stderr, "")
        receipt = json.loads(run.stdout)
        self.assertEqual(receipt["skill"], "cognitive_empathy")
        self.assertEqual(receipt["distinct_training_prompt_count"], 30)
        self.assertNotIn("Respond with only A, B, or C.", run.stdout)

    def test_invalid_mock_policy_and_seed_fail_before_room_exposure(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            run_demo(policy="private-gold-lookup", show_prompts=False)
        with self.assertRaisesRegex(ValueError, "seed"):
            run_demo(seed=-1, show_prompts=False)


if __name__ == "__main__":
    unittest.main()
