"""Nonreproducing fresh reload must be a failing process, not green CLI."""
import contextlib
import io
import unittest
from unittest.mock import patch
from labyrinth.hf_pilot import main


class ReloadQualificationCLITests(unittest.TestCase):
    def test_false_trial_match_is_nonzero(self):
        fake={"schema":"labyrinth.hf-lora-reload-eval.v2",
              "matches_original_candidate_trials":False,
              "process_id_differs":True}
        with patch("labyrinth.hf_pilot.reload_evaluate",return_value=fake):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as outcome:
                    main(["eval","--run-dir","fixture","--report","fixture.json"])
        self.assertNotEqual(outcome.exception.code,0)

    def test_same_pid_is_nonzero_even_when_trials_match(self):
        fake={"schema":"labyrinth.hf-lora-reload-eval.v2",
              "matches_original_candidate_trials":True,
              "process_id_differs":False}
        with patch("labyrinth.hf_pilot.reload_evaluate",return_value=fake):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as outcome:
                    main(["eval","--run-dir","fixture","--report","fixture.json"])
        self.assertNotEqual(outcome.exception.code,0)

    def test_verified_reproducing_reloaded_cli_remains_zero(self):
        fake={"schema":"labyrinth.hf-lora-reload-eval.v2",
              "matches_original_candidate_trials":True,
              "process_id_differs":True}
        with patch("labyrinth.hf_pilot.reload_evaluate",return_value=fake):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["eval","--run-dir","fixture",
                                       "--report","fixture.json"]),0)


if __name__=="__main__":
    unittest.main()
