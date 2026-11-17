"""Independent hostile process-boundary regressions against frozen Wave17 source.

These tests deliberately describe required behavior that the frozen implementation
does not meet; they are evidence, not modifications to the production runner.
"""

import json
import inspect
import os
import sys
import unittest
from unittest import mock

from scripts import novel_router_model_trial as trial


class Wave17ProcessDestroyer(unittest.TestCase):
    def test_generation_seed_in_child_receipt_was_actually_sent_to_candidate(self):
        """A provenance value cannot claim model RNG control if never delivered."""
        generation_seed = 271828
        config = trial.TrialConfig(case_seed=0, generation_seed=generation_seed,
                                  max_calls=1, call_timeout_seconds=1.,
                                  total_timeout_seconds=2.)
        self.assertEqual(config.generation_seed, generation_seed)
        view = trial.novel.ProbeSession(trial.novel.make_case(0)).view()
        public_params = inspect.signature(trial._public_request).parameters
        if "generation_seed" in public_params:
            raw_prompt = trial._public_request(view, 0, generation_seed)
        else:
            raw_prompt = trial._public_request(view, 0)
        prompt = json.loads(raw_prompt)
        argv = trial._argv(trial.AdapterSpec(mode="command",
                                             command_argv=(sys.executable, "-c", "pass"),
                                             authorize_command=True))
        environment = trial._child_environment()
        delivery = json.dumps({"prompt": prompt, "argv": argv,
                               "environment": environment}, sort_keys=True)

        self.assertIn(str(generation_seed), delivery,
                      "generation_seed is stamped in child receipt but never "
                      "sent to the child in prompt, argv, or environment")

    @unittest.skipUnless(os.name == "posix", "child process-group observation is POSIX-specific")
    def test_exited_parent_still_requires_descendant_process_group_cleanup(self):
        """Child exit does not imply that its spawned group descendants exited."""
        child = mock.Mock()
        child.pid = 875431
        child.poll.return_value = 0  # Parent exited; descendant may own pipe.
        with mock.patch.object(trial.os, "killpg") as group_kill:
            trial._stop_child(child)
        group_kill.assert_called_once_with(child.pid, trial.signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
