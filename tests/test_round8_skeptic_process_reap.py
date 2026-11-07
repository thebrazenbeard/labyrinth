"""Independent RED: child-kill reap must not silently bypass wall deadline.

Target root integration d4ee15d31e333c0027ebecd751b73cc0c5318728.
Uses a mock non-reaping child: no real child process or host side effect.
"""

import io
import subprocess
import time
import unittest
from unittest.mock import patch

from labyrinth.minotaur_process_adapter import ProcessAdapterFailure, ProcessModelAdapter


class UnboundedReapDeadline(unittest.TestCase):
    def test_child_killed_but_unbounded_reap_does_not_eat_wall_deadline(self):
        class ChildThatWillNotReap:
            pid = 424200
            returncode = -9
            stdin = io.BytesIO()
            stdout = io.BytesIO()

            def kill(self):
                return None

            def wait(self, timeout=None):
                if timeout is not None:
                    raise subprocess.TimeoutExpired(cmd="mock", timeout=timeout)
                # Simulate OS/resource race after kill while waiting to reap.
                time.sleep(0.55)
                return self.returncode

        started = time.monotonic()
        with patch("labyrinth.minotaur_process_adapter.subprocess.Popen",
                   return_value=ChildThatWillNotReap()):
            with self.assertRaises(ProcessAdapterFailure):
                ProcessModelAdapter(mode="probe", timeout_seconds=0.01).generate('{"turn":0}')
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.3,
                        "unbounded second wait() after kill violated claimed wall timeout")


if __name__ == "__main__":
    unittest.main()
