"""Even an exited child must not extend its allowed wall budget during cleanup."""

from io import BytesIO
from threading import Event
import subprocess
import time
import unittest
from unittest.mock import patch

from labyrinth.minotaur_process_adapter import ProcessAdapterFailure, ProcessModelAdapter


class ProcessDeadlineTests(unittest.TestCase):
    def test_nonreaping_child_gets_background_reap_without_blocking_caller(self):
        reaped = Event()

        class LateReapingChild:
            pid = 1050
            returncode = -9
            stdin = BytesIO()
            stdout = BytesIO()

            def wait(self, timeout=None):
                if timeout is not None:
                    raise subprocess.TimeoutExpired(cmd="mock-child", timeout=timeout)
                reaped.set()
                return self.returncode

            def kill(self):
                pass

        adapter = ProcessModelAdapter(mode="probe", timeout_seconds=.01)
        with patch("labyrinth.minotaur_process_adapter.subprocess.Popen",
                   return_value=LateReapingChild()):
            with self.assertRaises(ProcessAdapterFailure) as context:
                adapter.generate('{"turn":0}')
        self.assertEqual(context.exception.status, "TIMEOUT")
        self.assertTrue(reaped.wait(.5), "background wait must own final reap")

    def test_exited_child_with_blocked_writer_does_not_add_one_second(self):
        release = Event()

        class BlockedInput:
            def write(self, _data):
                release.wait(4)

            def flush(self):
                pass

            def close(self):
                pass

        class ExitedChild:
            pid = 1049
            returncode = 0
            stdin = BlockedInput()
            stdout = BytesIO(b"")

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        adapter = ProcessModelAdapter(mode="probe", timeout_seconds=.01)
        start = time.monotonic()
        try:
            with patch("labyrinth.minotaur_process_adapter.subprocess.Popen",
                       return_value=ExitedChild()):
                with self.assertRaises(ProcessAdapterFailure) as context:
                    adapter.generate('{"turn":0}')
                self.assertEqual(context.exception.status, "TIMEOUT")
        finally:
            release.set()

        self.assertLess((time.monotonic() - start) * 1000, 500)
        self.assertEqual(adapter.receipts[-1].status, "TIMEOUT")


if __name__ == "__main__":
    unittest.main()
