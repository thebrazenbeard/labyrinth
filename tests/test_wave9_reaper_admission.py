"""Fail-closed global admission when mock children never reap."""

from __future__ import annotations

import io
import subprocess
from threading import Event
from threading import Thread
import time
import unittest
from unittest.mock import patch

from labyrinth.minotaur_process_adapter import (
    MAX_OUTSTANDING_CHILD_CALLS, ProcessAdapterFailure, ProcessModelAdapter,
)


class NeverReapingPopen:
    """Mock Popen that accepts kill but never reaps before external release."""

    released = Event()
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.pid = 50_000 + len(type(self).instances)
        self.returncode = None
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.killed = False
        type(self).instances.append(self)

    def wait(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("mock-child", timeout)
        type(self).released.wait(timeout=3)
        if not type(self).released.is_set():
            raise RuntimeError("test mock did not receive release")
        self.returncode = -9
        return self.returncode

    def kill(self):
        self.killed = True


class BlockedPipeWriter:
    released = Event()

    def write(self, data):
        type(self).released.wait(timeout=3)
        return len(data)

    def flush(self):
        return None

    def close(self):
        return None


class ExitedChildWithBlockedWriter:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.pid = 60_000 + len(type(self).instances)
        self.returncode = 0  # OS child is gone, but pipe worker remains stuck
        self.stdin = BlockedPipeWriter()
        self.stdout = io.BytesIO()
        type(self).instances.append(self)

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


class ThreadStartupFailureChild:
    pid = 70_001

    def __init__(self, *_args, **_kwargs):
        self.returncode = None
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.killed = False

    def wait(self, timeout=None):
        if not self.killed:
            raise subprocess.TimeoutExpired("mock-child", timeout)
        self.returncode = -9
        return -9

    def kill(self):
        self.killed = True


class LateReapingStartupFailureChild(ThreadStartupFailureChild):
    released = Event()

    def wait(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("mock-child", timeout)
        type(self).released.wait(timeout=2)
        if not type(self).released.is_set():
            raise RuntimeError("test did not release late child")
        self.returncode = -9
        return -9


class ReaperAdmissionTests(unittest.TestCase):
    def test_second_worker_start_failure_late_reap_does_not_leak_slot(self):
        from labyrinth import minotaur_process_adapter as module
        child = LateReapingStartupFailureChild()
        LateReapingStartupFailureChild.released.clear()
        original_start = Thread.start
        calls = [0]

        def fail_only_writer_start(worker):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("mock writer thread cannot start")
            return original_start(worker)

        try:
            with patch("labyrinth.minotaur_process_adapter.subprocess.Popen", return_value=child):
                with patch("labyrinth.minotaur_process_adapter.Thread.start",
                           new=fail_only_writer_start):
                    adapter = ProcessModelAdapter(mode="probe", timeout_seconds=.01)
                    with self.assertRaises(ProcessAdapterFailure) as error:
                        adapter.generate('{"turn":0}')
            self.assertEqual(error.exception.status, "CHILD_ERROR")
            self.assertTrue(child.killed)
            self.assertEqual(adapter.receipts[0].resource_admission,
                             "PENDING_BOUNDED_REAPER")
        finally:
            LateReapingStartupFailureChild.released.set()
        deadline = time.monotonic() + 2
        while module._outstanding_child_call_count() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(module._outstanding_child_call_count(), 0)

    def test_thread_start_failure_attempts_kill_and_releases_reaped_slot(self):
        from labyrinth import minotaur_process_adapter as module
        child = ThreadStartupFailureChild()
        with patch("labyrinth.minotaur_process_adapter.subprocess.Popen", return_value=child):
            with patch("labyrinth.minotaur_process_adapter.Thread.start",
                       side_effect=RuntimeError("thread quota exhausted")):
                adapter = ProcessModelAdapter(mode="probe")
                with self.assertRaises(ProcessAdapterFailure) as error:
                    adapter.generate('{"turn":0}')
        self.assertEqual(error.exception.status, "CHILD_ERROR")
        self.assertTrue(child.killed)
        self.assertEqual(adapter.receipts[0].resource_admission,
                         "REAPED_SLOT_RELEASED")
        self.assertEqual(module._outstanding_child_call_count(), 0)

    def test_global_cap_denies_next_launch_and_later_recovers(self):
        NeverReapingPopen.released.clear()
        NeverReapingPopen.instances.clear()
        candidates = [ProcessModelAdapter(mode="probe", timeout_seconds=.01)
                      for _ in range(MAX_OUTSTANDING_CHILD_CALLS + 1)]
        try:
            with patch("labyrinth.minotaur_process_adapter.subprocess.Popen", NeverReapingPopen):
                for adapter in candidates[:MAX_OUTSTANDING_CHILD_CALLS]:
                    with self.assertRaises(ProcessAdapterFailure) as error:
                        adapter.generate('{"turn":0}')
                    self.assertEqual(error.exception.status, "TIMEOUT")
                self.assertEqual(len(NeverReapingPopen.instances), MAX_OUTSTANDING_CHILD_CALLS)
                with self.assertRaises(ProcessAdapterFailure) as error:
                    candidates[-1].generate('{"turn":0}')
                self.assertEqual(error.exception.status, "RESOURCE_LIMIT")
                self.assertEqual(len(NeverReapingPopen.instances), MAX_OUTSTANDING_CHILD_CALLS)
                denied_receipt = candidates[-1].receipts[0]
                self.assertEqual(denied_receipt.status, "RESOURCE_LIMIT")
                self.assertIsNone(denied_receipt.child_pid)
                self.assertEqual(denied_receipt.resource_admission, "DENIED_PROCESS_LOCAL_CHILD_LIMIT")
                self.assertEqual(denied_receipt.resource_scope,
                                 "PER_INTERPRETER_NOT_HOST_GLOBAL")
        finally:
            # Never leave our deliberately hung mock reapers around.
            NeverReapingPopen.released.set()
        from labyrinth import minotaur_process_adapter as module
        deadline = time.monotonic() + 2
        while module._outstanding_child_call_count() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(module._outstanding_child_call_count(), 0)
        next_result = ProcessModelAdapter(mode="probe").generate('{"turn":0}')
        self.assertIn('"ABSTAIN"', next_result)

    def test_exited_child_stuck_pipe_workers_remain_admission_bounded(self):
        BlockedPipeWriter.released.clear()
        ExitedChildWithBlockedWriter.instances.clear()
        adapters = [ProcessModelAdapter(mode="probe", timeout_seconds=.01)
                    for _ in range(MAX_OUTSTANDING_CHILD_CALLS + 1)]
        try:
            with patch("labyrinth.minotaur_process_adapter.subprocess.Popen",
                       ExitedChildWithBlockedWriter):
                for adapter in adapters[:MAX_OUTSTANDING_CHILD_CALLS]:
                    with self.assertRaises(ProcessAdapterFailure) as error:
                        adapter.generate('{"turn":0}')
                    self.assertEqual(error.exception.status, "TIMEOUT")
                with self.assertRaises(ProcessAdapterFailure) as error:
                    adapters[-1].generate('{"turn":0}')
                self.assertEqual(error.exception.status, "RESOURCE_LIMIT")
                self.assertEqual(len(ExitedChildWithBlockedWriter.instances),
                                 MAX_OUTSTANDING_CHILD_CALLS)
        finally:
            BlockedPipeWriter.released.set()
        from labyrinth import minotaur_process_adapter as module
        deadline = time.monotonic() + 2
        while module._outstanding_child_call_count() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(module._outstanding_child_call_count(), 0)


if __name__ == "__main__":
    unittest.main()
