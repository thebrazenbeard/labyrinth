"""Windows same-user candidate launch gate; not OS credential isolation.

A trusted Python bootstrap waits on an inherited unnamed Windows event until
the examiner assigns the bootstrap to a kill-on-close Job Object. Only then
does it launch the untrusted candidate. This removes the candidate-code
startup race within this cooperative bootstrap, NOT same-user interference,
kernel-level proof, independent file access denial, or process sandboxing.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys

from ._windows_candidate_job import attach_kill_on_close, close_kill_job

# This bootstrap has NO user-provided imports or code before the event wait.
_WINDOWS_BOOTSTRAP = r"""
import ctypes
from ctypes import wintypes
import json
import subprocess
import sys
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
k.WaitForSingleObject.restype = wintypes.DWORD
k.CloseHandle.argtypes = (wintypes.HANDLE,)
k.CloseHandle.restype = wintypes.BOOL
event = int(sys.argv[1])
result = k.WaitForSingleObject(event, 60000)
k.CloseHandle(event)
if result != 0:  # WAIT_OBJECT_0
    sys.exit(118)
command = json.loads(sys.argv[2])
if (type(command) is not list or not command
        or any(type(arg) is not str or not arg for arg in command)):
    sys.exit(119)
completed = subprocess.run(command, stdin=sys.stdin.buffer,
                           stdout=sys.stdout.buffer, stderr=subprocess.DEVNULL)
sys.exit(completed.returncode)
"""


def _kernel():
    if os.name != "nt":
        raise RuntimeError("Win32 gated candidate launcher requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateEventW.argtypes = (
        ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = (wintypes.HANDLE,)
    kernel.SetEvent.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel
def _close_gate(kernel, gate):
    if not kernel.CloseHandle(gate):
        raise OSError(ctypes.get_last_error(), "CloseHandle(launch gate) failed")


def start_candidate_inside_job(command, source, environment, cwd):
    """Return (bootstrap Popen, assigned job), never run code before attach.

    The inherited event-handle list still has a short inheritable-handle
    exposure during concurrent process creation, per CPython documentation.
    """
    kernel = _kernel()
    gate = kernel.CreateEventW(None, True, False, None)
    if not gate:
        raise OSError(ctypes.get_last_error(), "CreateEventW failed")
    child = None
    job = None
    try:
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {"handle_list": [int(gate)]}
        os.set_handle_inheritable(int(gate), True)
        try:
            child = subprocess.Popen(
                (sys.executable, "-I", "-S", "-B", "-c", _WINDOWS_BOOTSTRAP,
                 str(int(gate)), json.dumps(command)),
                stdin=source, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=environment, cwd=cwd, shell=False,
                startupinfo=startup, close_fds=True)
        finally:
            os.set_handle_inheritable(int(gate), False)
        job = attach_kill_on_close(child)
        if not kernel.SetEvent(gate):
            raise OSError(ctypes.get_last_error(), "SetEvent(launch gate) failed")
        # Close errors must still run the cleanup branch for the live job.
        _close_gate(kernel, gate)
        gate = None
        return child, job
    except BaseException:
        try:
            if job is not None:
                close_kill_job(job)
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait()
                child.stdout.close()
        raise
    finally:
        if gate is not None:
            _close_gate(kernel, gate)
