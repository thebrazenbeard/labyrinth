"""Best-effort Windows candidate-descendant cleanup; not a security sandbox.

Win32 Job Objects keep assigned descendants in a kill-on-close lifetime. This
does NOT prevent a child from running before assignment, confer a separate
security principal, or constrain same-user file access. Fail closed when a
job cannot be attached; never silently run an uncontained candidate.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os


class _BasicLimitInfo(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]

class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimitInfo(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInfo),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel():
    if os.name != "nt":
        raise RuntimeError("Win32 candidate job requires Windows")
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    k.SetInformationJobObject.restype = wintypes.BOOL
    k.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    k.AssignProcessToJobObject.restype = wintypes.BOOL
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    k.CloseHandle.restype = wintypes.BOOL
    return k

def attach_kill_on_close(child):
    """Attach an already-launched CPython Popen child to a private kill job.

    A spawn-to-attachment race remains: this is cleanup of assigned children,
    not a proof of start-suspended, credential-isolated model execution.
    """
    k = _kernel()
    job = k.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    try:
        limits = _ExtendedLimitInfo()
        limits.BasicLimitInformation.LimitFlags = 0x2000
        if not k.SetInformationJobObject(job, 9, ctypes.byref(limits),
                                         ctypes.sizeof(limits)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        process_handle = getattr(child, "_handle", None)
        if process_handle is None:
            raise RuntimeError("Windows child process handle unavailable")
        if not k.AssignProcessToJobObject(job, process_handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        return job
    except BaseException:
        k.CloseHandle(job)
        raise


def close_kill_job(job):
    """Closing last handle requests termination of all assigned descendants."""
    if not _kernel().CloseHandle(job):
        raise OSError(ctypes.get_last_error(), "CloseHandle(candidate job) failed")
