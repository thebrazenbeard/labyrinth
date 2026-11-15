# Minotaur Wave64 — Windows assigned-descendant cleanup (DRAFT/HOLD)

Date: 2026-11-14. Source parent: Labyrinth Draft PR #245
`fac5fd9d428920d9a4d53b98e70a30163561c569`.

**Classification:** candidate subprocess cleanup improvement; not independent
examiner custody, sandboxing, model execution attestation or training.
Lappy worktree: `D:\VERA\experiments\minotaur-wave64-windows-job-cleanup-20261114`.

## Real failure before implementation — RED

Added a Windows-only public adapter integration test spawning a synthetic
Python parent and an inherited-stdout grandchild. Parent waits 0.1s then
spawns its grandchild and writes an examiner-visible started marker. Grandchild
attempts a delayed marker write 1.6s later. Examiner deadline is 1.0s;
test waits a further 1.3s and requires the marker to be absent.

Before patch: `python -m unittest discover -s tests -p
test_minotaur_subprocess_adapter.py -k
windows_candidate_descendant_dies_after_parent_timeout -v` ran 1 test
in 2.323s, **FAIL exit1**: `surviving Windows candidate descendant
executed after cleanup`, marker actually present. No other user processes
or files were touched; this created only disposable synthetic test files.

## Implementation and second RED

New private `labyrinth/_windows_candidate_job.py` uses actual Win32
`CreateJobObjectW`, `SetInformationJobObject` with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, `AssignProcessToJobObject` for
CPython's child process handle, and `CloseHandle` on cleanup. No
Job Object is attached to an examiner-owned or unrelated process.
Assignment failure is not ignored: the child is terminated, output closed
and model call fails closed. No host-wide security policy changes.

The original hostile grandchild test then **GREEN exit0**, 1 test in 2.334s.
However, the complete subprocess-module run surfaced an additional
**RED**: oversized-output response became `MODEL_CALL_ERROR`, not
`MODEL_OUTPUT_INVALID`, even though the model wrote oversized output.
Direct adapter diagnostic reproduced `PermissionError [WinError 32]`
when Windows tried to delete its own ephemeral public request file as
the job's just-terminated child still held an inherited stdin handle.
This was not treated as a successful implementation.

Minimal follow-up: `_candidate_request_directory()` retries actual
`TemporaryDirectory.cleanup()` only for Win32 sharing violation 32,
for at most 0.75s with 20ms waits. On any other error or expired deadline
cleanup raises, not silently abandoned. No private file handling changed.
After correction: oversize-output **GREEN in 0.894s**, grandchild
termination **GREEN in 2.327s**; both original assertions unchanged.

## Completed targeted verification

Actual Windows Python3.14.5 on Lappy, isolated uncommitted candidate:
- `test_minotaur_subprocess_adapter.py`: **12 tests PASS** in 7.196s.
- `test_examiner_custody_acl.py`: **17 tests PASS** in 0.808s.
- `test_examiner_private_read_probe.py`: **6 tests PASS** in 0.476s.
- `compileall` and `git diff --check` **PASS**.
- WSL Ubuntu26.04 Python3 normal subprocess tests: **12 tests PASS,
  1 Windows-specific skipped**, 7.268s; optimized `-O` **12 tests PASS,
  1 skipped**, 7.141s.

Microsoft documentation for nested jobs/kill-on-close and real limitations:
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject

## Hostile limitations — HOLD

- **Spawn/attachment race is real:** the untrusted child begins executing
  before `AssignProcessToJobObject`. It could spawn and detach or escape
  through a creation mechanism outside the job before assignment.
- Windows Job Objects do **not** provide a distinct candidate security
  principal, filesystem/registry/network sandbox, model attestation or
  examiner-owned private-file read denial. Earlier real same-user leak
  in the subprocess test remains demonstrably possible.
- The cap reader's `PeekNamedPipe` synchronous-call worst-case latency is
  not formally proven. The targeted test does not prove denial of unknown
  malicious exit paths or resilience under host resource exhaustion.
- Exact-commit hosted CI and full Windows suite need separately documented
  results before broader qualification. Do not merge/deploy or graduate a model
  based on this source-only change.

## Immutable first-head full suite and CI-policy follow-up

Initial head `f2bcae9b3a653b7c50cb898d4b31305e6a338a21`:
- Native Lappy Windows Python3.14.5 full `unittest discover -s tests -q`:
  **1,279 PASS, 6 skipped, zero failures**, 325.507s, exit0,
  2026-11-14 19:06:48–19:12:24 EST; exact same start/end SHA.
- Hosted Python3.10 and 3.12 workflow #43656512198: both passed
  1,279 tests, **55 skipped** each; job logs verified.
- Hosted neural-CPU workflow #43656512155: **FAILED exit1**
  despite 141 unit assertions passing, because the new Windows-only
  test called `skipTest` on Linux and the dedicated workflow rejects
  *any* unexpected skip. See PR #246 comment #6886359693.

Subsequent test-only correction replaces this Linux-side `skipTest`
with the established suite convention `return` (Windows behavior and
adversarial test unchanged). Native Windows focused subprocess tests:
12 PASS, 0 skipped; WSL Ubuntu26.04 normal: 12 PASS, 0 skipped;
WSL optimized `-O`: 12 PASS, 0 skipped; diffcheck clean.
New immutable-commit hosted verdicts must be checked separately.
