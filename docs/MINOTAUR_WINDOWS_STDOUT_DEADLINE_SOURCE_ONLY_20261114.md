# Minotaur Windows inherited-stdout timeout: hostile source-only check (2026-11-14)

Base: Labyrinth Draft PR #241 head `9d955a8f2a63a71843e1f5afd1af7569df562335`.
Branch: `fix/minotaur-windows-stdout-deadline-20261114`.

## Reproduced failure on authorized Lappy, Windows Python 3.14.5

New Windows-only `test_windows_inherited_stdout_holder_respects_deadline`
uses the real `SubprocessCandidateAdapter.generate` with a 0.25-second
deadline. A synthetic, unprivileged candidate launches a grandchild that
inherits stdout, then exits. The grandchild sleeps 2.5 seconds and exits by
itself. No held-out Case, credentials, training or privileged effects are used.

- RED: with the old Windows daemon `BufferedReader.read(cap+1)` path,
  `child.stdout.close()` waited for the background reader's buffer lock.
  Test exited 1: elapsed 2.7712192s exceeds 1.4s bound.
- First modification: Win32 `PeekNamedPipe` + `os.read` polling removed the
  blocked reader but exposed a second failure: a living grandchild inherited
  the disposable request directory as its CWD, and Windows refused `rmdir`
  with `PermissionError [WinError 32]`. This was NOT marked green.
- GREEN after separating the Windows child CWD from its disposable request
  directory: same test exit 0; completed in 0.282s.

## Bounded change

- Windows uses Win32 anonymous-pipe `PeekNamedPipe` readiness with a
  deadline and max `MAX_RESPONSE_CHARACTERS * 4 + 1` byte read budget.
  It does not start a daemon reading thread, and the examiner's close no
  longer waits for a blocked Python buffered stream.
- Only Windows uses `tempfile.gettempdir()` as the child CWD. POSIX retains
  its previous per-request CWD and its prior `select`/process-group behavior.
- The child still receives a public JSON request through a temporary
  on-disk stdin handle. No private examiner-owned Case is passed.

## Limits and evidence classes

Source/test success does **not** establish Windows process-tree killing,
sandboxing, or access denial to same-user examiner files. An intentionally
malicious grandchild may inherit the original stdin file handle, write to
other same-user paths, or continue running; those are NOT contained here.
Win32 `PeekNamedPipe` itself is synchronous and cannot establish a formal
absolute wall-clock deadline under every hostile OS/device condition.
Native non-WSL Linux root/UID private-access qualification is still absent.
External task-mechanism integrity and independent examiner custody
(Labyrinth issues #147 and #146) remain OPEN and unqualified.

- TEST: Lappy Windows hostile RED -> intermediate cleanup error -> GREEN.
- TEST: ordinary Windows subprocess + POSIX UID boundary suites PASS.
- TEST: WSL Ubuntu26.04 subprocess normal and optimized, UID suites PASS.
- CI/full-suite qualification: report separately after exact-head readback.

This remains a DRAFT/HOLD source correction, not genuine neural learning,
model graduation, local install, deploy, or authorization to merge.
