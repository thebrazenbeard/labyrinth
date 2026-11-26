# Minotaur Wave65 — Windows inherited-event gated candidate launch (SOURCE-ONLY)

Date 2026-11-14. Stack base Labyrinth Draft PR #246 head
`c9af8ecf14a15a5c4d67346cc6456d29ee72fe9c`.
Isolated local branch `fix/minotaur-windows-gated-launch-20261114`.
Scope: a Windows *same-user subprocess lifecycle* improvement, not
independent examiner custody, model attestation, a private-file sandbox,
or actual Minotaur training.

## Hostile RED before source change

The earlier Windows Job Object approach started the untrusted command
first, then assigned it to a kill-on-close job. The first test using a
Python candidate and a 0.75s artificial attachment delay was inconclusive:
under severe Lappy memory pressure the model did not reliably start before
the observation. It therefore returned PASS without proving containment.

A replacement real-process synthetic test wrote a disposable `candidate.cmd`
that immediately wrote `executed.txt`, then ran an absolute Windows ping
command for about four seconds before emitting `attached`. It delayed
`attach_kill_on_close` by 0.75s and asserted the marker did not exist
*inside the attachment hook*, before assignment. On unmodified #246 source,
the marker WAS present: `AssertionError: True is not false: candidate
executed before Windows job attachment`. That assertion was genuinely
observed in the traceback before implementation, exit1. A secondary
WinError32 while cleaning up the synthetic child obscured the outermost
exception and is explicitly not counted as a successful test verdict.

## Minimum production change

`labyrinth/_windows_gated_launch.py` creates an unnamed, initially
unsignaled manual-reset Win32 event. CPython's documented
`subprocess.STARTUPINFO.lpAttributeList.handle_list` passes the
temporarily inheritable event handle to a fixed `python -I -S -B -c`
bootstrap. Only fixed stdlib import and `WaitForSingleObject` execute
before release; the untrusted command is not launched yet.

The examiner attaches that bootstrap to the already-existing
per-candidate kill-on-close Job Object, *then* signals the event.
After successful release, bootstrap launches the original bounded command
with the existing public-JSON stdin, output pipe and sanitized environment.
A rejected attach or failed event signal terminates the bootstrap without
releasing the candidate; handles are closed in failure and success paths.
Windows output cap/deadline, request-file cleanup and Job Object
descendant termination remain unchanged. POSIX still uses its original
process-group/UID code paths.

Documentation: https://docs.python.org/3/library/subprocess.html#subprocess.STARTUPINFO
https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createeventa
https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-setevent
https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject

## Actual GREEN and regressions

Post-change Windows hostile pre-attachment test: 1 PASS in 4.894s.
First full Windows subprocess run 13 cases had one test-harness failure:
the test patched `_windows_candidate_job.attach_kill_on_close` after
`_windows_gated_launch` had already imported the function alias; the
observation list was empty, not evidence of early candidate execution.
Corrected the *patch target* to the callable actually used by the launcher,
without removing/weakening the assertion. Native Windows subprocess module
then **13/13 PASS** in 12.202s.

Added characterization of a synthetic failed Job Object attachment:
`PermissionError` is propagated and the disposable untrusted command
never reaches its marker. Windows focused suite after addition:
**14 subprocess tests PASS in 12.452s**, **17 examiner ACL PASS in 0.538s**,
**6 private-read probes PASS in 0.135s**. Python compileall and Git diff
whitespace checks PASS.

WSL Ubuntu26.04 subprocess module **14 normal PASS in 6.744s** and
**14 optimized `-O` PASS in 6.805s**; zero unexpected skips.
These targeted tests do not imply exact immutable-head hosted CI or a
complete local Windows test-suite result; those require separate receipts.

## Unresolved hostile boundaries — DRAFT HOLD

- **Same security principal.** A candidate still executes as the Windows
  examiner account and can read a known examiner-owned file when Windows ACLs
  permit it. The negative test continues to demonstrate this.
- **Bootstrap trust and handle inheritance.** The fixed bootstrap uses the
  examiner's installed Python interpreter and same-user OS token. During
  `Popen`, the event handle must be temporarily inheritable; Python warns
  about handle leakage to unrelated concurrently launched inheriting
  processes. This is not a hardened child creation or separate-token
  design; same-user interference and host-code tampering are not excluded.
- **One launcher path, not total containment.** The tested ordered release
  closes the specific *candidate-code-before-job-assignment* window for this
  bootstrap path. It does not prove protection against malicious native
  binaries after launch, process access rights abuse, injections, other
  process creation mechanisms or all filesystem/network channels.
- **Honest model/mechanism evidence unresolved.** Real checkpoint identity,
  held-out private-case custody, semantic family independence, signed trial
  origin, generalization and graduation remain unqualified; see Labyrinth
  issues #146 and #147.
- **Source and tests, not production runtime.** No neural model downloaded,
  trained, graduated or deployed by this work. No Windows ACLs, shared
  services, protected branches or credentials were changed.
