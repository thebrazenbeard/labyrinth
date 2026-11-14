# Minotaur POSIX candidate identity drop — source-only (2026-11-14)

**Parent:** Labyrinth Draft PR #237, source head `4b1e1ea14c72fffb1a1301abc11f87955e0da4bd`.
**Status:** Tested selective POSIX DAC denial; NOT independent examiner custody,
not a sandbox, not installed, not model/generalization qualification.

## Opt-in contract

Use `SubprocessCandidateAdapter((absolute_executable, *args), model_id=...,
unix_user_id=65534, unix_group_id=65534)` from an authorized root-owned
POSIX examiner only. Both identifiers are mandatory, nonzero positive integers.
The adapter fails closed on Windows, WSL (Linux kernel release containing
`microsoft`), or without root privileges. In native POSIX only, the process
launch sets `user`, `group` and clears `extra_groups` through Python's
`subprocess.Popen`. Existing subprocess behavior remains the default.
The caller must still specify `require_process_separation=True` when invoking
`run_candidate` to prevent arbitrary same-process adapter callbacks.


## Reproducible physical probe

`tests/test_minotaur_posix_uid_boundary.py` uses a synthetic held-out
file in a freshly created private POSIX temporary directory. The root
examiner retains read access; the untrusted child receives the *known*
absolute path but runs as UID/GID 65534 with no supplementary groups.
The child attempts the read, catches `PermissionError`, and reports only
an allowed public bridge action. The test additionally asserts UID, GID,
empty supplementary groups and a replayable examiner receipt.

The initial source-first TDD test failed at the unimplemented keyword argument.
An early version passed a physical DAC check inside WSL Ubuntu 26.04 Python
3.14.4 as root: child UID/GID 65534 was denied that private file. That result
is valid for *that file* but was not a security boundary. A later real hostile
probe showed a child with UID/GID 65534 could successfully invoke Windows
`cmd.exe /c whoami` and obtain `lappy\\patri`, bypassing the Linux identity.
A new WSL-rejection test was RED (PermissionError not raised), then GREEN when
the constructor and launch path refused WSL UID mode. The POSIX physical test
now runs only on native Linux root; on WSL it is intentionally skipped. This
exact native-Linux production path has not yet been physically exercised.

> **HOSTILE LIMIT:** This verifies DAC for one specific private POSIX file.
> A process can still access other same-world-readable paths and local/network
> resources; no mount/network namespace, seccomp, capability bounding,
> no-new-privileges, executable attestation or descendant-process containment
> is established. WSL/Windows interoperability **was bypassed in a physical
> read-only identity probe**, so WSL UID mode is deliberately refused.


A native Linux examiner needs root privileges to set UID/GID, so arbitrary
candidate bytes must not run on an unisolated privileged host. Even native
Linux UID separation can be defeated by host kernel/helper vulnerabilities or
exposed sockets; these are unqualified. The WSL experiment proved the Windows
account escape and is now guarded. Existing Windows same-principal tests show the
unmodified default subprocess can read a known private file.

**Do not close issue #146:** independent evaluator custody and an actual
untrusted inference sandbox remain unverified. Issue #147 (semantic graduation
integrity) also remains open. No real model was trained or tested here.
