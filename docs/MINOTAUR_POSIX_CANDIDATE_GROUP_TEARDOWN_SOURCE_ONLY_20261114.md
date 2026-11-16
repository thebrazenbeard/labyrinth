# Minotaur POSIX candidate group teardown — source-only (2026-11-14)

**Base:** Draft PR #238 exact head `a006c01e8d1ab1bd9280f8bd232171c45d05217b`.
**Scope:** One subprocess launch/cleanup change plus a deterministic hostile test.

## Reproduced defect (RED)

In the previous child-only cleanup, a candidate spawned a grandchild which
inherited stdout, then exited. The examiner waited for EOF, timed out, and
killed only the original child. That child had already exited. The grandchild
survived long enough to write to an owned temporary marker file.

The isolated WSL Ubuntu 26.04 regression `test_posix_timeout_kills_stdout_holding_grandchild`
failed with `AssertionError: True is not false : Timed-out candidate left
a live grandchild behind` (exit 1).

## Correction (GREEN)

On POSIX only, launch each candidate with `start_new_session=True` and use
`os.killpg(candidate_pid, SIGKILL)` during teardown, whether the direct child
has exited or is still running. Ignore `ProcessLookupError` if the group
already disappeared. Continue waiting and closing pipe resources afterward.

The identical hostile regression passed on WSL after the change. Existing
Windows subprocess semantics are unchanged; the POSIX process-group test does
not claim to exercise process-tree teardown on Windows.

## Limits and classification

- **TEST:** Demonstrates cleanup of a cooperative grandchild inheriting stdout
  in the same candidate process group, including parent-exited timeout case.
- **UNKNOWN:** A child may deliberately call `setsid` or equivalent and escape
  its process group. Descendant tracking, Linux sandboxing, resource limits,
  mount/network namespaces, capabilities, seccomp and executable custody are
  not provided by this patch.
- **UNKNOWN:** Windows subtree containment remains unimplemented. The
  Windows test only exercises the unchanged platform contract.
- **UNKNOWN:** PR #238's root-owned native non-WSL Linux UID/GID path remains
  unqualified on physical native Linux. SSH to the known Mint host returned
  `Permission denied (publickey,password)` using configured unattended auth.
- **UNVERIFIED:** Independent examiner custody, model learning or reasoning
  generalization, and model/weight provenance; issues #146 and #147 remain OPEN.

This is an experimental source-only cleanup improvement. No deployment,
privilege changes, filesystem ACL changes, credential changes, or merges.
