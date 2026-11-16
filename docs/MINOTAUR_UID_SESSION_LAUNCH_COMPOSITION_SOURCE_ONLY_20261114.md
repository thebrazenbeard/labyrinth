# Minotaur POSIX credential/session composition — source only (2026-11-14)

**Parent:** Draft Labyrinth PR #239, exact source head `617d7d891e2e275d4e07570761d8b8e8a97907b6`.

## Defect

The PR #239 candidate process-group cleanup starts an ordinary POSIX candidate
in its own process session and signals that process group on teardown. But the
opt-in non-root POSIX UID/GID branch replaced the complete launch kwargs dict
instead of extending it, dropping `start_new_session=True`.

For the privileged branch, `killpg(child.pid, SIGKILL)` therefore no longer
necessarily targeted the candidate process. A blocked child could escape the
new group cleanup or cause an unbounded examiner wait. The native non-WSL root
branch was never physically qualified and must not be described as secure.

## RED / GREEN

On WSL Ubuntu 26.04, introduced
`test_uid_drop_launch_composes_with_posix_session_isolation` to inspect
`subprocess.Popen` arguments at the intercepted boundary, without launching
a privileged candidate in WSL. Before the source correction: focused test
ERROR exit 1, `KeyError: 'start_new_session'`, on actual launch kwargs.
After the one-line `launch_identity.update(...)` correction: same test PASS.

The test asserts `start_new_session=True`, nonroot `user=65534`, `group=65534`
and empty `extra_groups` occur in the SAME child launch. Windows explicitly
rejects that UID/GID configuration and makes no POSIX security claim.

A separate disposable synthetic WSL-root test launched a normal Python child
with an initial 0700 private working directory, pre-opened public stdin,
`start_new_session=True`, and UID/GID=65534; the child reported UID/GID 65534,
read its inherited public JSON and exited 0. This tests POSIX mechanics only:
WSL's Win32 interop means Linux UID credentials are NOT independent isolation.
The probe script and root temp directory were removed.

## Boundaries

- **TEST:** Launch-option composition and the adjacent existing subprocess
  descendant lifecycle tests on WSL and Windows.
- **NOT TESTED:** Actual native non-WSL Linux root examiner and restricted UID
  child, or safe denial of read access to a private examiner file under it.
- **NOT GUARANTEED:** Detached descendants, Windows process-tree containment,
  independent examiner custody, cgroups/namespaces/seccomp, or auth of a model.
- **NOT CLAIMED:** Learned reasoning, novel solution discovery, real-world
  security isolation, deployment, PR merge or installed runtime effect.

The code remains experimental and draft/unmerged. Issues #146 and #147 remain
open pending independent custody and semantic-learning evaluation.
