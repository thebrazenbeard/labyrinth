# Minotaur detached-stdout deadline: hostile source-only test (2026-11-14)

Parent: Labyrinth Draft PR #240 head `2c3a556a6cf82982a5d5b51fae6f34b36c7290f4`.
Branch: `fix/minotaur-detached-stdout-cleanup-20261114`.

## Failure

The POSIX candidate runner uses a background `BufferedReader.read(cap+1)`
to await stdout. A deliberately detached grandchild can inherit its pipe and
outlive the process group killed on timeout. The reader's capped `join(0.5)`
then returns, but `child.stdout.close()` in the examiner calls into the same
Python buffer lock held by the reader. Thus a candidate configured with a
0.25-second deadline could block the examiner for substantially longer.

## Red/green physical evidence

The new unprivileged `test_detached_stdout_holder_does_not_stall_cleanup`
spawns a synthetic POSIX candidate that detaches a 2.5-second stdout holder.
The holder ends naturally; no model, private case or system permissions are
exposed. The test calls the real adapter's `generate` method and asserts
both `TimeoutError` and elapsed time under 1.4 seconds.

- RED on unmodified PR #240: test exit 1; elapsed 2.560021 seconds; expected
  deadline-bound assertion failure, not an import or fixture failure.
- GREEN after modification: same test exit 0; elapsed 0.253 seconds.

## Smallest implemented boundary improvement

On POSIX, stdout is now read by a single deadline-aware `select.select`
loop with `os.read`, capped at `MAX_RESPONSE_CHARACTERS * 4 + 1` bytes.
There is no reader thread holding a Python buffered stream lock, so the
examiner can safely close its pipe immediately after killing the candidate's
own group and waiting for the candidate PID. Windows retains the existing
daemon-reader path: Windows process-tree and inherited-pipe deadlines remain
unqualified.

## Qualification boundaries

- SOURCE_INSPECTION: exact launcher behavior, output cap and timeout.
- TEST: WSL Ubuntu 26.04 ordinary and optimized subprocess suite, UID test;
  Windows ordinary subprocess and UID suites (report exact counts in PR).
- NOT PROVED: killing or quarantining intentionally detached descendants;
  native non-WSL Linux root private-examiner access denial; Windows tree
  containment or Windows inherited-pipe deadlines; independent examiner
  custody, authentic model execution, or neural transfer.
- HOLD: PR remains DRAFT and unmerged. Issues #146 and #147 are not closed.
  Passing unit tests cannot grant the model a graduation qualification.
