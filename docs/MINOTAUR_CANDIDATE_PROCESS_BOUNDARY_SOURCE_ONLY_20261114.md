# Minotaur candidate subprocess boundary — source-only, November 14, 2026

**Status:** Source and actual same-user Windows tests only. NOT independent custody.
Parent source: Labyrinth Draft PR #228, commit `388eab6aa92d72a3caba675cd1f0cc8a1b06b871`.
Related unresolved blockers: #146 (examiner custody) and #147 (graduation semantics).

## Contract

`SubprocessCandidateAdapter((absolute_executable, *args), model_id=..., timeout_seconds=...)`
accepts bounded argv and runs one local child per public-turn request.
Each child consumes one UTF-8 JSON object on stdin:
`{"prompt": <the validated public request object>, "seed": <integer or null>}`.
It writes one bridge decision JSON object to stdout. The existing strict bridge parser
validates the returned decision and the examiner independently replays its effects.
The accumulated public history is in each prompt; there is no private Case in child memory.

Call `run_candidate(public, examiner_factory, adapter,
authorize_local_adapter=True, require_process_separation=True)` to refuse
arbitrary same-process Python callbacks. No change to existing default behavior.
Provenance is `MODEL_SUBPROCESS_UNATTESTED`, NEVER independently attested.
## Actual controls and limits

The child is launched without a shell; its directory contains only a public request.
Only the Windows SystemRoot/WINDIR loader variables are copied from the host
environment. Standard output is read with a bounded cap of 4x MAX_RESPONSE_CHARACTERS
plus one byte; excess output terminates the child and becomes invalid model output.
Timeouts, nonzero exits, and invalid UTF-8 are model-call errors.
Stderr is not copied into any receipt, and all child paths must be explicit.

> **HOSTILE FALSIFIER:** The Windows test supplied a known examiner private file
> path to a child of the SAME OS user. The child successfully read it. That
> result is recorded; process separation does not imply private-file denial.

Not provided: different OS principal, ACL/token confinement, network sandbox,
descendant-process containment, file syscall interception, model weight identity,
CPU/RAM quotas for child work, or independent trial custody. The child may be
malicious and cause external effects with the examiner user's ambient OS rights.
Do not run arbitrary untrusted executables on an examiner holding private data.

## Red / green evidence and remaining hold

Tests: `tests/test_minotaur_subprocess_adapter.py`. Four initial tests RED
on missing adapter/process-only contract (4 errors); GREEN 4/4 after first
implementation. A separate stdout-pressure test was RED on actual wrong
`MODEL_CALL_ERROR` after timeout; bounded read corrected it to
`MODEL_OUTPUT_INVALID`, 5/5 GREEN. Existing bridge 23/23 GREEN.
Two additional hostile tests were RED against the first implementation: a
caller could assign an in-process `generate` method to the subprocess adapter
instance, or monkeypatch its class method so `require_process_separation=True`
ran attacker Python in examiner memory. Slot-restricted instances plus a
host-pinned method dispatch made those tests GREEN; 7/7 now pass.

This is not a trained model run, standalone securely isolated examiner,
graduation receipt, independent review, or license for source merge.
Even the pinned method is a mutable Python module global if other untrusted
Python code is loaded into the examiner process: never do that.
