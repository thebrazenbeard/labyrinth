# Minotaur Wave63 — Windows subprocess full-suite timing discrimination

**Date:** 2026-11-14. **Stacked source:** Labyrinth Draft PR #244,
`8cd023f9670902f4d4bb955aea98fddaefe22d53`.
**Scope:** Tests only. No production adapter, protocol, examiner state,
credential, permission, model weights or runtime installation changed.

## Actual failed baseline: preserve as RED

Prior [Labyrinth PR #243](https://github.com/thebrazenbeard/labyrinth/pull/243)
exact-head Lappy Windows full suite ran 1,275 tests, **3 failures and
2 errors**, 6 skipped, exit 1. Five outcomes came from the Windows
subprocess-adapter tests: two blank-response JSON decode errors and
three unexpected MODEL_CALL_ERROR classifications. A focused module
rerun passed 10/10 on both PR #243 and #244; this is a true unresolved
intermittent/full-suite-context blocker, not a known production bug.

## Fresh independent diagnostics

- Live Lappy at ~18:15 EST: RAM 32 GB; free physical memory about
  2.2 GB; committed 68.9 GB/103.6 GB; measured memory paging
  approximately 7,374 pages/second. This sample occurred **after**
  the prior full-suite RED, not simultaneously: plausible pressure,
  but **causal attribution UNKNOWN**. No other workloads terminated.
- Discovery prefix through the exact preceding 577 tests ran
  577/577 with 5 skipped, followed by the same-process Windows
  subprocess module 10/10 PASS; explicit exception tracing showed
  only expected 0.25-second TimeoutError.

- Another 20 consecutive Windows module repetitions of 10 tests
  each: 200/200 PASS. Observed only expected 0.25-second timeouts.
  This rejects a reliably deterministic failure under those conditions;
  it does not erase the previous full-suite RED.

## Deterministic test-contract mismatch, red before green

`SubprocessCandidateAdapter` has always defaulted to a **10-second**
response deadline. The synthetic `probe()` helper quietly overrode
it to **2 seconds** even for successful child-output protocol tests.
A synthetic child loading input, sleeping 2.4 seconds, then printing a
valid ACTION produced `MODEL_CALL_ERROR` before the patch:
focused RED exit 1, expected ACTION. Aligning the helper's default
with the production adapter's existing 10 seconds made the identical
public-interface test GREEN on Windows Python3.14.5.

The explicit 0.25-second timeout and inherited-pipe tests remain
unchanged; they continue to test fail-closed behavior. The cap test
now writes 200,000 characters and sleeps for 30 seconds, sets a
10-second deadline, and requires MODEL_OUTPUT_INVALID in **under
8 seconds**: no waiting for EOF or hanging child can falsely pass.

Post-change Windows 11/11 subprocess tests PASS; WSL Ubuntu26.04
11/11 normal and 11/11 optimized PASS. No weakened assertion was
removed; new delayed-child and elapsed-time assertions added.

## Qualification boundary

This is a **test-fixture reliability improvement**, not proof the
earlier Windows full-suite failure is explained or production reader
reliability is qualified. Full Windows and immutable-head GitHub CI
results must be supplied separately after execution. Examiner
private-case custody #146, semantic mechanism independence #147,
real model inference/training and generalization remain UNKNOWN.
**DRAFT/HOLD; do not merge, deploy or graduate.**
