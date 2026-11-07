# Wave17 hostile receipt review — exact red source

**Source:** Labyrinth PR161 batch branch exact Git HEAD
`7ecbaf9e5029023fd76206da0fea2e3770389f77`.

**Evidence:** actual local fixed-child process batch, executable test-only
mutation of the returned receipt, no host model installation or neural update.
The genuine fixed batch (seed 1, generation seed 10, one process call) returns
`COMPLETED` and `verify_batch_receipt=True`.

## S17-01 — outer Git provenance field accepts arbitrary text

**FAILS.** The batch emits `source_commit` as a dict with exact keys `sha`
and `worktree_dirty`; the verifier never validates that field. Replacing it
with a plain string of 40 zeros and recomputing only the unkeyed outer
`batch_receipt_sha256` still yields `verify_batch_receipt=True`.
Independent test: `test_outer_source_commit_schema_must_reject_rehashed_string`.

## S17-02 — nested trial Git provenance can be rewritten independently

**FAILS.** Replacing `cases[0].trial.source_commit` with the unstructured
string `made-up-source`, recomputing the nested trial receipt hash and then
the outer batch hash, also verifies true. No model output, action, or oracle
decision was altered; this isolates provenance/schema failure from model
grading. Independent test:
`test_nested_trial_source_commit_schema_must_reject_rehashed_string`.

Command, from an exact checkout:

```bash
PYTHONPATH=. python3 -B -m unittest discover \
  -s tests -p test_wave17_batch_skeptic.py -v
PYTHONPATH=. python3 -O -B -m unittest discover \
  -s tests -p test_wave17_batch_skeptic.py -v
```

Both provenance tests yield **2 FAIL** on this frozen SHA. Source shape should
be validated in both locations: an exact builtin dictionary with exactly
`sha` and `worktree_dirty`, and each value either a valid lower-case 40-hex
SHA plus exact bool or a paired explicit `None` for unavailable Git state.
This is a schema consistency invariant, **not host Git attestation**:
anyone able to rehash an unkeyed receipt can still substitute a plausible
syntactically valid SHA. The verifier must never promote its value to
independently authenticated Git custody.

## S17-03 — decisive false model win from child ABSTAIN bytes

**FAILS (P0 false attribution, reproducible at frozen `7ecbaf9`).** A
genuine local fixed child returns `ABSTAIN`, and a separate source-only
oracle yields a valid WIN action sequence for seed 0. A test mutates the
*receipt only* into a purported command-mode trial (the argv is a named
fixture and **never executed**), preserving the exact original ABSTAIN
raw bytes, response length, SHA-256, and process receipt semantics. It
relabels `turn.decision` as the independently sourced PLAN, supplies a
valid adjudication and replayable session receipt for that plan, then
rehashes the nested trial, manifest, and outer batch.

`verify_batch_receipt(forged) == True`, with `source_verified_wins=1`.
This is an evaluator false positive: the actual child's output contained
no plan. It demonstrates that `_wire_turn_valid` proves consistency of
raw bytes with claimed child digests while `_trial_reverified` **does not
independently decode those bytes and bind their decision to `turn.decision`**.
Single trust-boundary omission, not evidence of an actual model or
external command escape.

### S17-03b — malformed double-key JSON passes as a parsed child response

**FAILS.** The genuine fixed child is recorded as parsed ABSTAIN. A
deep-copied receipt replaces its raw response with
`{"kind":"ABSTAIN","kind":"PLAN","actions":[]}`—duplicate JSON
`kind` keys, invalid PLAN length—and updates *all* legitimate response
digests, framed child byte count, trial and batch self-hashes. The batch
still reports `SOURCE_PROCESS_TRIAL_REPLAY_VALID` and verifies true.
Receipt consistency without an independent strict JSON grammar allows
this corrupted trial to be counted as a valid completion. Standalone RED
`test_duplicate_key_wire_json_cannot_be_verified_as_parsed_child`.

Unique red test:
`test_raw_abstain_cannot_be_relabelled_a_winning_command_plan`.
Requires independent, strict, bounded raw JSON parsing in the batch
verifier and canonical semantic comparison with the scored turn decision;
sharing the trial producer's parser would recreate a common-mode fault.

## S17-04 — nested process call budget accepts booleans

**FAILS on exact frozen source.** Genuine one-call fixed trial is
`COMPLETED`; change either `trial.process_calls` or
`trial.configuration.max_model_calls` from integer `1` to Boolean `True`,
recomputing the affected config, trial and batch hashes. The batch still
returns `verify_batch_receipt=True` because ordinary Python equality treats
`True == 1`. Regression
`test_nested_process_call_and_call_budget_must_be_exact_integers` has two
RED subtests. This is source receipt type/attempt-accounting laundering,
not evidence of an actually uncharged model call.

The complete Wave17 independent file now has **seven test methods**:
five RED methods with six exact falsifiers (two provenance, two budget,
fake child WIN, malformed double-key raw model output), plus two PASS-as-
negative-controls. The controls verify that a deep-copied second child
receipt or undeclared second process call is rejected and that merely
supplying an argv does not grant command launch authority. These controls
launch no command. All reproduce on frozen `7ecbaf9` normal and optimized
Python.

**Verdict: FAILS typed provenance and child-output attribution; physical
source and candidate model identity remain UNKNOWN.**
