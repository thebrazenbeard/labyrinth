# Wave15 runner hostile review — independent test-only artifact

Target: Experimenter frozen PRE-FIX source
`9629bc44a7dc9c13c5d1822a6475c677ce460815`.

Isolated review branch: `wave15-runner-skeptic`; new file
`tests/test_wave15_runner_skeptic.py`. No changes to trial production code,
repository protections, remote PRs, or model/host services.

Reproduction:

```bash
python -B -m unittest discover -s tests -p test_wave15_runner_skeptic.py -v
python -O -B -m unittest discover -s tests -p test_wave15_runner_skeptic.py -v
```

## RED: S15-R1, unbound receipt to case and actual model answer

`run_trial` checked `verify_receipt_replay(source_receipt)` but did not bind
the replayed receipt's `seed`, `case_id`, `probe_trace` or underlying proposal
to the current trial Case and candidate's exact submitted decisions.

The unchanged fixed-process child actually returned `ABSTAIN`, but one
patched **host-side** `ProbeSession.submit` returned a genuine replayable WIN
receipt from a separate case or earlier same-case run. The report then
published `source_receipt_replay_valid=true`, `verified_success=true`.
Three independent tests preserve (i) wrong-case seed, (ii) stale same-case
PLAN in place of ABSTAIN, and (iii) wrong probe history after one charged
probe. These are defects in receipt/case/model causality, not claims that a
separately executed child escaped OS isolation.

A fifth RED tests transcript substitution rather than a stale receipt:
the unchanged command child emits `ACTION 0` followed by a valid winning
PLAN, while one patched `ProbeSession.probe` records `WAIT` as the charged
action. The source receipt is internally valid but must not be credited to
the model's distinct action sequence.

## RED: S15-R2, no raw-output-to-decision independent replay

Another single-function mutation of the host `_parse_response` interpreted
the unchanged child `ABSTAIN` output as a valid winning PLAN. The report
recorded the actual SHA-256 hash of the *ABSTAIN* bytes, then certified the
forged parsed PLAN as model success. The raw bounded response was not kept
for independent reconstruction of the parsed model decision.

Corrective source must retain bounded raw model output (host-private if
needed), separately reconstruct the parsed decision, bind each action to
the charged public ProbeSession transcript, and reject any source receipt
that does not match the exact Case/answer/observations. Merely replaying
the scorer's self-reported source receipt is insufficient.

## GREEN: bounded parser and model identity negative controls

- Duplicate JSON fields, NaN/nonfinite numeric encodings, multiple documents
  and bool plan actions are rejected by the existing strict parser.
- Missing absolute executable fails before child launch.
- Two different Python scripts at the same path have identical executable
  (interpreter) and argv hashes while producing different candidate text.
  The runner correctly marks `model_neural_identity_verified=false` and
  `model_weight_update_verified=false`; executable hash is **not** a
  cryptographic binding to loaded script/weights.

## Unproven boundaries

The child shares the OS user with its parent; this is not a sandbox.
`generation_seed_independence=UNATTESTED` and public deterministic source
permit seed/catalogue contamination. The report includes a host-only case
seed and unkeyed self-hash; it must not be released to candidates during
evaluation or treated as independently signed model provenance.

No actual neural model, trained checkpoint, private evaluator, Lappy model
server or independent credential custodian was exercised by these tests.
The runner's candid source-only limitations must remain explicit.

**Baseline verdict:** FAILS receipt-to-decision and raw-output causality;
source-only parser/resource bounds SURVIVE the scoped negative controls;
neural learning/generalization/custody UNKNOWN.

## After-fix independent readback

Experimenter source follow-up **`af54765c8a23a3c9c88ec04e500372e4556acecc`**
was fetched into a *separate detached worktree*, leaving the pre-fix test
branch unchanged. Independent hostile suite **8/8 PASS** under both normal
Python and `python -O`; the same suite on the frozen original `9629bc4`
continues **5 RED/3 GREEN**. Corrections bind the source receipt to the
current case ID/seed, actual proposal and charged probe trajectory. A
separate wire decoder independently reinterprets the actual bounded child
response and rejects a discrepancy with the original decision parser.

**After-fix verdict:** SURVIVES these five reproducible fault injections,
with real fixed and command subprocesses used. This is still a source-public
test with unauthenticated process/weight identity and **not** a qualified
neural inference or hidden independent evaluation. Output hashes and the
host-side report self-hash do not constitute cryptographic chain of custody.
