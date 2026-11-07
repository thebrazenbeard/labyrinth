# Switchback source-public preregistered process batch (Wave 16)

**Status:** executable source-only test instrument. No inference of neural
learning, adaptation, private heldout success, or independent examiner custody.
This report intentionally preserves all predeclared cases in its denominator,
including failed/invalid/time-skipped cases.

## Zero-update process control (no model or host access)

```sh
python3 -m scripts.switchback_batch_trial --seeds 0 1 2 \
  --generation-seeds 17 18 19 --mode fixed --fixed-actions 0 RESET \
  --max-calls-per-case 3 --max-total-calls 9 \
  --call-timeout 1 --max-total-seconds 15
python3 -m unittest discover -s tests -p test_switchback_batch_trial.py -q
python3 -O -m unittest discover -s tests -p test_switchback_batch_trial.py -q
```

The CLI prints one JSON batch receipt. Frozen `BatchConfig` binds the ordered
distinct case seeds, independent generation seeds, per-case limits, total call
reservation, global time reservation, and response size. `AdapterSpec` binds
the type of subprocess and optional model artifact path. Both manifest and
batch report have deterministic SHA-256 integrity digests, **not** signatures
or tamperproof storage.

An opted-in local command can run by giving `--mode command --authorize-local
--command PROGRAM ARGS...`; Ollama requires the same explicit opt-in and a
model name. Running the source or default fixed fixture never contacts Ollama,
Coinbase, a host machine, or paid compute. This is a same-user child process,
not a sandboxed independently authenticated model boundary.

## Admission and evidence

All cases receive a row even if global resources run out. Before each case the
runner reserves `max_calls_per_case` global calls and
`max_calls_per_case * call_timeout_seconds + 1` seconds. Unused reservation is
**not** recycled based on lucky early wins; otherwise case outcomes would
silently change which later cases were admitted. In addition there is a
best-effort monotonic wall-time admission check. These bounds are a
pre-admission accounting protocol, **not** an operating-system resource quota;
the total runtime can exceed the reservation during cleanup/review.

Each admitted case uses `run_trial` with only the public candidate prompt and
trusted `ProbeSession` adjudication. The batch independently checks individual
receipt hashes, case ID and seed, source replay, exact candidate action trace,
terminal model proposal, call count, configuration, and bounded process status.
The required updated lower-level runner preserves bounded `raw_response` for
each successful call; the batch checks its UTF-8 digest, then reconstructs the
child stdout hash/size from the exact command bytes or the Python child envelope
(including PID/frame fields). A lower-level runner version that omits raw text
is deliberately refused rather than having child provenance inferred from a
hash alone. For fixed actions it also checks decisions against the frozen
known script.
`verify_batch_receipt` rechecks row admissions, resource reservations, aggregate
counts, case bindings, and source receipts. A rehashed but fabricated win,
resource count or neural-update claim must not pass the verifier.

**Privacy/custody:** The JSON contains host-private case seed and raw model
output text, not merely public exam prompts. It must not be automatically
published, uploaded to an issue or shared outside the intended examiner.
Per-call digest reconstruction detects some altered same-user receipts; it
still cannot attest which physical executable originally produced the bytes.

Verdict denominators are *predeclared cases*, not just cases that ran. Source
families/seed relabelings are correlated; counting several seeds does not
establish independent mechanical family sample size (`n_eff`).

## Negative controls and deliberate limitations

- A fixed zero-update policy and source-aware single-case win cannot earn a
  perfect batch score by dropping the losing case.
- Duplicate seeds are rejected; case/generation seeds are immutable.
- A replay-valid receipt from case 0 cannot be reused for case 1.
- A faked 100% scorer, forged aggregate, malformed/uncharged receipt, or
  unauthenticated model-weight update flag cannot establish success.
- Partial per-case child timeout remains a non-win, never magically omitted.
- Source-visible case generation allows source-aware policies to identify
  cases from public probes. This is **open-book/source-public** testing, not
  blind heldout generalization.
- Same-process Python imports, parser helpers, mutable interpreter state and
  command-controlled model identity are part of a trust boundary not sealed by
  SHA-256. A genuine model trial requires a verified executable/weights digest,
  qualified local host, independent source custody, a frozen unseen mechanic
  split, and a later evaluator not writable by the candidate.
- A recorded win proves only that a particular declared finite puzzle witness
  replayed successfully under the current source rules. Neither arbitrary
  introspection nor a neural parameter update is inferred.

The original Wave15 release (`16f84839`) had an independently reproduced
shared `_read_json` scoring-parser vulnerability: actual ABSTAIN bytes could be
misread as a forged PLAN. The revised Wave15 runner and its bounded raw-response
receipt dependency must be included in any valid Wave16 build; using Wave16
alone atop the original release intentionally fails child-wire checks. Even
with that source fix, wire provenance is explicitly
`SAME_OS_USER_UNATTESTED`.
