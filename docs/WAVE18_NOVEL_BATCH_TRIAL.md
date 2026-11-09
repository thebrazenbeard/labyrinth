# Wave 18 — source-public Novel Router multi-case process trial

**Status:** executable, bounded, zero-cost source instrument built on the
authenticated PR165 Novel Router process source (`3e29c87c62bd0bffc8fb05c243e8db7bc6268e8c`).
This is neither privately heldout evaluation nor a neural learning result.

## Run a two-case zero-update child fixture

```bash
python3 -m scripts.novel_router_batch_trial --seeds 0 1 \
  --generation-seeds 41 42 --max-calls-per-case 1 \
  --max-total-calls 2 --call-timeout 2 --max-total-seconds 7 \
  --fixture-json '{"kind":"ABSTAIN"}'

python3 -m unittest discover -s tests -p test_novel_router_batch_trial.py -q
python3 -O -m unittest discover -s tests -p test_novel_router_batch_trial.py -q
```

The CLI prints a **redacted JSON summary by default**, with no private case
seeds, generation-seed list, candidate stdout, raw child bytes, source receipt,
or subprocess argv. To deliberately emit the complete **host-private** receipt,
add `--emit-private-json-stdout` *before* any `--command` option. Treat that
receipt as sensitive even for synthetic source-public examinations.

An explicit local command is supported with `--authorize-command --command
ABSOLUTE_EXECUTABLE [ARGS...]`. No shell, implicit provider access, model
launch, host transport, paid compute or external service is used by default.
The only zero-update fixture launches a separate Python child on the same OS
user; it is not a security sandbox. The parent sends the public view and
*public generation seed* but not the private case seed/rule/case ID.

## Frozen examination and budget semantics

The immutable `BatchConfig` preregisters **1–32 distinct case seeds**, one
generation seed per case, per-case model call count, global model call count,
per-call timeout, global wall-time admission budget and stdout byte limit.
The immutable source-provided `AdapterSpec` freezes a fixture transcript or
explicitly authorized argv. A candidate-supplied Python factory is not allowed.

Before a case is admitted, the examiner reserves **all** its possible model
calls and `max_calls_per_case * call_timeout_seconds + 1` seconds. Reservations
are never recycled after early lucky answers, preventing outcome-dependent
selection of later exam cases. Budget-exhausted, timed-out, invalid or skipped
cases retain their original manifest positions and count in the denominator.

The global budget is a **pre-admission accounting bound**, not a host-wide
kernel quota. Subprocess termination and cleanup are bounded best-effort
operations; physical OS custody and full descendant containment remain
unattested. A case must satisfy its own single-trial limits as well.

## Independent source-level evidence review

Every admitted case runs `scripts.novel_router_model_trial.run_trial` in a
bounded separate child process. Wave18 additionally checks:

- Exact declared case seed, public generation seed, adapter and call/time cap.
- Full per-call raw captured stdout bytes, SHA-256 and output size; strict
  separately implemented JSON grammar and terminal proposal correspondence.
- Every charged public `ACTION` and the source `ProbeSession` trace.
- Final source case ID and declared finite-rule PLAN verification.
- A fixed fixture's actual stdout must equal its committed zero-update script.
- The source runner's own complete replay must agree with the batch reviewer.
- All rows, reservations and aggregate win/valid/skip counts are reconstructed
  from the immutable manifest. Rehashed but contradictory batch statistics are
  rejected, including Python Boolean/int equality spoofing.

Both the per-case and batch SHA-256 receipts are **unkeyed checksums**. They
support detectable alteration of committed source receipts; they are not
independent signatures and cannot authenticate the physical executable, GPU,
source tree, user account, PID or stdout origin against an attacker who
controls both examiner layers. The source generating these cases is public and
enumerable, and mechanically correlated seeds cannot be counted as independent
unseen reasoning families.

## Negative controls and meaningful verdicts

The executable suite includes genuine fixed-child process runs and one
explicitly authorized local command, along with duplicate-seed rejection,
single-case cherry-picking, missing model calls, per-case timeout, forged
100% scorer, wrong-case receipt reuse, raw-output hash rewrite, rehashed
resource-count manipulation, malformed candidate JSON, fabricated neural
qualification, and private-output leakage controls. Passing these tests does
not imply an actual model was evaluated.

The source oracle can verify alternative valid plans, detect failures, retain
bounded uncertainty, and record potential novel rules without converting a
hypothesis into a verified discovery. `VERIFIED_WIN` describes **one synthetic
source-governed witness under declared rules**, not proof of neural transfer.

The batch always reports:

- `private_heldout_qualified: false`
- `neural_learning_verified: false`
- `model_weight_update_verified: false`
- `model_identity_qualified: false`
- `generalization_qualified: false`
- `generation_seed_effect_verified: false`
- `comparison_qualified: false`

All model-training/heldout qualifications remain **UNKNOWN** until a separate
qualified host process, authenticated weights, withheld structural mechanics,
matched nonlearning baselines and independent physical custody are actually
demonstrated. A source test cannot be elevated into that evidence class.
