# Wave18 — source-public behavioral integrity protocol

Evidence class: **SOURCE_INSPECTION + EXECUTED_LOCAL_TEST**, not model training,
private evaluation, independent custody or graduation. Frozen baseline Git
`3e29c87c62bd0bffc8fb05c243e8db7bc6268e8c` (PR #165 Wave17 source).
This Wave18 change adds a separate opt-in in-process protocol and does not
change the underlying Novel Router mechanics or existing child-process runner.

## Observable protocol

`labyrinth.novel_router_behavioral_protocol.precommit(pairs, policies, budget)`
freezes ordered `(TRAIN_seed, HELDOUT_seed)` pairs, closure-free candidate function
bytecode signatures, and a common `Budget(max_probes=20, max_calls=90,
max_seconds=1.0)` **before** scoring. A digest binds the self-declared roster
and functions. Both policies receive the *same ceilings on every case*, not
identical observations or actual consumed probes. Trial order is pair → split →
policy. This source-public split is not semantically or statistically independent.

Each policy receives a two-method `PublicObservationPort` (`view`, `probe`).
No seed, Case, private transition rule, `make_case`, `search`, adjudicator or
terminal `submit` is passed as a direct attribute. Policy returns only a typed
`CandidateChoice(kind, actions, solvable_probability)`. The host performs
`ProbeSession.submit`, replays charged actions in a fresh session and separately
checks any success against the independent primitive plan scorer. *Only after
the candidate terminates* does the host perform finite closure to label whether
any winning solution exists, enabling a Brier score for predicted **solvability**
(not just correctness of the submitted path).

The bounded port counts `view` and `probe` as calls, with separate max probes.
Wall time is checked between calls and after return. `elapsed_ms` excludes host
replay/closure. A timeout, invalid output, or exception becomes a charged
ABSTAIN, never a win. This is **post-hoc timing**, not a hard interrupt of
untrusted Python that loops forever without calling the port.

Scoring: `Brier = (p_solvable - source_truth)^2`; loss adds `0.4` per ABSTAIN,
`1.0` per failed submitted PLAN, and `1.0` per false CLAIM_UNSAT. Recording
these components separately prevents treating unknown/abstention as success.
These hand-selected costs are explicit research choices, **not** calibrated
decision-theoretic utility. The baseline probabilities `.5` when uncertain and
`1.0` after observing a winning path are heuristics, not trained calibration.

Source receipts are held in frozen trial records as canonical JSON strings,
bound by SHA-256. `verify_local_record()` recomputes a case-bound replay of
the full probe trace and final decision; mutations fail even if an adversary
recalculates the standalone SHA. This proves consistency with current local
source only; it does **not** witness actual elapsed time, candidate emissions,
model identity, or independent custody.

## Executed source-only 100-pair check

Invocation from repo root (using built-in zero-update controls):

```python
from labyrinth.novel_router_behavioral_protocol import (
    precommit, evaluate, fixed_witness, feedback_witness)
policies = (fixed_witness, feedback_witness)
manifest = precommit(tuple((i, i + 100) for i in range(100)), policies)
report = evaluate(manifest, policies)
```

Manifest SHA-256: `be8a8745e0377044a966beec50f703ec2d301eedb479a67aa558e3039cb72dae`.
Both source ranges contain 62 source-verifiable winnable and 38 finite-UNSAT
cases. Scores below count **actual PLAN success**, not oracle solvability.

| Zero-update candidate | TRAIN wins / abstains | HELDOUT wins / abstains | TRAIN probes / calls | HELDOUT probes / calls | False wins / UNSAT |
|---|---:|---:|---:|---:|---:|
| Fixed observed witness | 5 / 95 | 6 / 94 | 499 / 599 | 491 / 591 | 0 / 0 |
| Feedback observed witness | 12 / 88 | 8 / 92 | 724 / 1022 | 700 / 989 | 0 / 0 |

Combined: 400 policy-case trials, 31 verified wins, 369 abstentions, 0 failed
submitted plans, 2,414 probes, 3,201 charged API calls, 0 optimizer updates.
Brier sum 92.25, abstention penalty 147.60, total declared loss 239.85.

Exact focused regressions, normal and Python optimization:

```
python -m unittest discover -s tests -p test_novel_router_behavioral_protocol.py -q
python -O -m unittest discover -s tests -p test_novel_router_behavioral_protocol.py -q
```

Both reported **11 tests OK** during Wave18 implementation. Hostile cases:

- False 100%-confident `PLAN COMMIT` was scored failed, never promoted to WIN.
- Unsupported impossibility claims were penalized against a winnable source case.
- Attempted extra probes hit an equal, explicit budget and became ABSTAIN.
- After-commit changes to policy bytecode identity rejected.
- Altered source trace rejected even if its standalone SHA was recalculated.
- **Critical unresolved limitation reproduced:** a closure-free policy reading
  mutable *global variables* can change after bytecode-only precommit without
  invalidating the manifest. Same-process policy closures are introspectable.
- Slow policy was labelled time-budget exhausted **after** returning, not
  preemptively interrupted.

## Verdicts and necessary future qualifications

**SURVIVES:** local mechanical replay; honest counted candidate calls/probes;
zero-update control separation; explicit false-plan, impossibility, abstention
and loss accounting; source-correct outcomes on the tested seeds.

**FAILS as independent custody:** bytecode-only commitment does not freeze
mutable globals, imported code, RNG/caches, source data or filesystem snapshots.
Python reflection can inspect port method closures. No external signer or
host-owned append-only witness exists here. Seed sets 0–99/100–199 are from
the same public generator and can be memorized or reconstructed. As a result,
the 400 rows cannot be treated as 400 independent mechanism families.

**UNKNOWN:** neural parameter learning, trained-vs-untrained transfer, genuine
unseen-mechanism performance, confidence calibration under new populations,
private case isolation, artifact identity or model-grade host limits.

For serious model qualification, freeze a hidden source-independent trial roster
and code/runtime artifacts under separate host custody, launch actual candidates
in qualified constrained child processes, independently witness time and I/O,
and validate structurally novel rule families before calculating effective n.

## Wave20 hostile manifest re-admission (stacked on PR #174)

An adversarial review reproduced five additional **source-level FAILS** in the
source-public protocol. These are distinct from the historical Wave18 counts:

1. `precommit(((0, 0),), ...)` rejected a duplicate train/heldout seed, but
   `evaluate()` accepted a manually constructed manifest with the same duplicate
   pair and recalculated self-declared SHA. Both slots then scored the identical
   case as if it constituted heldout evidence.
2. `verify_local_report()` accepted an empty manifest and fabricated empty
   zero-valued report because no individual record reached its replay checker.
3. A custom `str` subclass implementing deceptive equality as the choice
   `kind` made the evaluator index an empty `PLAN` actions tuple and abort the
   batch with `IndexError`, instead of charging `INVALID_DECISION`.
4. A `MappingProxyType` wrapping a totals dict whose `str`-subclass key later
   changed its hash behavior raised an uncaught `RuntimeError` inside
   `verify_local_report()` rather than returning a failed verification.
5. An independent Skeptic reattack found `verify_local_report()` comparing
   untrusted row `pair_index`, `split`, and `policy` before primitive type
   admission. `int` and `str` subclasses overriding `__ne__` could raise a
   `RuntimeError` and abort report verification. All three fields now have
   exact-type guards before comparison.

The repair revalidates the exact tuple roster, unique train/heldout seeds,
bounded budget, unique policy identities and canonical manifest digest at each
consumption boundary. Decision kinds must be primitive strings; the witness
calculation additionally guards against empty actions. Report totals require
primitive string keys and their examination fails closed on anomalous
comparison/hash behavior. Each failure now has a source-level regression test,
including spoofed-subclass objects and a self-rehashed forged roster. Full
normal and optimized suite commands:

```
python -m unittest discover -s tests -q
python -O -m unittest discover -s tests -q
```

**Evidence ceiling remains unchanged:** these are local source-replay integrity
properties, not independent roster custody. A malicious producer can still
create a *different valid* manifest and compute its digest. A self-declared
digest cannot prove a trial was precommitted before observations, a policy
actually executed unaltered, or that candidate code lacked source access.
The built-in controls execute **zero optimizer updates** and no trainable model
checkpoint is loaded, compared, saved, or reloaded. Neither higher source wins
nor lower declared loss demonstrates neural learning or structural transfer.
