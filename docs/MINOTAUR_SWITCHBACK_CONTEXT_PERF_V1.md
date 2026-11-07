# Switchback: bounded canonical-case fastpath (source-only)

**Source base:** `f36ee55eb003692091904f6e9cc4f7f75a119558`.
**Scope:** `independent_transition_parity` only. This optimization does not
change declared rules, search logic, receipts, verdicts, custody or model
training. It runs in an isolated scratch checkout, not on Lappy.

For each parity traversal, the evaluator validates the exact canonical Case,
checks the State constructor and all before/after kernel truth cells, and
revalidates the Case after these operations. It then grants only this single
synchronous traversal a `ContextVar` validation scope. Every `oracle_step`
still executes and compares against the separate DUT transition; it skips
expensive canonical case regeneration **only** if:

1. The mutable scope record remains `active`.
2. Its owner thread ID matches the current execution thread.
3. The candidate Case is the **same Python object** (`is`).
4. Its full primitive Case/Rule fields equal the prevalidated snapshot,
   including **exact Python types**. Ordinary tuple equality would wrongly
   treat `False` or `0.0` as equivalent to the canonical integer `0`.
   The enclosing `Case`, `before Rule`, and `after Rule` classes must also
   remain exact; same-field subclasses are not canonical source objects.

The `finally` block invalidates the shared scope record **before** resetting
its ContextVar token, so `copy_context()` child copies cannot retain
authority after the parity invocation finishes. Direct `oracle_step` calls,
new sessions, other threads, same-seed different objects, and altered Cases
still execute normal canonical validation. There is **no global cache**.

## Measured before/after

Single exact seed-0 winning PLAN + session receipt independent replay:

```bash
python3 -m labyrinth.switchback_perf_benchmark
```

| `cProfile` observation | Clean base | Scoped fastpath |
|---|---:|---:|
| Profiled wall time | 1.508556 s | 0.247628 s |
| Total profile calls | 4,007,103 | 345,589 |
| `_valid_case` calls | 4,688 | 47 |
| `oracle_case` calls | 4,692 | 51 |
| `_rule_output` calls | 301,641 | 4,617 |
| `oracle_step` calls | 4,663 | 4,663 |
| `independent_transition_parity` calls | 7 | 7 |
| Verdict / replay | VERIFIED_WIN / true | VERIFIED_WIN / true |

The ~6.09× ratio is for these two profiled source-only runs, **not** a
general throughput forecast or trained-model result. The exact counts are
reproducible with the script, but elapsed time depends on current execution
load. Three further optimized samples were `0.236202`, `0.280832`, and
`0.255422` seconds. No inference about a private evaluator, neural learning or Lappy host
speed follows from these microbenchmarks.

Hostile source tests include copied-context lifetime and cross-thread checks,
same-seed wrong object, type-confusable in-place `seed` mutations
(`False`/`0.0`), same-field subclass substitutions, mutations during/preceding parity, exception
cleanup, direct calls outside scope, and retained independent adjudication.
Same-interpreter arbitrary monkeypatching of the final verifier or interpreter
remains outside this source-only TCB qualification.
