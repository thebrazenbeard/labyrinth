# Wave 16 — a mechanically new, source-public transfer challenge

**Status:** executable synthetic challenge; independent tests green locally.
**Starting source:** `thebrazenbeard/labyrinth`
`16f84839df869d2a89243d55baa212cf92527944`.

Code: `labyrinth/switchback_novel_transfer.py`.
Independent verifier: `tests/test_switchback_novel_transfer.py`.

The name links this as an *additional* Minotaur transfer challenge. It does
not patch, relabel or replace any of the existing Switchback source APIs,
training runs, official Qwen checkpoints, or protected grader gates.

## New causal state machine

The old Switchback has one three-bit register, binary input symbols, WAIT,
and a timed switch among five three-bit update families. This new task has
**two four-level reservoirs**, a **delayed packet in flight**, **routing
orientation**, and **two separate consumable resources**. Its control
alphabet is `LOAD`, `ROUTE`, `TICK`, `PURGE`, `COMMIT`, with `RESET` available
only for charged public observation trials. No action 0 or 1 exists.

The private rule at each phase is a triple `(lane_bias, coupling,
payload_shift)` generated from a deterministic SHA-256 seed. At a LOAD:

\[
p = 1 + ((L + 2R + o + \mathrm{payload\_shift}) \bmod 3).
\]

LOAD consumes one load, increments the tick, and queues `p` without changing
either visible reservoir. A later TICK uses the **rule active at delivery**,
which can differ from the rule active at loading. The delivery port depends
on orientation, a rule bias, and the parity of the two reservoirs. Delivery
adds the packet plus coupling to the chosen reservoir modulo four, and may
cross-couple one unit into the *other* reservoir. ROUTE toggles orientation,
while PURGE discards an undelivered packet. Both consume ticks, and ROUTE has
its own limited resource budget.

A winning COMMIT must arrive by the 11-action horizon with both reservoirs
at their declared target, no packet pending, and **at least one delivery
after the unknown routing-law change**. Exhaustive search considers every
legal action branch; returning `PROVEN_UNSAT_UNDER_DECLARED_RULES` requires
full finite closure. A deliberately capped search returns `NO_KNOWN_WIN`,
never proof of impossibility. Novel rule suggestions are recorded as
`NOVEL_RULESET_CHALLENGE`, not incorrectly rejected as impossible.

This creates an experimentally different causal problem:

1. LOAD is inert on both visible reservoir levels but changes a hidden
   delayed-state variable. Subsequent TICK outputs depend on that queued
   state. A memoryless three-bit update over the visible reservoir values
   cannot reproduce this two-action delayed causal dependency.
2. One generated case can reach all **16** distinct visible reservoir pairs
   (seed 99), versus eight states in the old single-register representation.
   That establishes greater *observed state capacity*, not a proof that no
   sufficiently augmented old-language emulator could simulate it.
3. Two canonical cases with identical initial public metadata (seeds **0
   and 668**) have distinguishable hidden rules: one authorized `LOAD`
   observation separates their pending packets. Mechanic novelty is not
   merely a new label assigned to otherwise identical experiments.
4. The prompt, action vocabulary, budgets, state variables, delivery laws,
   and success criterion differ. Natural-language robustness or neural
   structural transfer still must be measured separately.

## Independent test evidence

Run:

```bash
python -m unittest discover -s tests -p test_switchback_novel_transfer.py -q
python -O -m unittest discover -s tests -p test_switchback_novel_transfer.py -q
```

| Test | Observed local result |
|---|---|
| Focused source/public challenge tests | **14/14 PASS** normal and optimized `-O` |
| Generated cases independently adjudicated | **200** |
| Independent finite wins / bounded UNSAT | **124 / 76** |
| Disagreement with production finite search | **0** |
| Maximum independent backward-DP states per case | **743** |
| Maximum production BFS nodes per case | **885** |
| Differential transitions, independently reachable states | **83,768** across 40 seeds |
| Demonstrated public hidden-rule discriminator | Seeds **0 vs 668**; `LOAD` |
| Maximum distinct visible reservoir pairs reached in one case | **16**, seed 99 |
| Forged canonical Case via patched `__eq__` | **Rejected** |
| Forged finite closure via patched `State.__eq__` / hash | **Rejected** |
| Altered seed mechanics via patched `RoutingRule.__eq__` | **Rejected** after genesis correction |
| False WIN from corrupt `verify_plan` scorer | **Rejected** by separate tuple replay |
| False UNSAT from corrupt source `step` | **Rejected** by independent DFS closure |
| Fabricated public probe from corrupt `step` | **Blocked** by per-transition source parity |

The independent solver implements routing arithmetic from first principles
on primitive tuples, in different control order, without calling the
production step, search, or rule helper. It validates production witnesses,
reconstructs all reachable transitions for forty seeds, and rejects false
UNSAT if a known winning counterexample exists.

The production scorer now also has a **distinct primitive tuple-state
countercheck**. A forged `verify_plan` result cannot single-handedly turn an
invalid path into a verified win; a corrupted source `step` cannot turn a
known winning case into a finite UNSAT certificate, and probe transitions
must match separately recomputed raw-state physics. These are in-process
single-fault controls; they are not a physically independent verifier or a
defense against an attacker who controls both source implementations.

**Alternative-win acceptance:** On seed zero, production search returns
`ROUTE,TICK,LOAD,TICK,COMMIT`. The independent backward solver also finds
`TICK,TICK,TICK,TICK,TICK,TICK,TICK,ROUTE,LOAD,TICK,COMMIT`. The latter is
longer but legal; the adjudicator accepts both as `VERIFIED_WIN` without
requiring that a candidate match the cached shortest path.

## Evidence and custody boundaries

Every receipt explicitly marks:

- `evidence_class: SOURCE_PUBLIC_SYNTHETIC`
- `source_commit: UNBOUND`
- `private_custody: false`
- `neural_learning_verified: false`
- `graduation_eligible: false`

The SHA-256 receipt is an **unkeyed integrity checksum**, not a host-issued
signature or independent witness. ProbeSession is an in-process pedagogical
API, not a security sandbox: candidate code with Python process privileges
can read the module and reconstruct hidden seeded rules. The challenge
must therefore be separately randomized, custodied, and kept out of model
training before it could count as a truly withheld transfer examination.

No neural model, checkpoint, adapter, training optimizer, Lappy, or qualified
host was run by these tests. Parameter updates and actual candidate
generalization remain `UNKNOWN`.

## Hostile verdicts

- **SURVIVES** — bounded finite closure and alternative valid wins across
  the observed 200-seed/reference-DP and 40-seed transition suites.
- **SURVIVES** — demonstrably different source mechanism, action grammar,
  delayed state and candidate-visible counterfactuals relative to the five
  known three-bit Switchback kernels.
- **FAILS** — any claim that this source-public task is an independently
  secret examiner or by itself establishes neural structural transfer.
- **UNKNOWN** — unseen-language robustness, real model training, off-family
  generalization, independent evaluator custody, and population-level
  statistical power.
