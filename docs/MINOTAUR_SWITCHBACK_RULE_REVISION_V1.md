# Minotaur Switchback Circuit — generated rule-revision exam

**Status:** executable, source-public synthetic examination; not a private
held-out test, neural learning result, host qualification, signed evaluator,
or authorized graduation artifact.

Build source: Labyrinth combined v7 tree `140840be58b9a8cf1df93dafea744306d4c03a06`,
whose Git tree matched the earlier lost local combined checkpoint. New code is
isolated in `labyrinth/switchback_{circuit,oracle,controls,cli}.py` with
`tests/test_switchback_circuit_exam.py`. No production service is involved.

## Mechanics and the changing rules

The world contains a three-bit causal register, an action clock, a bounded
number of input pulses and a mid-episode rule replacement. Each input bit
updates the register using one of **five genuinely different kernels**:

1. Parity-feedback shifting.
2. Rotation and modular addition.
3. Threshold-gated nonlinear bit masks.
4. State-dependent carry accumulation.
5. Majority-controlled mask and XOR.

Every generated instance selects the two full kernel parameter sets and the
switch threshold from a deterministic seed. `0`/`1` apply the current kernel;
`WAIT` advances time without injection; `COMMIT` wins only on the target value
after the revision and at least one post-revision input. The pulse budget and
nine-action horizon are independently checked. There may be valid wins or a
provably closed finite-state loss. A candidate can submit *any* valid plan:
the verifier does not require the designer's cached shortest sequence.

The separately implemented oracle regenerates each Case without using the
simulator's `make_case` or `kernel`, computes transitions from distinct
expressions, replays plans and searches finite states. A second memoized
depth-first finite closure independently challenges the breadth-first search
before any UNSAT certificate is promoted. Thus a search routine that falsely
reports `complete:true` cannot make a known-winnable case unsatisfiable;
verifiable alternate winning plans count as counterexamples. These two search
algorithms still share the declared oracle transition rules, so independent
external review of those rules remains necessary. To catch this shared-physics
failure, any terminal adjudication first cross-checks all 32 before/after
kernel input/output cells and **every action edge in the entire finite
DUT-reachable state graph**, without sampling only selected clock values.
Seed 0 has **166 reachable state tuples and 664 checked action edges** under
this traversal; other seeds generate different counts. The state ledger uses
exact primitive field tuples, not overloadable object equality. A mutated oracle
transition that contradicts the DUT must yield `UNKNOWN`/evaluator
disagreement, never a finite UNSAT certificate; both implementations can still
share a specification mistake, so this is not formal model qualification.
Because both engines use the same `State` type, every bounded primitive
register/clock/pulse/flag constructor combination is also checked against the
requested exact tuple before adjudication. This catches ordinary shared
constructor perturbations, but a deliberately context-dependent malicious
mutation of the live Python interpreter or final verifier remains within the
same-process trusted computing base.
Likewise, verified PLAN wins are independently replayed through the DUT
without calling the oracle `verify_plan` scorer; mutating only that scorer to
award a false win fails closed instead of self-certifying the false label.
A resource-capped search
returns `NO_KNOWN_WIN` with `complete:false`, never a proof of impossibility;
exhaustive search can return `PROVEN_UNSAT_UNDER_DECLARED_RULES` for the finite
action alphabet and nine-move horizon, **not** global metaphysical impossibility.
Untyped rule-change descriptions remain `UNKNOWN` and pending external review,
not rewarded as discoveries. False UNSAT claims retain actual winning
counterexample paths. A correct certificate is distinct from an escape win.

## Mechanically distinct versus merely relabeled

For the precommitted seeds 0–99, executable source tests count:

| Quantity | Measured count |
|---|---:|
| Distinct full paired transition-table signatures | 100 |
| Distinct *relabeling-invariant* summaries | 84 |
| Cases with a verified winning plan | 67 |
| Exhaustively unsatisfiable under declared horizon | 33 |

The invariant records, for each pre/post law and each input, its transition
image size and number of fixed states, sorted over input labels. Those values
are preserved under arbitrary state renaming and input swaps, so **at least
84 groups cannot all be mere label permutations**. This is a lower bound on
structural differences, not statistical independence or 84 neural-transfer
tests. Exact sampling independence and a qualified effective sample size
`n_eff` remain `UNKNOWN` without independently precommitted semantic custody.
In a separate proposed HELDOUT block (seeds 100–199), **88** distinct
invariant groups appear, **71** not seen among TRAIN invariant groups;
**78/100 cases** belong to those unseen groups. This is concrete structural
novelty relative to the frozen 0–99 dataset, not secrecy: the generator is in
this public source. A prewritten source-aware solver can still reason about it.

The public metadata is insufficient to identify all hidden laws: some cases
with identical public initial/target/budget settings have different solvability
verdicts (for example, seed 13 versus 14). In a separately bounded sample
of seeds 0–999, the public fingerprint yields **569** equivalence buckets;
**80** contain both WIN and UNSAT cases. Even a retrospective best majority
label per bucket reaches only **916/1000 (91.6%)**, which is an *in-sample*
zero-probe ceiling, not a prospective generalization bound. The totals are
717 wins and 283 finite UNSAT results; no model was run. Seed 13 and 14 have
identical initial public metadata and opposite results, but their observed
register values **differ after one legal `0` probe**, demonstrating that a
candidate can acquire information absent from the initial prompt.
Further hostile near-twins demonstrate different meanings for the
discriminators: seeds **3 versus 2513** have identical complete initial
public views and the same *family-pair labels*, but different transition
tables and opposite outcomes; seeds **44 versus 3566** have an identical
mechanic signature, but different initial/target conditions and opposite
outcomes. Therefore neither family names nor the mechanic hash determines
the correct case verdict. This generator uses one canonical presentation;
actual natural-language paraphrase resilience remains `UNKNOWN`.

Public `ProbeSession` offers up to
16 charged probes (`0`, `1`, `WAIT`, `RESET`) and returns only observed register
states, not private Rule objects. All 16 are usable before the final one-shot
answer; requesting a 17th probe forfeits that submission. A crashing,
ill-typed, or independent-oracle-disagreeing DUT probe is charged and seals
the session, rather than providing false candidate feedback or a free retry.
Each observation distinguishes a recognized command (`valid`) from a command
that actually advanced/reset the circuit (`applied`); an input blocked by the
pulse budget can be `valid:true` and `applied:false`, without pretending it
affected the register. Both flags are checked in independent probe replay.
The one-shot submission cannot be changed after adjudication. Observation and
caller-proposal aliases are copied; **two different trace validators** (oracle
and raw DUT replay) check each reported observation. Mutating only the oracle
trace validator cannot thereby self-certify a forged transcript. A
session-specific receipt binds the **exact probe count and
transcript** with a deterministic digest, so zero-probe and two-probe
submissions cannot claim identical provenance. This is an API boundary **only**:
the public JSON advertises separate `probe_actions` and `plan_actions`
(`COMMIT` is a plan submission action; `RESET` is probe-only).
Python callers with access to the private Case object or this open source can
inspect or recompute all underlying mechanics.

## Explicit non-learning controls

- A frozen seed-0 plan wins **7/100** generated cases by coincidence. It has
  zero parameter updates and no observations; benchmark successes must not be
  interpreted as reasoning.
- A privileged deterministic symbolic BFS finds the **67** finite wins and
  independently proves the **33** bounded UNSAT cases. It directly receives
  hidden case configuration; it is **not** a candidate-public-only model and
  cannot establish a language-model transfer rate.
- An adversarial source-aware **zero-probe lookup** memorizes the complete
  TRAIN 0–99 catalogue and uses only public initial/target/switch/pulse fields
  to match cases. Those four public fields uniquely identify **83/100** TRAIN
  entries; the other **17** lie in eight ambiguous collision groups. On TRAIN
  the memorizer gets **56 verified wins + 27 correct UNSAT certificates**, with
  **17 UNKNOWN** abstentions and no model updates. On proposed HELDOUT
  seeds 100–199, it returns **86 UNKNOWN, 12 VERIFIED_FAILURE, 1 VERIFIED_WIN,
  and 1 correct UNSAT certificate**. Perfect-looking TRAIN performance on
  answered cases is therefore reproducible contamination, not understanding.
- A stronger independently constructed source-aware lookup for seeds
  100–199 recovers the exact seed in **84/100** cases from initial public
  metadata; after **one legal `1` probe** it recovers **100/100**. A `0`
  probe identifies 98/100. These are fixed-block source-public catalog
  lookups, **zero neural updates**. The proposed HELDOUT label therefore
  does not confer true secrecy or an uncontaminated transfer test.
- Seed 0 has at least two distinct verified winning plans:
  `0,0,WAIT,0,COMMIT` and `0,WAIT,0,0,COMMIT`. Both pass independent replay.
  The second is not the shortest cached plan, which demonstrates acceptance
  of a valid alternative, not evidence of a historical novel model discovery.

Every core adjudication receipt marks `evidence_class:SYNTHETIC_SOURCE_TEST`,
`neural_learning_verified:false`, `private_custody:false`,
`graduation_eligible:false`, `source_commit:UNBOUND`, and
`semantic_family_independence:UNQUALIFIED`. A session wrapper also explicitly
marks nonlearning status and binds probe trace, count, and the core receipt.
Its SHA-256 is a reproducibility checksum, **not** an authenticated signature;
a controller of source and receipt can forge both. A separately custodied evaluator
is required before claims of real candidate learning, withheld mechanics,
or graduation.

## Run — zero external costs

```bash
python3 -m labyrinth.switchback_cli --seed 0
python3 -m labyrinth.switchback_cli --seed 2
python3 -m labyrinth.switchback_cli --seed 0 --control memorized
python3 -m unittest discover -s tests -p 'test_switchback_circuit_exam.py' -q
python3 -O -m unittest discover -s tests -p 'test_switchback_circuit_exam.py' -q
python3 -m unittest discover -s tests -q
```

The CLI prints one JSON receipt, with no network, credentials, model, paid
compute, persistent service or filesystem output. Tests include explicit
mutation of the DUT kernel, independent seed generation/differential rule
checks, invalid actions, budget enforcement, finite UNSAT versus UNKNOWN,
alternative valid wins, one-shot submissions and replay-tampering attacks.
