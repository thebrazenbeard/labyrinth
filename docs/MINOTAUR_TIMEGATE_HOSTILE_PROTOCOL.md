# Timegate adversarial Minotaur experiment

Research artifact based on live `thebrazenbeard/labyrinth` Draft PR #140
(`129aa7de4f46caa0f432196195a64877e5891a61`) and Draft PR #139
(`51eca4d273e113849d4dc64996c5c89a90a37c4c`), both read on 2026-11-07.

Purpose: a repeatable but *mechanically different* challenge in which a single
cell must be collected, activated during a configurable phase, then used before
its energy expires at a second configurable phase. Clock phase advances on
every attempted action, including invalid attempts. There are both an
irreversibly sealed panel control and an insufficient-energy control. This
tests timing, consumable resources, irreversible action, time-dependent state,
and counterfactual rule changes. No neural model is trained or evaluated.

```bash
python3 -m unittest discover -s tests -p test_timegate_exam.py -v
```

Observed deterministic baseline for seeds 0–119 (not statistically
independent task families): 120/120 public open cases have shortest witnesses
found by symbolic breadth-first search and independently replayed. The same
fixed script found at seed 0 succeeds in 16/120 (13.33%) of those cases.
120/120 sealed twins and 120/120 weak-cell twins are closed-world UNSAT; a
separate tuple-based implementation checks complete reachable closures.
There are 86 unique finite-rule configurations, but **one underlying concept**.
This does not constitute model generalization evidence.

A stronger deterministic **zero-update public-only baseline** reads the
candidate-visible natural-language phase requirements, computes waits, and
replays through the public session interface. It wins **120/120 open cases**
without neural training or hidden Case access and abstains as `UNKNOWN` for
**120/120 sealed and 120/120 weak-cell cases**. This demonstrates that even
perfect performance on this toy family's open controls is insufficient for
claims of learned reasoning or structural transfer.
One broader manual control ran the same deterministic parser across **2,000
open seeds: 2,000/2,000 VERIFIED_WIN**, with no model or parameter update.
More successes from this single known mechanism do not create independent
evidence of general reasoning. Honest reporting should count the conceptual
family once, not each seed as a new reasoning skill.

The `NOVEL_RULESET_CHALLENGE` verdict preserves an external rule-change
proposal for independent evaluation. It never silently declares a new action
valid or impossible under the existing six-action closure.

**Explicit candidate claim channel:** `CandidateSession.submit()` accepts
exactly one `WITNESS`, `ABSTAIN`, `UNSAT_CERTIFICATE`, or `NOVEL_RULE` claim.
The independent referee combines any earlier charged attempts with a submitted
witness, preventing transcript laundering. `ABSTAIN` with `UNKNOWN` or
`NO_KNOWN_WIN` is recorded without crediting a win; a complete finite-state
certificate receives only the qualified closed-world unsatisfiability label.
The evaluator records escape victory and certificate correctness separately:
`escape_verified=True`/`verified_success=True` only for a replayed escape,
while a complete UNSAT certificate returns `certificate_verified=True` with
`escape_verified=False`/`verified_success=False`. An admission or graduation
gate must not treat a proof of no escape as a successful escape.
After a terminal win or exhausted attempt budget, a new submission is rejected
rather than allowed to overwrite that outcome.
On every view/action, the in-process world's entire `(room, phase, carrying,
available, energy)` state is compared with independent replay of **all**
charged attempts. A mismatch—including an environment bug that suppresses an
otherwise independently verified escape—ends the session with internal
`EVALUATOR_DISAGREEMENT`, not a candidate failure. Transition and observation
exceptions likewise fail closed and preserve the charged attempted sequence.
Non-text or empty observation payloads also terminate as evaluator faults;
whether an otherwise well-formed observation is semantically truthful remains
an open independent-oracle question.
This integrity result is an examiner fault, not an additional candidate
learning/verdict category.
The `UNSAT_CERTIFICATE` claim must explicitly include the complete assumed
rule parameters (`period`, `charge_phase`, `pass_phase`, `initial_phase`,
`lifetime`, `sealed`), matching the examined world; this prevents a true
closure from silently certifying a different hidden rule. The seal is
deliberately not given away in the prompt, so matching those assumptions
does **not** independently prove the candidate discovered or inferred the seal.
Novel rule proposals are preserved pending independent review, not silently
discarded or counted as successful escapes. This is a source-level contract
only, not a secure evaluator service.

`test_hypothetical_manual_release()` is an adversarial **conditional**
counterexample: a sealed world that is correctly UNSAT under six declared
actions becomes winnable if a previously undeclared `manual_release` action
can open the gate while a traveler still carries the cell. An independent
replay then finds the proposed escape, but the result remains
`NOVEL_RULESET_CHALLENGE` pending real rule-custodian review, not a retroactive
"wrong proof" against the original exact six-action model. This demonstrates
the intended handling of unforeseen rule amendments without pretending they
have already been empirically established.

The novelty sandbox also requires the literal `manual_release` string;
objects that merely compare equal to that string are rejected. This matters
for the in-process Python API, though the public challenge still requires a
separately isolated and authenticated runtime before it can qualify models.

`CandidateSession` is **in-process and not secure** against Python object
inspection. It must never be used as a secret-held-out examiner in this form.
The independent verifier is a second code path from the same author and source
tree—not independent custody or a formal proof of natural-language semantics.

Labels are deliberately separate:

- `VERIFIED_WIN`: witnessed replay of a legal path with ≤16 attempts.
- `VERIFIED_FAILURE`: concrete attempted sequence did not reach outside.
- `NO_KNOWN_WIN`: truncated search or no victory within attempt budget,
  never advertised as a proof of globally impossible escape.
- `PROVEN_UNSAT_UNDER_DECLARED_RULES`: exhaustive reachable closure of the
  exact finite six-action world without a goal state.
- `UNKNOWN`: malformed/unadjudicable input or unfinished in-progress session.
- `NOVEL_RULESET_CHALLENGE`: proposed rule amendment preserved for review,
  neither accepted nor rejected by the closed-world verifier.

Hostile claim still unproven: that a neural Minotaur learns, adapts, transfers
to genuinely new mechanics, resists contamination, or finds overlooked rules.
