# Minotaur Temporal Repair Tribunal — v1 (source-public)

## Mechanically distinct recurring dilemma

A single machine must complete every declared repair job. Each job has a
release tick, finishing deadline, execution duration, energy cost, and an
optional prerequisite set. The machine has a bounded battery; `WAIT` consumes
an action and one clock tick while restoring one energy, up to capacity.
Executing job `i` consumes an action, its duration, and its energy cost. The
exam requires all jobs to finish before their individual deadlines, a global
time horizon, and an action budget. Job order and the timing of waits can
change the outcome. This is *not* a renaming of static exact cover, hidden
navigation, or causal world selection.

The candidate submits an open-ended tuple: each nonnegative integer is a
declared job index and `-1` is a charged `WAIT` action. Repeating a job,
running before a release/prerequisite, running without energy, missing any
deadline, or exceeding either budget produces `VERIFIED_FAILURE`.

`public_view(s, variant=0..7)` supplies eight perspectives on the same
mechanics. All variants retain the same mechanics fingerprint: paraphrases
are **not** eight independent structural evaluation families.

## Source API

```python
from labyrinth.temporal_repair_tribunal import (
    families, public_view, search_schedule, verify_schedule,
    adjudicate_unexpected, novel_rule_challenge,
    independent_reachability, independent_witness_check,
    catalogue_control, fixed_script_control,
)

s = families()["recharge_window"]
challenge = public_view(s, variant=3)
result = search_schedule(s)
assert result["verdict"] == "VERIFIED_WIN"
assert verify_schedule(s, result["schedule"])["verdict"] == "VERIFIED_WIN"
assert independent_witness_check(s, result["schedule"])
```

`families()` provides four declared public examples: two SAT (`recharge_window`,
`release_pivot`) and two finite UNSAT (`dependency_cycle`,
`deadline_collision`). A bounded solver performs BFS over finite
`(time, energy, completed jobs)` states. Wins require direct sequential replay,
a separately written exact-witness interpreter, and an independent time-layer
DP existence check. A `PROVEN_UNSAT_UNDER_DECLARED_RULES` result is only issued
after the BFS exhausts and both the time-layer DP **and** a separately written
exhaustive DFS independently find no declared solution. State or node cap
exhaustion never becomes an impossibility certificate:

| Verdict | Meaning |
| --- | --- |
| `VERIFIED_WIN` | A concrete declared schedule passed independent replay |
| `VERIFIED_FAILURE` | A submitted schedule violates the declared rules |
| `NO_KNOWN_WIN` | Bounded source solver was interrupted without a proof |
| `PROVEN_UNSAT_UNDER_DECLARED_RULES` | Complete finite search, with independent negative checks, found no declared schedule |
| `UNKNOWN` | Independent implementations disagree or input/evidence is inadequate |
| `NOVEL_RULESET_CHALLENGE` | A proposed new capability changes the game and requires separate review |

`adjudicate_unexpected` independently checks an alternative schedule even when
the prior verdict was UNSAT. The exact source replay and separately coded
specific-witness checker must agree on the verdict and rules fingerprint. An
inconsistent positive **or** inconsistent negative yields `UNKNOWN`, never
automatic rejection of an overlooked valid solution. A valid alternative may
overturn the prior verdict. `novel_rule_challenge` records an undeclared
affordance, rejecting already-declared `RUN`/`WAIT`, but neither treats a new
proposal as an automatic win nor claims it is impossible.

## What the execution does — and does not — prove

The module uses **no model inference or optimizer update**. A fixed-script
control and a source-public catalogue memorizer are provided as explicit
zero-update baselines. A perfect score on these known examples is **not**
evidence of model learning or structural transfer. The proof scope is solely
the finite jobs, machine, energy, budgets, release/deadline windows, and
prerequisites supplied in the scenario. New outside resources, simultaneous
machines, or altered physical rules need new adjudication.

The protected structural-transfer screen still requires eight mechanically
distinct families per evaluation split with disjoint custody. Even after this
fourth family is wired into an opt-in CLI, public fixtures and self-generated
receipts cannot graduate a model. The training/evaluation controls and
official TrainingDay checkpoint authority remain separate.

## Reproduction

```shell
python -B -m unittest tests.test_temporal_repair_tribunal -q
python -O -B -m unittest tests.test_temporal_repair_tribunal -q
python -B -m unittest discover -s tests -q
python -O -B -m unittest discover -s tests -q
```

Hostile tests include 160 generated causal/time/energy graphs compared against
a *test-owned* independent exhaustive DFS, not the production solver's
transition implementation. Success is qualified only for those exact inputs;
source-public verification is not independent private examination custody.
