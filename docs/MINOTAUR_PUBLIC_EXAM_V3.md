# Minotaur source-public examination v3 (opt-in)

Run from the repository root after installing no optional dependencies:

```bash
python -m labyrinth minotaur-exam --seed 17 --variant 2 --exact-cover --temporal-repair
python -O -m labyrinth minotaur-exam --seed 17 --variant 2 --exact-cover --temporal-repair
```

The `--temporal-repair` switch requires `--exact-cover`. An invocation without
either switch retains the original **8 cases/2 families/v1** JSON contract.
`--exact-cover` alone retains the **12 cases/3 families/v2** contract. The
two-switch v3 run adds four temporal-repair cases for **16 public cases and four
distinct declared mechanics**. None of these counts represents a blind model
evaluation, model parameter update, or structural generalization.

The fourth family involves a single repair machine, bounded energy that can
be replenished by a charged WAIT action, release times, deadlines, prerequisite
jobs, an action budget, and a total time horizon. Source-search policy results
are checked against the *specific* proposed sequence through a direct replay
and a separate exact-witness interpreter. An asserted finite impossibility
must agree with two independently implemented bounded reachability searches.
Incomplete searches return `NO_KNOWN_WIN` or `UNKNOWN`, never a global UNSAT
proof. A valid unexpected schedule can overturn a previous source claim; an
undeclared new affordance becomes `NOVEL_RULESET_CHALLENGE` for further review.

The public four-fixture control has two finite-rule wins and two declared-rule
UNSAT cases. Fixed scripts and a memorizing public catalogue are zero-update
controls. Changing the wording variant does not change mechanics. Even a
perfect public score could be achieved by a zero-update memorizer, and four
mechanics cannot satisfy the separate structural-transfer screen's requirement
for at least eight disjoint held-out and eight disjoint transfer mechanics.

The current source-public runner invokes **no local LLM**, accesses no private
candidate checkpoint, performs **zero neural weight updates**, and does not
establish genuine reasoning learning, unknown-rules impossibility, or private
evaluation custody. See `MINOTAUR_PUBLIC_EXAM_V2.md` and
`STRUCTURAL_TRANSFER_SCREEN_V1.md` for the preceding source and evidence gates.
