# Minotaur source-public examination v2 (opt-in)

Run from the repository root:

```bash
python -m labyrinth minotaur-exam --seed 17 --variant 2 --exact-cover
python -O -m labyrinth minotaur-exam --seed 17 --variant 2 --exact-cover
```

The legacy invocation **without** `--exact-cover` intentionally preserves its
`minotaur-source-public-v1` contract: 2 families and 8 cases. The flag yields
`minotaur-source-public-v2`, 3 *mechanically distinct* families and 12 cases:

1. `constant_exam`: finite navigation and offering.
2. `veiled_fork`: contingent planning under incomplete observation.
3. `exact_cover`: disjoint three-site crew allocation within an action budget.

The new family replays any claimed winning allocation, separately checks that
specific witness with bitwise site accounting, and uses independently coded
subset enumeration for positive and negative *existence* checks. For a claimed
finite UNSAT result, a **third**, iterative bitmask reachability DP also checks
for valid coverage. Exhausting that bounded DP's work cap yields `UNKNOWN`,
not impossibility. Merely proving that some winning allocation exists does
**not** validate the proposed witness. Disagreement on a claimed win or finite
impossibility is vetoed to `UNKNOWN`.
Incomplete search is `NO_KNOWN_WIN`, never global impossibility.
The exact-cover tribunal separately supports independent adjudication of
unexpected proposed allocations and labels undeclared new crews
`NOVEL_RULESET_CHALLENGE` without automatically accepting them.

For these four **public fixtures**, the symbolic solver finds 2 wins and 2
finite-rule UNSAT cases; a single fixed-script zero-update control wins 0/4.
An exact-public-fixture catalogue can memorize all 4/4 without any learning.
Changing `--seed` or `--variant` changes wording or perspective, **not** the
underlying mechanics or independence of evaluation cases. Twelve public cases
are only **three declared mechanics**, not twelve independent heldout families.
They cannot meet `structural_transfer_screen`'s at-least-eight genuinely
independent heldout **and** eight distinct transfer-mechanic requirement, with
no mechanics reused across splits. A memorizer can still score perfectly on
this public source without a parameter update. No model is called, no neural
parameters change, and no private or blind examination is involved. Do **not**
use this receipt as model graduation, neural-learning, structural transfer or
absolute real-world impossibility evidence.

The v2 flag creates no files unless `--report NEW_FILE.json` is supplied;
the existing write-once report protection applies. The CLI exits nonzero
when an independent check disagrees; that is not a model failure score.
