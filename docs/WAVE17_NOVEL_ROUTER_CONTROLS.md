# Wave 17 — source-public Novel Router behavioral controls

**Evidence class:** executable synthetic source tests only. No neural model,
weights, checkpoint, optimizer, private examiner, authenticated signatures or
host training. Exact base `thebrazenbeard/labyrinth`
`13a5d122a3c5a577023a38a6e32c891676362471`.

New source-only controls: `labyrinth/novel_router_public_controls.py`.
Tests: `tests/test_novel_router_public_controls.py`.

## Candidate contract

Policy code receives a narrow wrapper with **only** `view`, `probe`, and
`submit` methods; it never receives a target Case, hidden seed, before/after
rules, case ID, mechanism signature, or search result. The Python process is
not a secure custody barrier. Both source and finite mechanics are public.

Three bounded zero-update policies:

1. `blind_delayed_delivery` uses no observations. It schedules TICK until the
   declared switch, then LOAD/TICK/COMMIT. Wrong answers remain failures.
2. `deterministic_trial_witness` tries up to three predeclared legal schedules,
   resetting between probes when its twenty-probe budget allows. It submits
   only a candidate-visible witnessed path, otherwise ABSTAIN.
3. `seeded_feedback_witness` uses a public strategy seed (default 7) and
   post-intervention reservoir observations to make repeatable route choices.
   It submits a witnessed path or ABSTAIN. The strategy seed is not the
   private target seed, and **no weights update**.

## Fixed reproducible train / held-out results

All cases have source-replayed receipts; counts below are from 100 distinct
seeds in each block. The finite independent evaluator finds **62 winnable**
and **38 bounded UNSAT** per block. Policies never claim UNSAT.

| Policy | TRAIN 0–99 WIN / FAIL / UNKNOWN | HELDOUT 100–199 WIN / FAIL / UNKNOWN | Public probe totals TRAIN / HELDOUT |
|---|---|---|---|
| Blind fixed delivery | 5 / 95 / 0 | 6 / 94 / 0 | 0 / 0 |
| Deterministic witnessed trial | 13 / 0 / 87 | 17 / 0 / 83 | 1,513 / 1,456 |
| Seeded feedback (seed 7) | 12 / 0 / 88 | 13 / 0 / 87 | 749 / 720 |

Each policy emits **zero invalid actions** on the measured blocks, obeys the
11-action plan and 20-probe limits, and has zero model updates. The feedback
policies do not produce false verified wins; they abstain frequently.

Budget matching is necessary to avoid treating different source resource
limits as equivalent. Pairing deterministically by exact public
`(switch_at, maximum_loads, maximum_routes)` yields **76 train/held-out pairs**
across 18 budget cells. Within those matched pairs, the oracle finds 52
winnable TRAIN and 47 winnable HELDOUT cases. WIN counts are:

| Policy | TRAIN paired WIN | HELDOUT paired WIN |
|---|---:|---:|
| Blind | 5 | 6 |
| Deterministic trial | 13 | 13 |
| Seeded feedback | 12 | 8 |

## Anti-contamination, source provenance and remaining unknowns

- TRAIN/HELDOUT are disjoint **seed ranges**, not proof of independently
  designed mechanism families or private custody. Both blocks use the same
  published delayed-router generator, with up to 24 law configurations per
  phase. Distinct signatures are 100/100 per block; exact before/after rule
  pairs number 88 in TRAIN and 91 in HELDOUT. None establishes n_eff=100.
- The evaluator has a separate finite closure and synthetic source-checksum
  receipts. SHA-256 here is an unkeyed integrity check, not a signature.
- A publicly witnessed solution is a concrete verified path **under declared
  rules**. It is not evidence of a neural weight update, independent language
  understanding, or transfer to an unseen physical law.
- A bare `UNKNOWN`/ABSTAIN does not prove impossibility. Existing finite
  closure certificates remain separate from behavioral baselines.
- Invalid probes consume budget; returned public views and finalized receipt
  histories are checked for detached mutation in the test suite.

Run source-only focused tests:

```bash
python -m unittest discover -s tests -p test_novel_router_public_controls.py -q
python -O -m unittest discover -s tests -p test_novel_router_public_controls.py -q
```

**Verdict:** bounded, observable behavioral baselines survive these tests;
actual model generalization and independent examiner qualification remain
**UNKNOWN**.
