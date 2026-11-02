# Pinned Continuum Calc public training challenges

**Status: source-only, development use, no protected evaluation or TRAIN grant.**

`labyrinth.calc_challenges.generate_calc_training_challenges` creates seeded
public TRAIN cases using two exact operations from
[`thebrazenbeard/calc`](https://github.com/thebrazenbeard/calc) commit
`17e12d0f659b42732fb98e5730732aba9ff1b488`: exact rational polynomial
derivative evaluation and exact 2×2 matrix determinant. It checks the installed
`calc.algebra` source-file SHA-256 against
`6a1b269d14ddb34fbe4e8e5a9f56ac98c401fa98636cd3b13ab423cba1bead08`
before invoking either operation. The public task ID, prompt and metadata are
stable for a given seed/count. Independent elementary algebra checks in the
tests are the separate mathematical oracle for the Calc implementation.

To qualify with Calc source checked out at that exact commit, from this
repository's root use:

```bash
PYTHONPATH=<pinned-calc-checkout>/src:. python3 -m unittest -v tests.test_calc_challenges
```

Core Labyrinth still works without Calc installed; the optional real
integration test is then reported **SKIP**.

The generated `Challenge` objects can be passed to the existing
`labyrinth.evaluation.score_response`, which supplies only `challenge.prompt`
to a model and keeps `challenge.answer` on the local scoring side. The bridge
is deliberately **not wired into** the main `generate_maze` CLI. Its tier 5
is only a development identifier, not a calibrated difficulty level or a new
Q6 witness. Existing levels 1–4 and saved formats do not change.

This code refuses `eval`, `dev`, and other non-TRAIN splits. Its generator is
fully public and therefore unsuitable as a secret holdout. A separately
protected evaluator with inaccessible future seeds and answer custody is
needed before using such problems to claim generalization. It neither issues
corpus rights, verifies publisher identity, authenticates model weights, owns a
GPU lease, updates neural weights, nor authorizes a model promotion. Source
hash equality is an observation in the current Python process, not an
independent host signature or anti-tamper guarantee.

Calc currently presents a Python source package without a top-level license
file in the checked-out commit. The bridge imports the separately installed
code rather than copying it; any redistribution or external use needs an
explicit license/rights decision from the relevant owner. The mathematical
facts themselves are not a substitute for dataset provenance or consent.
