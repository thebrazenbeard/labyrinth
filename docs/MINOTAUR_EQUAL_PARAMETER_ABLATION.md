# Minotaur equal-parameter representation ablation

**Status: public synthetic diagnostic only. Qualification `NOT_ESTABLISHED`.**

The optional `labyrinth.minotaur_algebraic_ablation.run_ablation` measures
how much the compact modulo-97 head's transfer depends on its supplied
multiplication feature. It does not modify the installed/default Minotaur
architecture, train a neural network, install a checkpoint, or confer
graduation authority.

## Comparison

Both arms fit exactly four float64 coefficients for each of ADD, SUB, MUL
(12 total) using least squares on the same 12 nonwrapping examples per
operation. The operation token and modulo-97 output projection are fixed.

| Arm | Four features |
| --- | --- |
| `bilinear` | `[1, x/8, y/8, xy/64]` |
| `no_xy` | `[1, x/8, y/8, x²/64]` |

The `no_xy` arm substitutes a single-variable quadratic feature, preserving
parameter count while removing the cross-operand multiplication prior.
Each design matrix must have rank four or the study stops with an error.

Run without writing artifacts:

```bash
python -c 'import json, torch; from labyrinth.minotaur_algebraic_ablation import run_ablation; print(json.dumps(run_ablation(torch), sort_keys=True))'
```

The result binds source identity through file-content SHA-256 hashes, **not**
through a Git branch or commit name. An external reviewer must independently
associate those hashes with the intended source revision. It also includes training-sample digests,
per-operation full-domain accuracy, supervised-atom-excluded accuracy,
modular-wrap accuracy, and exact/stepwise free-running program scores by
length (one through six). The same deterministic program cases are used
for each arm. Only the offline evaluator computes full-domain arithmetic
and gold program trajectories; the prediction function receives no teacher
state or oracle label.

## Reproduced public measurements

With training data seeds 47, 42, and 2026 and program seed 808:

| Seed | `bilinear` full ADD/SUB/MUL | `no_xy` full ADD/SUB/MUL | Exact programs: bilinear / no_xy |
| --- | --- | --- | --- |
| 47 | 9409 / 9409 / 9409 | 9409 / 9409 / 82 | 128 / 45 |
| 42 | 9409 / 9409 / 9409 | 9409 / 9409 / 99 | 128 / 44 |
| 2026 | 9409 / 9409 / 9409 | 9409 / 9409 / 92 | 128 / 44 |

These results are representation-capacity comparisons under a complete
algebraic hypothesis class. The bilinear arm already contains `xy`, and
both arms are handed the modulo projection. Perfect synthetic transfer
therefore does **not** establish unaided arithmetic discovery, broad
reasoning, robustness to new operations/moduli, or an independently sealed
private evaluation.

The next controlled experiment should train both arms with matched neural
optimizer budgets, multiple independently selected initializations, and
preregistered unseen cases. Report actual closure/gradient calls and
model-selection exposure; never promote a public diagnostic to graduation.
