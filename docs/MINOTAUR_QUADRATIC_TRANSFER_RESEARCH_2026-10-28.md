# Public quadratic-activation transfer probe (research only)

## Exact subject

An experimental, random-initialized 33-parameter network consumes only two
numeric operand inputs. A trainable affine projection, square activation, and
trainable linear output are optimized with Adam. There is **no supplied `x*y`
input feature**, atomic lookup table, or multiplication routine in the forward
pass. The **square activation is itself a strong, deliberately chosen
polynomial inductive bias**: it creates a hypothesis class containing a global
quadratic multiplication function. This is not a generic reasoning learner.

Train: 121 even-even pairs where both operands range from 0 through 20.
Public holdout: 100 odd-odd pairs with disjoint operand identities.
Out-of-range: 144 pairs with operands 21 through 32.
Signed-range shift: 144 pairs with operands -12 through -1.

The supervision labels are unwrapped integer products, not just modulo-97
residues. The reported modulo-97 and modulo-101 answers both use an **external
deterministic modulo projection** after the network's scalar prediction.

## Exact reproducible results

Each update arm performs 3,000 Adam steps from a same-seed initialization;
frozen, zero-learning-rate matched-work, and corrupted-label controls are
scored on identical public cases.

| Seed | Arm | Odd in-range | Outside 21–32 | Negative -12..-1 | Mod 101 with external projection | Doubled input-coordinate units |
|---|---|---:|---:|---:|---:|---:|
| 41 | trained | 100/100 | 144/144 | 144/144 | 144/144 | 0/100 |
| 41 | frozen | 0/100 | 2/144 | 1/144 | 1/144 | 2/100 |
| 41 | zero LR | 0/100 | 2/144 | 1/144 | 1/144 | 2/100 |
| 41 | corrupted | 0/100 | 2/144 | 1/144 | 0/144 | 3/100 |
| 42 | trained | 100/100 | 144/144 | 144/144 | 144/144 | 0/100 |
| 42 | frozen | 1/100 | 3/144 | 1/144 | 1/144 | 1/100 |
| 42 | zero LR | 1/100 | 3/144 | 1/144 | 1/144 | 1/100 |
| 42 | corrupted | 0/100 | 2/144 | 0/144 | 3/144 | 2/100 |

An additional eight public exploratory seeds (43–50, same 3,000-step schedule)
each scored 100/100 odd, 144/144 out-of-range, 144/144 negative,
144/144 with externally supplied modulus 101, and 0/100 under the doubled
input-coordinate encoding. These additional public runs strengthen
reproducibility of this *specific polynomial architecture*, not independent
qualification or transfer beyond its function family.

## Interpretation and limits

The trained coefficients implement a quadratic mapping that extrapolates over
these numerical intervals. The data show a *learned coefficient effect under a
hand-chosen polynomial representation*, not autonomous discovery of general
multiplication or a transferable reasoning mechanism. The external modulo
projection makes the modulus-101 result especially narrow. Doubled coordinate
units without an adapter fail completely, establishing a clear representation
fragility. Numeric sign/range shifts are not natural-language representations.

Posthoc decomposition of the trained square-layer network on seed 41 recovers
the normalized polynomial coefficients `xy = 1.0`, with the `x²`, `y²`, `x`,
`y`, and constant terms all below approximately `1e-15` in absolute value.
This provides mechanistic evidence for the learned *quadratic coefficient*
within the hand-chosen function family; it does not remove the inductive-bias
limitation. The decomposition is never used to produce predictions.

The public split, architecture, hyperparameters and evaluation are not
independently sealed. More adversarial tests should vary the target family,
representation, and data-generation rule with an independently administered
holdout before any stronger claim. No model is graduated or deployed.

## Reproduce

```text
python -m labyrinth.minotaur_transfer_next --seed 41 --steps 3000
python -m labyrinth.minotaur_transfer_next --seed 42 --steps 3000
python -m unittest tests.test_minotaur_transfer_next -v
```

This source is additive and uncommitted in an isolated checkout.
