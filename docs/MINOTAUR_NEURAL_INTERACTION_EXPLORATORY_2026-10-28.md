# Minotaur neural-interaction exploratory study (not qualification)

Source base: Labyrinth PR #32 at `48c6419beb67f031e59613cb0fbef57e633b2398`.
Local uncommitted additions: `labyrinth/minotaur_neural_interaction.py` and
`tests/test_minotaur_neural_interaction.py`.

## Question

Can a compact neural function learn numerical multiplication behavior from
incomplete examples without receiving the engineered `x*y` input feature of
the existing 12-parameter algebraic head?

## Setup

- Random-initialized 1,185-parameter NumPy MLP, two scalar inputs, two tanh
  layers, linear output. No atomic table or supplied product feature in forward.
- Supervision: all 121 pairs from even-numbered operands 0, 2, ..., 20.
- Public interpolation holdout: 100 pairs from odd-numbered operands 1, 3,
  ..., 19, with **zero** shared operand identities.
- Public extrapolation challenge: 144 pairs whose two operands are 21..32.
- Full-batch Adam; 10,000 training iterations for trained, corrupted-label,
  and zero-learning-rate matched-compute arms. Frozen initialization is scored.
- Output: rounded real-valued prediction followed by supplied modulo-97
  projection; labels are unwrapped integer products, not merely mod-97 values.
- Two independently initialized seeds: 41 and 42.

## Fresh local results

Exact modulo-97 accuracy (correct / total):

| Seed | Arm | Odd-operand interpolation | 21..32 extrapolation |
| --- | --- | ---: | ---: |
| 41 | Trained | 99/100 | 0/144 |
| 41 | Frozen | 0/100 | 0/144 |
| 41 | Zero-LR sham | 0/100 | 0/144 |
| 41 | Corrupted labels | 2/100 | 0/144 |
| 42 | Trained | 75/100 | 1/144 |
| 42 | Frozen | 2/100 | 2/144 |
| 42 | Zero-LR sham | 2/100 | 2/144 |
| 42 | Corrupted labels | 3/100 | 1/144 |

Commands:

```console
python -m labyrinth.minotaur_neural_interaction --seed 41 --steps 10000
python -m labyrinth.minotaur_neural_interaction --seed 42 --steps 10000
python -m unittest discover -s tests -p test_minotaur_neural_interaction.py -q
python -m unittest discover -s tests -q
```

At this snapshot: focused 3/3 tests passed in 10.2 seconds, full suite 154
tests passed with 11 skips in 14.6 seconds. The two CLI studies completed
in approximately 6.5 seconds wall time combined on the local CPU.

## Interpretation and blockers

The trained network acquired a narrow, nontrivial interpolation ability from
operand-disjoint examples, unlike its matched controls. It **did not** learn
reliable multiplication extrapolation outside the numerical training range.

This experiment is exploratory and its public odd-operand score influenced
development iteration. It is not a sealed blind test. A modulo projection and
unwrapped product labels are supplied priors. It uses a scratch MLP rather
than a pretrained Vera model or the official private Minotaur examiner.
No model is qualified, graduated, deployed, or authorized for promotion.

Next experiment: freeze architecture and hyperparameters before a privately
held-out family and representation shift, use only task-legitimate modular
labels, run more seeds with paired controls, and include compound-program
traces rather than only isolated atomic tuples.
