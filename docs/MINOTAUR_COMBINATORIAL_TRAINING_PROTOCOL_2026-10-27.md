# Minotaur V2 combinatorial training protocol — 2026-10-27

Status: frozen public synthetic development protocol; **not qualification or production evidence**.

The executable protocol is `labyrinth/minotaur_combinatorial_training.py`. It trains a
fresh Minotaur V2 dual-head Transformer on exactly 128 balanced cases from
`labyrinth/minotaur_combinatorial_curriculum.py`, using program lengths 1 through 6.
It does not import or sample the legacy four-template generator. Dev and
`structure_disjoint_transfer` each contain 64 balanced cases whose canonical operation
sequences are disjoint from training and from each other.

Two deterministic random initializations, seeds 20261027 and 20261028, run on CPU for
24 epochs. Each seed has a hard optimizer budget below 90 seconds. The binary head
predicts approve/reject and the location head predicts the earliest erroneous step.
Before both loss and prediction, every location logit beyond that example's real trace
length is replaced by negative infinity. Gold labels are constructed separately and are
never included in tokenization.

All holdout labels are checked and all predictions are scored through
`labyrinth/minotaur_independent_eval.py`, which replays printed programs instead of
trusting generator labels. Reported metrics include exact first-error accuracy, false
approvals, false rejections, and selective coverage. Frozen-random, always-approve, and
always-reject-at-step-1 controls use the same evaluator. No holdout tunes a threshold.

The implementation, tests, evaluator compatibility repair, and this protocol must be
committed first. Only then may the CLI run from that clean commit. The JSON evidence must
record the exact Git SHA, source-file SHA-256 values, per-split curriculum commitments,
dataset hashes, balance, lengths, structures, fault families, runtime, losses, and weight
digests. Evidence is committed separately. The experiment never saves weights and its
promotion recommendation remains `DO_NOT_PROMOTE`, especially if results are weak.

Commands:

```powershell
py -3 -m unittest tests.test_minotaur_combinatorial_training -v
py -3 -m unittest discover -s tests -q
py -3 -m labyrinth.minotaur_combinatorial_training --output docs/evidence/MINOTAUR_COMBINATORIAL_TRAINING_2026-10-27.json
```
