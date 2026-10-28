# Minotaur V5 algebraic evaluator

Added `labyrinth/minotaur_algebraic_eval_v5.py`, a dependency-free evaluator that does not import Minotaur generators, training code, models, or prior evaluators.

`evaluate_atomic_predictions` recomputes each modulo-97 ADD/SUB/MUL result from `(op, x, y)`, rejects any supplied output label that disagrees, and fails closed for malformed cases, duplicate IDs, and missing or unexpected prediction IDs. It reports exact accuracy per operation and for unseen-pair, diagonal, and zero-operand slices. It also reports a best constant-output baseline, a copy-`x` baseline, and exact training-pair/operation/group overlap.

`evaluate_program_rollouts` accepts only an atomic callback with `(op, x, y)`, never a gold state or label. Its separate public replayer audits 1--6 step clean and corrupted traces, then reports exact first-error agreement, abstention, false approvals, and false rejections. Invalid callback outputs or callback exceptions refuse the batch.

No confidence interval is computed: group and paired dependence are explicitly retained rather than assumed away. This adds no neural training, no private holdout assertion, and no broad-mastery or graduation claim.

Verification completed on base `e9efd41f8ee2e25cfb5535b605371ad1801fcee6`:

```
py -3 -m unittest tests.test_minotaur_algebraic_eval_v5 tests.test_minotaur_independent_eval tests.test_minotaur_transition_eval
```

Result: 14 tests passed.
