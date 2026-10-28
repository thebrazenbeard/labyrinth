# Minotaur independent evaluator

`labyrinth.minotaur_independent_eval` evaluates first-error predictions for the
printed modulo-97 register trace language without importing a dataset generator,
training labels, model code, or the older guarded checker. It independently
parses the program and observed states, derives the earliest mismatch, and
requires every supplied claimed label to match before scoring begins.

Use `TraceEvaluationCase` and `evaluate_predictions`. A label is `None` (or
prediction `ALL_CORRECT`) for a clean trace, and `1..N` (or
`ERROR_STEP:<n>`) for a first-error location. A missing or malformed prediction
is an abstention, never a correct answer. The report includes exact first-error
accuracy, selective coverage and selective accuracy, false approvals/rejections,
a sparse confusion table, structural-shape overlap with supplied training
shapes, and always-yes, always-no-step-1, and position-cycle negative controls.

The whole batch returns `REFUSED` if any case is malformed, has duplicate IDs,
or has a claimed label or optional declared structure that disagrees with the
independently replayed program. This prevents untrusted labels from silently
entering a denominator.

Confidence intervals are absent unless the caller passes
`iid_binomial_attested=True`; Wilson intervals are then explicitly conditional
on that caller attestation. Public deterministic suites normally do not justify
an IID claim. This public narrow-domain evaluator is not a private blind
holdout, neural-generalization evidence, or graduation authorization.

## Public-fixture audit (2026-10-27)

The evaluator was run against the current 96 `generate_process_cases(90210,
16)` public fixtures. The prediction map was populated from the fixtures' own
claimed labels only to exercise label cross-checking; it is not a model score.
Independent replay accepted all 96 claimed labels (zero audit failures) and
derived 12 program shapes. Against the four known public train shapes, it found
exactly four overlaps, which is expected because the fixture includes a
fresh-number train-shape slice. The controls' exact accuracies were 0.5000
(always-yes), 0.1354 (always-no-step-1), and 0.1146 (position-cycle). No
confidence interval was calculated. This is reproducibility/audit evidence for
the public dataset format only, explicitly not a private blind holdout.
