# BT2 Minotaur TRAIN routing — L8 controlled effectiveness probe

**Evidence class:** SOURCE TEST with deterministic synthetic TRAIN outcomes.
Not a physical taskmaster neural-weight update, independent learning outcome,
held-out evaluation, or graduation qualification.

The exact test subject is `labyrinth.minotaur_policy_learning.MinotaurTrainingPolicy`
as consumed by the real `minotaur_model.plan_training_room` adapter seam.
The test file is `tests/test_minotaur_policy_effectiveness_l8.py`.

## Controlled comparison

Frozen declaration: `dual-skill-corridor`, domain `reasoning`, approved skills
`arithmetic` and `program_reasoning`, specificity 3, scale 2, three cycles.
All observations were pre-admitted TRAIN-only case/prompt commitments. A static
operator adapter always proposed `(practice, 3)`. The no-model fallback was
deterministic and unchanged for the same metadata, regardless of feedback.

| Feedback to policy | Adaptive TRAIN proposal | Static proposal |
| --- | --- | --- |
| Cold start, arithmetic | practice, difficulty 3 | practice, difficulty 3 |
| Three practice successes, arithmetic | contrast, difficulty 4 | practice, difficulty 3 |
| Three practice failures, program reasoning | practice, difficulty 1 | practice, difficulty 3 |
| Three arithmetic contrast successes after practice mastery | transfer, difficulty 4 | practice, difficulty 3 |
| Four arithmetic transfer failures | transfer, difficulty 1 | practice, difficulty 3 |

Three extra old-practice successes were not allowed to erase an observed
current-transfer weakness. Feedback for one skill did not alter the other
skill's room suggestion. A zero-update sham retained its initial state and
proposal; an inverted-label control generated a distinct, weaker proposal.

An explicitly approved new-cycle TRAIN registry preserved the preceding
adaptive policy state, admitted one new occurrence, and then allowed a new
observation to increase arithmetic's proposed contrast difficulty to 5.
Unapproved cases, held-out/private-final split labels and wrong objectives
failed closed without changing the policy's state hash.

An additional matched live-corridor test instantiated **two separate
`TrainingCorridorRunner` instances from the same exact arithmetic maze**.
After two genuinely scored TRAIN successes in one versus two deliberately
incorrect responses in the other, their third materialized learner-facing
questions diverged: the successful corridor presented an `Expression A is`
contrast challenge, while the failing corridor remained on a `Calculate`
practice question. Both views excluded the answer key. This is direct
source-level evidence that a feedback-conditioned proposal changed actual
TRAIN prompt content, not merely planner metadata.

## Interpretation and limits

These tests **demonstrate feedback-dependent routing** through a real source
integration seam, and distinguish that from unchanging controls. All correctness
bits were authored by the test; no learner model produced them, no neural
Minotaur weights changed, and no independent grader verified correctness.
The tabular policy is operator-provisioned, and its proposals are accepted only
after TRAIN data validation. A separate training planner must consume the
suggestions for generated question content to change.

The tests do not measure learning speed, genuine weakness detection against
real learner behavior, adaptive efficacy beyond the deterministic synthetic
schedule, transfer, retention, or final examination authority.
