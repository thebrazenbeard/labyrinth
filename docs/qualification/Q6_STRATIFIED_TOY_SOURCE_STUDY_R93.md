# Q6 R93: stratified toy extrapolation and immediate retention control

**Status: `SOURCE_ONLY_TOY_Q6_UNQUALIFIED`.** This experiment neither measures
a pretrained language model nor grants a Labyrinth/Minotaur qualification,
independent evaluation, graduation, merge, or deployment decision.

Source: `labyrinth/q6_stratified_toy_study_r93.py`

Run (from the repository root):

```python
from labyrinth.q6_stratified_toy_study_r93 import run_q6_toy_study
result = run_q6_toy_study(seed=7, per_profile=6)
assert result["q6_independently_qualified"] is False
```

## Fixed study design

- Four separately generated cohorts: stage-A TRAIN, stage-B TRAIN, lower-range
  HELDOUT, and higher-range TRANSFER.
- Exactly four balanced arithmetic strata in each cohort: addition requiring
  unit-digit carry, addition without it, subtraction requiring unit-digit
  borrowing, and subtraction without it. Cohorts contain `4 * per_profile`
  cases each. All operands in both TRAIN stages and HELDOUT are 11–499;
  TRANSFER operands are 800–999. No answer labels are passed to `generate()`.
- Generator seeds and rejection sampling are deterministic and independent of
  learner responses. A split validator refuses duplicate case IDs, exact
  prompt bytes, and algebraic-equation aliases, including swapped addition
  operands and inverse subtraction (`a+b=c`, `c-a=b`, `c-b=a`, or a negative
  rearrangement). This guards accidental contamination
  *within this claimed run*, not externally authenticated public/private bank
  custody. Caller-supplied previously exposed prompts, including their
  commutative semantic aliases, can be refused but
  cannot establish a durable, nonrollback global exposure register.
- `run_q6_toy_study(previously_exposed_eval_prompts=...)` now checks that
  caller-owned history against **all four current partitions (TRAIN-A,
  TRAIN-B, heldout, transfer) before any optimizer call**. Its report exports both
  `evaluated_prompts` and `evaluation_prompt_sha256` so a caller can reconcile
  exact and algebraically equivalent expressions across studies. The runner
  cannot prove the history is complete; absent/cleared history replays public
  seeds. In particular, exporting only prior evaluation prompts does not rule
  out earlier TRAIN cases reappearing in a later heldout/transfer cohort. A
  multi-run independent holdout requires a complete, independently custodied
  exposure history across TRAIN and evaluation. This is never a
  sealed-private-exam authorization service.
- R94 source hardening snapshots and validates the supplied public history at
  runner entry, before cohort generation (which may invoke caller-supplied or
  substituted code). Clearing or changing the original set after entry cannot
  erase an already exposed arithmetic problem from the study's validation.
  An optional content pin can be checked before generation:

  ```python
  from labyrinth.q6_stratified_toy_study_r93 import (
      q6_exposure_history_sha256, run_q6_toy_study,
  )
  public_prior = {"Calculate 12 + 23.\nRespond with exactly one line: ANSWER: <integer>."}
  owner_supplied_pin = q6_exposure_history_sha256(public_prior)
  result = run_q6_toy_study(
      seed=7, per_profile=3,
      previously_exposed_eval_prompts=public_prior,
      expected_exposure_history_sha256=owner_supplied_pin,
  )
  assert result["caller_supplied_history_pin_checked"] is True
  assert result["q6_independently_qualified"] is False
  ```

  The source digest is canonical for a given prompt set and exposed in the
  result with a count for reconciliation. **It is not a signature, externally
  pinned monotonic head, proof of completeness, or evidence of physical
  custody.** Reuse, malformed history, malformed pin, mismatch and bounded
  history exhaustion fail with explicit `Q6StudyError` variants. An actual
  independent owner must retain and authenticate the prior register and its
  head outside this runner. Concurrent mutation during the initial snapshot
  still requires host-atomic custody; the local snapshot cannot supply it.
- All arms start from the same `ToyLinearAdapter` seed. The trained arm uses
  true TRAIN targets and real normalized SGD. A zero-update arm makes matched
  case visits without parameter writes. An inverted-label arm also performs
  real SGD against a deranged permutation of training labels.
- Both evaluation splits are scored at initialization, after stage A, and
  after stage B, without adding evaluation gold to the optimizer. The trained
  checkpoint is exclusively written and freshly reloaded after each stage;
  SHA-based local readback is **integrity**, not trusted custody.
- `retention_delta_after_stage_b` is the heldout-score change from just after
  stage A to just after B. No long-term elapsed-time retention is tested.

At default seed 7/per-profile 6 with 80+80 epochs, a local Python3.12 source
probe measured trained HELDOUT `0 -> 24 -> 24` and high-range TRANSFER
`0 -> 24 -> 24`, while zero-update and deranged-label controls scored zero in
each stage; the trained arm made 1,198 toy parameter updates. These are
**four hand-engineered coefficients with public arithmetic prompts**: the toy
linearly extrapolates very easily. A perfect score here says nothing about
independent pretrained model Q6 capability.

## Falsifiers and remaining gates

The R93 regression suite deliberately constructs a commutative prompt alias
with a new case ID and different bytes: it is rejected as semantic overlap.
It also mutates evaluation gold and checks an explicit previously-exposed
prompt, commutative alias and inverse-subtraction alias, which are rejected. A separate test demonstrates that identical
public-seed cohorts are replayable. There is no proof that an outside
examiner selected the seed before training or retained the private gold.
Across seeds 0–11, the public evaluation history intersects seed 12's
evaluation semantics; the R93 negative test reproduces that overlap and
verifies refusal before a learner's `train()` could execute. This protects a
supplied public history, not cross-host global uniqueness.

**Still required before Q6:** independent pretrained checkpoints and baselines,
unexposed material controlled by a separate examiner, fixed preregistered
control budgets, multi-seed/semantic robustness under an operator-owned
replay register, durable optimizer/RNG restarts, and a no-regression retention
threshold observed on a preexisting capability. Q7/Q8/Q9 remain separate
authority gates.
