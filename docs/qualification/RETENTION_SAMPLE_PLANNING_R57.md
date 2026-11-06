# Fixed-sample retention planning (R57, source-only)

`labyrinth.retention_sample_planning_r57.plan_prior_skill_retention_samples`
computes how many **independent, previously designated semantic situations per
prior skill** are minimally necessary to keep the exact one-sided
Clopper–Pearson *deterioration-event* upper bound at or below the declared
margin, given a fixed maximum number of anticipated deteriorations.

It uses the same `one_sided_exact_deterioration_upper` implementation as
`labyrinth.retention_qualification.decide_retention` and the same familywise
Bonferroni adjustment, `alpha / number_of_prior_skills`. The formula for
**zero** deteriorations is:

`upper(n,0) = 1 - (alpha / skills)^(1/n)`.

For default `alpha=0.05`, deterioration margin `0.05`, and minimum policy
roster size `64`:

| Number of prior skills | Zero-deterioration minimum per skill | Notes |
| ---: | ---: | --- |
| 1 | 64 | Policy minimum dominates the approximately 58.4 statistical minimum. |
| 2 | **72** | 71 gives an upper bound about 0.0506294; 72 gives 0.0499441. |
| 3 | 80 | The per-skill alpha is 0.05/3. |
| 12 | 107 | The per-skill alpha is 0.05/12. |

The frequently repeated estimate of **73** cases for two skills and zero
losses is one too high for this exact policy. The actual first passing `n`
is **72**. The project implementation and an independent closed-form
calculation agree.

The function also accepts *predeclared maximum deterioration counts* (for
example, one permitted loss on a previous skill) and finds the first feasible
integer `n` within `64..512`; it rejects impossible plans. Passing the
planning function only means that a specific fixed-size study **could**
meet the numerical upper-bound criterion if its assumptions and anticipated
loss count hold. It does not measure any actual retention or accuracy.

## Required independent controls

1. Freeze the skill roster, case count, loss budget and statistical policy
   **before** any final learner evaluation. Do not repeatedly add cases until
   significance appears; optional stopping changes the inference.
2. The independent examiner must control two different forms for each
   semantic situation and exclude TRAIN prompts and structure families.
3. Establish the prior skill's *before* proficiency independently. The
   numerical deterioration bound cannot replace the existing baseline
   accuracy floor of at least 75%.
4. Evaluate fixed before/after learner checkpoints with separately verified
   model and evaluator identities. Neither this function nor a signed source
   receipt attests to physical execution or independent case custody.
5. Apply `decide_retention` to the actual, precommitted roster and signed
   evaluation observations; a planning result is not a retention PASS.

The returned immutable tuple-backed value always has
`status=SOURCE_ONLY_UNATTESTED_SAMPLE_PLAN` and all authority/provenance
booleans `False`. Only the six policy inputs are stored; the constructor
validates them against the exact binomial planning calculation and sample
limits, while the minimum exact units per skill, Bonferroni alpha, and total
are read-only derived properties. Authority flags and status cannot be set
through `object.__setattr__`, the constructor, `dataclasses.replace`, or
`_replace`; the prior frozen-dataclass representation did allow raw
`object.__setattr__` mutation and has been retired. This is **source-object
consistency only**, not a Python process security boundary: a caller can
still submit a self-consistent policy or
roster, and a source-only value proves neither independent preregistration
nor physical execution. It cannot grant training, graduation, merge,
deployment, host access, identity, consent, or any permission. Q5–Q9 remain
separate gates; **ROUTING != AUTHORITY**.

## Example (non-authorizing)

```python
from labyrinth.retention_sample_planning_r57 import plan_prior_skill_retention_samples

proposal = plan_prior_skill_retention_samples(
    ("prior_math", "prior_language"), (0, 0)
)
assert proposal.required_units_per_skill == (72, 72)
assert proposal.graduation_authorized is False
```

Testing: `python -m unittest tests.test_retention_sample_planning_r57 -v`.
