# Synthetic appraisal-head mechanism experiment V1

**Status: pre-registered toy mechanism test; no LLM or subjective-experience claim.**

## Hypothesis and falsifier

H1: In a two-action contextual bandit whose *training* rewards are initially hidden from the policy, outcome-aligned updates to a small persisted appraisal head improve expected payoff on an independently seeded, structurally different *evaluation* context sampler. A frozen head, a head fitted to feedback shuffled independently of context/action, and a disabled-head policy should fail to match that improvement. A complete fresh-process reload must preserve the learned policy.

Falsify H1 if any of three preselected seeds (11, 29, 71) yields correct-feedback expected payoff no more than 0.08 above the frozen and disabled heads, or no more than 0.06 above shuffled feedback, on 1,200 unseen held-out cases; if the learned head's checkpoint fails to replay identically after fresh interpreter reload; or if the action probabilities aren't logged and valid.

## Frozen synthetic design

- **Training generator:** independent seeded continuous uniform contexts `x ∈ [-1,1]`; a Bernoulli outcome is drawn *after* the policy chooses action `a∈{0,1}`. Expected reward for action 0 is `0.5 + 0.4x`; for action 1 it is `0.5 - 0.4x`. The policy is not given that formula or the other action's reward.
- **Evaluation generator:** independent seeded discrete grid cells with small random jitter, disjoint from the training sampler's sequence, with independent outcome draws. The structural context generator differs but the latent conditional payoff relationship is intentionally **shared**; this is in-family transfer only, not external or OOD transfer.
- **Training:** 3,200 online updates per seed; two weights (bias/slope) per action; squared-error feedback updates on the chosen action only; epsilon-softmax action exploration (`epsilon=0.2`), propensity logged *before* observing each payoff. No importance correction or policy-gradient claim.
- **Controls:** (a) zero-weight frozen head, (b) shuffled feedback over the exact candidate's logged state/action trajectories with the same label multiset and update budget, (c) candidate trained head disabled at evaluation so actions tie, and (d) a deterministic oracle knowing the hidden payoff function, reported only as an upper-bound reference.
- **Evaluation:** evaluate all arms on exactly the same 1,200 held-out cases. Use deterministic greedy choice, tie-breaking by equal expected mixture; report expected payoff as primary metric and independently generated realized Bernoulli rewards as secondary diagnostics. The expected-value metric intentionally removes outcome sampling variance; no confidence interval or population-generalization inference is made.
- **Persistence:** save only version, exact bounded weights and a SHA-256 payload checksum to a new JSON path; reject overwrite and malformed/nonfinite weights. Test a separate Python interpreter loading the saved checkpoint with no training trace.

## Boundaries

The reward generator and linear model class are deliberately aligned. A passing result proves a limited source mechanism can learn a synthetic signal, persist it, and beat chosen controls. It does **not** prove an LLM learned, independently invented an appraisal signal, improved general intelligence, experienced reward, or acquired agency. The held-out generator is source-visible; evaluation is not secret or blinded. No user conversation, private corpus, protected evaluator, GPU, model weights, external service, or billed API is accessed.

## Observed pilot (after frozen design)

`python -m unittest discover -s tests -q` on local Python 3.12: **244 tests run, 42 skipped, exit 0**. These include eight new source-level appraisal tests, a real fresh Python interpreter checkpoint-readback test, logged propensity reconstruction, and matched control comparisons.

Each independent run used 3,200 training actions and 1,200 evaluation contexts. Primary metric is mean *expected* reward, whose oracle is coded in this synthetic harness. Candidate-minus-control effect sizes:

| Seed | Candidate expected payoff | Frozen/disabled | Shuffled feedback | Candidate − frozen | Candidate − shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.7031 | 0.4917 | 0.4919 | +0.2114 | +0.2113 |
| 29 | 0.6712 | 0.5011 | 0.3064 | +0.1701 | +0.3648 |
| 71 | 0.7067 | 0.5033 | 0.4973 | +0.2035 | +0.2094 |

**Exploratory uncertainty added after pilot, not pre-registered:** conditional on the fitted heads, exact toy payoff function and independent held-out sampler draws, each paired expected-payoff contrast has range at most [-0.8, +0.8]. Hoeffding with a Bonferroni bound across nine comparisons gives a conservative simultaneous 95% **±0.07924** radius. This is an algebraic *conditional sampling* bound, not a generalization guarantee, independent external evaluation, or confidence interval over model classes or seed choice. Three preselected seeds are insufficient to make a broad empirical claim. The frozen and disabled arms share the same policy (intentional negative control), but the shuffled arm has its own weights and receives the same logged actions with labels permuted independently of contexts.

**Falsification limits:** This is a favorable, matched linear hypothesis class. Different latent payoff rules, distribution shifts, different optimizer seeds, adversarial reward generators, and genuine language-model heads remain untested. In particular, the strong negative shuffled result on seed 29 need not replicate outside these exact seeds.
