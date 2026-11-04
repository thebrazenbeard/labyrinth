# Synthetic appraisal V2 — independent online shuffled-feedback nulls

**Protocol chronology:** The design was drafted locally on 2026-11-04 before the first V2 execution, but it had **no independently sealed GitHub commit/PR timestamp before the run**. Its comparison thresholds are therefore **protocol-defined retrospective pilot criteria, not independently verified pre-registration**. **Status:** source-only toy falsification of V1, not LLM learning.

## Why V1 is insufficient

V1's shuffled-feedback head is updated over the candidate's *own correct-feedback-adaptive* context/action trajectory. The action distribution therefore encodes prior genuine rewards, even after feedback-label permutation. The single shuffled head is also subject to high random variability. Independent hostile exploratory runs found that some shuffled-online null heads approached or exceeded V1's candidate. V1's favorable comparison therefore does not qualify a strong causal contrast between independently trained learning policies.

This document supplements, does not rewrite, V1's source protocol or thresholds. V1 also lacks a separate immutable pre-result design-only head; its pilot should be described as **protocol-defined retrospective**, not independently preregistered. All V1 results remain evidence with their original limited claim.

## V2 fixed design, predictions and falsifiers

- Three unchanged candidate training seeds: **11, 29, 71**; 3,200 steps each. Candidate policy remains exactly V1 with no tuning.
- Candidate rewards form a 3,200-element binary label tape. For each candidate seed, produce exactly **100 independently seeded null heads**, each starting at zero. Each null receives a separately shuffled copy of the same candidate label multiset, with reward labels independent of its own contexts/actions.
- Every null sees the candidate's *context-generator sequence* (matched exposure) but chooses **its own actions** from its own evolving epsilon-softmax head with an independently seeded action RNG. Log its actual pre-feedback action probabilities, update it online with the next shuffled tape label, and use the exact same learning rate, step count, and feature class as the candidate. No trained weights or actions are imported from the candidate.
- **Primary evaluation**: a new independently seeded heldout **triangular context generator B**, `x = (uniform(-1,1) + uniform(-1,1))/2`, 1,200 cases per seed. Its source is visible and its latent payoff formula is intentionally identical to V1's: in-family context-distribution transfer only. No optimizer sees evaluation rewards.
- **Secondary sensitivity only**: V1's grid-with-jitter evaluation generator A, 1,200 cases. Do not select between samplers or thresholds using results.
- Each arm is evaluated with deterministic greedy choice and equal-tie parity as in V1. Primary quantity is the expected synthetic payoff. Report candidate, frozen, null 5th/median/95th/max and fraction of nulls scoring within 0.06 of or above candidate, for all three seeds and both samplers.
- **Strict protocol-defined strong-causal criterion (must pass every seed)**: candidate minus frozen expected payoff must exceed 0.10 on generator B **AND** candidate must beat the empirical null 95th percentile on generator B by at least 0.05. If either fails for any seed, the strong-causal claim is **FAIL** even if candidate exceeds null median. No retrying/cherry-picking seeds or null repeats.
- **Narrow mechanism criterion**: V1 frozen/disabled and fresh-process reload controls still validate that head weights causally affect in-family toy decisions, independent of whether shuffled-online null significance passes.

## Evidence and claim limits

100 null repeats characterize stochastic control variability for *these fixed toy seeds and generators*, not a population confidence interval. Null heads share one candidate label multiset and are not independent of the candidate data collection process in that respect; shuffling breaks temporal action/context association but not all possible selection biases. A source-visible triangular generator is not sealed/independently blinded. Two linear synthetic generators sharing the same payoff rule do not prove out-of-distribution reasoning transfer, neural-head learning or subjective reward.

**Prohibited in this source test:** real model weights, new GPU jobs, private conversations or data, provider writes, model training, host installs, and claims of consciousness or durable identity.

## Fixed pilot results (appended after the design above)

**The strong-causal criterion FAILED on all three preselected seeds.** The result was retained without tuning/replacing seeds or thresholds. Each result uses 100 independent online shuffled-feedback policies, each with 3,200 own stochastic actions and pre-feedback propensities, evaluated on the same 1,200 heldout contexts per sampler.

| Seed | B candidate | B frozen | B null median | B null 95th | B null max | B nulls ≥ candidate | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 11 | .6218 | .4913 | .4991 | .6269 | .6329 | 8/100 | **FAIL** |
| 29 | .5815 | .4924 | .5079 | .6300 | .6393 | 16/100 | **FAIL** |
| 71 | .6308 | .4994 | .4969 | .6285 | .6339 | 5/100 | **FAIL** |

Seed 29 also fails the candidate-minus-frozen >0.10 criterion (observed +0.0891). All three fail the candidate-minus-null-95th ≥0.05 requirement; for seeds 11 and 29 the null 95th *exceeds* the candidate, and for seed 71 the margin is only +0.0023.

Sensitivity generator A (V1 grid) also shows substantial null overlap: candidate expected payoff .7031/.6712/.7067; null 95th .7050/.7095/.7058, null max .7084/.7136/.7097, respectively. Thus favorable single shuffled-arm V1 results were **not robust** to an independently adapting online shuffled null distribution.

Local Python 3.12 full `unittest discover`: **248 tests run, 42 skipped, exit 0**. The four new V2 tests verify independent action and probability traces, label-multiset preservation, distinct null replicates, deterministic source behavior and fixed 100-null evaluation protocol. These are source mechanism tests; an external blinded evaluation has not occurred.

> **Interpretation:** Weight updates and fresh-process persistence still work for the constructed linear bandit, and switching the trained head off remains a meaningful within-toy intervention. What fails is the *stronger causal claim* that the single candidate's good heldout score cannot also be obtained under independent shuffled online feedback. The greedy evaluator can turn tiny chance slopes into near-optimal decisions. This is a falsifier, not an inconvenience to tune away after seeing results.
