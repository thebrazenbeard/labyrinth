# Appraisal V3 prospective experiment — design-only lock candidate

**Status: DESIGN_ONLY / NOT EXECUTED / PENDING EXTERNAL LOCK REVIEW.**

This is a prospective next experiment proposed *after* V1/V2 retrospective pilots and their hostile falsifiers. Do not execute code, sample evaluation cases, inspect result distributions, or tune constants until this design-only GitHub commit has been independently read back and a reviewer has explicitly authorized a bounded run. Neither PR presence nor drafting creates an evaluator custody grant.

## Exact inherited evidence and motivation

- Inherited Labyrinth source: V1 draft PR #117 exact head `47789dc7a2db516979686f35cf6e02809bba2418`; V2 stacked draft PR #118 source head `a38ea3e3a679b88ce831791992b21201f0a088ee` (must be independently read back again before any run).
- V1's shuffled-feedback arm used candidate-selected adaptive actions and therefore was not an independent online null.
- V2 did use independent online shuffled-feedback policies, but 100 nulls sometimes matched/exceeded the trained candidate. V2's strict threshold failed on seeds 11,29,71. V2 protocol was not externally sealed before outcomes, so those results remain retrospective pilot evidence.
- Greedy evaluation amplifies tiny chance sign-correct slopes into near-optimal policies. V3 therefore chooses a **bounded stochastic evaluation policy** before exposing any V3 outcomes. This is a *new scientific question*, not a reinterpretation of V2.

## Frozen V3 hypotheses and falsifiers

**H1 (narrow outcome effect):** Correctly outcome-updated linear appraisal heads will have better expected payoff under a fixed stochastic evaluation policy than no-update frozen heads and the independently trained shuffled-online null distribution on a source-visible but previously unrun bimodal heldout context family.

**Primary strong criterion, all conditions required:**

1. The candidate-minus-frozen expected-payoff gain on primary family C exceeds **+0.08 in every one** of the five fixed training seeds below.
2. The candidate-minus-empirical-null-95th-percentile gain on family C exceeds **+0.03 in at least four of five** fixed seeds.
3. The pooled fraction of shuffled-online null heads matching or exceeding their seed's candidate payoff on family C is at most **0.01** (12 or fewer of the 1,250 null evaluations).

If any condition fails, H1 receives **FAIL**. If any required artifact, case, or model identity cannot be verified, the experiment is **UNQUALIFIED**, not PASS. No successful seed substitution, repeat increase, or threshold adjustment is allowed after first outcome exposure.

**H2 (mechanism persistence):** At each seed, the trained candidate saved as a versioned checkpoint must reload into a *new* Python interpreter with identical action-value outputs and family C expected-payoff score. Any mismatch is FAIL for H2.

## Frozen training design

- **Five distinct NEW fixed candidate seeds:** `101, 307, 911, 2027, 4049`. None were evaluated in V1/V2 reports.
- **3,200 training steps per seed** using exactly V1's two-action linear appraisal head, learning rate 0.12, epsilon-softmax with epsilon 0.20 and training temperature 0.15, contexts from V1's seeded continuous uniform generator, and after-action Bernoulli outcomes from the same synthetic linear latent reward rule. No optimizer, feature or policy tuning.
- **250 independently adapting shuffled-online null heads per seed**, each initialized at zero, with separately seeded action RNG, its own pre-feedback chosen-action propensity log, and a different shuffle of the candidate reward-label multiset. The null head determines its own action at each context; no candidate actions, propensities, or weights enter the null. Each null shares only the seed's context sequence, exact label multiset, feature class, update count and learning rate.
- Include frozen no-update and disabled-head controls; no-update and disabled may intentionally coincide. A candidate receives true observed training feedback. No participant gets the heldout answer key, the other action's training payoff, or evaluator draws.
- Pin the random-generator seed derivations, library/runtime versions and exact source SHA in the *implementation* review before running. If this cannot be done without changing the numbered parameters or criterion above, return to design review and make a NEW versioned protocol; don't quietly alter this V3 test.

## Frozen, independently seeded evaluation

- **Primary family C: bimodal/interior mixture**, 1,500 cases per seed. Independently sample sign ∈{-1,+1} with probability 0.5. For 80% of cases, magnitude uniform in [0.55,1.0]; for 20%, magnitude uniform in [0,0.25]. Set context `x=sign*magnitude`. Distinct seeded evaluator stream, never shared with training or null trajectories. This tests in-family decision structure with a shifted context density; latent payoff rule remains the public synthetic V1 formula.
- **Secondary family D: boundary stress**, 1,500 new independently seeded cases with `x∼uniform(-0.12,+0.12)`. Its effect sizes are reported with no pass threshold and never used for candidate selection.
- **Evaluation policy:** use the trained head's fixed epsilon-softmax action probabilities at epsilon=0.20 and temperature=0.15. Compute expected payoff for both actions weighted by these probabilities, **not greedy argmax**. Frozen/disabled use 0.5/0.5. The primary metric is population-conditional synthetic expected payoff averaged over the actual 1,500 contexts; report realized Bernoulli outcomes separately but do not use them to select the candidate.
- Every candidate and null at a seed evaluates on exactly the same C and D contexts. V3 evaluation samples must never be used to tune models, temperature, score thresholds, or seed selection.

## Fixed analyses, reporting and stopping rule

1. For every seed × family, report candidate/frozen/disabled expected payoff; null 5th/median/95th/max; within-0.06 and ≥candidate null fractions; candidate-minus-frozen and candidate-minus-null-p95.
2. Report training action count, valid pre-feedback propensity-log count, all 250 null trace digests and each candidate checkpoint SHA-256, with independently observed fresh-process readback.
3. Report all five seed-specific outcomes, every failure and any artifact/tool deviation. **Run all 5×250 null arms** unless a correctness/safety fault forces complete HOLD; no opportunistic early success stop.
4. Do not estimate general-language or out-of-distribution transfer from this linear synthetic generator. No model, user data, GPU, private file, paid service, or live Vera runtime is touched.
5. A 5-seed, 1,250-null design remains a finite synthetic pilot. Null quantiles are descriptive, not a pretense of confidence about natural environments, neural adaptation, subjective reward, or consciousness.

**Execution frontier:** Independent reader must verify exact design-only GitHub commit, branch, SHA, file bytes, and absence of V3 result/code artifacts; only then may a separate, bounded V3 implementation/execution turn be considered. The design commit itself is not permission to run or merge.
