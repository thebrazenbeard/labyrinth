# Appraisal V3 implementation-only review — execution hold

**Status: SOURCE-ONLY; V3 training, heldout context sampling, outcome
evaluation, model checkpoint generation, and score exposure have NOT occurred.**
The immutable scientific design subject is the one-file [V3 prospective
design lock](APPRAISAL_V3_DESIGN_LOCK_PENDING.md), GitHub source PR #119,
exact head `f412103a3b290ff1019ca4e044afb80e9fad66bb`. This implementation
is a NEW review subject. No PR, CI, argument, or documentation grants run
authorization; require the parent's independent exact-head approval before a
bounded read-only experimental run. A CLI SHA check is a consistency guard,
not cryptographic authentication of authorization.

## Implementation pins, *before* exposure

| Parameter | Source pin and invariant |
|---|---|
| Interpreter | CPython **3.12.14** exact `sys.version.split()[0]`; no unpinned alternative; inherited V1 `labyrinth/appraisal_experiment.py` source blob **`5f73eab99b0a3981782944f227ff3e60106141fd`** via the stacked Labyrinth PR history |
| Source | Run `--approved-source-head` requires exact full current Git SHA-1; a moved head requires a NEW independent review |
| Seeds | Candidate `(101,307,911,2027,4049)`, null replicate `0..249` per seed; no early selection |
| Training | Exactly 3,200 sequential candidate reward updates plus 250×3,200 independent-policy null updates **per seed** |
| Null percentile | Nearest-rank 95th percentile for n=250: `ceil(0.95×250)-1 = 237` **zero-indexed** of ascending null scores; p05 index 12; median average indices 124,125 |
| Randomness | `random.Random` (CPython 3.12.14 Mersenne Twister), numeric injective packing `(domain_id << 42) | (seed << 10) | arm`; seed in `[0,2³¹)`, arm in `[0,1024)`; unique domain IDs declared in source |
| Training contexts | Candidate and each null share V1's same seed-derived context tape, but null draws its *own* action using evolving null-policy probabilities, separate action seed and separately shuffled candidate reward-label multiset; no candidate action replay |
| C/D evaluators | Both sample exactly 1,500 fixed contexts per seed in independent domain-separated streams. Canonical context commitment is SHA256 of sorted-minimal JSON containing family, seed, and ordered Python `float.hex` context strings. The exact hashes **cannot be disclosed without forbidden pre-approval evaluation-context sampling**; emission occurs at the approved run only |
| Scoring | Predeclared epsilon 0.20 and temperature 0.15, expected payoff under stochastic policy is primary. Separately sample one action and Bernoulli outcome per context/arm using two disjoint RNGs for **descriptive-only** realized scores; these cannot select candidates or override H1 |
| H2 | At all five seeds checkpoint must save via exclusive-create, SHA256 and version; **new Python interpreter** independently loads, and exact Python float64 equality is required at ordered 15 probes `(-1,-.9,-.75,-.5,-.25,-.12,-.01,0,.01,.12,.25,.5,.75,.9,1)` and C expected-payoff score. Tolerance **zero**; process ID differs |
| Work bounds | `5×3,200×251 = 4,016,000` training steps; `5×2×1,500×252 = 3,780,000` scored context/arm pairs and descriptive reward draws; 1,250 independently logged null propensity-trace SHA256s, each committing 3,200 pre-feedback samples. Abort as UNQUALIFIED if 900s wall budget exhausted |

The frozen H1 criterion is unchanged: every C candidate-minus-frozen >0.08,
at least four C candidate-minus-null-p95 >0.03, and at most 12 of 1,250
null scores at/above candidate. **No V3 outcome has yet been calculated.**

## Important interpretation limit

The null heads are separately adapting *online behaviors* with distinct action
RNG and propensities, correcting V1's off-policy action replay. However, all
250 nulls per seed use shuffled variants of the **same candidate-generated
label multiset** and shared context tape. They are dependent synthetic
reference policies, **not 250 causally independent trials or independent
environment replications**. A low empirical fraction is a finite in-family
control comparison, not a validated p-value or claim of transfer to LLMs,
Vera, psychological pleasure, subjective reward, or consciousness. V2's
strong criterion failed; the new V3 experiment must be allowed to fail too.

## Execution frontier

1. External independent reviewer reads back this **implementation-only PR's
   exact GitHub head and source bytes** and its unchanged parent design head.
2. Reviewer assesses the code, pure-contract tests and all pins. Any new code
   commit changes the review subject. No evaluation contexts, outcomes, or
   model scores may be sampled during this review.
3. Only after exact-head release may an explicitly authorized owner execute
   `python -m labyrinth.appraisal_v3 run --approved-source-head <exact-head>`
   with a bounded $0 CPU lease, collect full five-seed results, and publish
   BOTH successes and failures without adjusting design or code.

No private user data, model weights, GPU use, paid services, deployment, or
runtime identity mutation is in scope.
