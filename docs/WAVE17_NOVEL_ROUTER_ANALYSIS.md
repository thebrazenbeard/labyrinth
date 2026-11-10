# Wave 17 — hostile novelty and leakage audit of the delayed router

**Exact source:** Labyrinth novel-router PR #159 integrated local head
`13a5d122a3c5a577023a38a6e32c891676362471`.

**Scope:** source-public synthetic mechanics only. The independent analysis
implementation is `labyrinth/novel_router_analysis.py` and its validation
is `tests/test_novel_router_analysis.py`. Neither imports or calls production
`step`, `verify_plan`, `search`, `_deliver`, `_load_payload`, or a source
oracle to obtain its answers. The tests call production search and public
sessions **after** generating independent labels and predictions to check
agreement. No changes are made to novel-router production code.

Run:

```bash
python -m unittest discover -s tests -p test_novel_router_analysis.py -q
python -O -m unittest discover -s tests -p test_novel_router_analysis.py -q
```

## Measured source-public inventory

| Scope | Independent observation |
|---|---:|
| Complete primitive Rule parameterizations | **24** |
| Distinct complete primitive Rule functions | **24** (no aliases) |
| Different before/after pairs in fixed seed-zero public context | **552** |
| Distinct bounded observable pair behaviors in that context | **552** |
| Generated seed universe independently adjudicated | **1,000**, seeds `[0,1000)` |
| Finite wins / finite UNSAT | **598 / 402** |
| TRAIN `[0,100)` wins / UNSAT | **62 / 38** |
| Proposed HELDOUT `[100,200)` wins / UNSAT | **62 / 38** |
| Public initial-metadata buckets across 1,000 seeds | **944** |
| Ambiguous initial metadata buckets | **52** |
| Case pairs sharing identical initial public metadata | **61** |
| Such pairs distinguishable by legal bounded probes | **61 / 61** |
| Same-public-metadata pairs proven probe-indistinguishable | **0** |
| Maximum shortest discriminating trace length | **3** charged probes |
| Distinct primitive law values in TRAIN / HELDOUT | **24 / 24** |
| Distinct source law-pair combinations in TRAIN / HELDOUT | **88 / 91** |
| Shared TRAIN and HELDOUT law-pair combinations | **12** |
| New ordered HELDOUT law-pair combinations | **79** |
| HELDOUT cases containing a TRAIN-unseen ordered pair | **85 / 100** |
| Retrospectively identifiable HELDOUT catalogue entries from public metadata alone | **98 / 100** |
| Same after a best fixed 3-probe sequence | **100 / 100** |

Unlike the prior three-bit puzzle's 840 parameterizations collapsing to 378
functions, these 24 full delayed-router rule tuples are *genuinely different
complete local functions*. Even fixing the public initial conditions and
the two resource budgets of seed zero, all 552 ordered **changed-law** pairs
have distinct complete bounded public-experiment trees. That is positive
evidence for mechanically meaningful novelty **within this declared router
grammar**, not for new generator families or independently sampled tasks.

In the actual `[0,1000)` generated catalogue, 52 public metadata buckets
contain multiple cases. A complete paired-state search found legal
discriminators for every one of the 61 same-metadata pairs, with no witness
requiring more than three probes. In this sample there is **no** identified
observational-equivalence false diversity. The sample is finite; it cannot
establish that no such aliases occur at other seeds, resource caps, or future
mechanism designs.

## What novelty must mean

The source genuinely changes the primitive causal puzzle: unlike the old
single three-bit register, this system has two four-level reservoirs, an
in-flight packet, orientation, load/route resources, delayed delivery and
cross-coupling. Yet those facts do **not** imply every seeded scenario is a
new independent mechanism. The observable task is determined by the
candidate-visible initial state, available operations, budget, and all
possible outputs—not the presence of distinct SHA hashes.

The independent analysis tests three levels:

1. **Complete local law equivalence:** enumerate all `2×4×3 = 24`
   `(lane_bias,coupling,payload_shift)` tuples, comparing the complete LOAD
   output and TICK-delivery output for every reservoir pair, orientation and
   pending payload. This distinguishes truly different rule functions from
   parameter aliases without assuming case hashes imply semantic diversity.
2. **Finite solution completeness:** a separate tuple-state backward-DP
   enumerates TICK/PURGE/ROUTE/LOAD under the exact eleven-action horizon,
   conserved packet and resource constraints. It decides WIN versus UNSAT
   without production BFS. An absent witness is only a proof under these
   declared rules; unknown external mechanics remain UNKNOWN.
3. **Candidate-visible equivalence:** for cases with identical initial public
   metadata, explore the pair-product graph of all legal charged probe
   actions, including RESET, up to twenty probes. An `applied` mismatch or
   differing public state is a separating counterexample. An exhausted
   pair-product with no difference establishes bounded observational
   equivalence for that pair. Invalid probes and shared evaluator identity
   are not treated as independent signal.

`RESET` returns a case to the same public initial state and consumes one
probe, while a probe episode cannot advance more than eleven ticks. Pair
exploration still includes RESET directly, so a purported indistinguishable
pair is not based on omitting a candidate-authorized action.

## Why seed holdout is inadequate

TRAIN seeds `0–99` and HELDOUT `100–199` both use the same public deterministic
case generator and the same 24-value source law space. No genuinely new law
generator was withheld. The heldout block *does* include many new ordered
combinations of existing primitive rules (**85 of 100 heldout cases** have
TRAIN-unseen ordered pairs; only 12 of 91 distinct heldout rule pairs overlap
the 88 train pairs), so compositional recombination is a
real property worth testing. It is not the same as an independently authored
new law family. A source-aware controller can enumerate a claimed
finite seed catalogue and use initial metadata plus charged probes as a
case-identity lookup; such retrospective identification is contamination,
not neural structural transfer. Discovery of a distinguishable rule pair
after a small number of probes demonstrates *observability*, not learning.

For catalogue metrics the denominator is exactly **100 preselected heldout
seeds**. The test reports the number of seeds identified uniquely among that
known list under a fixed observation sequence, and exhaustively searches
four named actions for the best fixed one-, two-, and three-probe sequences.
These are explicitly **retrospective in-sample** catalogue results. The
candidate does not receive a legitimately secret set of possible seeds;
future performance has no population confidence interval from these counts.
The empirically strongest fixed three-step sequence in this bounded block is
`ROUTE,LOAD,TICK`; it identifies **all 100** indexed heldout cases from public
observations. This does not mean a candidate would know that the true trial
must be in `[100,200)` without an explicit leaked inventory assumption.

## Minimum acceptance requirements for a serious transfer examination

1. Independently authored *new causal families*, not only different seeds,
   constants, rule pairs, or prompt paraphrases from the public generator.
2. A sealed evaluator and withheld trial inventory custodied outside the
   candidate's read/write privileges, with exact source/model/checkpoint
   fingerprints and immutable per-case receipts.
3. Paired matched budgets and observations for symbolic source prior,
   catalogue memorizer, zero-update model, corrupted-label control, and
   genuinely updated candidate, without policy-selection exposure to holdout.
4. Distinguish PLAN wins, complete finite UNSAT certificates, wrong answers,
   NO_KNOWN_WIN, and abstentions. Penalize false impossibility claims and
   preserve valid alternative wins.
5. Independent semantic-family units for uncertainty estimates and power,
   not nominal seed count. Clustered bootstrap/permutation or a precommitted
   independent family-sampling procedure is needed before claiming transfer.
6. Hidden near-twins that differ in *reachable* causal mechanics, as well as
   deliberately observationally equivalent pairs. Require correct
   uncertainty where the available probes cannot distinguish hypotheses.

## Evidence verdict boundaries

- **SURVIVES** only for finite source labels or observational classifications
  explicitly checked by these bounded reference algorithms.
- **FAILS** for any automatic inference from distinct parameter hashes or
  heldout numeric seeds to novel, independent, private mechanics.
- **UNKNOWN** for true effective sample size, neural parameter learning,
  externally sealed custody, authentic private graduation, and general
  intelligence transfer to newly authored mechanism families.

## Separate hostile source-integrity failure (not a novelty metric)

**FAILS under a single in-process constructor mutation at this exact head.**
I independently reproduced the Destroyer's separate source-level
counterexample against `13a5d12`: for seed **16**, the candidate plan
`LOAD,ROUTE,TICK,LOAD,TICK,COMMIT` is correctly a `VERIFIED_FAILURE`
without mutation. Monkeypatching only `State.__init__` to replace every
constructed `left` field with `(left+1)%4` makes `adjudicate` return
`VERIFIED_WIN` and `verified_success:true`. Both the production scorer and
its supposed independent tuple verifier still obtain genesis from the
same corrupted `initial_state`/`State` constructor. This is a direct
single-fault shared-trust false-positive, **not** evidence of model
learning or a legitimate candidate solution under unmodified rules.

The production fix belongs to the separate source owner; Wave17 has made
no edits to the shared novel-router implementation. Until the fix is
independently reattacked, the source-level `VERIFIED_WIN` label cannot be
treated as robust against this adversarial runtime mutation, even though
the unmodified finite mathematics and public observability checks survive.
