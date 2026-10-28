# Minotaur 100-pair public neural representation study: observed result

Date: 2026-10-28. **Public protocol hypothesis: PASS**.
**Private qualification: NOT_ESTABLISHED. No graduation or deployment.**

## Frozen source, policy and output custody

- Frozen protocol source: `thebrazenbeard/labyrinth` draft PR #49, head
  `054744bf3e851cfd68521999f22c95d3196d07fd`, tree
  `970926c1a54b1e36a8cf169c1f2ed0ce4e5c4390`. The protocol was
  published before any reserved-pair fitting, and hosted exact-head CI passed.
- Canonical protocol SHA-256:
  `b37ac57d4f93fc4148b2bee5d18d6efb3dcb0d52646796cf35ae1142d69217ba`.
- Frozen 100-pair seed manifest SHA-256:
  `664737a816fca6f8afeecb04051b3a0d692a0141b5fe127b6912f5e77d467c58`.
- Source-manifest SHA-256, supplied separately to the offline CLI and
  checked before and after execution:
  `1f4b54d1b0d228e74f7ce6afd45d1b79da69ba779d39b585c5248677b8f0a732`.
- Raw full JSON report SHA-256:
  `44d607f57f256f2b78a854f4a04e2590e1ff6fc553e4c117ecba21bcdb6b23a7`.
- Deterministically compressed artifact:
  `research/artifacts/minotaur_public_100_pair_results_2026-10-28.json.gz`;
  gzip SHA-256:
  `bfdd4018b11128f8f1642028f8a95a17a712fa2d986cdc35b768cb9c63d33bf6`.
  Decompress the archive and verify the raw JSON SHA-256 before analysis.

## Primary and controls

One hundred unique matched dataset/initialization pairs were run exactly once
from the frozen manifest. For each pair, two learned 12-coefficient float64
CPU heads (bilinear `xy` versus equal-sized no-cross-term `x²`) shared
supervision, coefficient initialization, optimizer closure cap, and 128 unique
six-step heldout register programs. Two negative controls were separately
evaluated: frozen bilinear initialization and bilinear wrong-label training.

| Measure | Bilinear | No-cross-term | Frozen | Wrong-label |
| --- | ---: | ---: | ---: | ---: |
| Exact whole six-step programs (out of 12,800) | **12,800** | 1,190 | 0 | 0 |
| Per-pair exact-program range (out of 128) | **128–128** | 4–21 | 0–0 | 0–0 |
| Successfully scored arms (out of 100) | 100 | 100 | 100 | 100 |

The precommitted primary event required the bilinear arm to complete at least
13 more entire programs out of 128 per pair (strictly >10 percentage points).
All **100/100** matched pairs met it, with differences of 107–124 programs.
The nominal one-sided exact binomial value is
`2^-100 = 7.888609052210118e-31`, under the stated operational independence
assumption for pseudorandom seed-pair outcomes. The 59/100 precommitted
threshold was exceeded. The mean paired improvement of 90.703125 percentage
points is descriptive, not a separately tested hypothesis.

Additional descriptive evidence: bilinear correctly scored all 28,191 exact
training-excluded atomic tuples for every pair (2,819,100 across 100 paired
runs). Those are repeated publicly defined arithmetic cases, **not**
independent statistical replicates. Observed bilinear L-BFGS closure counts
ranged 33–315, below the hard cap of 500. All 400 arms returned `SCORED`;
no failed or replaced pairs, missing controls, or reported source drift.

## Independent review and interpretation boundary

Two read-only review agents independently recomputed all 100 primary wins,
the exact p-value, ordered seed identities, program/training-data digests,
no-overlap checks, scored-arm totals, and closure bounds from the report.
Both found the reported public classification internally consistent. This
is **artifact consistency review**, not an independently trusted examiner
or replay of every neural prediction. The archive contains coefficient
digests but not full checkpoint tensors or per-program prediction traces.

The feature family explicitly supplies multiplication (`xy`) and modulo-97
projection; the public task and seeds are not a sealed private challenge.
The model demonstrates narrow coefficient learning under those priors, not
arithmetic discovery or broad general reasoning. The pseudorandom seed
generator does not by itself guarantee independent sampling. Source hashes
can be fabricated by a hostile producer; these results do not exercise
private evaluator custody, signed exam receipts, monotonic attempt state,
rollback anchoring, Vera installation, or runtime effectiveness.

`public_primary_gate=PASS` is **only** the defined public representation
comparison. The full artifact states `qualification=NOT_ESTABLISHED`,
`graduation_authority=none`, `minotaur_graduated=false`, and `deployed=false`.
