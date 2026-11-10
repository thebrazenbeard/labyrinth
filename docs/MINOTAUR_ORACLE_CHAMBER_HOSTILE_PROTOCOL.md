# Minotaur Oracle Chamber — hostile information-theoretic examination

**Status: public executable experimental source, NOT a qualified learning exam.**
Source orientation: `thebrazenbeard/labyrinth` main at
`b33a517daa47c13f8692a898cd277460dfca968a`, freshly rechecked on
2026-11-07. Existing Draft PRs #139/#140 introduce route and item puzzles;
the separate Timegate experiment introduces timing and consumable power.
Oracle Chamber changes the actual problem class to **adaptive information
acquisition with a bounded adversarial liar**.

## The recurring dilemma

The opponent hides one of four two-bit secrets, `0` through `3`. You may make
at most **five binary queries**. `alpha` reveals bit 0, `beta` reveals bit 1,
and `parity` reveals their XOR—except an adversary may invert a bounded number
of answers. Invalid queries consume budget and return no secret information.

Two declared-rules near twins change exactly one constraint:

- **Open:** At most **one** answer can be inverted. There is a fixed schedule
  `alpha, alpha, beta, beta, parity` whose codewords have minimum distance 3;
  nearest-codeword decoding corrects every single inversion. An independent
  verifier enumerates all **4 secrets × 6 lie placements = 24** possibilities.
- **Sealed:** At most **two** answers can be inverted. For *any deterministic
  adaptive binary-question policy*, each secret has
  `1 + C(5,1) + C(5,2) = 16` distinct possible response transcripts. Four
  secrets require 64 disjoint transcripts, but five binary answers provide at
  most `2⁵ = 32`. A guaranteed identification strategy is impossible
  **under precisely these declared rules**. This is not a proof that an
  unmodeled sensory affordance cannot exist.

The verifier checks the exact counting assumptions. On the open twin the same
volume test finds only `4 × (1+5) = 24 ≤ 32` and returns **UNKNOWN**, not an
impossibility certificate. The impossibility claim is about universal
guarantees against the stated adversary, not whether a particular individual
guess can happen to be correct in a sealed instance.

## Explicit candidate claims

`CandidateSession` allows charged queries followed by exactly one submission:

| Claim | Verification | Task evidence (never graduation credit) |
| --- | --- | --- |
| `COMMIT` | Correctness on one sampled secret | **None**, despite `VERIFIED_WIN` instance label |
| `POLICY` | Exhaustive replay across *all* secrets and allowable lie patterns | `policy_verified=True` if every test survives; `verified_success=False` pending external qualification |
| `ABSTAIN` | `UNKNOWN` or `NO_KNOWN_WIN` recorded | None |
| `UNSAT_CERTIFICATE` | Recomputed exact declared-rule information bound | `certificate_verified=True`, **not** escape or policy success |
| `NOVEL_RULE` | Exact proposed change preserved | None until independently established and reexamined |

**No claim in this source-only experiment sets generic `verified_success` to
true.** An external admission/graduation gate must require controlled new
mechanics, actual model adaptation and independent custody, not this public
toy family's deterministic correctness.

The supported policy representation includes a fixed question schedule with
nearest-codeword decoding, or a query tree indexed by the previously seen
binary prefix plus an explicit terminal decoder. A failed strategy produces a
replayable counterexample identifying secret, liar slots, query sequence and
the incorrect prediction. Unsupported/incomplete policies return UNKNOWN.

The source also illustrates a proposed **new attested four-outcome sensor**.
If such a noiseless two-bit query genuinely existed, the old 32-binary-path
bound would no longer apply. The proposal remains a conditional
`NOVEL_RULESET_CHALLENGE`, not a retroactively accepted counterexample to the
old declared-rules theorem. A future independent reviewer must reconstruct
and validate any such changed affordance before changing the ruleset.

## Independent controls and negative tests

```bash
python3 -m unittest discover -s tests -p test_oracle_chamber_exam.py -v
python3 -O -m unittest discover -s tests -p test_oracle_chamber_exam.py -q
python3 -m unittest discover -s tests -q
```

- `oracle_chamber_exam.py` implements candidate-facing question results.
- `oracle_chamber_verifier.py` independently tabulates predicate truths,
  replays charged attempts, exhaustively attacks policies, and checks the
  Hamming-volume bound. Same author/source means **separate codepaths, not
  independent reviewer or custodian**.
- `oracle_chamber_baseline.py` is deterministic, parses only candidate-visible
  instructions and uses no optimizer/model/hidden sample. It produces a fully
  verified strategy on all tested open cases and a properly qualified
  impossibility certificate on the sealed twins.
- Adversarial tests cover all 24 open and 64 sealed sampled-adversary
  combinations in separate engine/verifier replay, an intentionally corrupted
  transition, invalid inputs, certificate forgery, decoder counterexamples,
  terminality, paraphrase invariance, and novel ruleset proposals.
- A third fixed hand-calculated truth-table check catches shared engine/oracle
  arithmetic mistakes; all **243** possible nonadaptive five-query schedules
  are also challenged against the two-lie twin. The broader adaptive
  impossibility verdict rests on the disjoint-response counting argument.

### Fresh source-only execution (2026-11-07)

- Focused new tests: **34/34 PASS**.
- Full Labyrinth main plus new files: **270 tests OK, 42 optional skipped**.
- Zero-update public-only parser, one-off extended deterministic seed sweep:
  **2,000/2,000 open `VERIFIED_WIN` policies**, and **2,000/2,000 sealed
  `PROVEN_UNSAT_UNDER_DECLARED_RULES` certificates**. A correct certificate has
  `verified_success=False`; a single correct commit has
  `instance_win_verified=True` but no policy/graduation credit. All 2,000
  seeds belong to the **same conceptual family**, not 2,000 independent tasks.

### What results do **not** show

1. There is **no neural Minotaur execution, tensor change, checkpoint,
   zero-update neural comparison, or structural transfer measurement** in this
   experiment. A hand-coded parser's perfect score is a warning against
   equating test accuracy with learning.
2. Public deterministic seeds, paraphrase variants and this known two-bit
   code are one conceptual family. Do not count each seed as independent
   learned reasoning or a hidden holdout.
3. `CandidateSession` is same-process Python. Its `_case` fields and source
   can be inspected. It is **not** an isolated adversarial service; untrusted
   candidates require a separate process/custody boundary and signed receipts.
4. A mathematically verified finite declared-rules bound does **not** rule out
   genuinely new rules, new question alphabets, new observations, or another
   designer mistake. Preserve and independently examine credible surprises.
5. Evaluator/program agreement remains insufficient for external empirical
   claims; an independent implementation, blinded task selection and genuine
   model-control receipts remain graduation prerequisites.
