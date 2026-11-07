# Wave 15 — hostile mathematical review of Switchback heldout claims

**Source under test:** `thebrazenbeard/labyrinth` local integrated Git
`2345c9c69ad0882cb32c8a584a3d0677af986123` (PR #155 integration).

**Evidence class:** local source inspection + independent deterministic
finite-state tests. This does **not** represent qualified host custody, a
private sealed exam, a neural update, or model generalization.

Run the isolated adversarial tests:

```sh
python -m unittest discover -s tests -p test_wave15_math.py -q
python -O -m unittest discover -s tests -p test_wave15_math.py -q
```

## Independent observed results (public synthetic scope)

| Examination | Reproduced finding |
|---|---:|
| Parameterized Rule values | 840 |
| Distinct complete 16-cell Rule truth tables | **378** |
| Rule equivalence-class multiplicities | 243 singleton, 119 triple, 8 ninefold, 8 twenty-onefold |
| Seeds independently reclassified | **1,000** |
| Verified potential wins / finite UNSAT in all seeds | **717 / 283** |
| TRAIN 0–99 potential wins / UNSAT | **67 / 33** |
| Proposed HELDOUT 100–199 potential wins / UNSAT | **72 / 28** |
| Maximum independent memoized DP states per case | **192** |
| Public initial-metadata buckets over 1,000 seeds | **569** |
| Buckets containing conflicting win/no-win labels | **80** |
| Same-public-metadata case pairs | **608** |
| Such pairs with opposite win/no-win labels | **136** |
| Probe-indistinguishable same-metadata pairs | **1** |
| Maximum shortest identifying probe sequence among pairs | **5** |
| TRAIN / HELDOUT observed family-pair categories | **25 / 24** |
| HELDOUT family pairs already present in TRAIN | **24/24** |
| Descriptive Simpson pair-category diversity TRAIN / HELDOUT | **21.277 / 20.243** |
| HELDOUT exact-seed retrospective identification using initial metadata | **84/100** |
| Same, after one legal `1` probe | **100/100** |

The only pair indistinguishable by **every legal bounded public probe trace**
in the 0–999 sample is seeds **118 and 555**. They have different private
mechanism signatures and different complete law truth tables, but the same
public starting metadata and the same finite solvability label. The unused
transition differences never become visible under their resource limits.
Thus even distinct *full* rule functions do not necessarily define separate
observable examination units. All 136 opposite-label pairs sharing public
initial metadata had a distinguishing trace. This finite statement follows
from exhaustively exploring the paired reachable-state product, not from
checking a few sampled candidate probe paths.

An initial-metadata-only retrospective majority classifier has a ceiling
of **916/1,000 correct labels** *on this particular inspected set*. That is
not a new heldout result and is not a confidence guarantee: the classifier
was allowed to see every label before forming its majority key table.

## Formal decision boundary

The declared puzzle has an eight-value register, a bounded number of pulses,
an exact post-change-input flag, and at most nine actions including COMMIT.
All legal nonterminal actions advance a tick; a COMMIT ends the plan. The
independent memoized backward recurrence in `tests/test_wave15_math.py` has
states `(register, ticks, pulses, post_input)`. It enumerates WAIT and both
binary inputs when legal; it accepts only a COMMIT that (a) has an available
action slot, (b) occurs after the switch, (c) follows a post-switch input,
and (d) matches the target. Its recursion strictly increases ticks, hence
terminates after finite depth without relying on the production BFS.

For a given canonical `Case`, an absent witness after this exhaustive
recurrence is a finite closed-world `PROVEN_UNSAT_UNDER_DECLARED_RULES` claim.
It is **not** a statement about changed semantics, longer budgets, or
unforeseen external actions. The tests independently recompute all five
three-bit law equations and compare with the production search only after
computing the reference answer. The sampled-seed test cannot prove that all
possible seeds are correctly generated or that independently authored source
rules match the actual problem intended by a human.

## Observability is not parameter identity

- Rule objects carry family, mask, bias and threshold. Distinct parameter
  tuples can generate the **same complete 16-entry truth table**. Counting
  parameterizations as independent structures exaggerates diversity.
- Different two-stage rule tables can produce identical public observations
  when the public initial state, target, switch time and pulse cap restrict
  which transitions are reachable. A model cannot be credited for identifying
  an underlying law that no allowed experiment distinguishes.
- The pair-product search checks public observational equivalence for cases
  with equal initial public metadata. A difference in visible register output
  is a discriminator. RESET never increases the *existence* of a distinguishing
  sequence: after RESET, both machines return to the same shared initial
  state, so the suffix could have been executed from the start. Invalid
  actions also yield no new mechanical observation. The legal visible
  evolutions terminate after nine ticks, within the 16 charged probes.
- A different case hash, or even a different mechanism signature, does not
  alone imply a different candidate-visible or statistically independent
  challenge. The converse is also true: identical mechanism tables with
  different initial/target conditions can change whether a win exists.

## Heldout and leakage

The `0–99` and `100–199` blocks use the **same published five law-family
generators**. Changing a numeric seed is not mechanism-family holdout.
The entire seed-to-case algorithm is public. Given a known bounded catalogue,
public initial metadata and one observation can identify many or all exact
cases; such a lookup can exhibit perfect apparent test performance without
training or genuine transfer. Any retrospective catalogue match is a
contamination demonstration, never a valid predictor of future unseen seeds.
The Simpson pair-category diversity numbers above quantify frequency
concentration *among named family pairs only*; they are **not** estimates of
statistically independent trials or model-learning sample size.

The existing Wave12 public-source adaptive **non-neural** control reports
44/100 TRAIN and 39/100 HELDOUT verified wins (source testimony, not an
independent new policy execution in this review). The independently verified
numbers of winnable cases change the meaningful recall denominators:

| Source-aware control | TRAIN | HELDOUT |
|---|---:|---:|
| All-case win fraction | 44/100 = 44% | 39/100 = 39% |
| Recall among actually winnable cases | 44/67 = 65.7% | 39/72 = 54.2% |
| *Illustrative* IID Wilson 95% interval for win recall | 53.7–75.9% | 42.7–65.2% |

The 11.5-percentage-point observed recall difference is not proof of a
distribution shift, and the illustrative intervals are **not** legitimate
external transfer confidence intervals: all 200 cases arise from the same
public five-family generator. Family-pair concentration and correlations
shrink the independent evidence relative to a naive `n=200`. No mechanism-
population sampling process has been independently specified, so the true
effective sample size is `UNKNOWN`.

## Statistical and semantic acceptance rules for a future trial

1. **Freeze before exposure.** Precommit the candidate executable, weights,
   adapter, memory, tools, prompts, training cases, and evaluator source; keep
   the heldout cases and generator seeds outside the candidate's privileges.
2. **Declare the independent unit.** Count genuinely different generative
   mechanisms or externally adjudicated semantic equivalence classes, not
   paraphrases, case hashes, 840 overlapping parameterizations, or 100 seeds
   from the same five published families. Report family-pair clusters but do
   not assume cluster labels alone establish independence.
3. **Match controls.** Run zero-update, memorized catalogue, public-source
   symbolic, and qualified trained/model candidates on identical sealed cases,
   observation budgets, context and runtime, with no information asymmetry.
4. **Score abstentions and false claims.** Distinguish verified wins, valid
   declared-rules UNSAT certificates, incomplete NO_KNOWN_WIN, wrong plans,
   and abstentions. Report win recall conditioned on *independently* verified
   winnable cases as well as all-case pass rate; do not reward false UNSAT.
5. **Use paired, clustered uncertainty.** Preserve per-case paired outcomes;
   compare matched candidates within mechanism clusters, with confidence
   intervals or permutation tests that respect independent generating units.
   Clopper–Pearson/Wilson intervals under IID cases may be descriptive but
   cannot claim population-level transfer when cases share generators.
6. **Test genuinely new laws.** Introduce mechanism families that are not
   instantiations of the published five, plus adversarial near-twin cases,
   changed resource limits and hidden rule reversals. Require independent
   verifier custody before considering graduation.
7. **Record honest evidence.** Keep Git SHA, exact command, model/checkpoint
   hashes, seeds, trial-level traces, chain of custody and independent signers.
   Source-only synthetic receipts (`source_commit:UNBOUND` or unkeyed SHA-256)
   are reproducibility diagnostics, not authenticated private evidence.

## Verdict scope

- **SURVIVES:** a specific sampled finite-state classification if the test
  comparisons pass on the exact source above.
- **FAILS:** the premise that seed-disjoint public-generator cases are an
  independent hidden-mechanism test, or that distinct parameters imply
  distinct observable causal laws.
- **UNKNOWN:** genuine neural learning, transfer to independently authored
  unknown mechanisms, externally qualified graduation, and true effective
  sample size for a real problem population.
