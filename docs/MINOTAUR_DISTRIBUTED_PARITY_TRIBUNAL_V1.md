# Minotaur Distributed Parity Tribunal V1 — fifth source-public family

## Exact declared mechanics

Two isolated agents are assigned local observations `x` and `y`. Neither sees
the other's value and neither may communicate. Before the world is selected,
each agent commits to a deterministic binary lookup table, `L[x]` and `R[y]`.
All outputs happen simultaneously. For **every** declared possible world
`(x,y,target)`, the pair must satisfy `L[x] XOR R[y] = target`. The sum of
asserted-one entries across both lookup tables must be at most
`total_on_budget`. A single observed success is not sufficient.

This is an asymmetric-information **joint policy synthesis** problem. It is
not navigation, single-agent causal intervention, exact-cover item assignment,
or timed task scheduling. Changing the framing does not create new mechanics.
Allowing communication, shared observation, pre-shared randomness, adaptive
responses, or more actions changes the game and needs a new declared ruleset.

## Reproducible scope

Source: `labyrinth/distributed_parity_tribunal.py`

Tests: `tests/test_distributed_parity_tribunal.py`

```bash
python -B -m unittest discover -s tests -p test_distributed_parity_tribunal.py -v
python -O -B -m unittest discover -s tests -p test_distributed_parity_tribunal.py -v
python -B -m unittest discover -s tests -q
python -O -B -m unittest discover -s tests -q
```

Four public source fixtures:

| Fixture | Grounded verdict | Mechanism |
| --- | --- | --- |
| `coordinated_parity` | `VERIFIED_WIN` | Four worlds target `x XOR y`; exact two asserted-one table entries |
| `missing_corner` | `VERIFIED_WIN` | AND-style target with the hard corner absent; all-zero tables work |
| `and_paradox` | `PROVEN_UNSAT_UNDER_DECLARED_RULES` | Four worlds target `x AND y`; XOR across all four edges contradicts |
| `inconsistent_reports` | `PROVEN_UNSAT_UNDER_DECLARED_RULES` | Same local observations require both target bits |

For `and_paradox`, equations on worlds `(0,0)`, `(0,1)`, `(1,0)` force
`L(1) XOR R(1) = 0`. The fourth world requires it to equal `1`; finite
contradiction. That is a theorem **under this public deterministic/no-channel
ruleset only**, not a claim that no undiscovered physically different rescue
is possible.

## Independent evidence, disagreement, and uncertainty

- **Source implementation:** enumerates all `2^(left+right)` lookup policies.
- **Independent witness replay:** separately checks exact types, cardinality,
  and each world without calling the source replay.
- **Independent existence checker A:** Gaussian elimination over GF(2), plus
  free-variable enumeration for the non-linear budget.
- **Independent existence checker B:** independently coded constraint-pruning
  DFS that enforces budget before accepting a complete assignment.
- **Test-owned oracle:** another exhaustive bit-vector enumeration, wholly in
  the test module, checks 512 deterministic seeded scenarios and observation
  relabelings. This provides mechanical cross-checking, not statistical evidence
  of model learning.

`VERIFIED_WIN` requires exact replay and three affirmative checks. Finite
`PROVEN_UNSAT_UNDER_DECLARED_RULES` requires exhaustive source enumeration
and two independent negative existence results. A cap reached before search
completes is `NO_KNOWN_WIN` or `UNKNOWN`, never UNSAT. Any disagreement between
source and independent validators is `UNKNOWN` and should be investigated.
The finite exhaustion receipt records the policy-space size, the total
enumerated policies, and zero valid policies; it is checkable by re-executing
all policies against the exact fingerprinted declared world list. It does not
claim to be an independently checked general theorem prover.

An unexpected exact policy is independently replayed even if a prior answer
claimed UNSAT. Valid surprises can overturn that answer. A proposed extra
communication channel is `NOVEL_RULESET_CHALLENGE`, not a default WIN or proof
that existing rules were solvable. The proposal itself is NOT_PERFORMED as an
independently verified physical mechanism.

## Hostile regression found during initial red-green run

The first implementation of independent DFS checked `index == n` before
checking `used > total_on_budget`. This incorrectly accepted over-budget
complete policies. The randomized independent test found seven failures
(seeded cases 88, 187, 355, 367, 383, 393, 468). Corrected by moving the
budget rejection before terminal success. A minimized permanent regression
uses one world `(0,0,target=1)` with `total_on_budget=0`: the target requires at
least one asserted-one entry and is formally UNSAT under the zero budget.
This also demonstrates why a green
source enumerator alone is insufficient evidence of a valid independent oracle.

## Honest controls and learning ceiling

The one fixed-script policy wins 0/4 public fixtures; the deterministic
symbolic solver wins 2/4. The public fingerprint memorizer scores 4/4 exact
verdict lookups and yields `UNKNOWN` on unseen fingerprints. Both controls
report **zero optimizer updates**. The apparent memorizer 100% is deliberately
not presented as reasoning or transfer.

This is an executable source-public benchmark and adversarial test target,
not a hidden independent examination. No local model was run or trained by
this module. Four fixtures and same-mechanic metamorphisms do not constitute
cross-mechanic generalization; model weight differences, checkpoint reload,
contamination exclusions, and matched model controls require separate
qualified experiments. Integrate this fifth mechanic into the existing runner
only after independent review; this patch does not change the runner/CLI.
