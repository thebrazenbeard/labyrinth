# Wave 18 — mechanically new causal-threshold out-of-distribution family

**Base:** exact Labyrinth novel-source chain
`3e29c87c62bd0bffc8fb05c243e8db7bc6268e8c` (PR #165).

**Implemented:** `labyrinth/novel_router_ood_family.py`.
**Independent test oracle:** `tests/test_novel_router_ood_family.py`.

This is a real runnable additional examination family. No old register,
Switchback Circuit, or delayed-router source function is imported or used.
No host, GPU, trained model, checkpoint or neural optimizer is invoked.

## Mechanism — not another seeded packet router

The prior exams have one three-bit arithmetic register or two numeric
reservoirs moving delayed packets. This one is a **five-node directed
Boolean causal dependency network**, evolving by **synchronous nonlinear
threshold updates** with **irreversible interventions**.

The hidden causal rule consists of five directed parent masks and five
node-specific thresholds. The public state has five binary node levels and
a separate permanent seal mask. The allowed plan actions are:

- `TOGGLE_i`: intervene on one unsealed Boolean node, directly changing its
  value without running the causal network.
- `SEAL_i`: irreversibly lock a node at its current value, consuming one of
  two nonrenewable seal slots.
- `PULSE`: simultaneously update **every unsealed node from the same old
  global network snapshot**, using hidden parent dependencies and local
  thresholds. Sealed nodes do not update.
- `COMMIT`: terminate with both the declared target bitmask and the exact
  declared pair of sealed nodes, after a causal PULSE executed with the two
  irreversible seals in place.

For an unsealed node `i`, the one-step causal law is:

\[
x_i(t+1) = \mathbf 1\left[\sum_{j \in \mathrm{Parents}(i)} x_j(t)
\geq \theta_i\right].
\]

The action horizon is **six including COMMIT**; a candidate can spend at most
two seals, and a public exploration session gets sixteen charged probes.
The source also offers RESET during probing, with a one-shot submission
afterwards. The private dependency graph is not placed in the ordinary
candidate `view()` output, but the generator itself remains open Python
source. This is not a secure private evaluation boundary.

### Falsification changed the construction

The first source rehearsal had a nine-move horizon and a loose goal requiring
only one sealed node. **All 200 tested seeds were winnable**—a useless
impossibility examination that could reward generic toggling. Tightening the
game to six moves, exactly two specified permanent seals, a mandatory
post-seal pulse, and two public goal bitmasks produced both real wins and
finite UNSAT cases. This is an observed design correction, not a predictive
claim that the new distribution is maximally difficult.

The independent oracle explicitly demonstrated that a synchronous network
update differs from incorrectly updating nodes one at a time: on seed zero,
starting bits `00001`, the parallel result is `00100` (decimal 4) whereas
sequential updating yields `00000` (0). The graph interpretation matters.
Permanently sealed nodes also remain fixed across later pulses, and
attempting to toggle a sealed node is illegal.

## Executed verification

```bash
python -m unittest discover -s tests -p test_novel_router_ood_family.py -q
python -O -m unittest discover -s tests -p test_novel_router_ood_family.py -q
python -m labyrinth.novel_router_ood_family --seed 2
python -m labyrinth.novel_router_ood_family --seed 2 --plan SEAL_0 SEAL_3 PULSE TOGGLE_4 COMMIT
python -m labyrinth.novel_router_ood_family --seed 0 --claim-unsat
```

| Evidence | Measured result |
|---|---:|
| Focused tests normal / optimized | **10/10 PASS / 10/10 PASS** |
| Generated source cases independently adjudicated | **200** |
| Finite wins | **119** |
| Finite UNSAT under declared rules | **81** |
| Disagreements between independent tuple-state DP and source BFS | **0** |
| Maximum independently explored DP states per case | **1,583** |
| Maximum source BFS nodes per case | **2,409** |
| Independently checked reachable source transitions | **705,661** across 30 seeds |
| Distinct hidden dependency/threshold rules among 200 seeds | **200** |
| Source scorer false-WIN mutation | **Blocked** by independent raw-state check |
| Shared State constructor genesis mutation | **Blocked** by primitive genesis |
| Corrupted source step falsely proving UNSAT | **Blocked** by independent closure |

An independently implemented Boolean threshold reference on primitive
tuples performs backward dynamic programming over all legal remaining
actions. It does not call the production search, step, or pulse helper.
Production forward search separately uses exact primitive state keys and
must finish closure before issuing `PROVEN_UNSAT_UNDER_DECLARED_RULES`.
Capped search returns `NO_KNOWN_WIN`. The adjudicator cross-checks any
PLAN outcome against a second raw-state replay and any UNSAT claim against
independent finite DFS, retaining valid alternative solutions. The public
probe path also checks source transition parity before reporting effects.

## Leakage and qualification — explicitly negative

The proposed seed block `100–199` is **not a private heldout inventory**.
Within that preknown 100-seed catalogue, all **100/100 case identities** are
uniquely recoverable from the initial public `(initial, target, required
sealed-mask, budget)` tuple. The source generator is public and deterministic,
so any model with access to that specific alleged heldout seed range could
precompute a label/plan catalogue without learning graph mechanisms. This
does **not** establish that the seed could be recovered uniquely among all
`2**31` possible seeds without knowing the block. The broader posterior has
not been measured; unconditional identity leakage remains UNKNOWN.

Therefore the source-public family is **mechanically novel but not a
qualified private test**. Receipts explicitly report:

```
evidence_class: SOURCE_PUBLIC_SYNTHETIC
source_commit: UNBOUND
private_custody: false
neural_learning_verified: false
graduation_eligible: false
```

Their SHA-256 values are deterministic unkeyed checksums, not signatures.
No candidate model was trained, adapted, or run; no claims about neural
generalization, independent human surprise, or genuine private graduation
are supported by this source-only benchmark.

## A credible subsequent heldout protocol

1. Generate challenge-family source **after** freezing model weights,
   adapters, tools, memory, training corpus, and experiment-selection rules.
2. Custody evaluator mechanics and random seeds independently; neither
   source code nor exact trial inventory may be readable by the candidate.
3. Independently derive complete reachable-state labels and preserve legal
   counterexamples for every candidate UNSAT claim.
4. Sample several independently authored causal graph families (not merely
   200 hashes from this one public threshold generator). Include adversarial
   near-twins, rule changes, misleading premises and observationally
   equivalent cases.
5. Run matched zero-update, catalogue memorizer, symbolic source solver,
   corrupted-label model, and genuinely updated model with equal budgets;
   cluster significance tests by **independent family**, not numeric seed.
6. Keep an UNKNOWN verdict when a source-private rule expansion is claimed
   without qualified independent adjudication. Accept any verified valid
   alternate path, even when the designer did not foresee it.

**Verdicts:**

- `SURVIVES` — distinct Boolean causal graph family and the exact tested
  119/81 finite labels, source-only.
- `FAILS` — the earlier all-winnable rehearsal and any claim of hidden
  evaluation/secrecy for the published deterministic seed block.
- `UNKNOWN` — model learning or transfer, public-source population power,
  authentic private custody, and effectiveness against unseen independently
  authored mechanism families.
