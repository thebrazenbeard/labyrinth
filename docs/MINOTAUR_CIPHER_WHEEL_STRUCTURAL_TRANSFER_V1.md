# Minotaur Cipher Wheel: structural transfer and hostile controls

**Evidence class:** `SYNTHETIC_SOURCE_TEST`. Public source-only finite-state
examination. No private custody, GPU/model execution, checkpoint update,
structural neural transfer, or real-world novel discovery is demonstrated.

Source base: Labyrinth unified hostile checkpoint
`b43daaa469fa2fc629459da1d54ccffcef1bbde0` inspected 2026-11-07.
This challenge has no dependency on the earlier two-step rebound law or the
seven binary Epistemic Shift kernels. Its action composition is *noncommutative*:
executing A then B is generally not equivalent to B then A.

## Two genuinely distinct mechanical families

The hidden device has four internal states, independently relabeled for each
bounded seed as colored indicators. Candidates see only the current color,
three permitted controls (`A`, `B`, `RESET`), charged public observations,
remaining attempts, and a paraphrased task. `RESET` always restores state 0.
They must submit a full eight-row transition table: for each color, where A
and B lead. There are at most 64 attempted actions; malformed actions count
toward the budget and leave state unchanged. Table submission terminates.

The TRAIN mechanism combines a four-cycle and a reflection. Their generated
permutation group has **8 elements (D4)**. The HELDOUT mechanism combines a
three-cycle (fixing one state) and a transposition connecting that fixed state
to the cycle. Its generated group has **24 elements (S4)**. Those distinct
group orders and fixed-point structures are executable structural invariants,
not differences in wording, rotation seed, or prompt color labels.

The deterministic seed shuffle is source-public. Label identity changes
candidate-facing transition tables, but it **does not create unlimited distinct
semantic mechanics**: only two underlying group mechanisms are declared. A
seed permutation alone never counts as a new transfer family.
The effective semantic sample size is **one mechanical family within TRAIN**
and **one within HELDOUT**, even if thousands of seeds are evaluated. Neither
group can be reclassified as a set of independent graduation families.

A deterministic 100-seed control sweep (seeds `0..99`, zero neural updates)
demonstrates the danger: a frozen table memorized for TRAIN seed 0 still wins
**6/100 TRAIN relabelings** because some permutations produce the same visible
transition map. It wins **0/100 HELDOUT** cases. The public-observation
symbolic explorer wins **100/100 TRAIN** and **100/100 HELDOUT** cases within
the action budget. These are synthetic solver/control scores, not model
learning results, and the 100 relabelings are not 100 independent families.

## Controls and oracle

- `symbolic_explorer`: a **non-learning**, deterministic graph learner uses
  only the public `view()` and `act()` interface to navigate, observe, and
  construct the complete table. It has no access to seed, split, hidden state
  index, simulator's transition arrays, or the independent oracle.
- `memorized_train_seed_0`: a frozen, zero-update table for one published
  training seed. It performs no observations. It illustrates how exact
  memorization can score a perfect observed case while failing a different
  mechanism or seed permutation.
- Independent finite verifier: reconstructs seeded labels without calling the
  simulator generator and computes transitions from separate explicit
  mathematical formulas. It checks every charged observation and all eight
  entries of a submitted table, returning either `VERIFIED_WIN` or
  `VERIFIED_FAILURE` with the first falsifying transition. Active cases are
  `UNKNOWN`; simulator/oracle disagreement fails closed to
  `EVALUATOR_DISAGREEMENT`.

The `candidate_view` result deliberately excludes seed, heldout flag,
world ID and transition tables. The evaluator receipt includes those values
for reproducibility. This is interface separation, **not information security**:
the code and source-known mechanisms are accessible to any local caller, and
a malicious Python caller holding a `CandidateSession` object can introspect
private attributes. Only model-facing public JSON is treated as an input.

## Reproduce locally, with no model or service

```bash
python3 -m labyrinth.cipher_wheel_cli --seed 0
python3 -m labyrinth.cipher_wheel_cli --seed 0 --heldout
python3 -m labyrinth.cipher_wheel_cli --seed 0 --heldout --control memorized
python3 -m unittest discover -s tests -p 'test_cipher_wheel_exam.py' -q
python3 -O -m unittest discover -s tests -p 'test_cipher_wheel_exam.py' -q
python3 -m unittest discover -s tests -q
```

The CLI prints a deterministic canonical JSON receipt; it does not call a
local or remote model, start a server, write files, or spend compute credits.
`verify_receipt_replay()` reconstructs the source-known world, validates all
observed transitions, exact submitted table, verdict and receipt checksum.
Every bounded JSON action records its exact canonical value **and top-level
type**; `true`, `0`, `null` and invalid long strings cannot collapse into the
same charged evidence. A recursive exact-type gate also rejects native Python
values that JSON would silently coerce, such as integer/boolean/null dictionary
keys and nested tuples converted into arrays. An unserializable, noncanonical,
or unbounded Python action is charged and then quarantined as
`BLOCKED_UNSERIALIZABLE_TRACE`; no subsequent table can turn it into a win and
the quarantined trace is intentionally not replay-attested. A malformed or
unserializable Python **table submission** is likewise a single terminal
submission, `BLOCKED_UNSERIALIZABLE_SUBMISSION`; it cannot be retried for free
and cannot later be redeemed into verified success. Ordinary well-formed but
wrong tables instead receive a replayable `VERIFIED_FAILURE` and falsifier.
Public `view()` history and independently returned receipts **deep-copy**
nested action and submission objects: modifying a displayed `B` into `RESET`
must not rewrite the charged private ledger even when both controls happen
to leave the current state unchanged.
Receipts explicitly say `neural_learning_verified: false`,
`independent_custody: false`, `graduation_eligible: false`,
`public_source_oracle: true`, `effective_semantic_families_in_receipt: 1`, and
`source_commit: UNBOUND`. A SHA-256 digest is not a signature, and source
replay is not externally authenticated custody.

## Hostile epistemic limits

1. A caller who reads this public source can construct the true table for any
   seed and split **without performing a single intervention**. The test suite
   preserves that shortcut as a negative control: `verified_success: true`
   means only that the **declared finite table matches**, not that a neural
   model learned, discovered, or generalized. Graduation remains ineligible.
2. A deterministic symbolic solver should perform well on this closed system.
   Neural success must be compared against it and against frozen memorization;
   no such neural run is represented by these receipts.
3. Two declared structural families are not a statistically useful private
   heldout distribution, especially when both algorithms reside in source.
   A future evaluator must independently precommit unseen generated mechanics,
   isolate test inputs, and govern model/training custody before graduation.
4. Full-table verification closes only the stated four-state/three-action
   finite mechanics; it cannot prove absence of valid explanations or
   solutions to undeclared or changed rules.
5. The Python interface is not a hostile-code sandbox, and the two oracle
   implementations remain in one source repository. A separate reviewer and
   external signed custody would be required for independent model evidence.

The intended contribution is an executable structural *falsifier* for
sequence memorization and a tractable independent replay exercise, not a
new claim that Minotaur learned to reason.
