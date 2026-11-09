# Rule-Reversal Tribunal v1 — causal adaptation, not a hidden-label quiz

**Status:** Source-public standalone examination. No neural model, private
examiner process, model-training artifact, or graduation qualification.

## Distinct mechanism

The world contains two indicator bits and two physical actuators, A and B.
Both indicator bits start off, as explicitly disclosed to the candidate.
An intervention changes the actual state by a two-bit XOR effect. After at
least one pre-shift intervention, the candidate must invoke `SHIFT`. The
effects of A and B then change under a hidden per-world causal law. A candidate
must intervene after the shift and can `COMMIT` only when both indicators are
on. All four action types cost one finite attempt, and the shift is mandatory
and allowed once. An observation is feedback from a *state-changing action*;
there are no free information probes.

The exam asks for **one public, observation-dependent policy** that works for
every world compatible with the observed history. It does not reward a
clairvoyant action string chosen separately for each hidden world. A winning
policy is replayed unchanged under every world; no predetermined gold trace
or designer-provided answer key is required for credit.

This is distinct from Veiled Fork's static probes and irreversible final fork:
in Tribunal the candidate must alter a physical state, survive an epoch change,
and sometimes correct a causal intervention that became counterproductive.
The five counterfactual fixtures are **one causal-revision structural family**,
not five independent transfer samples.

## Reproduced test-of-tester failure

The first synthetic two-world pair was unsuitable: exhaustive enumeration
found a fixed script that won both worlds, and the no-feedback case was
winnable. The intended adaptive advantage was therefore **FAILS**, even though
the source solver returned a true winning policy. The regression tests were
kept, and the actual post-shift effects were changed. The corrected pair now
has the following exact masks (bit 0 = first indicator):

| World | A/B before SHIFT | A/B after SHIFT |
|---|---|---|
| 0 | `01` / `10` | `10` / `01` |
| 1 | `01` / `10` | `11` / `10` |

One robust strategy intervenes with B before SHIFT, invokes SHIFT, then uses B
again. The observed response determines whether to `COMMIT` immediately or
correct with A before committing. All eligible fixed, noncontingent action
sequences of at most five actions fail at least one world. The fixed four-action
script `A, SHIFT, A, COMMIT` wins only one of two worlds; the source-public
adaptive symbolic policy wins both. A **zero-update fingerprint catalogue**
recites all five source-public fixture verdicts correctly (5/5) while
abstaining on an unseen fingerprint. Neither result proves neural learning.
**Optimizer updates: zero.**

## Honesty of proof and new discoveries

Supported result categories are:

- `VERIFIED_WIN`: candidate's frozen policy replayed to terminal success in
  every declared hidden world within budget;
- `VERIFIED_FAILURE`: invalid or failing policy, including malformed or
  excessive actions (each charged);
- `NO_KNOWN_WIN`: explicitly truncated source search, not proof of loss;
- `PROVEN_UNSAT_UNDER_DECLARED_RULES`: finite belief-state search completed
  with no all-world winning policy within the declared horizon;
- `UNKNOWN`: malformed search request or internal disagreement;
- `NOVEL_RULESET_CHALLENGE`: proposed unlisted intervention, quarantined for
  separate semantics and independent validation, **never** auto-certified.

`public_view` reveals a rule description, the action vocabulary, budget, sensor
type and initial observation. It does not reveal private world deltas or the
host-owned fingerprint. The input policy is recursively copied as exact JSON
primitive dictionaries/strings before any world replay, rejecting dynamic
Python object keys, deceptive subclasses, cycles and excess depth. This is
source input hygiene, **not** independent OS or model attestation.

Independent native Skeptic reproduced unbounded novel-rule response
amplification: an action name of one million characters was accepted as a
challenge and echoed verbatim. The input boundary now rejects new action
names longer than 128 characters before constructing the response; 129 and
1,000,000-character reproductions are preserved as regression tests.

## Independent evidence

`brute_force_oracle` is a second Boolean belief-state implementation using
primitive tuples and raw bitmask transitions. It never calls the production
transition function or replayer. The tests compare both verdicts across:

- **1,024 exhaustive** two-world post-shift mask combinations, two budgets,
  and full versus sealed sensor feedback;
- **300 seeded randomized** worlds/budgets/sensors;
- independently enumerated fixed scripts of lengths one through five;
- invalid action budgets, malformed policy objects, novel ruleset claims,
  no-feedback UNSAT, genuine incomplete search, unchanged positive controls,
  and the 5/5 source-public catalogue contamination baseline.

The mutation-killing test deliberately makes the source transition claim a
nonexistent B-actuator capability after SHIFT. Its **correlated source solver
and replay both report `VERIFIED_WIN`**, but the separately implemented oracle
still rejects the declared no-y-causality scenario. Restoring the unmutated
transition restores the UNSAT verdict. This is a concrete example of why
replay and solver agreement alone cannot certify a proof.

Run:

```bash
python -m unittest discover -s tests -p test_rule_reversal_tribunal.py -q
python -O -m unittest discover -s tests -p test_rule_reversal_tribunal.py -q
python -m unittest discover -s tests -q
python -O -m unittest discover -s tests -q
```

## Limits and future qualification

This source-public scenario is reproducible by a deterministic solver. A
zero-update catalogue or a symbolic planner can solve it without acquiring
neural reasoning. It has no held-out private model execution, no authenticated
checkpoint identity and no matched neural update measurement. A future
Minotaur model trial must freeze source, evaluation seeds, model weights and
prompts outside the model's accessible environment; compare matched controls;
and test genuinely unfamiliar mechanisms after model freeze. A new solution
outside these finite actions should be reconstructed and checked rather than
rejected as impossible by definition.
