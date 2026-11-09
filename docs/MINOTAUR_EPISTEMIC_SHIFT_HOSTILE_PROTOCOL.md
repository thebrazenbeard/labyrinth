# Minotaur Epistemic Shift — causal model revision under changed rules

**Status: public executable developmental examination. No neural learning
qualification or private challenge custody.**

Source orientation: `thebrazenbeard/labyrinth` main SHA
`b33a517daa47c13f8692a898cd277460dfca968a`, freshly checked 2026-11-07.
This is a fifth conceptual mechanics family, separate from item/room movement,
timed consumable gates and corrupted binary-secret interrogation.

## What the examinee must confront

A machine accepts interventions `(x, y)` with binary causes and gives a
**truthful** binary effect. Invalid interventions consume one of ten attempts
but cannot advance causal state. For the first three **valid** interventions,
the effect is `x`. In changed-law cases, the machine silently switches to a
different mechanism. No physical travel, stored battery or adversarial liar
is involved. An examinee may record a provisional hypothesis, gather
falsifying interventions, revise that hypothesis and submit a final causal
claim. The evaluator can independently replay it and challenge the final law
against every three-step sequence of future interventions.

The candidate-facing text says the law may change, but deliberately omits the
actual hidden mechanism, exact source split and change-point. The public
source still reveals those details and is **not** a private held-out test.

## Precommitted semantic split (not wording variants)

| Split | Underlying post-change mechanism | Causal structure |
| --- | --- | --- |
| `train` | XOR; XNOR | Stateless affine Boolean functions |
| `dev` | AND; OR | Stateless nonlinear functions |
| `test` | `y AND (x XOR previous_x)`; `x AND previous_y` | Stateful order-sensitive functions |

Each split has two exact rule families. Seeds select which family; wording
variants do not alter causal rules. `semantic_fingerprint` records the
mechanism rather than treating rephrased copies as new transfer tasks.
The split mapping is now frozen in source, but this source creation was not
independently time-stamped as a blinded pre-registration before exploratory
engineering. It must not be described as a privately precommitted model test.

Every changed-law case has a one-rule no-change near-twin: the otherwise
identical machine continues emitting `x` after the usual switch point. A
candidate must not infer a rule change solely because the prompt says it
*might* happen. This is a causal change-detection challenge, not an
impossibility certificate; no unsatisfiability claim is made.

## Claims and independent examiner

- `PREDICT`: a two- or three-intervention forecast, independently replayed.
  Even a correct forecast gives `prediction_verified=True` without learning
  credit—one lucky forecast is not generalized reasoning.
- `HYPOTHESIS`: a proposed kernel tested against **64** future three-step
  action sequences (up to 192 individual transition checks). Incorrect
  mechanisms receive a concrete falsifying intervention sequence.
- `REVISION`: requires an initially recorded `copy` belief, a subsequent
  observation contradicting that belief, and a newly recorded hypothesis
  after the contradiction. The revised kernel must also survive all
  counterfactual probes. A retrospective scripted revision is still not
  proof of neural adaptation.
- `ABSTAIN`: record `UNKNOWN` or `NO_KNOWN_WIN` with no success credit.
- `NOVEL_RULE`: preserve exact proposed semantics. Optional four immediate
  predictions are independently checked for local consistency, and a false
  prediction returns a counterexample. A local fit does **not** establish a
  new global law; external review and rule reconstruction are required.

No claim in this public artifact sets generic `verified_success=True`. It
only records narrow task-specific `hypothesis_verified`,
`revision_verified` or `prediction_verified` results. Actual training and
held-out model qualification require separate custody and controls.

The environment and independent verifier implement the causal transition
rules separately. After **every** attempted intervention, a full independent
replay checks the state registers and observations. A mismatch or runtime
exception terminates with `EVALUATOR_DISAGREEMENT`, never silently assigns
candidate failure or allows a terminal result to be overwritten.

## Deterministic non-learning controls

`epistemic_shift_baseline.py` fixes an intervention schedule *before* seeing
the evaluated sample. A symbolic classifier considers both stateless and
stateful candidates and revises `copy` only after observed contradictions.
It uses no optimizer, neural weights or hidden `Case` fields. A prelaw-frozen
control insists that the law remains `copy`; another insists that the
training XOR rule always applies. Both are separately evaluated on the
stateful test split. The general symbolic parser can solve the public test
family without learning precisely because the mechanism catalogue and
schedule are built into its source. **Perfect scores here do not demonstrate
generalization by a neural Minotaur.**

## Reproduce

```bash
python3 -m unittest discover -s tests -p test_epistemic_shift_exam.py -v
python3 -O -m unittest discover -s tests -p test_epistemic_shift_exam.py -q
python3 -m unittest discover -s tests -q
```

Hostile checks include split disjointness, no-change and rule-change twins,
hand-derived stateful output signatures independent of both implementations,
malformed action accounting, false-positive change detection, stale-belief
controls, future counterexamples, evidence-backed belief revision, injected
transition/replay faults, and local testing of proposed new rules.

### Independent hostile reattack of the first implementation

Destroyer and Skeptic independently broke the initial examiner at source
commit `644244f67aeccc32519e28d5c0fa47cd3eccd37d` despite its green tests:

1. A crafted nonbinary output with hostile `__eq__` concealed a replay
   disagreement. One witness escalated this to a false `VERIFIED_WIN` for a
   purported causal revision in a sealed machine whose law never changed.
2. Missing, non-text and whitespace-only hypothesis kernels were incorrectly
   returned as `NOVEL_RULESET_CHALLENGE`, sometimes with an unserializable
   raw object in the receipt.
3. Mechanically identical sealed copies received different semantic
   fingerprints, incorrectly inflating family-diversity accounting.
4. In the independently called replay API, a list subclass lied about its
   length and bypassed the ten-intervention ceiling. A tuple subclass could
   also lie about iteration. The ordinary CandidateSession always uses an
   exact list, but the verifier's public contract still required hardening.
5. Two distinct causal worlds (stateful EDGE and LAG) produced identical
   early contradictory observations. A blind guess of EDGE nevertheless won
   `revision_verified` on the EDGE world. Correct future-law identification
   and evidence that the observed history actually distinguishes candidate
   rules are separate claims. A revised hypothesis now needs a unique fit
   among the seven declared post-laws to receive revision credit; otherwise
   it receives `UNKNOWN` and `observationally_identified=False`. Direct
   `HYPOTHESIS` oracle correctness remains a separate narrow result.

The revised implementation checks exact binary observation types, typed
state values, and snapshot-bound replay budgets; malformed causal-law labels
return `UNKNOWN`, while genuine novel textual proposals remain reviewable.
These are regression-tested, not grounds to claim tamper-proof independent
execution or neural learning. Even unique identification is relative to the
finite declared rule catalogue, not proof against arbitrary unknown laws.
The hostile findings came from separate
agent-owned test surfaces, then were reproduced in the unified checkout.

### Fresh executed source-only results, 2026-11-07

- Focused new suite: **32/32 PASS**; Python `-O`: **32/32 PASS**.
- Labyrinth main plus new suite: **268 tests OK, 42 optional skipped**.
- Seeds 0–199 for each precommitted semantic split: the deterministic
  catalogue-aware baseline achieved **200/200 verified causal revisions per
  split**. The frozen pre-change law failed **200/200 per split**. A frozen
  training-XOR law passed **100/200 train**, but failed **200/200 dev** and
  **200/200 test**. These are diagnostic control results, not empirical
  evidence of any neural model's structural transfer.

## Qualification limitations

1. Source/test behavior does not show any actual neural weight updates,
   checkpoint reload, unseen-model performance or structural transfer.
2. Only two postlaws per split exist; public code can memorize all six.
   Repeated seeds and paraphrases are not independent conceptual skills.
3. The candidate and examiner share a Python process. Hidden `_case` fields,
   provisional belief history and verifier code are inspectable; isolate
   execution and authenticate receipts before treating this as a real exam.
4. The verifier exhausts the **declared finite** three-step future horizon,
   not all possible natural-language meanings, new sensors, new interventions
   or mechanisms that no designer anticipated. New rules deserve independent
   testing rather than rejection based on the current answer key.
