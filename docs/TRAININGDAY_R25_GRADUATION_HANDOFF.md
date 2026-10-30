# R25: Labyrinth to Minotaur evidence handoff (source-only)

`labyrinth.trainingday_graduation_bridge.assess_graduation_handoff` provides a
small model-agnostic integration boundary over the existing
`labyrinth.graduation_gate.decide_graduation` policy. It is **not** another
grader, training runner, deployment control or private-bank custodian.

The bridge binds four independently specified roles:

1. **Declared intent:** objective ID/kind and immutable intent content digest.
2. **Admitted curriculum:** training-only manifest and corpus digests, the
   frozen intent/source/tokenizer/initial-adapter commitments, declared rights
   clearance, no train/evaluation overlap, separately expected examiner source
   and conditions digests.
3. **Adapter training receipt:** matching manifest, base and trained adapter
   digests, positive optimizer steps and finite elapsed time within the budget.
   A 1 ms/step declared-timing sanity floor rejects zero or subnormal total
   elapsed claims, but does not independently witness any optimizer execution.
4. **Signed independent examination:** existing Minotaur gate checks its own
   complete private-exam receipt schema, HMAC, four arms, matched controls,
   fresh process and holdout/transfer evidence, and custodian-owned attempt
   state. A trainer's unsigned PASS is not an examination.

Absent an adapter receipt, invalidated historical R18 time budget, unlicensed
or eval-overlapping curriculum, forged/unsigned pass, wrong adapter or source
identity, the bridge stays in the Labyrinth. A complete *declared* candidate
without independent examination returns `EXAM_REQUIRED` and requests the
Minotaur examination without claiming graduation. Even if an independently
signed **synthetic** receipt causes the existing gate to return `GRADUATED`,
the bridge returns `SIGNED_GATE_REPORTED_GRADUATED` and
`REQUIRES_INDEPENDENT_CUSTODY_REVIEW`; `promotion_allowed` remains false.
If a custodian-provided state was already graduated, Minotaur may return that
prior result without authenticating a new supplied receipt. The bridge labels
this separately as `PRIOR_TRUSTED_GATE_GRADUATION_REPORTED`, never as a fresh
exam or new neural evidence.

**Trust boundary:** Digests are only content identifiers, not evidence that
the model/adapter bytes were opened or that a run occurred. Caller flags for
rights, no leakage and budget status are not independently authenticated by
this module. The HMAC key can be passed by a caller; this source cannot prove
the caller is a distinct OS principal or genuinely unseen evaluator. A real
workflow needs independent admission, run and private-exam custody, a durable
Minotaur `GateState`, OS-enforced principal isolation, original artifact hash
readback and host leases. Actual Qwen3.5-4B skill mastery, learning effect,
release permission and quantized deployment remain **UNKNOWN / HOLD**.

Run synthetic CPU tests, without fetching models or private cases:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest -q \
  test_trainingday_graduation_bridge test_graduation_gate
```

No user credentials, private evaluation examples or model outputs belong in
this handoff report or in a public coordination-bus issue.
