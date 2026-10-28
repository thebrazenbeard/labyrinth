# Trainee training lifecycle

`labyrinth.training_lifecycle` provides only routing for a trainee:

```text
CHALLENGES --objective public feedback--> TARGETED_RETRAINING
CHALLENGES --no identified target--------> EXTERNAL_REEXAM
TARGETED_RETRAINING --all targets done---> EXTERNAL_REEXAM
EXTERNAL_REEXAM --signed failed exam-----> TARGETED_RETRAINING
EXTERNAL_REEXAM --attempts exhausted----> REEXAM_EXHAUSTED (blocked)
EXTERNAL_REEXAM --custody says GRADUATED-> CUSTODY_AUTHORIZED
```

Public bounded checkers and any neural/advisory score can only select a
training target. They never authorize exit, including at a perfect score.
Malformed, absent, reused, or unsigned exam evidence leaves the lifecycle at
`EXTERNAL_REEXAM` and is blocked by the existing gate/custody path.

The sole code path that can reach `CUSTODY_AUTHORIZED` calls
`graduation_state.decide_campaign`, which invokes the signed graduation gate
inside its policy-bound, HMAC-protected atomic custody transaction. The
lifecycle accepts receipt bytes and a detached signature, not a boolean or a
`GateDecision` supplied by a caller. It creates no signing or evaluator key,
and it performs no self-signed examination.

`CUSTODY_AUTHORIZED` is a routing status, not a portable release token. A real
exit operator must retain the external custody database and independent key
governance. Unit tests use synthetic keys and synthetic receipts to test the
cryptographic protocol. That demonstrates gate mechanics only; it is not proof
that the signer is an independent outside examiner or that its questions were
private. The production qualification claim additionally requires the external
examiner and key custody described by `graduation_gate.py` and
`private_examiner.py`; the bundled private examiner is synthetic-only and is
therefore intentionally rejected by the graduation gate.
