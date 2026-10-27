# Minotaur V2 scratch architecture proposal (2026-10-27)

Status: implementation and software tests only. No behavioral improvement is
claimed, and there is no V2 trained checkpoint or blind evaluation evidence.

## V1 audit and smallest change

V1 tokenizes observed `A,B` values after each instruction, but has one
seven-class head that must jointly decide whether the whole trace is correct
and, if not, locate the earliest error. V2 keeps the same tiny bounded
TransformerEncoder and random initialization, while making the representation
and objectives explicit:

- Every program step receives a distinct ordered step token and an
  `OBSERVED_STATE` token directly before its observed `A,B` register tokens.
- A binary `correctness_head` predicts correct versus incorrect trace.
- An `earliest_error_head` has six location classes and is trained only on
  incorrect examples. It cannot turn a correct-trace label into a location.
- Advice carries a confidence in `[0, 1]`; below the configurable 0.50--0.99
  threshold it is `ABSTAIN`. All advice states `graduation_authority: none`
  and `qualification: NOT_ESTABLISHED`.

This is an independent experimental module,
`labyrinth.minotaur_scratch_v2`. It imports PyTorch lazily, creates only
random parameters from a caller-controlled seed, has no pretrained weights,
does not load or modify V1 artifacts, and has no signer or release capability.
The independent guardian remains unchanged and retains its own custody.

## Verification scope

`tests/test_minotaur_scratch_v2.py` covers the explicit marker/order encoding,
padding-mask invariance, output shapes, bounded abstention, and one actual
AdamW forward/backward/update on an eight-case CPU batch when PyTorch is
available. In a no-Torch checkout, the same smoke executes a tiny random
mean-embedding two-head network with analytic cross-entropy gradients, so an
actual parameter update is still exercised without an installation. The smoke
has a strict 30-second ceiling and checks a complete parameter digest changes.
Those checks establish only interface and update mechanics, not accuracy,
calibration, false-approval behavior, or any generalization claim. Any future
comparison must use frozen, blind data and report false approvals/rejections
and abstention coverage before claiming an improvement.
