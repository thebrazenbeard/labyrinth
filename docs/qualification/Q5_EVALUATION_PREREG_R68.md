# Q5 independent-evaluation preregistration — source-only gate (R68)

This is a **source-input consistency gate**, not an independent evaluation or
permission to train. The existing `five_arm_statistical_screen` computes the
SHA-256 of case pseudonyms that arrived **with** each seed's correctness bits.
An operator can change an evaluation case after observing its result, recompute
that same submitted digest, and still obtain `source_screen_pass=True`.

`labyrinth/q5_evaluation_preregistration_r68.py` adds an optional outer gate:

1. **Before any evaluation outcomes are visible**, the evaluation custodian
   chooses all seeds and each development/transfer case roster, including exact
   order, and records the immutable `FrozenEvaluationPrecommit` digest outside
   the run/trainer report directory. The commitment binds the full seed identity,
   source five-arm plan digest, both case rosters, and all eight seed-level
   Holm/sign-test parameters and control names.
2. A separately governed verifier must preserve and authenticate the pre-outcome
   digest and witness. At evaluation time, the caller supplies the expected
   digest from that source (not from the run or outcome inventory), the witness,
   and a verifier callback. There is **no built-in identity, trusted clock,
   signature authority, or replay protection** in this Python module.
3. `screen_predeclared_evaluation` verifies that the external digest matches the
   frozen precommit, asks the caller's verifier to check the witness, then checks
   the **complete, ordered** seed/split/case roster against submitted outcomes
   before invoking the existing source-only statistical screen.

The verifier callback is caller-supplied Python code. A `frozen=True` dataclass
alone is not immutable against `object.__setattr__`; this gate therefore seals
separate, nested roster objects before invoking the callback, verifies that
the original inputs were not mutated by it, and performs all remaining source
comparisons against the sealed copy. Because same-interpreter code can inspect
caller frames and mutate even that snapshot, the gate **revalidates the sealed
canonical digest and its external pin immediately after the callback and again
before constructing the result**. Tests reproduce attacks against the outer
tuple, nested roster, and caller-frame snapshot. Even so, **untrusted Python
running in the same interpreter cannot be a security boundary or trusted
witness**: the actual external pre-outcome authority must live in independently
governed host infrastructure, outside this callback/process. No result from
this module can establish such custody.

The precommit can be manufactured retroactively by anyone able to choose the
external digest and witness verifier; passing this check proves **only source
consistency given those supplied inputs**. Separately required for a genuine
independent pretrained-model claim are immutable host-owned pre-outcome custody,
independent evaluator and holdout authority, observed pretrained model weights
and inference, semantic leakage review, measured retention, and explicit
governed graduation decisions. None are implemented here. Every returned
authority/verification flag remains `False`, including when the statistical
source screen reports a pass.

Run the bounded regression suite:

```bash
python3 -m unittest tests.test_q5_evaluation_preregistration_r68 -q
```

Negative coverage includes post-hoc case substitution with a recomputed
self-digest that the legacy screen accepts, an externally pinned digest mismatch,
missing/malicious witness callbacks (including callback-induced TOCTOU mutation
of outer/nested frozen fields and frame-inspection mutation of both sealed and
original values), case/seed omission and reordering,
duplicate/undersized roster rejection, and statistical policy mutation.

This change launches **no model**, touches **no host, GPU, private bank, NAS, or
secret**, writes no evidence outside ordinary source-controlled fixtures, and
does not alter `.github` CI, deployment, learning, retention, or graduation.
