# TrainingDay R26 — synthetic model/Minotaur rollback rehearsal

Status: **SOURCE/CPU REHEARSAL ONLY — NO RELEASE OR GRADUATION AUTHORITY**.

This module exercises a small immutable state machine for a hypothetical
quantized model/adapter cutover coupled to a Minotaur graduation state and
its evidence index. It is deliberately isolated from deployed services.

## State and transition

- `ModelMinotaurBundle` binds five values in one canonical SHA256 commitment:
  model bytes **claimed** SHA, adapter bytes **claimed** SHA, graduation-state
  **claimed** SHA, evidence-index **claimed** SHA and monotonically increasing
  generation.
- `CutoverCandidate` precommits to the exact previous bundle, proposes an
  entire new bundle, and includes a self-declared R25 graduation-handoff
  digest. All values are schema-checked, but **none are independently signed**.
- `rehearse_cutover` stages all three logical surfaces and swaps only one
  frozen bundle. The old bundle remains available as an immutable rollback
  target, including the previous adapter and graduation/evidence hashes.
- A deterministic failure can be injected into every phase: validation,
  model staging, gate staging, evidence staging, precommit verification,
  swap, post-swap verification or journal finalization.
- A failed phase rolls back all components. Explicit synthetic **forward**
  recovery is permitted only if the atomic swap already occurred. A failure
  before swap never activates an uncommitted proposal. The return receipt
  contains an exact canonical SHA256 over the input commitments, executed
  phase trace, chosen recovery policy and outcome.
- Stale previous commitment, replayed transaction ID, generation skip,
  missing independent evidence changes, malformed hashes and unexpected
  policies fail closed. Exact dataclass input types are required: untrusted
  subclasses overriding digest field access are rejected before staging, and
  the accepted previous-state commitment is rechecked before journaling.
  Failure phases and recovery policies require actual strings rather than
  objects whose overloaded equality can impersonate an allowed value.

## Local evidence

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s tests -p test_trainingday_atomic_cutover.py -q
```

Tests include all failure-injection points, after-swap forward recovery,
pre-swap forward refusal, hash-binding across all bundle fields, stale
generation/replay and malformed declarations. The test fixtures are public
hexadecimal strings, not actual model or evaluator artifacts. A hostile
frozen-dataclass subclass whose model digest flips after initial validation
was reproduced as an integrity bug in the first candidate and added as a
negative regression before correction. A second hostile regression covers
equality-overloaded objects falsely matching valid failure phases or
recovery policies before journal serialization.

## Trust boundaries / blockers

This is a **single-process in-memory copy-on-write model**, not a real
two-phase commit across a serving process, an evaluator custodian and an
evidence database. It has no durability under OS/process crash, no
cross-process lock, no signed independent observer, no host-enforced lease,
no adapter bytes or fresh-process reload, and no applied quantized release.
Its journal SHA is a content hash, not an authenticated or append-only audit.
The old model's neural weights are not erased or changed by this simulation.

In all outcomes `deployment_allowed=false`, `promotion_allowed=false`,
`host_custody_verified=false` and `physical_rollback_verified=false`.
Genuine release requires separately authenticated artifact custody,
real multi-service atomic rollback validation on isolated test endpoints,
the qualified Minotaur evaluation, host lease and independent review.
Historical R18 MeZO `TIME_BUDGET_OVERRUN_INVALID` remains invalid.
