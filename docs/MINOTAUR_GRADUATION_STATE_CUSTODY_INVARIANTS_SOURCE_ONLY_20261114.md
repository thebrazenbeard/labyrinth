# Minotaur Wave61 — persisted graduation-state consistency (2026-11-14)

Base: Labyrinth Draft PR #242 head `f8355d657076e584a89c63e97411e0350e15667c`.
Branch: `fix/minotaur-graduated-state-consistency-20261114`.

## Hostile red-first falsifier (Lappy, Windows Python 3.14.5)

A signed receipt is necessary to advance qualification, but a persisted
`GateState` with a matching policy fingerprint could be manually constructed
with `graduated=True` and no consumed run/suite/case/prompt/family history.
`decide_graduation(..., receipt_bytes=None, signature_hex=None)` returned
`GRADUATED` in **six** inconsistent-state cases; hostile test exit 1,
six assertion failures. This bypasses source-level state bookkeeping *if* an
actor can replace supposedly trusted durable state. This is NOT proof an
external actor controls the independent state custodian.

A second attack used a legitimate synthetic first signed attempt that failed
the objective, then removed `consumed_eval_case_ids` from the resulting
persisted `STAY` state. Submitting an otherwise valid synthetic signed second
attempt returned `GRADUATED` despite the corrupted history; separate test
exit 1. This breaks the intended consumed-case ledger if trusted persistence
is faulty or tampered with.

## Minimal correction

Before interpreting an already-graduated state or evaluating a subsequent
signed attempt, require necessary internally consistent evidence:

- Completed attempts = `next_attempt - 1`, not exceeding the frozen
  `max_attempts`.
- Distinct consumed run IDs and suite hashes each total one per attempt.
- Consumed evaluation case IDs and prompt commitments have equal counts,
  at least the per-slice frozen minimum times attempts.
- Distinct consumed semantic families total at least their frozen per-slice
  minimum times attempts, and never exceed case count.
- A state with no completed attempt cannot claim graduation or contain
  consumed evaluation cases/families.
- Inconsistencies return `BLOCKED`, without altering trusted state or
  exposing per-case scores.

The existing successful signed decision and subsequent persisted-state
readback remain legal. The canonical v2 receipt schema and policy fingerprint
are unchanged; no new trust in signer-provided metadata is introduced.

## Evidence and limits

- TDD RED: 6 forged-state cases incorrectly `GRADUATED`; second corrupted
  `STAY` state plus synthetic signed attempt incorrectly `GRADUATED`.
- GREEN: 2 new hostile tests pass, and all 26 graduation-gate tests pass
  on Windows 3.14.5 (normal mode).
- WSL / full CI outcomes: append only after fresh verification.

**Not qualified:** independent durable-state custody, honest signer
classification, semantic mechanism independence, real model/checkpoint
execution, native non-WSL Linux private-file isolation, or neural learning.
A party able to forge an internally consistent state or false signed receipt
is not stopped by these structural checks. Open Labyrinth issues #146 and
#147 remain unresolved. This is source-only, DRAFT/HOLD, not model graduation
or deployment. No secrets or actual private examiner material are read.
