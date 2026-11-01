# Q6 per-exposure CAS — source-only reference, not host authority

This reference describes the transaction ordering needed after a Q6 campaign
has been pinned and reserved by the separately verified interface in
`labyrinth/q6_governed_campaign_contract.py`.

The campaign manifest declares every source-generated TRAIN and EVAL case, the
three experimental arms, expected phases and each permitted visit. It does not
include private answers. `Q6SlotCasReference` derives an opaque, deterministic
slot ID from the signed campaign commitment, reserved campaign ID, case raw and
arithmetic-semantic commitments, arm, phase, and visit index.

## Required slot lifecycle

1. `RESERVED`: campaign reservation acknowledged and read back. A slot has no
   exposure yet. Merely reserving a campaign does **not** release a prompt.
2. `REVEALED_OR_CONSUMED`: `begin_exposure(slot, expected_head, version=0)`
   atomically advances the reference state and global head **before** its
   caller obtains a reveal receipt. `begin_then_call` invokes the supplied
   callback only after that transition. A model crash, callback exception or
   lost ACK leaves the case consumed; a second BEGIN fails.
3. `COMPLETED`: `complete_exposure` binds one result, execution commitment,
   checkpoint commitment, and completion nonce to an already revealed slot.
   Identical idempotent completion returns the existing receipt without a
   second score; changed outcomes fail.

The reference keeps independent version and global head CAS predicates.
Concurrent attempts to begin the same or different slots using a stale global
head cannot both commit. Read-only `reconcile` provides the last reference
state after an uncertain acknowledgement. A missing or stale external high-
water witness is not proof of freshness.

Repeating observations of a fixed public EVAL panel within one approved Q6
campaign is expressed by *different preregistered arm/phase slots*. Repeating
TRAIN on an approved case is expressed by distinct planned visit indices. A
new campaign's freshness must be checked against an **independently retained**
case register; a local caller's omitted history is not authoritative.

## What these tests prove

- Exact typed source schedule, raw and arithmetic-semantic slot commitments;
- no scoring callback before the BEGIN transition;
- no caller-owned slot substitution after BEGIN but before callback dispatch;
- no second reveal after loss of ACK or model callback failure;
- result and checkpoint pinning, idempotent completion and conflict denial;
- stale head/epoch/fence and competing worker rejection;
- in-process receipt mutation does not alter primitive ledger records;
- gold-free slot API and false-only authority/graduation fields.

## What they explicitly do **not** prove

The state in this module is ordinary **process-local Python memory**. A
snapshot, admin or model process with the ability to replace the object or
restore the entire process state can roll back its history. An unkeyed hash is
not an authenticated witness. The synthetic signing port used in tests is
not an independent host or hardware attestation. Python callbacks can access
their own enclosing process and bypass this wrapper entirely.

Physical Q6 qualification still requires an **owner-operated transaction
service** which combines slot state, globally complete semantic exposure
history, per-slot reveal/consume CAS, result completion CAS, signing-key
custody, and an externally anchored monotonic high-water mark outside the
trainer, model, filesystem, VM snapshot and credential domains. It must
prevent private question bytes being delivered before successful readback,
and prove fail-closed restart/reconciliation on real hosts and transport.

This reference never returns a graduation/learning flag, never opens a sealed
exam, and cannot certify physical Qwen weight training or release a checkpoint.
The integrated pinned neural CPU workflow now requires all 45 source tests
without skips. Passing that gate proves only this in-memory reference's tested
source behavior; it does **not** provide rollback-proof host custody, independent
evaluation authority, model graduation, or a physical training result.

## Scope separation from the cognitive perspective TRAIN runtime

`labyrinth/cognitive_objective_training_run.py` owns a separate source-local
cognitive perspective TRAIN route, including its own `_opened_*` bookkeeping.
That runtime is **not** backed by `Q6SlotCasReference` and does not inherit its
pre-reveal CAS guarantees. Q6 per-slot reference state covers only the typed
arithmetic/program Q6 campaign manifest. Neither route has an independent
host-owned exposure authority, rollback-resistant cross-campaign witness, or
graduation permission. A future cross-lane service requires separate design,
integration, and independent hostile tests before any authority claim.
