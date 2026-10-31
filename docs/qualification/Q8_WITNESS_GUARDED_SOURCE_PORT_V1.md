# Labyrinth / Minotaur Q8: independently signed witness guard (SOURCE ONLY)

**Scope:** A source-level adapter for an externally controlled monotonic
witness, not an installed host authorization service and not a graduation
license. The existing SQLite authority remains a *proposal journal*; the
witness must decide independently whether its exact signed proposal can be
accepted once. The two keys must be distinct.

## Flow

1. An operator outside the learner's trust domain pins the exact run, event,
   exam, model artifact, scope, prior checkpoint, cycle, epoch, fence and head.
2. The client reads the witness head, not the trainer's cloned SQLite head.
   It refuses stale one-shot claims before any local operation.
3. The SQLite authority makes a signed *local proposal*. No permission is
   released yet.
4. An independently hosted witness atomically checks the same pre-admitted
   intent and CAS head, consumes the event/exam/nonce once, and signs a
   `WitnessCommit` binding the local receipt's exact bytes and new head.
5. The adapter verifies both Ed25519 signatures, distinct pinned public keys,
   exact request/receipt binding, witness event/exam/nonce, and witness head
   readback. It returns a receipt only after this validation.
6. Once the local proposal may have committed, any witness error, missing or
   malformed ACK, or divergent readback is `OUTCOME_UNKNOWN`; **no automatic
   retry or training is permitted**. Operator reconciliation is required.

`WitnessGuardedReservationPort` is an `ExternalAuthorityPort` suitable for
the existing `ExternalMonotonicReservationClient`. It exposes no operator
admission, signer or final-exam authority to the learner.

## Falsification tests (source-only)

Run:

```
python -m unittest discover -s tests -p 'test_witness_guarded_reservation_source.py' -q
```

The test-only `TestOnlySharedWitness` deliberately uses ordinary in-process
memory and an exportable signing key. It demonstrates:

- Two cloned local SQLite ledgers and one **shared** witness yield only one
  accepted external reservation; the second client's stale preflight fails.
- A forged witness signature or acknowledgement for a different receipt is
  treated as uncertain after local commit, never as a safe retry.
- Reusing the same local/witness signing identity is refused.
- Cloning *both* the witness state and its signing key produces two accepted
  reservations of the same event. This is a counterexample to any claim of
  host-owned anti-rollback from software tests alone.

## Physical qualification obligations (all unproven here)

1. The witness process/OS identity, monotonic state, private signing key and
   operator intent admission must be controlled independently of the trainer
   and of the local SQLite proposal journal. An in-process object or a copied
   file is not separation.
2. The witness must durably reject replay even after a trainer snapshot restore
   **and after its own service restart**, subject to a rollback-resistant
   anchor verified separately. A cloneable SQLite DB plus a cloneable Ed25519
   key fails this obligation.
3. The transport must authenticate the exact pinned witness principal and
   endpoint; a source-level public-key parameter does not prove who selected
   it. Credential/key rotation, failover and recovery require operator-owned
   control and independent observation.
4. Verify atomic admission and CAS of event/exam/nonce/candidate identity,
   including concurrent clients, lost ACK, crash between local and witness
   commits, and protected post-crash reconciliation without blind replay.
5. Perform the test on an actually authorized host with a released training
   lease, verified process identity, bounded resource use and preserved logs.
   No Lappy/VN1/VN2 process, token or tunnel is touched by this artifact.

## Qualification

- Source adapter and synthetic counterexamples: **testable here**.
- Independently hosted, nonrollbackable witness principal: **NOT PROVEN**.
- Current cloneable witness anti-rollback design: **FAIL**.
- Physical Q8 host probe: **NOT RUN**.
- Neural weight updates, hidden final-exam custody and graduation: **NOT RUN**.
- Merge/deployment/graduation: **HOLD**.

This source cannot grant a protected witness existence claim, however
plausible the signatures or passing tests appear.
