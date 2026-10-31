# Labyrinth Q8 — independently held witness, source falsifiers

**Result: SOURCE_TEST PASS; cloneable-authority design Q8 FAIL; physical Q8
HOST_PROBE NOT RUN; release HOLD.** These tests are not a host-owned
authority, a hardware monotonic counter, or a deployable transport.

The existing source SQLite CAS + Ed25519 identity is insufficient if both its
database and signer can be cloned. `tests/test_q8_witness_boundary_source.py`
reproduces that double-authorization, then verifies that two independent
clients sharing *one* witness state accept one request and reject its replay.
It also demonstrates that rolling back that supposedly common witness defeats
the property again. An imposter signer with the correct provider name cannot
pass the pinned-key check. A trainer-invented candidate fails intent pinning.

The minimal physical authority boundary has these independent requirements:

1. The trainer gets only an authenticated reservation endpoint and immutable,
   operator-pinned provider/public-key/exam/candidate bindings. The signer,
   owner intent admission, authority database, and backups remain under a
   distinct administrator/OS principal and cannot be accessed by training
   jobs. Local `SQLiteReservationPort` must remain a SOURCE fixture.
2. The *single* witness atomically consumes `(run, event, exam, nonce)` with a
   monotonic epoch/fence/index/head across all clients. Its history must not be
   recoverable to an older accepted position, including snapshots/backups,
   failover nodes, stale leaders, and copied signing credentials.
3. Host operators choose an actual rollback-resistant anchor (for example,
   protected append-only external service or secure monotonic hardware) and
   independently prove its identity/custody/restore semantics. No such device
   or service is verified by this source contract; no brand or primitive is
   presumed present on Lappy, VN1, or VN2.
4. If witness or receipt acknowledgement is lost after a possible commit,
   the caller must return UNKNOWN and reconcile read-only. No retry of the
   mutable action, no fallback to a cached local SQLite head, and no
   graduation when witness health or provenance is uncertain.
5. Independently fault-inject concurrency, replay, clone-and-restore, signing
   identity duplication, stale-leader failover, host compromise, clock reset,
   lost ACK, and recovery before marking physical Q8 PASS.

The source tests use multiple in-process SQLite stores and a key in test RAM.
They prove neither role isolation nor a non-cloneable witness. A second local
database with the same signer is *not* an external authority.

```bash
python -m unittest discover -s tests -p test_q8_witness_boundary_source.py -v
```

> **Hostile release challenge:** If an operator can restore both an authority
> database and its private signing identity to a previously accepted position,
> or can impersonate a witness by redirecting the trainer's endpoint, what
> independent mechanism prevents two different signed `index=1` reservations?
> Until a physical test answers that, Q8 remains FAIL/HOLD and Q9 stays HOLD.
