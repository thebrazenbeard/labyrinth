# Labyrinth/Minotaur external reservation — SQLite source prototype

**Status: SOURCE-ONLY. Not an installed or independently controlled authority.**

This optional repository experiment implements the `ExternalAuthorityPort` storage side of
Labyrinth's `ExternalMonotonicReservationClient`. It is deliberately **not**
wired to Executor, TRAININGDAY hosts, a network endpoint, a private final exam
bank, or the active Labyrinth Draft PR. No host process was started.

Historical source contract pinned for this isolated packaging worktree on
2026-10-31; **not a claim about the current remote Draft PR head**:

- [Labyrinth PR #62](https://github.com/thebrazenbeard/labyrinth/pull/62)
  `e417631f8afbbdc4de0130cce9a13ae965df9bc7` (Draft/open/unmerged).
- `labyrinth/external_monotonic_authority_contract.py`, SHA-256
  `c506fb04939da7524f47b7d4b27da126675f0357a5fa668e4b70b248bb75b6cd`.
- Host coordination [Bus #406](https://github.com/thebrazenbeard/chat-communication-bus/issues/406)
  has no explicit R19-01 host-lease release. No remote experiments allowed.

## Observable source behavior

`SQLiteReservationPort` uses one SQLite DB file with WAL and `synchronous=FULL`:

1. A privileged *operator API* provisions the initial run head and separately
   admits the exact typed training/exam reservation request. A trainer-invented
   request with the same plausible digest shape is not admitted.
2. `reserve_once()` acquires `BEGIN IMMEDIATE`, checks the admitted request,
   exact epoch/fence/index/head, and unique event/exam/nonce before committing.
3. The authority signer signs the typed receipt's exact domain-separated
   Ed25519 payload. The port verifies that signature against an explicitly
   pinned public key **before** committing the new head and audit receipt.
   Invalid but correctly sized signatures roll back without consuming events.
4. The existing Labyrinth client verifies Ed25519, receipt/request binding,
   and independent readback. A lost acknowledgment yields UNKNOWN, not retry.
5. `read_receipt()` supports read-only after-crash reconciliation; its
   returned bytes must still be verified independently against the pinned
   Ed25519 key and expected request before any downstream decision.

There is **no network listener or deployment API** here. Exposing privileged
operator methods through an unauthenticated RPC endpoint would be unsafe.
Direct and ancestor symlinks (including detected Windows junctions) are
rejected before opening a database. This path check is not an OS-level
protection against filesystem handle races or rollback.
The signer is injected into the source prototype and is not protected from
code executing in the same process. Admin and learner roles are not separated
by this source implementation.

## Run tests

From this Labyrinth checkout's repository root (the source module is under
`experiments/`, and is not installed as an active service):

```bash
python -m unittest discover -s tests -p test_labyrinth_sqlite_authority_source.py -v
```

Requirements: Python 3.10+; `cryptography` for real Ed25519 verification; no
Torch/GPU/network/device access. Source-only exact tests: 13 cases; relevant
Labyrinth external authority/Journey tests and the complete repository suite
should be run on the exact integration candidate before publication.

Tests cover real Ed25519 signer/client compatibility, exact host-pinned intent,
one-shot replay denial, duplicate exam/nonce admission refusal, competing
pre-admitted event CAS, concurrent independent SQLite connections, signer
failure rollback, invalid-signature non-consumption, symlink-path rejection,
restart/readback, lost ACK reconciliation, and malicious
`str` subclass identity probes.

## Falsifier — cloned authority still double-authorizes

The test **intentionally copies the DB twice before reservation**, then signs
the same request against each copy using the same cloned key. Both independent
stores accept commit index 1. This is a material **Q8 physical FAIL**.

> A single SQLite database can enforce local atomicity but cannot establish
> globally monotonic authority if its database and signing identity can be
> restored or cloned. Real Q8 needs a host-owned, rollback-resistant primitive
> and independently witnessed provider/key binding, plus validation of custody,
> recovery, and compromise behavior under physical fault injection.

## Release and integration boundary

The next integration seam is a separately installed service with authenticated
transport, immutable role/provider bindings, independent OS principal and
private key, protected operator intent admission, atomic durable ledger, an
external monotonic anti-rollback anchor, host lease, and independent reviewer.
Do not expose `operator_bootstrap_run()` or `operator_stage_intent()` to the
trainer. Do not stage a private key on Lappy merely to make tests pass.

SOURCE_TEST: PASS in temporary SQLite databases. CLONED_DB_AND_SIGNER_Q8: FAIL
(both copied stores authorized the same event). REAL_HOST_PROBE: NOT RUN.
FORMAL_VERIFY: NOT RUN. PHYSICAL_HOST_Q8: NOT RUN. GRADUATION/DEPLOY: HOLD.
