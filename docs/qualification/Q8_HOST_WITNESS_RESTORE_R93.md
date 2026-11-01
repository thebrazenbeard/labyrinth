# R93 Q8 host/witness crash-recovery source contract

**Classification:** `SOURCE_ONLY_HOST_CAS_CONTRACT_NOT_ATTESTED`.

`labyrinth/q8_witness_restore_r93.py` defines the caller-facing shape for a
host-owned one-shot compare-and-swap of a typed *plastic-model-component*
state transition, with a separately signed high-watermark from a witness.
This module contains **no local authority service, signing key, signer factory,
private evaluation bank, host lease, deployment or model-graduation effect**.

## Typed effect boundary

The `Q8PlasticChange` includes the exact run, event, target component, prior
and successor SHA-256, evidence hash, host-issued authority-lease hash,
`state_family=plastic_model`, `change_class=model_component_update`, and
`reversible=True`. Identity, memory and configuration mutations are outside
this contract and fail closed. A caller-supplied lease hash is **not** proof
that a host-issued lease exists or was valid; only the real operator can
validate it against their protected current lease/roster.

The witness record binds host/run, epoch, fencing token, commit index, head,
journal generation/hash, event and change digest, preceding witness digest,
and witness counter. Service and witness sign distinct domain-separated
messages over those same canonical fields. Both public keys, the provider and
host IDs, and the prior high-watermark digest/counter must be pinned **outside
the learner/trainer's rollback domain**. The client snapshots those objects
before verifying, freezes the original change digest, and passes **separate
transport copies** of both the intent and witnessed predecessor to the
adversarial port. Replacing port arguments with `object.__setattr__` cannot
relax the retained signed postcommit comparisons.

## Execute-one-CAS observations

1. Read witness and service heads and verify exact public-key, provider/run,
   predecessor high-watermark and signature matches before calling the host.
2. Call `compare_and_swap_once(change, expected_witness_record)` exactly once.
   The external operator must independently enforce privilege, state-family
   policy, event uniqueness, checkpoint identity, journal durability, and
   atomic compare-and-swap across concurrently contending clients.
3. Verify two signatures and exact epoch, fence, +1 commit index, +1 witness
   counter, +1 journal generation, distinct journal/head hashes and predecessor
   witness digest, then independently read the witness and service heads back.
4. Any exception **after** invoking CAS yields `Q8OutcomeUnknown`: the event
   may have committed, so an agent must reconcile through the actual host. No
   blind retry or learner-side replay is authorized.

## Reproduced local counterexamples

The 16 tests in `tests/test_q8_witness_restore_r93.py` use test-generated
Ed25519 keys and a **toy SQLite database in the test workspace**. They prove
the *client source behavior* below, not host isolation:

- A real SQLite commit survives closing and reopening the test process object.
- A simulated crash after disk commit and before ACK returns outcome UNKNOWN.
- A previously pinned high-watermark rejects a clone whose SQLite file is
  rolled back to an earlier generation.
- Reusing a consumed event after restart does not advance the toy ledger.
- A one-key fake dual-role pin, foreign witness signer, signed-but-stale
  journal generation and post-signature mutation are rejected or UNKNOWN.
- A malicious port cannot mutate the passed state-change intent to substitute
  a different signed successor digest. Nor can it reduce the *passed* prior
  counter, index and generation to present a signed **zero-advance** record as
  a one-step CAS. Both attacks were reproduced RED, then rejected GREEN.
- Wrong state families, nonreversible updates, and no-op successor hashes are
  rejected before the CAS invocation.

> **Critical hostile limitation:** If the same operator can roll back both the
> service database and the purportedly external witness/pin, this source
> verifier cannot detect the rollback. A test co-owned SQLite backend is not
> independently witnessed and must NEVER be deployed as Q8 authority. Keys,
> lease state, high-watermark, service principal, replica and physical storage
> control must be independently inspected, isolated and exercised after real
> power/process failure before any Q8 authorization is considered.

Neither `Q8SourceCASObservation` nor a test-passing source code path carries
permission, consent, identity, graduation, merge or deployment authority.

## Host release gate still absent

Require separately verified operator and witness identities, current scoped
lease, authenticated host route, protected persistent append-only/CAS store,
nonrollback independent witness, race and true crash-recovery traces, exact
source/commit pins, and independent audit of Q8-A1 through Q8-A10. Q7
independent sealed evaluation and Q5/Q6 learning evidence remain separate
requirements. **Q8 status remains HOLD.**
