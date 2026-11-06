# R68: Labyrinth/Rezon two-model source partition (non-authorizing)

Status: **SOURCE_ONLY_TWO_ADAPTERS_UNATTESTED**. This module is a
Labyrinth-owned, pure source-evidence comparison, not a live Rezon adapter,
distributed inference consumer, model identity service, host authorization
service, checkpoint verifier, or graduation oracle. No host, network, model,
cross-repository or workflow operation occurs.

The implementation in `labyrinth/rezon_two_adapter_source_partition_r68.py`
extends the R57 source-only comparison using **two distinct model-adapter
source claims**. R57's accepted Labyrinth pointer, Rezon claim, and *separate
expected source pins* remain prerequisites. The R68 claim for each adapter
contains only four bounded source fields:

| Field | Invariant | Explicitly not established |
| --- | --- | --- |
| `adapter_id` | Lower-case, bounded source key; distinct A and B | Authenticated actor/principal or permission subject |
| `model_source_head` | Exactly 40 lower-case Git hex digits | Host executable, image signature or model runtime |
| `model_artifact_sha256` | Exactly 64 lower-case hex digits, **different between A and B** | Verified model bytes or checkpoint identity |
| `adapter_contract_sha256` | Exactly 64 lower-case hex digits | Verified consent, capability binding or adapter authenticity |

Each adapter also takes an `ExpectedAdapterSourceV1` pinned separately from
the offered `AdapterSourceClaimV1`. The comparison rejects mismatched fields,
swapped adapters, an identical adapter key, or two claims of the same model
artifact digest. Independently pinned, genuinely distinct model sources can
share *one provenance observation*, not each other's identity or authority.
This permits a source comparison even if adapters share a code contract or a
repository head; the **model artifact digest must differ**.

## Contract and owners

| Owner/source | What R68 checks | What the owner must still establish |
| --- | --- | --- |
| Labyrinth source producer | R57 verifies exact Labyrinth pointer against independently supplied expected head/tree and all bounded digest pins | Provenance of expectations, source integrity and host execution |
| Rezon source producer | R57 checks separately expected Rezon claims and comparable cross-source references | Real receipt signatures, signer identity, host lease, replay prevention and runtime coordination |
| Model A adapter owner | Four exact fields match A's independently supplied expected pins | A's own model identity, checkpoint bytes, consent, grants and host authorization |
| Model B adapter owner | Four exact fields match B's independently supplied expected pins; A and B keys/models differ | B's **separately** established identity, checkpoint bytes, consent, grants and host authorization |
| Graduation/deployment authority | No authority inputs or output tokens accepted | Independent per-model gate decision and all physical qualification evidence |

An application cannot make source expectations independent simply by copying
the offered claims into `Expected*` objects. R68 **cannot prove** that an
actual owner was consulted or enforce durable host-level registry bindings.
Treat such claims as untrusted until independently confirmed outside this
module. The test fixtures derive synthetic expected objects from synthetic
claims solely to exercise comparisons; they are **not owner attestation**.

Both disputed field relationships remain **UNRESOLVED**, with neither
equality nor semantic conversion performed:

| Labyrinth field | Rezon field | Result |
| --- | --- | --- |
| `train_exposure_sha256` | `maze_exposure_sha256` | Unresolved exposure-object meaning/coverage |
| `checkpoint_manifest_sha256` | `checkpoint_sha256` | Unresolved manifest versus checkpoint-byte meaning |

## Returned observation and negative controls

`compare_two_model_adapter_sources(...)` returns the immutable, tuple-backed
`AdapterPairSourceObservationV1`. Its sole constructor field is
`source_pair_sha256`: a deterministic, **untrusted** fingerprint binding the
accepted pointer bytes, Rezon claims, both distinct adapter source claims and
source run labels. It is not a token, signature, proof of identity or grant.
There is no model ID, principal, consent, permission, lease, checkpoint
identity, trained model, or deployment command in the return value. Runtime
must never authorize based on this hash.

Under the unmodified class definition, its qualification state always reads
`SOURCE_ONLY_TWO_ADAPTERS_UNATTESTED`; model/checkpoint identity, principal,
consent, permissions, cross-model grants/identity transfer, training,
graduation, merge and deployment flags always read **false**. These are
read-only properties, not constructor fields, and `object.__setattr__` cannot
flip them on an instance. **This is an application-level type guard, not a
security boundary against hostile code running in the same Python process.**
An attacker who can monkeypatch the class, imports or functions can replace
property behavior; only the host-owned authorization service may make real
authority decisions.

## Post-validation mutation and snapshot discipline

`frozen=True` on a dataclass does not prevent `object.__setattr__`. The R68
implementation originally checked `observed_alpha` against independently
expected pins, then re-read the mutable frozen object's fields during `asdict`
to compute its source digest. A deterministic `sys.settrace` hook could mutate
the offered `model_artifact_sha256` between checks, producing an apparently
successful source fingerprint for an unmatched model artifact. The same gap
existed for Rezon claims and adapter expectations.

The R69 hardening captures **deep, immutable primitive snapshots** of each
source claim and expected-pin record, validates model pairs only from sealed
snapshots, and hashes only sealed source values. A second read immediately
before return rejects any input that differs from its sealed snapshot,
including mutations after the R57 source check. Rezon control-arm tuples are
copied into immutable primitive tuples, and other mutable source-field types
are rejected. Hostile injection tests reproduce the original RED behavior
for offered adapter, Rezon and expected-adapter objects and are GREEN under
this implementation.

This second-read fence cannot exclude a mutation that occurs *after* the final
check, an object deliberately changed then restored, or arbitrary function/
class monkeypatching by hostile same-process code. The result remains an
**untrusted source observation** and never grants authorization.

The following operations **must fail** at this boundary:

1. Replaying B's observed adapter claim under A's independently expected pins,
   reversing both claims, or changing a model/contract SHA without the
   corresponding independently governed expected pin.
2. Presenting identical model artifact digests or identical adapter keys as
   two distinct consumers, even if both offered/expected objects agree.
3. Passing malformed Git hashes, hashes of the wrong length/case/type, unsafe
   adapter IDs, or unsupported extra identity/consent/permission fields.
4. Trying to mutate/construct a positive authority/graduation/identity flag.
5. Presenting a Labyrinth pointer or Rezon claim that fails existing R57
   independently expected source checks.

No security principal, consent, permissions, model identity, graduation
right, checkpoint identity or host lease **ever transfers from A to B** merely
because they share a Labyrinth/Rezon source-pointer observation. A real
runtime consumer requires per-model host-owned bindings and a separately
verified authorization service; implementing or granting that is outside R68.

## Reproduction and boundary

The RED test ran before the new module was added and failed with
`ModuleNotFoundError`. After implementation:

```bash
python3 -m unittest tests.test_rezon_two_adapter_source_partition_r68 -v
python3 -m unittest tests.test_rezon_pointer_mapping_r57 tests.test_evidence_pointer -v
```

R69 repeated RED→GREEN with three deterministic post-validation mutation
tests; without snapshot fencing, all three failed because the comparator
accepted mutated source inputs. With snapshot fencing, all three reject.

Passing source tests provide no evidence of host training, checkpoint replay,
five-arm training success, model quality, real Rezon execution, physical
qualification, AGI graduation or safe merge/deployment. The existing Labyrinth
and Rezon owner gates remain HOLD until independently satisfied.
