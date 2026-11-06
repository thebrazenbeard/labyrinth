# R76: executable *local toy* consumer of the Labyrinth/Rezon source partition

**Status: LOCAL_TOY_PROBE_SOURCE_ONLY_UNATTESTED.** This first deliberately
bounded consumer of `rezon_two_adapter_source_partition_r68.py` executes a
canonical arithmetic prompt on **two separate in-memory `ToyLinearAdapter`
instances**, each with different parameter content. It is not a Rezon worker,
distributed inference system, collaborative reasoning engine, host lease,
consent/permission verifier, model-identity service, neural qualification, or
training/graduation authority. R76 neither contacts Rezon nor installs code on
another host.

## What actually executes

`probe_two_toy_sources(...)` in
`labyrinth/rezon_toy_pair_consumer_r76.py` performs these bounded steps:

1. Validate the exact `TrainingScopeSpec` and require its current canonical
   `scope.fingerprint` to equal the separately pinned Labyrinth source
   `scope_sha256`. Changing only the declared objective/seed without changing
   the actual pointer **fails closed**.
2. Require **two distinct, exact `ToyLinearAdapter` runtime instances** and
   one canonical, prompt-only arithmetic request. Derive both actual
   `model_hash` values from each adapter's live parameter content. Reject
   offered source artifact digests that differ from those local hashes.
3. Call the existing R68 two-model source comparator. It requires the R57
   Labyrinth and Rezon source-pin alignment and **independently obtained**
   expectations for the Rezon claims and both adapter source records.
4. Invoke each model **separately**, with the same prompt, using
   `ToyLinearAdapter.generate(prompt)`. Return two individually labeled
   responses and local content digests. Check both live model hashes and the
   scope fingerprint again after generation and reject changes.

The resulting `ToyPairProbeV1` has only source/parameter content fingerprints,
scope fingerprint and two text answers as constructor fields. Its immutable
status flags are all false for model identity, principal, permissions,
consent, cross-model identity/grant transfers, distributed inference, and
training/graduation/merge/deployment authority. **This response is not a
signed receipt or authority token.** Different toy weights can produce the
same integer text for a particular prompt; their canonical parameter digests
must nevertheless be distinct.

## Provenance limits and unresolved owner obligations

- Unit-test records are deliberately synthetic; copying test observed claims
  to test expected pins demonstrates comparator behavior **only**, not
  independent owner review or attestation.
- The adapter `model_hash` is a hash of toy parameters/configuration, not a
  file checksum verified against host-owned immutable bytes. The offered
  `adapter_contract_sha256` and `model_source_head` are compared to expected
  source fields, **not** checked against executable code or a signed runtime.
- Rezon `receipt_bundle_sha256` remains an externally maintained opaque
  source reference; the R57 `maze_exposure_sha256` and
  `checkpoint_sha256` still differ in object semantics from Labyrinth's
  training exposure and checkpoint manifest pins.
- Hostile code in the same Python interpreter may monkeypatch classes,
  source comparators or `generate`; the before/after checks are only
  best-effort application fences. An owner-governed runtime is required for
  immutable source expectations, per-model identity, consent, permissions,
  receipts, physical measurements and durable replay prevention.
- Real Rezon distributed inference would require a separately scoped Rezon
  worker/scheduler protocol, host supervision, authorized principals and
  per-model state isolation. **None is implemented or inferred by R76.**

## Negative acceptance tests

```bash
python -m unittest -v tests.test_rezon_toy_pair_consumer_r76 \
    tests.test_rezon_two_adapter_source_partition_r68 \
    tests.test_rezon_pointer_mapping_r57
```

Tests cover positive execution from two distinct toy parameter sets, source
claim and expectation swaps, swapped live model objects, one object reused
under two labels, identical parameter content, runtime weight mutation,
cross-scope pointer replay, malformed prompt, and mismatched Rezon source
claims. Each negative path must reject; none provides a graduation verdict.
