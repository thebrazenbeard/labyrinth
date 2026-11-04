# R72 Minotaur registry admission lineage

Status: **source-only, non-authorizing**.

`MinotaurTrainingPolicy.extend_training_registry()` intentionally remains
backward compatible and returns the new approved-registry fingerprint. It can
change `policy.model_hash` even though no TRAIN outcome was scored. Therefore,
a sequence consisting only of `observe()` receipts can legitimately have a
predecessor discontinuity when a trainer admits regenerated rooms.

`extend_training_registry_with_receipt()` provides an **atomic local** record
of that separate state transition:

- previous and successor policy-state SHA-256;
- previous and successor approved-TRAIN registry SHA-256;
- exact newly admitted case IDs;
- a fixed non-learning transition kind and permanently non-authorizing
  *instance-level* properties.

The method uses the same in-process `RLock` as training observation,
checkpointing, registry validation, and policy hashing. A malformed or
duplicate registry extension fails without emitting a receipt or changing
state. The resulting policy remains checkpointable and reloadable with the
complete externally supplied approved-TRAIN registry.

This record is **not** a model-weight update, authorization lease, signature,
or independently sequenced event. The policy still trusts the caller-supplied
trainer registry. Python code in the same interpreter can monkeypatch its
classes and inspect or mutate state; tuple-style local receipt immutability is
not a host security boundary.

Before a host can use these transitions for distributed replay prevention or
graduation evaluation, it must independently enforce:

1. authenticated trainer/provider identity and exact TRAIN scope;
2. durable, monotonic reservation of each registry-admission event;
3. compare-and-swap of the stored predecessor policy and registry heads;
4. read-back of the committed event and checkpoint after process restart;
5. explicit separation between inventory admission and genuine model learning;
6. no permission, consent, identity, or examiner authority transfer from
   this source-only value.

Regression:

```bash
python3 -m unittest -q tests.test_r72_minotaur_registry_transition \
    tests.test_minotaur_policy_learning tests.test_r68_minotaur_policy_concurrency \
    tests.test_journey_policy_runner
```

This does not advance Q5–Q9 physical qualification, pretrained-model
improvement, or deployment readiness.
