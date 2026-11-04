# R69 byte-bound Labyrinth/Rezon source observation

**Status:** `SOURCE_BUNDLE_BYTES_MATCH_UNATTESTED`; source-only, nonauthorizing.

`labyrinth.rezon_pointer_mapping_r57` compares independently expected source
*strings* but leaves `receipt_bundle_sha256` opaque. The R69 adapter adds a
specific observation: it receives the exact bytes of a Rezon receipt bundle,
calculates SHA-256 and demands that those bytes match the expected Rezon
reference already cross-checked against the Labyrinth canonical pointer.

```python
from labyrinth.rezon_receipt_binding_r69 import compare_bundle_bytes

# These MUST come from independently governed source registries, not the
# offered pointer or bundle. The reader is responsible for safely opening
# the actual bundle and preserving its unmodified bytes.
observation = compare_bundle_bytes(
    pointer_bytes, receipt_bundle_bytes,
    expected_labyrinth=independent_labyrinth_pins,
    observed_rezon=offered_rezon_claims,
    expected_rezon=independent_rezon_pins,
)
assert observation.source_bundle_bytes_matched
assert not observation.source_receipts_verified
assert not observation.can_authorize_training
```

The module does **not** deserialize the bundle, verify Rezon's HMAC roles,
confirm bundle signature or producer custody, compare semantic exposure vs
checkpoint objects, register model identity/consent, or consult a host-owned
antirollback witness. An adversary who clones old expected pins can also
replay the matching old bundle successfully. Distinct real-time external
owner-governed currentness and Q8 signer/CAS are required before granting any
effect. Identity matching of opaque source bytes is not model learning,
distributed compute, or a deployment/merge gate.

The byte input is an exact `bytes` object with a 1 B–4 MiB bound. No local
file reads, network calls, subprocesses or mutations are performed by the
adapter. It can be used with Rezon's independent receipt owner without
introducing an import-time dependency on the Rezon Python package.

Focused local test:

```bash
python -m unittest tests.test_rezon_receipt_binding_r69 \
    tests.test_rezon_pointer_mapping_r57 tests.test_evidence_pointer -v
```

The synthetic fixtures prove source byte-substitution rejection only, not
actual Rezon runtime or a physical pretrained neural experiment.
