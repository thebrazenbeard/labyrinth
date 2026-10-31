# Q5: data-only signed source witness (R72)

Status: **SOURCE SIGNATURE CONTRACT PASS; EXTERNAL CHRONOLOGY/CUSTODY HOLD**.

`labyrinth.q5_signed_data_witness_r72` adds a source-verification path that
does **not accept or execute a caller-supplied Python witness callback**.
The caller supplies a canonical, Ed25519-signed public source record and a
separately pinned signer public key, preregistration hash, event ID, and
externally maintained spent-event set. The module verifies the signature and
exact public fields, then invokes the existing Q5 statistical source screen.

The signed canonical body has only these fields:

- `schema = labyrinth.q5-signed-preoutcome-source-witness.v1`
- `event_id`
- `precommit_sha256`
- `signer_public_key_sha256`
- `declared_stage = BEFORE_OUTCOMES`

The signature is over `b"labyrinth.q5-signed-preoutcome-source-witness.v1\0"`
followed by canonical JSON bytes terminated with a single newline. Inputs
with duplicate JSON keys, noncanonical bytes, invalid signatures, changed
hashes, unpinned keys or mismatched event IDs fail closed. A caller-provided
spent-event digest rejects that event at the source API boundary.

Importantly, **`declared_stage` is only a signed claim**. A signer can sign
the record after seeing results. A caller can supply a fresh key and its own
matching pin, or falsely claim an empty spent-event register. The source
module cannot tell whether signer custody was external, whether the signature
was anchored before outcomes, whether the event was consumed durably, whether
the evaluated cases were independently held out, or whether pretrained model
parameters changed. These require host-governed, nonrollback authority and
independently witnessed execution; no positive authority flags are produced.

As with other in-process Python verifiers, code allowed to monkeypatch module
classes or functions is inside the trusted interpreter boundary. This record
is not an authorization credential; its instance-level constant `False`
properties cannot substitute for a host authorization service.

Test source behavior:

```bash
python3 -m unittest -q \
  tests.test_q5_signed_data_witness_r72 \
  tests.test_q5_evaluation_preregistration_r68 \
  tests.test_final_exam_precommit
```

The signed source record is deliberately **not** presented as physical Q5
qualification, Q6 retention/transfer, Q7 sealed-final chronology, Q8 host
authority, or Q9 graduation. All those physical gates remain HOLD.
