# R26 — Labyrinth → Vera Mono evidence handoff (source-only)

Status: `SOURCE_CANDIDATE_NONAUTHORIZING`. This module is **not installed in
Vera Mono**, not a host verifier, not an examiner, not a neural-training run,
and not an evaluator custody mechanism.

The export is a versioned, canonical JSON envelope from
`labyrinth.training_evidence_handoff_r26`:

- `schema = labyrinth.training-evidence-handoff.v1`
- `record` contains exact declared Labyrinth Git commit, run and report hashes,
  learning intent, curriculum admission, pretrained base, initial and trained
  adapters, tokenizer, optimizer receipt, Minotaur policy and reported gate
  result, evaluator source/key/receipt hashes.
- `sha256` binds the canonical record bytes to a domain-separated hash. It is
  an **integrity checksum**, not a signature, independent witness or proof that
  any underlying model, checkpoint, examiner or report actually existed.

`export_handoff()` accepts existing typed
`LearningIntent`, `CurriculumAdmission`, `AdapterTrainingReceipt`,
`GraduationPolicy`, and `BridgeDecision`. It reruns the existing
pre-exam source binding validator before emitting. This rejects conflicting
objective/manifest/tokenizer/adapter/source and malformed provenance claims.

`inspect_handoff()` independently checks bounded canonical transport,
no duplicate fields, digest, strict schema, exact typed non-authorizing flags,
and a supplied `ExpectedConsumerBinding`. Every identity-bearing reported
field is now pinned: objective ID/kind, intent, curriculum, admitted corpus,
training source, tokenizer, base/initial/trained adapters, the exact positive
completed optimizer step count, evaluator source and
conditions, Minotaur policy, examiner public key and receipt, optimizer and
training reports, training run ID, Labyrinth Git commit, and *reported gate
status*. These are **mandatory**, with no unchecked legacy fallback, because
this is an uninstalled source candidate with no existing runtime consumer.
The caller must obtain those expectations **out-of-band** from an appropriately
controlled source. Merely passing strings from the same untrusted exporter
does not improve trust or create a host authority. A consumer that cannot pin
a field must hold that evidence rather than pretend its provenance is known.

The transported `minotaur.reported_gate_status` can report a source-only
`SIGNED_GATE_REPORTED_GRADUATED`; it must never be interpreted as a current
trusted graduation, independent private-bank result, installation or actual
permission. Source export and readback additionally enforce exact status to
disposition mappings (e.g. `GATE_BLOCKED` can only report `BLOCKED`, not
`GRADUATED`). A malicious producer changing an identity-bearing field and
recomputing the packet's unkeyed checksum still fails a correctly independently
pinned consumer comparison. All packets enforce:

```json
{
  "requested_effect": "NONE",
  "source_evidence_status": "DECLARED_UNATTESTED",
  "host_authority_verified": false,
  "physical_learning_verified": false,
  "graduation_authorized": false,
  "plastic_write_authorized": false,
  "deployment_authorized": false,
  "identity_consent_transferred": false
}
```

The R26 transport distinguishes `SOURCE_DECLARATION` from actual neural
weights, source history from lived/episodic memory, and receipt metadata from
permissioned functional/plastic effects. It follows Vera Mono's requirement
that admitted runtime mechanisms be copied into its self-contained source
with exact donor provenance; this Labyrinth module is **not** a runtime sibling
dependency for Vera Mono.

Before a receiver can use the artifact for any consequential admission, a
separately administered host service must authenticate the producer and
examiner, verify actual byte-range and model provenance, enforce monotonic
single-use state under rollback, and apply its own explicit authority and
resource leases. A source checksum, caller callback or public Minotaur PASS
does not satisfy those conditions.

Contract tests: `PYTHONPATH=tests:. python3 -m unittest -q
test_training_evidence_handoff_r26`. Independent cross-repo consumer and
physical deployment tests remain unperformed.
