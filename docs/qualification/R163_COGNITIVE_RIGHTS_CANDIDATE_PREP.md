# R163 cognitive TRAIN rights candidate (source-only)

`labyrinth.cognitive_rights_candidate_v1.prepare_cognitive_rights_candidate`
packages a **candidate for independent admission**, not an admission grant.

Input: one immutable `CognitiveTrainDataset` from the approved #109 synthetic
TRAIN exporter. The function verifies exact dataset SHA-256, canonical row
serialization, **exactly the full 15-row canonical roster**, objective and
cycle bindings, prompt SHA-256, recomputed prompt-derived semantic SHA-256,
unique case IDs and source bindings, and optional collision against *supplied*
protected-evaluation semantic hashes.

There are **two strictly separated outputs**:

- **PUBLIC:** `candidate.manifest_bytes` and `candidate.sha256` are the same
  constant `labyrinth.cognitive-rights-public-status.v1` rejection receipt for
  *every* candidate. It contains **only** fixed schema, `UNVERIFIED_REJECTED`,
  and `authorizes_training=false` / `authorizes_host_execution=false`. There
  are **zero** seed, cycle, source, contract, corpus, roster, row, prompt,
  semantic, or answer-correlated fields. Different valid datasets yield
  byte-identical public output.
- **PRIVATE:** A **separate function**
  `prepare_private_cognitive_steward_manifest(...)` returns an opaque
  `PrivateStewardManifest`; its explicitly private `.private_bytes()` accessor
  exposes `labyrinth.cognitive-rights-private-steward.v1`, including full
  labeled dataset digest, per-row/source-binding hashes, prompt/semantic
  hashes, Git/file claims, 15 TRAIN entries, absent DEV and rights/exclusion
  holds. This distinct object must remain entirely in authenticated private
  operator-to-steward custody. Never log/publish it or a raw test dump.

**API migration:** The prior
`candidate.private_steward_manifest_bytes()` was intentionally removed:
`repr=False` hid private bytes from printing but not
`dataclasses.asdict(candidate)` or recursive serializers. Public
`CognitiveRightsCandidate` now physically contains **only** the constant
status receipt, its constant SHA-256, and fixed 15-row count. The full object
is safe to `dataclasses.asdict`; it cannot embed private bytes. Operators who
need private evidence must explicitly call the separate private constructor
with the same source/exclusion arguments. Neither path authorizes training.

Crucial privacy boundary: even a supposedly answer-independent *prompt* or
semantic SHA-256—and the source contract fingerprint—can reveal the correct
A/B/C label by enumerating public deterministic producer seeds. Full labeled
dataset hashes, individual row hashes, and `source_binding_sha256` are also
low-entropy answer oracles. **No unkeyed data-specific commitment is public.**
Any durable private custody commitment needs an independently held key or an
authenticated operator-to-steward channel. This code does **not** implement
that channel. The synthetic generator itself is public source, so nothing
here claims that public repository code or all possible generated labels are
secret; the objective is to avoid a new public receipt becoming a label oracle.

## Explicit HOLD boundaries

- Private source file and Git revision hashes are unverified producer identity claims,
  not human authorship, copyright ownership, permission, or consent. An
  independent steward must replay the exact canonical 15-row producer output
  against a trusted Git revision **privately** before concluding provenance;
  caller-selected subsets or local hashes cannot substitute for that replay.
- A supplied protected-evaluation roster digest and local exclusion comparison
  do not prove roster completeness, custodian identity, or global exposure
  non-overlap. If an exclusion semantic collides, packaging rejects.
- Consent, license, and privacy reviews remain `NOT_ASSESSED`/`NOT_ADMITTED`.
  The source examples appear synthetic; this is not independent rights proof.
- External steward and evaluator signature-shaped strings do not grant
  admission. `require_cognitive_rights_admission` always raises
  `CandidateAdmissionUnavailable` until a separately authenticated issuer
  exists. `authorizes_training` and `authorizes_host_execution` are false.
- No DEV examples or independently held-out bank are generated. Never promote
  these labeled TRAIN records into the protected Q6 examiner.
- No GPU/host lease, NAS route grant, model load, optimizer, RNG checkpoint, or
  learning-gain claim is produced by this code.

An external rights steward must establish original source/license and consent
scope; an independent evaluator custodian must publish a sealed exclusion
commitment and exposure ledger; the host-owned broker must verify those
authorities and exact run/corpus/base bindings before any physical training.
This module intentionally cannot impersonate any of those issuers.
