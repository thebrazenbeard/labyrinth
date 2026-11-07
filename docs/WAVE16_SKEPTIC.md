# Wave16 Skeptic — model-trial evidence and common-mode trust

**Baseline:** isolated PR157 integration source
`16f84839df869d2a89243d55baa212cf92527944`.

**Artifacts:** `tests/test_wave16_skeptic.py` and this document only. The
candidate exercised here is an unchanged local fixed child subprocess; no
neural model is installed, modified, or scored as learned. No remote PR or
host service is touched.

Run: `python -B -m unittest discover -s tests -p test_wave16_skeptic.py -v`
and `python -O -B -m unittest discover -s tests -p test_wave16_skeptic.py -v`.

## S16-01 — common-mode JSON decoder falsely credits an abstaining model

**FAILS (reproduced).** The model-response scorer `_parse_response` and
supposed independent source-wire witness `_independent_wire_decision` both
call the identical `trial._read_json` function. One host-side mutation of
that function changes the parsed result of exact child `_ABSTAIN` bytes to
an already published correct seed-0 PLAN. The fixed subprocess never
proposed a plan, yet the runner returns:

- actual response hash = SHA-256 of the child **ABSTAIN** bytes;
- source receipt replay = true;
- trial binding = true;
- `verified_success = true`.

The two validators shared a single source implementation. Their agreement
was not independent physical or semantic evidence. A fix must independently
parse bounded raw output through a separate strict decoder path, preserving
failure-closed detection if one parser is mutated. A second call to the same
parser is not an independent witness.

This deliberately mutates trusted host Python, not the isolated child, so
it demonstrates the evaluator's *common-mode software-fault* boundary,
not a separately proven process escape.

### Follow-up S16-01b — shared stdlib decoder beneath separated wrappers

Root's first in-progress repair used a second `json.loads` invocation rather
than the scorer's `_read_json` wrapper. That stops the original single-helper
mutation but **still delegates both paths to the same mutable standard-library
`json.loads` function**. A second adversarial test patches only `json.loads`
for the actual fixed-child `ABSTAIN` bytes. The repaired WIP then also
reported a false `verified_success=true` despite a valid ABSTAIN response
hash. This is an intentionally strong *same-interpreter TCB* challenge,
not evidence that an external candidate could patch the host stdlib through
the child protocol. The genuine custody boundary remains unqualified.

## S16-02 — outer trial report SHA-256 is not an authenticator

**SURVIVES as explicitly unqualified evidence, not a secure receipt.** A
recipient can rewrite `model_neural_identity_verified`,
`model_weight_update_verified`, `generalization_qualified`, the trial
verdict, and success flag, then recompute an internally consistent unkeyed
`receipt_sha256` without any evaluator authority. The embedded source
receipt still disagrees with the fake outer WIN. This is a controlled
demonstration of why a hash alone cannot prove provenance; the source
correctly labels model identity unverified. Do not interpret a matching
checksum as independent signed custody.

## S16-03 — candidate-private versus host report disclosure

**SURVIVES at the configured JSON child boundary.** The child-facing public
view omits `seed` and `case_id`; the host's completed trial report includes
the exact case seed twice (`case_seed_host_only` and configuration). The
unqualified report is not an evaluation prompt and must not be distributed
to candidate models during an active withheld campaign. Model/source
contamination remains UNKNOWN until genuine hidden seed custody exists.

## Evidence limits

These are source and executable TEST results on exact commits, not trained
parameter updates, authenticated model artifacts, independent external
reviewer custody, or hidden-generation qualification. Public deterministic
Switchback source permits catalog/seed memorization. A classifier claiming
neural generalization or graduation from these tests would exceed the
evidence, even after every regression goes green.
