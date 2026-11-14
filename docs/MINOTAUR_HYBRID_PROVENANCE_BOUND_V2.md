# Wave47 — type-separated V2 hybrid answers and source-bound verifier

**Status: working offline finite-mechanics hybrid with source-only provenance replay.
Not a neural-learning qualification, independent examiner custody, loaded model
weight attestation, or new model inference.**

## Actual defect in Wave46 V1

The initial public-rule fallback in Draft PR #226 emitted mechanically corrected
answers with exactly the same `labyrinth.blind-mechanics-responses.v1` schema
used for the original model's raw answer roster. The source-only examiner
accepted the corrected file **even without its separate fallback report**.
Although the scorer explicitly labeled that result source-only/not model
attested, external consumers could mistake the selected 4/4 for a model's
original answer quality. Merely writing provenance first does not fix this
semantic type collision.

A red-first test reproduced the collision on the original implementation:
the selected schema was `labyrinth.blind-mechanics-responses.v1`, instead
of the required distinct V2 schema (focused test exit1). The original
source-only scorer's acceptance of V1 corrected responses is also established
by the actual Wave46 scorer receipts. This is a **provenance-laundering
integration risk**; not a claimed model-trust bypass of an attestation the
scorer never offered.

## V2 contract

Newly generated V2 corrected outputs have:

- **Selected answer file schema:**
  `labyrinth.public-mechanics-selected.v2`
- **Companion report schema:**
  `labyrinth.public-mechanics-hybrid-runtime.v2`
- Explicit per-case source: `CANDIDATE_PUBLICLY_VERIFIED` or
  `DETERMINISTIC_PUBLIC_FALLBACK`
- Original public/candidate byte hashes, original model HTTP observation
  SHA when supplied, selected output hash, counts, full public action traces
  and explicit nonattestation booleans.
- Same offline public-only candidate validation and symbolic correction as V1.
  A corrected answer is NEVER credited as new neural capability.

The existing `labyrinth.blind_structural_exam score` command accepts only
raw `labyrinth.blind-mechanics-responses.v1` candidate rosters, and now
**rejects** V2 corrected outputs at the schema boundary. That distinction
does not depend on the optional model HTTP observation.

To verify corrected V2 output, use the *different* read-only command:

    python -m labyrinth.hybrid_artifact_verifier --public PUBLIC_ORIGINAL.json --candidate RAW_CANDIDATE_ORIGINAL.json --selected SELECTED_HYBRID_V2.json --report HYBRID_PROVENANCE_V2.json --expected-public-sha256 PUBLIC_ORIGINAL_SHA256 --expected-report-sha256 ORIGINAL_HYBRID_REPORT_SHA256

When the source candidate came from one of the local-model observers,
also pass `--observation ORIGINAL_MODEL_HTTP.json` and
`--expected-observation-sha256 ORIGINAL_MODEL_HTTP_RECEIPT_SHA256`.
Both must be present together.

The verifier checks the held original public SHA; exact candidate byte hash;
strict V2 selected and companion schemas; exact original companion receipt
SHA and its canonical self-hash; exact selected output bytes/hash; consistency
with the model observation pin; nonattestation flags; and the original
response roster's case alignment. It then recomputes the *entire* finite
mechanics hybrid calculation in an isolated temporary directory **using
only the original public rules and raw candidate**, revalidating the original
model HTTP observation when supplied. Both recomputed original file bytes
must exactly match those under verification.

Successful output has status
`PUBLIC_HYBRID_SOURCE_VERIFIED_NOT_MODEL_ATTESTED`, and separate counts for
candidate-accepted responses, deterministic corrections, and publicly
valid selected answers. It never claims that the final source-only score
is a neural model score. The verifier does **not read the private answer
key**; tests physically hid it while verifying.

## Actual replay on Lappy, 2026-11-14

At exact Wave47 source `ef7407cc9ec12d1e7957f7345a37be7801652df0`,
the original held four-case Wave45 model response files were reprocessed
with the V2 builder and **then verified with the separate V2 reader**.
The model observation files, public cases and private-case precommit were
not modified.

| Original observed candidate | Original correct | Candidate answers accepted | Symbolic corrections | Final *source-only* valid |
|---|---:|---:|---:|---:|
| Qwen2.5 Coder 1.5B q8 | 1/4 | 1 | 3 | 4/4 |
| Qwen3 4B JSON-enum | 0/4 | 0 | 4 | 4/4 |
| Source-only all-UNSAT 12-case control | 6/12 | 6 | 6 | 12/12 |

Original Qwen coder model receipt SHA
`064ca5105e75f3594025cd26c88e576616a3768bee6cea4394e3aecd26c5e186`;
new V2 hybrid provenance receipt SHA
`2c777da8211185922e059eecdd08657fbbf1ad9606155a6b6ca354b00a57d768`.

Original Qwen3 4B JSON observation SHA
`1f991dac831a178484a73d2bc0be33b8f8e7aaaed5a2809af33f3cac9207c9e2`;
new V2 hybrid provenance receipt SHA
`78b15d3eecaa8cc334d092710bca45f5c8762636d2e5112bfa16efb07f2479d8`.

Original source-defined 12-case public SHA
`598e70248f7fb84e3f48a0cb7301cbf0ff797055311fd3053be15ab27542c4e7`;
new V2 hybrid report SHA
`4e9ecab28209b6fa47e241674a93e6ca494c677eff4e466e3771bee10c1bbbc0`.

The real-file V2 Qwen3 4B selected output was also submitted directly to
`python -m labyrinth.blind_structural_exam score` using the original
private/public/precommit paths. The original scorer **raised**
`ValueError: candidate response roster is invalid`, exited 1 and
published **no score file**. The V2 verifier had already accepted that
same exact selected file as an explicitly labeled public-only hybrid
with 0 neural candidates accepted and 4 symbolic corrections.

Off-Git originals and new artifacts reside at:
`D:\VERA\experiments\minotaur-wave45-20261114`,
`D:\VERA\experiments\minotaur-wave47-real-v2-20261114`, and
`D:\VERA\experiments\minotaur-wave47-fresh12-v2-20261114`.

## Threat ceiling

> **SURVIVES:** the tested accidental raw-model schema collision is removed
> for newly generated V2 outputs; the reader rejects absent, inconsistent,
> modified or schema-swapped source/provenance files; source-only action
> reconstruction produces mechanically valid answers without private keys.
>
> **FAILS:** any claim that symbolic fallback performance is a neural
> reasoning or weight-training result. Qwen3 4B remains an observed 0/4
> original candidate and Qwen coder 1/4 on the reused four-case examples.
>
> **UNKNOWN:** trust in the original caller-controlled receipt pin,
> independently administered examiner/candidate identities, effective
> access-denial proof, served weight page/adapter identity, and semantic
> generalization beyond the existing finite mechanics.

An adversary who controls the original candidate *and* original held
receipt pin can still forge their own local observations, and someone can
deliberately strip or rewrite a V2 schema. **This is not a security boundary
against a malicious owner of all files.** V2 prevents the specific ordinary
source-scoring pathway from silently accepting corrected hybrid artifacts.
The prior V1 model-compatible corrected artifacts remain historical and
MUST NOT be retroactively labeled neural evidence. Two separate publication
files are not an atomic transaction; use the strict V2 verifier before
consuming them.

The official model-training PR #351 Qwen TrainingDay physical launch HOLD
remains unchanged, with zero weight updates in this wave.

Focused verification:

    python -m unittest discover -s tests -p test_hybrid_artifact_verifier.py -q
    python -m unittest discover -s tests -p test_public_mechanics_runtime.py -q
    python -O -m unittest discover -s tests -p test_hybrid_artifact_verifier.py -q
    python -O -m unittest discover -s tests -p test_public_mechanics_runtime.py -q
    python -m unittest discover -s tests -q
