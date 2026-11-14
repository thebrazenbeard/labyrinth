# Wave46 — executable public-rule verified mechanics runtime

This is a *functional deterministic game-mechanics runtime* that preserves
separate provenance for a candidate's proposed answer and a public transition
solver's correction. It is **NOT neural learning, independent examiner custody,
a generalization qualification, or a weight attestation**.

## Execution

The input is the source-defined public mechanics exam JSON and a candidate
response roster. The runtime verifies both the exact original public-byte
SHA256 and the exact source-defined prompt for each transition system. It
does not load the private examiner answer key, signing key, model checkpoint,
or the source-exam precommit.

For each candidate answer, the runtime:
1. Replays the candidate's proposed PLAN through the public transition
   tables. A candidate UNSAT is accepted only when the public exhaustive
   solver independently confirms no bounded winning trace.
2. Retains an accepted candidate answer **unchanged**, even if it is a valid
   but non-shortest solution.
3. For every incorrect, malformed or unsupported candidate answer, computes a
   deterministic public-rule fallback with dual BFS and exhaustive finite
   enumeration. This does not update a neural weight.
4. Emits the original response schema for compatibility with existing
   source-only action scoring, plus a **separate original unkeyed receipt**
   with exact counts and per-case source labels:
   CANDIDATE_PUBLICLY_VERIFIED or DETERMINISTIC_PUBLIC_FALLBACK.

Every output includes a public action-by-action trace, original candidate
answer hash, selected answer hash, original public world hash, model observation
class (if supplied), and explicit:
neural_learning_verified=false,
independent_custody_verified=false,
served_weight_bytes_attested=false.

An optional *original* actual-model HTTP observation and its independently
retained original receipt digest can be supplied. The runtime reuses the
strict original free-text V1/V2 and JSON enum evidence verifiers from prior
waves. A wrong, forged, incomplete, reordered, or inconsistently pinned
model observation fails **before** publishing runtime outputs.

Example:

    python -m labyrinth.public_mechanics_runtime --public C:/exam/public.json --candidate C:/exam/model-responses.json --expected-public-sha256 ORIGINAL_PUBLIC_SHA256 --observation C:/exam/model-http-observation.json --expected-observation-sha256 ORIGINAL_MODEL_RECEIPT_SHA256 --responses-out C:/exam/hybrid-responses.json --report-out C:/exam/hybrid-provenance.json

The runner is **offline**: no model inference, network access or training
happens during the hybrid execution. Its outputs are exclusive-create
artifacts. The original candidate observations, public tables, source
precommit and scores remain unmodified.

### Important provenance boundary

The selected response roster uses the existing exam candidate-response
format for source-only mechanics replay. **Do not** report its 4/4 or 12/12
as a neural model's score. The model's *original* candidate file and HTTP
receipt must remain separate from the corrected hybrid output, and the
hybrid provenance receipt must accompany any runtime result. The source-only
scorer itself reports SOURCE_ONLY_RESPONSES_NOT_MODEL_ATTESTED and will not
certify learning, independently served weights or custody. This is not an
independently trusted binding: an operator controlling evidence could discard
or forge the hybrid provenance report.

## Actual replay of previous neural model observations, 2026-11-14

The exact unchanged four development-only authored counterfactual cases
from Wave38/45 were used. Public source SHA256:
bb64c0a1984377beb4ad4d6b0598fbb53668977a752d5aee1d6449c6af26f0d3.
Original examiner precommit:
79511e606a58099435ef79f37f3207946caeb29b13ee8dfd26fdb8f2fc33ba3a.

| Actual earlier candidate | Original neural correct | Accepted by public check | Deterministic corrections | Final source-only score |
|---|---:|---:|---:|---:|
| Qwen2.5 Coder 1.5B q8, original V2 HTTP | 1/4 | 1 | 3 | 4/4 |
| Qwen3 4B Instruct, JSON-enum HTTP | 0/4 | 0 | 4 | 4/4 |

These are **actual original model responses replayed after inference**; no
new model inference was performed in this Wave46 hybrid pass. The separate
real local Ollama evidence was verified against each response file's original
held receipt digest. Model responses were not rewritten in-place.

- Qwen2.5 Coder hybrid provenance receipt SHA256:
  12ec559577f2c9244ac453c62bba6612bf97e0dff295e37c965b1860195f7962.
  Original model invocation:
  LOCAL_HTTP_SELF_REPORTED_NOT_ATTESTED.
- Qwen3 4B constrained hybrid provenance receipt SHA256:
  6e6ed12ae1cf11764210ee6fe3f487267f38b179e5467cc85d4d0636e938e6e0.
  Original model invocation:
  LOCAL_HTTP_JSON_ENUM_SELF_REPORTED_NOT_ATTESTED.
- Both source-only corrected response rosters were separately evaluated by
  the original, unchanged blind action replay; both scored 4/4.
  This is deterministic correction, not improved model behavior.
- Raw original reports and new corrected results remain outside Git at:
  D:\VERA\experiments\minotaur-wave45-20261114 and
  D:\VERA\experiments\minotaur-wave46-real-replay-20261114.

## Additional 12-case on-host smoke run

A **fresh source-generated** 12-case exam (seed 46012) was produced on Lappy
for runtime functionality. Its case outcomes were balanced by the
source-defined generator, six SAT and six UNSAT. This is NOT an independent
holdout because generation and scoring occurred under the same Windows account.

- Original public source SHA256:
  598e70248f7fb84e3f48a0cb7301cbf0ff797055311fd3053be15ab27542c4e7
- Original precommit SHA256:
  c8cff3c6a14c5da11069971b46ab3c84ee286fca1d0a859ab0c9f2a2178f11f4
- Input candidate was the deliberately trivial *source-only* always-UNSAT
  roster, not a real model response: 6/12 accepted, six corrected by the
  symbolic engine, and 12/12 final answers verified by original source-only
  examiner action replay.
- Hybrid provenance receipt SHA256:
  63794f11f157113f3c9a75e6fe4a0b3868d257f38abd7e3a725a008325cc8f62
- Exact output:
  candidate_invocation_evidence=NOT_SUPPLIED;
  neural_learning_verified=false.
- Raw files retained outside Git at:
  D:\VERA\experiments\minotaur-wave46-fresh12-20261114.

The run demonstrates that a user-facing finite mechanics runtime can return
correct, verified actions despite a weak candidate. It does NOT demonstrate
that the neural model learned to reason or passed a graduation benchmark.

## Threats and boundaries

> SURVIVES: valid candidate plans preserved, erroneous candidate UNSAT or
> blocked plans corrected, original model receipt and public-source hash
> checked, selected answers accompanied by exact per-case fallback count and
> public action traces, no private exam needed at execution.

> FAILS: any assertion that 4/4 corrected results or 12/12 source-only
> fallback results represent neural model capability. The original models
> remain 1/4 and 0/4, respectively.

> UNKNOWN: effective generalization, separately held examiner custody,
> original model weight bytes or adapter provenance, and transfer beyond
> this finite two-phase transition problem family. The fallback algorithm
> and scorer intentionally share source-defined mechanics; model agreement
> is not independently verified evidence.

Full tests:

    python -m unittest discover -s tests -p test_public_mechanics_runtime.py -q
    python -O -m unittest discover -s tests -p test_public_mechanics_runtime.py -q
    python -m unittest discover -s tests -q
