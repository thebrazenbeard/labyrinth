# Live model observation V1 — bounded execution, not qualification

This tool replaces a synthetic-only model-output claim with an actual, bounded
localhost HTTP observation and a write-once receipt. It does **not** update weights,
control a protected evaluator, certify model-serving weights, or graduate Minotaur.
It cannot release the official TrainingDay physical-launch HOLD.

## Three operations

1. `commit` writes a case-set SHA-256 commitment **before inference**.
2. `observe` reads that exact case-set, hashes a local regular GGUF weight file,
   checks that the already-running loopback llama.cpp server advertises that
   exact file path, sends bounded prompts and saves full raw model responses,
   strict exact-answer scores, response hashes and execution times.
3. `compare` checks two write-once observation receipts against the same
   committed case set. By default, its correctness totals are **self-reported**
   and paired win/loss counts are withheld (`null`). Pass **both** `--cases`
   and `--commitment` to replay actual model text against the exact locally
   precommitted answer-key bytes before reporting paired win/loss counts.
   Different weight hashes are *not* optimizer provenance and do not prove
   the files were served.

Case file, input limits: 1–16 cases, at most 128 KiB JSON, each case has
`case_id`, `prompt`, and a short integer `answer`. No case answers go
into model HTTP requests. The model must produce `ANSWER: N` with no other
content to receive exact credit. This is a format-scoped arithmetic probe,
not a general semantic reasoning evaluator.

The case file is controlled by the evaluator/operator and must remain outside
the candidate's training context. Do not commit a claimed secret exam or its
answers to public Git. Keep the cases and original receipt read-only after
precommit; only a truly independent authority could supply stronger custody.

A separately retained `receipt_sha256` from the original `commit` output
can be supplied to `compare` via `--expected-precommit-sha256` **together
with** `--cases` and `--commitment`. This rejects an attacker replacing
the case file, recreating its precommitment, and rewriting both observations,
*provided* the original expected SHA was preserved outside the attacker's
control. An attacker who controls the expected value can simply pin the
forgery. The CLI cannot independently establish who retained that value.
`precommit_pin_evidence=CALLER_SUPPLIED_SHA256` is a literal input record,
not an independent-custody or cryptographic-signature claim.

The accepted precommit schema is exact: a `NOT_EXECUTED` status and
`LOCAL_DEVELOPMENT_NOT_INDEPENDENT` custody class. Rehashed receipts with
additional or more favorable custody declarations are rejected.

## Invocations on an evaluator host

Use an already running `llama-server` exposing the OpenAI-compatible API on
127.0.0.1. The tool does **not** start or stop any server and cannot send
requests to remote, redirected or proxy endpoints.

```powershell
python -m labyrinth.live_model_probe commit --cases C:/audit/private_cases.json --out C:/audit/commit.json

python -m labyrinth.live_model_probe observe --cases C:/audit/private_cases.json --commitment C:/audit/commit.json --model-file C:/models/before.gguf --endpoint http://127.0.0.1:11435 --out C:/audit/before.json
```

After a separately authorized, genuinely trained checkpoint is **actually
materialized, hashed and served** under a separately verified local endpoint:

```powershell
python -m labyrinth.live_model_probe observe --cases C:/audit/private_cases.json --commitment C:/audit/commit.json --model-file C:/models/after.gguf --endpoint http://127.0.0.1:11436 --out C:/audit/after.json

python -m labyrinth.live_model_probe compare --before C:/audit/before.json --after C:/audit/after.json --cases C:/audit/private_cases.json --commitment C:/audit/commit.json --expected-precommit-sha256 ORIGINAL_SEPARATELY_RETAINED_64_HEX_SHA256 --out C:/audit/paired.json
```

Expected case-file structure (PUBLIC DEMO ONLY, not held-out):

```json
{"schema":"labyrinth.live-probe-cases.v1","cases":[
  {"case_id":"demo1","prompt":"Reply only ANSWER: N. What is 5 + 8?","answer":"13"}
]}
```

The comparison's `score_evidence` distinguishes
`SELF_REPORTED_ONLY` from `REPLAYED_CALLER_SUPPLIED_CASES`.
The latter recomputes each `correct` flag from the exact prompt and answer in
the supplied case file, checks the original case/precommit hashes, and rejects
substituted prompts or fabricated scores. It is **not** a trusted external
evaluator: someone controlling both the case file and its unkeyed commitment
can manufacture another self-consistent pair. Without `--cases` and
`--commitment`, reported before/after accuracy totals remain unverified
claims and `paired_candidate_only_wins`/`losses` are withheld rather than
implicitly treated as observed improvements.

All receipts are exclusive-create and cannot overwrite existing files. The
precommitment hashes original case bytes and canonical cases. The report binds
the observed response to its exact prompt SHA, plus the local model-file SHA.
Case and output JSON are bounded and reject duplicate keys/nonfinite values.
The serving identity check is *declarative* path matching, not verified process
mapping to live weight pages.

## Evidence classes and hard blockers

- **REAL_SYSTEM_PROBE** — HTTP 200, model text, response SHA, exact score, and
  local model-file hash observed from a live localhost service.
- **SOURCE/TEST** — unit-test mock server proves API and receipt semantics,
  not a learned model.
- **UNKNOWN** — independently secured case custody, signed execution witness,
  model-serving weight binding, training run provenance and protected semantics.
- **HOLD** — neural-learning and model graduation until authentic checkpoint
  lineage, matched baseline/zero/corrupted controls, independent evaluation
  and across-seed evidence are read back.

`OBSERVED_SINGLE_MODEL_ONLY` means no before/after pair.
`SAME_DECLARED_FILE_HASH_NOT_TRAINING_ATTESTED` means both receipts declare the same model-file SHA, but no runtime
weight pages or adapters are independently attested; **neither the absence nor
presence of a parameter update is proven**. `weight_update_status=UNKNOWN`
is explicit for both equal and different declared model-file hashes.
`PAIRED_BEHAVIOR_NOT_TRAINING_ATTESTED` means different local model-file
hashes and paired claimed scores (replayed only when source files are supplied), **not** proof the second model was trained
from the first. In all three cases `neural_learning_verified=false` and
`independent_custody_verified=false`.
