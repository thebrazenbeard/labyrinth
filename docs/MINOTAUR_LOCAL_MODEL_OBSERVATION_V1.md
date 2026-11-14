# Wave39–41: local model invocation evidence for blind structural exams

**Scope:** bounded existing local Ollama inference, answer-withheld public questions, pinned response-file readback. **No model-weight attestation, independent examiner custody, or neural learning certificate.**

The blind mechanics scorer in Wave38 accepts candidate responses as data. Without another record, there is no way to distinguish actual local model text from a hand-written file. This tool adds an optional localhost HTTP observation and an independently saved hash *value* to the scoring workflow. It does **not** turn an untrusted operator into a trusted witness.

## How to run

Prepare the public and private exam on the examiner side using [the blind mechanics protocol](MINOTAUR_BLIND_STRUCTURAL_EVALUATOR_V1.md). Save the precommit digest outside the model/candidate process. The model runner receives **only the public questions path** and does not take a private key file, private case path, answer-key field, or reference to the expected verdict.

On a host with an **already running** Ollama HTTP service bound to a literal `127.0.0.1` port, use:

```powershell
python -m labyrinth.local_neural_observer --public C:/exam/public.json --responses-out C:/exam/model-responses.json --observation-out C:/exam/model-http-observation.json --endpoint http://127.0.0.1:11572 --model qwen3.5:0.8b --timeout 30 --max-tokens 96
```

The runner does not start the service, download or update any weights, access non-loopback endpoints, use a proxy, follow redirects, or send private answers. For **new V2 receipts**, it issues `GET /api/tags` once initially and immediately **before and after each `POST /api/generate`**, a total of `1 + 2 * trial_count` digest observations. This detects advertised model changes between cases, even when the server restores its original tag by the end of the examination. Each generation uses `stream=false`, `think=false`, `num_gpu=0`, `temperature=0`, `seed=17`, bounded tokens, and `keep_alive=0s`. Requests are only sent to the literal IPv4 loopback address; hostnames and credentials in URLs are rejected.

Generation must have `done=true` **and** `done_reason=stop`. An incomplete or token-cap-terminated reply (`done_reason=length`) is rejected even if its visible text happens to parse as a valid winning plan. The tool never scores a partial model invocation as a completed action sequence.

The tool refuses changed model name, invalid/truncated JSON, changed declared model tag digest, missing completion, extra private-answer/public fields, malformed public world or duplicate case IDs. It writes two **exclusive-create files**: the original raw candidate responses in the unchanged blind exam response schema, and a separate receipt with the original public-file hash, exact response-file hash, SHA-256 of every prompt and model response, local timing and the *server-advertised* model tag/digest. The receipt is itself unkeyed SHA-256; retain its printed `receipt_sha256` outside the candidate process before scoring. If the HTTP exchange fails, no score or success record is generated; a filesystem write failure may leave one partial artifact, not a qualified trial.

To replay those exact responses with the original retained observation digest:

```powershell
python -m labyrinth.blind_structural_exam score --private C:/exam/private.json --public C:/exam/public.json --precommit C:/exam/precommit.json --responses C:/exam/model-responses.json --expected-precommit-sha256 ORIGINAL_PRECOMMIT_SHA256 --observation C:/exam/model-http-observation.json --expected-observation-sha256 ORIGINAL_OBSERVATION_SHA256 --out C:/exam/model-score.json
```

When supplied together, scorer checks actual raw file bytes and trial-level response/prompt hashes against the original observation and precommit. Its output adds:

- `invocation_evidence=LOCAL_HTTP_SELF_REPORTED_NOT_ATTESTED` for new V2 receipts
- `observed_model_digest` and `observed_model_tag` from the local server
- `observation_receipt_sha256` of the exact verified observation

The receipt is versioned. New observations use strict `labyrinth.blind-model-local-http.v2` and record `model_tag_check_policy=INITIAL_AND_PER_TRIAL_BEFORE_AFTER` and an exact integer check count. Historical **V1** receipts remain readable using their original schema, file bytes, signatures and pinned hashes, but are explicitly downgraded to `invocation_evidence=LOCAL_HTTP_V1_FINAL_ONLY_NOT_ATTESTED` rather than receiving V2's never-executed per-trial assurances. V1 replies that actually ended `done_reason=length` are **not admissible as completed model observations**. This nonpromotion preserves the valid prior Qwen trial while refusing unsafe truncated historical output.

When no observation is supplied, `invocation_evidence=NOT_SUPPLIED`, leaving historical no-observation score paths intact. Partial observation flags fail closed. No signature, key or independent-custody assertion is inferred from local HTTP.

## Adversarial limitations

A caller who controls the local Ollama service can send a fake model response and report an arbitrary model digest. `GET /api/tags` is not an independent memory-page or adapter hash. **Even per-trial before/after tag checks cannot detect a model that is swapped within a single generation request and restored before the next tag read**, nor a server that simply lies about its model tag. A malicious caller who can replace both evidence files *and* supply a different original hash defeats self-attestation. A local observation, even with a signed case-set precommit, is **not** a signed model execution transcript. To graduate, an examiner other than the candidate/operator must independently preserve the case/key and authentic model-serving receipts, and matched before/after controls must have verified training lineage.

Reproduce:

```bash
python -m unittest discover -s tests -p 'test_local_neural_observer.py' -q
python -O -m unittest discover -s tests -p 'test_local_neural_observer.py' -q
python -m unittest discover -s tests -q
```
