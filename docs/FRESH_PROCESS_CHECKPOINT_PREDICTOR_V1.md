# W0 versus Wt fresh-process predictor — DEVELOPMENT_ONLY

**Status:** optional, bounded, local-source experimental harness. This does not
train Qwen, show a neural gradient step, authenticate a remote model service,
attest independent evaluator custody, or grant graduation.

## What the code actually verifies

`labyrinth.checkpoint_predictor_verification.compare_fresh_process_checkpoints`
requires a **write-once EntranceState** manifest from
`labyrinth.entrance_state_v1`, an existing successor adapter directory, an
explicit operator-pinned Wt state SHA-256, and 1–32 matched public
`devprobe-*` arithmetic prompts and expected answers. It does not silently
select a changed file or accept a privately sealed final examination.

The verifier rehashes **every base shard** and associated model
configuration/tokenizer file and both complete adapter file inventories,
not merely one `model.safetensors` or a claimed model ID. It starts a **new
Python process for W0 and another for Wt**, using unique nonces, and pins the
worker's exact source-code SHA. Each worker independently rehashes all
referenced source bytes before and after inference. The parent rejects stale
nonce, reused/current PID, changed worker source, wrong Wt pin, mutated
model/adapter bytes, reused probes or unexpected answers.

The same public probes, scorer, and generation contract are used at both
checkpoints. Results include the checkpoint state digests, process IDs,
source digest, each prediction and score, and scoped score delta. A measured
public gain is **not** a claim that training caused it; other controls,
provenance and later fresh transfer are still needed. In particular, a
counterexample test loads two different checkpoint byte strings, returns only
their SHA as a predictor output, and correctly reports **zero skill gain**.

The parent scorer retains the `expected` answer values. The predictor
subprocess receives only development case identity, skill, prompt and prompt
hash; it explicitly rejects score/gold fields even if called directly. This
separation prevents accidental answer-key exposure through the fresh-process
inference seam. It is **not** an independent blinded examiner: these remain
public developmental probes and parent/worker source are controlled locally.

## Explicit predictor modes

| Mode | Uses genuine local checkpoint bytes? | Claim boundary |
| --- | --- | --- |
| `mock-local-file-arithmetic` | Yes, tiny deterministic deliberately non-neural test bytes | Validates fresh-process plumbing only |
| `mock-bytes-digest-only` | Yes, but never solves a prompt | Proves that changed weights/hash alone cannot establish learning |
| `hf-local-file` | Optional `transformers` + `peft` local CPU checkpoint loader; `local_files_only=True`, no remote code | A real local inference observation **if dependencies and full checkpoint are available**, not currently exercised on Qwen |
| `localhost-self-attested` | Loopback-only API, with no-proxy client and exact state-hash echo | **Unverified service self-report**; it might serve stale/different weights despite echoing the expected hash |

File loaders run with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and proxy
environment variables removed. Localhost accepts only `http://127.0.0.1:PORT`
or `http://[::1]:PORT` (`/infer` optional); redirects, userinfo, remote hosts,
HTTPS and `file://` are rejected. Its internal service contract is a bounded
JSON POST containing `checkpoint_state_sha256` and `prompt`, returning
`checkpoint_state_sha256` and `answer`. Do not treat the service's hash echo
as independent evidence that its loaded model really matches those bytes.

The actual Qwen3.5 pretrained snapshot on Lappy may contain multiple
`.safetensors` shards, an index and additional tokenizer files. This source
loader demands a compatible complete local `EntranceState` inventory, **not
just the first shard**; an unsupported layout must fail rather than silently
hashing one file. The optional HF inference path has not been executed with
official Qwen weights, so no neural model improvement is claimed.

## Spark measurement and the saga

The real goal is a short, genuine gradient/LoRA **Spark** in a TRAIN chamber:
an exact pre/post tensor digest, completed optimizer step, training loss,
finite gradient evidence, checkpoint/optimizer/scheduler/RNG availability,
and lineage to that chamber's approved prompt. The result must be saved and
reloaded before comparison. A before/after hash alone proves neither a
gradient step nor improved skill. Independently scheduled fresh *development*
probes at W0 and Wt may then detect a narrow ability change. The protected
private Minotaur graduation gate remains separate.

This harness is the inference-measurement seam after Spark execution; it does
not generate, apply or attest any Spark.

## Host split: VN1 coordination, Lappy heavy worker

For Patrick's $0 architecture, **VN1 is the Labyrinth coordinator**. Lappy is
the worker for memory/GPU-heavy model loading and inference, addressed over
an already authorized LAN/Tailscale transport. VN2 is not the coordinator.
This code contains no transport or host orchestration and performs **no host
probe/deployment**. A future authenticated job envelope should pin:

- Objective/scope, exact job ID, coordinator-issued nonce and expiration.
- Full W0 EntranceState manifest digest, Wt adapter/base inventory digest,
  exact local model path bindings (mapped only by the Lappy worker), source
  commit and predictor-worker SHA, probe commitment/partition and budget.
- Worker response signed by its host identity with loaded file digests,
  actual process ID/start, source hash, ordered outputs/scores, elapsed time,
  error status, and no private evaluator material.
- VN1 verifies signature, request/job nonce, replay registry and receipt
  lineage; independent source custody must not be inferred merely from
  Lappy's local SHA report. No new tunnel or paid compute is authorized here.

The worker's local PID and nonce offer replay resistance *within this run*;
they are not a remotely verifiable process attestation. GitHub CI on fake
test weights only proves harness source behavior, not Lappy or VN1 runtime.

## Reproduction

```bash
python -m unittest -q tests.test_checkpoint_predictor_verification
```

The tests use fake `.safetensors` file bytes and short subprocess calls;
they make no external connections, create no actual LLM or optimizer,
download nothing and never inspect private graduation material.
