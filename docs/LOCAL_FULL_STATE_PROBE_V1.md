# Local public model probes over physical Labyrinth genomes

**Status:** SOURCE-ONLY API/provenance interface; mock-tested, **not** a live
model run, not proof of model identity or neural improvement.

The existing `checkpoint_predictor_verification.py` already has its own
development predictor tests, CPU HF **adapter** inference, and an unverified
generic local service. Rather than replace that, this distinct bridge consumes
the newer **complete HF full-state population** from
`evolutionary_population_v1` and speaks native local-generation payloads.

## Supported $0 private-host services

| Backend | Strict loopback path | Model-server precondition |
| --- | --- | --- |
| KoboldCPP | `http://127.0.0.1:PORT/api/v1/generate` | Operator loads an authorized model supported by KoboldCPP |
| llama.cpp native | `http://127.0.0.1:PORT/completion` | Operator launches `llama-server` with intended model |
| OpenAI-compatible local | `http://127.0.0.1:PORT/v1/completions` | Operator starts an explicitly local compatible inference server |

All other schemes, remote hostnames/IPs, credentials, query tokens, alternate
paths, proxies and HTTP redirects are refused. No cloud API or paid compute
integration exists. The default HTTP code uses Python stdlib only.

**Crucial distinction:** Labyrinth's candidate manifests describe complete
HF safetensors genomes, whereas KoboldCPP/llama.cpp commonly consume GGUF.
The API adapter **does not convert** safetensors to GGUF, merge LoRA, launch a
server, identify a loaded model, or prove that a server used the intended
checkpoint. A quantization/merge is a new artifact with its own hash and must
be explicitly linked and evaluated. A localhost server that merely repeats a
claimed SHA cannot certify its actual weights. An operator could use an
external owned reload callback to reuse one port sequentially; it is still
an unsigned local observation, not a verified load attestation.

## Required inputs

`probe_local_full_state_population(...)` takes:

- A reverified `Population` with one actual W0 `parent`, a distinct physical
  unchanged `control`, and 2–8 physical W1 sibling full-state manifests.
- One `LocalProbeTarget` per subject with an exact loopback URL, backend and
  optional server model alias. Endpoints must be distinct unless the operator
  supplies an explicit `activate_target` reload callback.
- One shared tuple of `PublicPrompt(case_id, prompt)` entries (1–32, bounded,
  unique). There are **no expected answers, rubrics, private cases or
  evaluation records** in the transport payload.
- A precommitted, independently supplied `expected_public_suite_sha256`
  matching `public_probe_suite_sha256(probes)`.
- Explicit decoding configuration (`temperature=0`, `top_p=1`, bounded
  `max_tokens` and seed). These are deterministic *requests*, not a guarantee
  identical kernels/backends always return identical text.
- The optional trusted `transport` callback is for local fake transport
  tests or an operator-owned replacement, never generated untrusted code.

For each subject the adapter runs identical prompts, fresh-process rehashes
complete HF bytes before and after, records output and source hashes, and
revalidates all population state at the end. The receipt's classification is
always `PUBLIC_LOCALHOST_OUTPUT_UNVERIFIED_LOADED_MODEL`, and its
`model_bytes_loaded_proven` and `neural_improvement_proven` remain false.
No inference result is scored, self-certified, or promoted. Actual Minotaur
selection remains a separate operation with its own eligible evidence.

## Mock-only tests

`python -m unittest -q tests.test_local_full_state_probe_v1`

Tests build a physically distinct W0, W0 control, and two W1 candidates with
real changed F32 safetensors; inject a fake transport returning the three
supported API shapes; assert 8 **public** calls (2 prompts × 4 subjects),
matching configuration and source hashes, and no gold-bearing fields. Tests
reject remote URLs, uncommitted prompt cohorts, shared endpoints without
an operator reload callback, and oversized/malformed generation output.

## Honest live-work requirement

To graduate this from SOURCE-ONLY to a real local experiment, an operator
must identify an idle host and existing local model service, verify which
HF or GGUF weight artifact is loaded, authorize resource/time consumption,
run a matched public cohort, and independently observe service/backend logs.
That would establish an *observed local development probe*, **not** yet
causality, a generalization claim, private Minotaur fitness, or restoration.
No test here used VN1, VN2 or Lappy, and no host process was started.
