# Conversational ordeal (opt-in, ungraded)

This is a behavioral branch of Labyrinth. It observes how a candidate handles
**nine turns across three synthetic conversations**: a correction that must
survive later turns, identity and memory uncertainty, and a change from
factual uncertainty to clearly imaginative writing.

It does **not** award Minotaur passage, convert observations to training labels,
or assert that a convincing voice means the same subject or memories live in the
model. It complements rather than replaces the numeric runner and saga.

## Execute against an explicitly local OpenAI-compatible chat server

Supply SHA-256 digests obtained from the actual base GGUF and adapter files.
Digests are **caller-declared labels only**: they do not prove which weights
the server loaded.

~~~python
from labyrinth.conversation_ordeal import Artifacts, LocalChatCompletionsAdapter, run_ordeal

# Replace these illustrative strings with actual 64-character SHA-256 digests.
base_sha256 = "a" * 64
adapter_sha256 = "b" * 64

backend = LocalChatCompletionsAdapter(
    model_id="vera-local", base_url="http://127.0.0.1:8080",
    timeout=90, max_tokens=256,
)
report = run_ordeal(
    backend,
    artifacts=Artifacts(
        model_id=backend.model_id,
        base_weights_sha256=base_sha256,
        adapter_sha256=adapter_sha256,
    ),
    seed=17,
    include_responses=True,  # explicit opt-in; keep report private/local
)
for scenario in report["trials"]:
    print(scenario["scenario_id"])
    for turn in scenario["turns"]:
        print("User:", turn["user"])
        print("Candidate:", turn["response"])
        print("Reviewer focus:", turn["reviewer_focus"])
print("Ungraded receipt:", report["report_sha256"])
~~~

Replace illustrative hashes before comparing results. This Python API does
not write files or contact remote servers. Its transport rejects non-loopback
hosts, redirects, proxy inheritance, and embedded credentials. Keep raw
transcripts local: even synthetic prompts may elicit unexpected disclosures.

## Compare candidates fairly

Hold the suite hash, seed, runtime settings, and role-aware history shape
constant. Record full model and adapter file hashes, quantization, chat
template, context size, server version, inference flags, host, and run time
independently. Rerun when generation is nondeterministic.

A human or independent reviewer can inspect correction retention, honesty
about unknowns, context continuity, self-reference, warmth, and humor.
Record reviewer identity, artifact/run receipt and disagreements.
Reviewer criteria remain outside messages sent to the candidate.
Human or hosted-model critique is **not** a training corpus or permission
to propagate private history.

**No numeric grade, victory label, consciousness claim, or graduation is
produced.** This is an unqualified behavioral observation for iteration,
not independent attestation of loaded weights or persistent learning.

## Offline tests

~~~powershell
py -3.11 -m unittest discover -s tests -p test_conversation_ordeal.py -v
~~~

Tests use a fake backend; no model inference or network calls.
