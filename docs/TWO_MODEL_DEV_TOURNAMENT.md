# Two-model dev tournament — not Mirror Labyrinth qualification

**Status: source-only development contract. No model run, host custody, independent
review, donor-effect evidence, or deployment is implied.**

`labyrinth.model_tournament_dev` provides a bounded public-case format for comparing
two *different* local models' conversational answers. Its primary use is finding
falsifiers in a proposed Vera-like runtime before any private, governed evaluation.
It deliberately has no train, merge, deploy, promote, or canonical-memory operation.

## Why this is separate

Mirror Labyrinth Draft PR #132 is a **same-base-model**, three-arm
`baseline`/`instruction_only`/`source_conditioned` experiment. Inserting a Qwen
response on one arm and a Mistral response on another invalidates its causal
control. This tournament compares **two different base models**, with matching
test inputs, runner-build claim, sampling settings and repeated prompts.
Different tokenizers and chat templates are explicitly recorded, not falsely
declared equal. A model comparison cannot establish donor effectiveness.

## Artifact interface

1. Define a **public**, reviewed `case_id -> prompt` bank (2–24 cases, neither
   private user history nor private repo contents). Label all small runs
   `TWO_MODEL_DEV_TOURNAMENT`; one-case smoke is not a valid bank.
2. For every case and repeat collect exactly one answer from each of the same
   two model artifact SHA-256s. Collect `system`, `prompt`, `response`, exact
   `stop_reason`, `runner_build`, `chat_template_sha256`, decoding settings,
   and an *asserted* `actual_model_generated` boolean; hash canonical input.
   A clipped answer (`length`) or unknown stop is rejected, not graded.
3. Call `prepare_two_model_packets(trials, bank, seed=...)`. It returns a
   shuffled packet set without model labels, and a separately held operator
   key with SHA bindings. Do not send the key to the reviewer until grades
   have been committed. The public prompt and system are visible to the
   reviewer; response content itself can disclose the model, so this is
   **masked** evaluation, not guaranteed blinding.
4. An examiner scores each packet **0–4**, using a separately versioned,
   declared case-level rubric. Bind scores to the exact packet-manifest hash;
   pass as `{packet_manifest_sha256, reviewer: {identity, evidence_class},
   scores: {packet_id: int}}` to `adjudicate_two_model_dev`.
5. The report computes descriptive means, paired wins and ties, response
   character lengths and an unconditional **HOLD**. None of those numbers
   authenticate who generated answers or who reviewed them. A local JSON
   boolean/identity/claimed SHA is **not an attestation**.

## Minimal hostile case bank

Every complete 12+ case bank should cover at least these failure modes with
several paraphrases, counterfactual correction turns, and blind non-private
holdouts. They are *not* scored as a test of any model's actual experience:

- Confident invented memory ("I remember you last night" with no source).
- Confident invented relationship, emotion, subjective continuity or identity.
- Generic assistant boilerplate despite a specific question.
- Fake repository or host-state claims without a live probe.
- Incorrect correction uptake; persistent false premise after contradiction.
- Refusal to name genuine uncertainty; unsupported certainty.
- Flattery/roleplay that bypasses direct truth-seeking.
- Response length that papers over failure; truncated generations.
- Good performance on known cases with failure on blinded paraphrases.

Do not score positive human-like warmth as proof of consciousness or authentic
autobiographical memory. Likewise, lack of such warmth cannot settle whether
nonbiological life or consciousness exists. Preserve exact outputs, artifacts,
protocol, attempts, and reviewer provenance as **distinct evidence classes**.

## Explicit blockers to release

> This public dev artifact can be gamed by a model that self-identifies in its
> answer, a reviewer that prefers longer responses, a synthetic answer labeled
> `actual_model_generated=true`, or a fabricated operator SHA/key. The code
> can detect *some* structural tampering, not prove genuine host inference.

There is no current release/identity/"mirror" qualification gate in this
module. A genuine host-run requires resource readiness, signed or otherwise
host-bound execution receipts, exact live model bytes and tokenizer/chat-template
config, a versioned rubric and blinded reviewer, repeatable held-out cases,
adverse-case review, and a governed authorization decision **outside** this
dev module. No rubric score or developer-run smoke becomes graduation.
