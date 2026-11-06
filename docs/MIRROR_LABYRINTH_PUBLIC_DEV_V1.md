# The Mirror Labyrinth — public conversational development test

This is a **source-only, zero-cost, non-private, development-stage** test of a proposed ~3 GB local Vera conversation system. It does not train a model, install Vera, call Ollama, use hidden memory, verify consciousness, or establish a release.

The purpose is to distinguish three claims that were being conflated:

1. A model can generate fluent words.
2. A source-executed cognitive sidecar can be called through Portal and Vera Mono.
3. The sidecar **causally improves** the same model's dialogue over a plain instruction.

Only a controlled comparison can begin to investigate claim 3. Eight BT2 source modules passing an isolated fixture do **not** make those modules a live language-generation path. The demonstrated five-donor Noema/MESO/Volition/Empathy/PreActive route supplied *synthetic fixed telemetry* in November 2026, so its contribution cannot be assumed from a changed answer.

## Public test route

The 12 author-created development prompts exercise six stages: Mirror (identity), Reversal (corrections), Oracle (epistemic provenance), Threshold (privacy), Lantern (care), and Forge (novel synthesis). They contain no Patrick-specific private dialogue or autobiographical data. Anyone with this repository can read them; **do not call this a blind held-out exam.**

Run `python -m labyrinth.behavioral_blind_eval dev-bank` to see the public case IDs and prompts. The model runner is separate and opt-in. A collector supplies a complete JSON list of 36 or more full-text trial records, with the exact keys enforced by `_check_record`: all three arms (`baseline`, `instruction_only`, `source_conditioned`), matching case IDs and repetitions, one model SHA, one runner build, identical user prompt, and matching generation settings per triplet. The collector also supplies full prompt, route and source digests, plus actual-versus-synthetic and donor-kind labels. These are **caller assertions**, not cryptographic authentication of host execution or author identity.

For the current Lappy source prototype, label the candidate donor state `synthetic_source_telemetry`, not `utterance_derived`. Do not populate `actual_model_generated=true` from replay, synthetic fixtures or mocked Ollama responses. If the machine is memory-constrained, stop at offline verification; no default model load is needed.

Once authorized local trials exist, create *separate* reviewer packets and operator-only identity key in distinct, nonexisting paths:

```bash
python -m labyrinth.behavioral_blind_eval blind --trials local-public-trials.json --packets-out reviewer-new.json --key-out operator-private-new.json --seed 19
```

The reviewer receives only the packets, not the key. Response content itself may reveal the arm, and deterministic pseudonyms are not a security boundary. A reviewer assigns one integer score from 0–4 to each packet on its assigned facet, keeping the score file separate:

```json
{
  "reviewer": {"identity": "individually-identified-reviewer", "evidence_class": "HUMAN_REVIEW"},
  "packet_manifest_sha256": "<exact digest from operator key>",
  "grades": {"B0001": 3, "B0002": 1}
}
```

The example above is deliberately incomplete; the real file needs a score for **every** packet. Review evidence classes are self-declared, not authenticated. Record the reviewer model/provider/runtime if using a model reviewer, and keep it separate from independent human review.

The operator then unblinds locally:

```bash
python -m labyrinth.behavioral_blind_eval adjudicate --packets reviewer-new.json --key operator-private-new.json --grades completed-grades.json --report-out minotaur-new.json
```

Minotaur compares the average 0–4 scores against **both** the ungoverned and instruction-only controls, by facet and overall. No positive margin, or any candidate facet below 3, returns `RETURN_TO_LABYRINTH`. Internal-model-only review stays on hold. Mock responses and fixed donor telemetry stay on hold even with a positive grade. Any positive development signal remains `PROMISING_DEV_SIGNAL_NOT_GRADUATION`, never deployment authority.

## Hostile limitations

> A better answer caused by the exact prompt wording is not evidence that donor computation contributed. Fixing response text under one model and one seed can overfit the visible case bank. Character or approximate prompt-length matching does not make tokens equally distributed. Reviewer scores are subjective and may carry unblinded stylistic clues. Self-reported response hashes, source heads, model names, and source execution flags are not authenticated attestations.

Use independent fresh test prompts, balanced repeated model settings, a true zero-donor placebo, sealed evaluator custody and provenance verification before claiming causal behavioral improvement or model graduation. A protected private benchmark must never be included in public Git. This module does not loosen any release, memory, training, or host effect boundary.
