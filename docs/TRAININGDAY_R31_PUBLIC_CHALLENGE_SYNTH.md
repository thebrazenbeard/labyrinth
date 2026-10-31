# TRAININGDAY R31 — Objective-conditioned PUBLIC TOY exercise content (HOLD)

`labyrinth.r31_public_challenge_synth` consumes a validated R29
`PublicRegenerationReceipt`, its formal objective, and any required externally
supplied predecessor chain. Unlike R28/R29's generated **case identities only**,
R31 materializes small, deterministic **actual public prompt text** for three
explicit skill families:

- `arithmetic`: two- or three-step bounded integer expressions.
- `source_priority`: a fictional versioned record versus an undated recap,
  with the versioned toy record *declared* higher priority.
- `program_reasoning`: execute three ordered arithmetic operations.

Only supported formal skill IDs are accepted. Unknown objectives, malformed
R29 commitments, absent G2+ ancestry, and oversized case groups fail closed.
R31 also computes a separate canonical **semantic** fingerprint from the
actual operation or conflicting source labels, excluding the R29 case-ID
wrapper. All development and challenge cases share one uniqueness set;
deterministic bounded resampling rejects repeated semantic tasks rather than
confusing new case IDs with new questions. Exhausting the bounded content
domain fails closed.
The declared shortcut category changes question structure; public template
families, frozen objective, generation, seed and case IDs change each question's
bounded operands and wording. The content is deterministic and reproducible,
not sampled from an unreleased private examination.

The `development` split includes public practice answers; the `challenge`
split stores no expected answer. `serialize_public_challenge_jsonl()` requires
the exact contract, R29 receipt and regenerated content, validates the bundle,
and serializes only a whitelist of prompt-only fields from freshly regenerated
private strings. It cannot preserve or expose a private answer because **there
is no private answer bank in this module**. This is a public test/demo; an
answer can of course be computed by a reader of the prompt.

The two hash domains must not be confused:

1. `plan_prompt_identity_sha256` is R29's synthetic case identifier commitment.
2. `prompt_sha256` is SHA-256 over R31's actual UTF-8 prompt bytes.
3. `semantic_sha256` commits the underlying public question semantics without
   treating a new case ID or paraphrase as an independent task.

These are intentionally not interchangeable. `content_sha256` commits the
caller-declared public bundle, including development answers and answer-free
challenge prompts, but does **not** authenticate a source, rights, examiner,
or human consent. Hashes can be recomputed by whoever controls all inputs.

### Local verification

```bash
python -B -m unittest -q tests.test_r31_public_challenge_synth
python -B -m unittest discover -s tests -q
git diff --check
```

The verification API returns only `PUBLIC_TOY_CONTENT_CONSISTENT_ONLY` or
`PUBLIC_TOY_CONTENT_INVALID`. Every training/graduation/promotion/independent
exam authorization is false. There is no Qwen3.5-4B model load, actual neural
weight update, private/untouched OOD bank, genuine hidden oracle, host lease,
independent rights attestation or deployable Minotaur graduation qualification.
This module is a tiny public corridor generator intended for future governed
Labyrinth task adapters; it is not one of those adapters yet.
