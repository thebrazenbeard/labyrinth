# Labyrinth/Minotaur cross-training provenance bridge (V1 historical pins)

Status: source-only Labyrinth integration candidate and local Python unit tests; **not** an
installed service, physical learner probe,
independent examiner, host authorization, or graduation decision.

Use: `python -m unittest discover -s tests -p 'test_cross_training_provenance.py' -v` from the Labyrinth repository root.
No dependencies outside Python standard library.

The bridge pins three independent exact source identities:

| Campaign | Source head | Objective / model | Source relationship |
| --- | --- | --- | --- |
| `labyrinth-pr59-corpus-r16` | `1d38e4f25b5663bf8a84ff5e139875d4152e3680` | Frozen verified integer-chain **corpus**, no model | R16 corpus producer |
| `labyrinth-pr62-regenerative-minilm` | `b698ede24d9a69e551e1cad83dee83a825feb9a6` | Regenerative sentiment, MiniLM/AdamW | Independent campaign |
| `model-training-pr143-r18` | `a14118e87dbba9fd436ef2ae6ad04293966d4d8b` | Integer-chain, Qwen3.5-4B/MeZO-LoRA or QLoRA | Exactly consumes PR59 R16 corpus |

PR59/R18 pinned corpus manifest SHA256:
`90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836`.

Profiles fix source repository/head, corpus repository/head/manifest, bridge-local
task ID and control-set ID, actual model family/revision, objective family,
training method taxonomy and development-eval source. A validated envelope
binds a control arm, plan digest, baseline digest, optional result digest and
evaluation-custody digest where applicable. Its canonical digest fingerprints
the *metadata object*; it does not verify the referenced bytes or that model
weights really changed. Those require separate runtime evidence.

Only two explicit relations exist:

1. `R16_CORPUS_INPUT`: PR59 source metadata → R18 corpus input metadata.
2. `MATCHED_CONTROL_METADATA`: two control arms in the **same** campaign,
   method, plan, baseline, and eval-custody setting.

No relation implies model interchangeability, safe transfer, physical training,
private examination, authorization or graduation. Source metadata outright
rejects any flag claiming those authorities.

## Hostile limitations (must remain visible in integration)

> A user-supplied 64-character hex string is only digest-shaped; this module
> does not read a checkpoint or hash a live artifact. A forged envelope may be
> structurally valid. Integration must verify bytes against independent custody.

> `MATCHED_CONTROL_METADATA` compares declared context, not paired random seeds,
> data content, GPU numerical outcomes, score deltas, retention or forgetting.
> Claims of learning or generalization require the actual experimental record.

> Every source head above is pinned as observed during W26 on 2026-10-31.
> Moving upstream heads require a separately reviewed profile/version; do not
> silently retarget these source-only evidence objects.

This source-only module is packaged as `labyrinth/cross_training_provenance.py` with regression tests in `tests/test_cross_training_provenance.py`. A future model-training adapter may submit a typed envelope, but this integration validates only source-pinned metadata. It cannot attest host authority or satisfy any Q0–Q9 neural qualification gate. Historical PR62 pin `b698ede...` deliberately remains distinct from a later PR62 head: `25089c0...`; changing pins requires a separately reviewed schema/profile revision.
