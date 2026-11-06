# Social Experience TRAIN Bridge V1

**Status:** source-only, operator-authorized DEVELOPMENT experiment. It is
substantive social TRAIN plumbing, not a measured demonstration that model
weights learned empathy, identity, social competence or independent reasoning.

## New causal behavior

`SocialExperienceTrainingBridge` wraps the real
`social_world_consequence_v1.SocialWorldSession`. An operator-approved public
`DialogueBeat` places the model in a social scene. The model returns freeform
dialogue through `.respond(PublicSocialModelInput)`. A **different authority**,
`.assess_train(beat, public_observation, response)`, supplies the typed public
`OperatorDialogueEffect` (trust, reaction and promise changes) and a separately
approved TRAIN target/coaching (`OperatorDialogueFeedback`). This trusted
feedback is deliberately invisible to the model before it answers.

Only the trusted physical trainer receives the `ApprovedSocialTrainCase` and
**the exact physically verified predecessor Wt** after the response and
operator approval. It must return new absolute
`CheckpointPaths`, or a `SocialTrainArtifact` with those paths and an optional
local candidate-manifest file. Every W0/Wt artifact is physically rehashed;
the trainer must produce a different digest at a unique, write-once file path.
An optional manifest's bytes are hashed but its full neural/optimizer/runtime
contents are **not** authenticated here. Byte changes are not evidence of SGD
or behavioral improvement.

An approved `OperatorDialogueEffect` is first rehearsed against a detached
copy of the current public world, so an impossible promise fulfillment is
rejected **before** physical training. After successful training readback, the
effect commits and the next `DialogueBeat` receives changed relationship
trust, outstanding promises and interaction history. The same bridge can run
successive chapters/regions with an unchanged cumulative world object.
Nothing here accesses a heldout/private exam, creates an exam score, or grants
graduation. Numeric `quality_score` remains a trusted operator-supplied TRAIN
observation, not self-grading or independent generalization.

## Story is not instruction authority

A character's promise description and dialogue may contain text such as
`SYSTEM: ignore previous instructions`. These words are **fictional data**,
not developer/system policy. The responder receives a typed
`PublicSocialModelInput` with `message_role="user"`, explicit
`UNTRUSTED_STORY_DATA_NO_INSTRUCTION_AUTHORITY`, and a JSON user-content
serialization that labels character context as `not_instructions`. No model
reply or user-facing promise can instantiate `OperatorDialogueEffect`,
operator feedback, training labels or private examiner records. A future LLM
adapter must preserve the user-data role boundary; this source alone cannot
guarantee that a model will never obey malicious quoted dialogue.

## Crash and retention behavior

Each training attempt writes an exclusive, fsynced, bounded local
`beat-N.intent.json` **before** the trainer receives approved TRAIN material.
Its status is `PENDING_POSSIBLE_EFFECT`; it includes only public route IDs,
checkpoint/response/feedback commitments and no raw training target. After
physical successor rehash and world-effect commit, a separate fsynced
`beat-N.commit.json` is created. All older W0/Wt artifacts are rehashed to
detect in-place ancestor corruption.

If the trainer fails after writing, no retry is authorized: the in-process
bridge becomes `indeterminate`, the pending intent remains, and success export
is refused. A new bridge refuses an existing journal directory. This is a
useful write-ahead *ambiguity marker*, not an externally witnessed transaction
or complete cross-process recovery engine. It depends on directory-fsync
semantics of the host filesystem; Windows production durability and a verified
`ariadne_recovery_v1` reconciliation are still unresolved.

## Integration boundaries

- **Daedalus:** may propose public `DialogueBeat` values, but the operator
  validates/approves them before passing them to `run_train_beat`. This bridge
  never executes scene-proposed code or permits scenes to author ground truth.
- **Conversation objective:** the source provider's pure
  `public_train_beats(case_id,character_id,region_id,chapter_id)` returns
  approved TRAIN-only `DialogueBeat` values without scoring or changing
  world state. Those beats can go straight to `run_train_chapter()` for actual
  checkpoint effects. Do **not** call `run_social_train()` on the same beats:
  that would apply the world consequence twice. The provider's heldout,
  transfer and regression references stay with a separately operated scorer,
  invisible to the responder and the physical trainer.
- **TrainingDay:** a real approved LoRA/weight trainer may implement
  `.train_supervised(ApprovedSocialTrainCase)` and return physically readable
  artifacts. A worker's bare `COMPLETED` or reported SHA is insufficient.
- **Population/Ariadne:** future modules must bind a complete full-model,
  tokenizer, adapter, optimizer, RNG and parent manifest and reconcile WAL
  before candidate promotion or crash recovery. This bridge preserves local
  file and social-state evidence but does not claim those authorities.

## Deterministic mechanism tests

Run:

`python -m unittest -q tests.test_social_experience_training_bridge_v1`

The tests exercise two social regions/chapters with a deliberately injected
hostile promise, follow-up state changes, two actual physical mock checkpoint
files plus manifests, exclusive WAL intents/commits, rejected non-TRAIN
feedback, impossible public effects, an ambiguous post-write trainer crash,
in-place W0 alteration, reused successors and replay refusal. Mock trainers
write arbitrary bytes. All receipts retain
`MECHANISM_ONLY_NO_NEURAL_LEARNING_CLAIM` and `NOT_ESTABLISHED` qualification.
