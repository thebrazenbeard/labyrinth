# R104 — Cognitive perspective objective TRAIN execution (source-only)

## Scope

`labyrinth.cognitive_objective_training_run.open_approved_objective_training_run`
accepts a canonical `ObjectiveContract` and current `JourneyState` in the
`TRAINING` phase. For the exact allowlisted `empathy` /
`("cognitive_empathy",)` objective with rubric `perspective_choice_train_v1`,
the factory selects `CognitiveObjectiveTrainingRun` and the fixed
`synthetic_cognitive_perspective_train_v1` provider. Arithmetic and program
reasoning retain their existing `ObjectiveTrainingRun` scoring path.

This separate A/B/C scoring path never parses `ANSWER: <integer>` responses.
Its operator-private 100/0 choice rater sees the source gold. Only
`TraineeRoomView` (room ID, skill, case ID, public vignette/options, prompt
digest) goes to an approved learner callback. This callback **must be trusted**:
Python frame/closure inspection or direct trainer-object mutation defeats
same-process secrecy. An untrusted learner requires separate process/host
isolation and independently held private gold. Feedback contains local TRAIN
correct/incorrect counts and sanitized observations; it is not an evaluator
score or a certified learning measurement.

## Bounded corridor and source replay exclusions

- A source-approved plan has at most 15 rooms, drawn from 30 unique scenario
  identities. Cycle 0 and cycle 1 use distinct halves. A third cycle fails
  closed when the source bank is exhausted.
- The constructor verifies prior `JourneyState` attempts against regenerated
  canonical case IDs and prompt commitments before exposing the next room.
  Current and sealed TRAIN/DEV IDs and prompt digests are excluded.
- `forbidden_semantic_sha256` accepts immutable, canonical operator-provided
  semantic digests; the source checks the supplied digest plus historical
  rewrite aliases. This operator input is **not** host-authenticated, and
  source normalization cannot detect arbitrary paraphrases.
- A pending prompt remains exposed if the callback fails. It must be scored
  once before opening the next room. Each room is revalidated against current
  canonical source bytes before opening, scoring, and completing a receipt.
- The response, returned views, observations, and completion receipts cannot
  mutate the primitive source ledger through `object.__setattr__`. The
  completion digest is pinned after the first successful receipt call.

## Authority boundaries

`authorizes_host_execution`, `authorizes_training`, `authorizes_exam`, and
`authorizes_graduation` are all **false**. A `TrainingCompletionReceipt` with a
changed checkpoint SHA-256 remains a self-reported source-level handoff. No
model weights are read or updated; no Q6 host exposure CAS, Q7 witness, Q8
restore, signed checkpoint, live GPU lease, private evaluator, or model
graduation is established by this implementation.

The synthetic development-failure receipt used in the test harness exists
solely to exercise `JourneyState` cycle regeneration. It is deliberately
unauthenticated and must not be confused with a real Minotaur exam.

## Required verification

`tests/test_cognitive_objective_training_run.py` covers the 15-room public
view/private scoring handoff; wrong choice semantics; callback failure;
separate 15-room regeneration; source bank exhaustion; source replay exclusion;
forged state, view, observation and receipt; and provider substitution.
The pinned neural CPU CI invokes this module by exact name and rejects any
test skip or count drift. Both PR and main-push path filters include source,
tests, and this qualification note.
