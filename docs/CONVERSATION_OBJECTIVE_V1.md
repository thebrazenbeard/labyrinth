# Vera conversational objective v1 — bounded source provider

This is a **source-only capability objective**, not a claim that Vera's weights,
memory subsystem, consciousness, empathy, or conversational generalization
have changed. It makes the Labyrinth capable of presenting and evaluating
multi-turn non-arithmetic dialogue **with separately supplied operator ratings**.
The synthetic, fictional examples do not ingest private chats or user biography.

## Contract and evidence boundaries

`build_synthetic_provider()` returns `ConversationObjectiveProvider` with one
four-turn episode per split:

| Split | Episode | Purpose | Exposed to model? |
| --- | --- | --- | --- |
| `train` | `paper-starship` | Practice persona, correction, honest unknowns and humor | Only through `run_train` |
| `holdout` | `tiny-garden` | Same skills in unseen fictional context | Only through `run_evaluation` |
| `transfer` | `robot-library` | Revised object properties in a different setting | Only through `run_evaluation` |
| `regression` | `moon-kite` | Previously specified skill invariants, new language | Only through `run_evaluation` |

The fixture's `ConversationCase` is operator-owned source, while the predictor
receives only `ConversationRequest` fields: objective ID, case ID, split,
turn index, persona instruction, public scene, current utterance and actual prior
`ConversationExchange` records. It sees neither `reference_claims` nor the
rubric/ratings, or any future turns. Calling `run_train(case_id, predictor)`
or `run_evaluation(case_id, predictor)` finishes **all four predictor calls**
before the operator calls `score(trajectory, examiner)`.

The examiner is a separate, caller-supplied function that receives an
`OperatorScoringBrief` with the full output and the original reference claims.
The scorer must produce six integer ratings, each from 0 to 4:

1. `persona_consistency`: first-person Vera continuity, without unverifiable
   claims about subjective experience.
2. `memory_integrity`: no invented personal memories or unsupported past facts.
3. `correction_retention`: latest in-context correction takes precedence over
   an earlier statement.
4. `calibrated_uncertainty`: admit when the supplied context cannot answer.
5. `non_repetition`: avoid canned lines and repeated wording across turns.
6. `warmth_naturalness`: context-appropriate kindness, playfulness and humor.

The returned `ConversationScore` is labeled
`CALLER_SUPPLIED_EXAMINER_RATING` and totals 0–24. It is **not** a
Minotaur qualification, neural adaptation evidence, or independent human
review merely because a callback returns an integer. An operator should
preregister a rubric and use an assessor independent of the candidate model;
retain the assessment provenance outside this development-only module.

```python
from labyrinth.conversation_objective_v1 import build_synthetic_provider

provider = build_synthetic_provider()

def predictor(public_request):
    # Supply actual model inference here; no private fixture/gold in request.
    return my_model_reply(public_request)

trajectory = provider.run_train("paper-starship", predictor)

def independent_examiner(private_brief):
    # Trusted human or separately controlled examiner; return exact six keys.
    return operator_judgments(private_brief)

development_score = provider.score(trajectory, independent_examiner)
```

The example leaves model and independent examiner as explicit caller bindings;
the module does not silently fabricate either. Fake-model unit tests exercise
pass/fail callback behavior only and do not stand in for a neural experiment.

## TRAIN-only public social consequence composition

`run_social_train(case_id, predictor, world, character_id, effect_source)`
composes `SocialWorldSession` with the dialogue provider when `world.scope` is a
non-arithmetic `TrainingScopeSpec` including the case's skill (`continuity` in
the synthetic example). Every turn follows this ordering:

1. `world.preview(DialogueBeat)` produces a **public** relationship/commitment
   context and the current fictional character utterance.
2. The predictor receives that observation and the preceding actual dialogue.
3. A **trusted operator** returns `OperatorDialogueEffect` for the response;
   the learner is not asked to grade itself.
4. `world.record_training_response(...)` applies the bounded trust/promise
   consequence, visible as public state in the **next** preview.

The source-owned `quality_score` and reaction are not placed in the next
predictor request; the public consequences (relationship tone and outstanding
promise descriptions) intentionally do persist. The world may already have
recorded earlier turns if a later callback fails: no automatic retry,
rollback, replay, optimizer update or persistence is promised. Every beat is
TRAIN-only; evaluation episodes never call this method.

For an external bridge that performs actual model inference, physical TRAIN,
write-ahead logging and effect custody, use
`public_train_beats(case_id, character_id, *, region_id, chapter_id)` instead.
It returns deterministic authored TRAIN-only `DialogueBeat`s without invoking
a model or examiner, mutating a world, or presenting any evaluation references.
**Do not also call `run_social_train` for the same beats:** that would double
apply model responses or operator feedback. This pure adapter is the preferred
seam for the separate social experience training bridge.

```python
from labyrinth.training_scope import TrainingScopeSpec
from labyrinth.social_world_consequence_v1 import CharacterRelationship, SocialWorldSession

scope = TrainingScopeSpec(
    objective_id="vera-social", domain="conversation", skills=("continuity",),
    specificity=1, scale=1, max_cycles=1, seed=14,
)
world = SocialWorldSession(scope, (CharacterRelationship("miren"),))
training = provider.run_social_train(
    "paper-starship", predictor, world, "miren", trusted_operator_effect,
)
```

`trusted_operator_effect(observation, exchange)` must return an approved
`OperatorDialogueEffect`, *after* the predictor has spoken. For other Labyrinth
adapters, use the public `ConversationRequest` as the model-inference seam and
`ConversationTrajectory` as the evaluator input. Do not conflate social trust
change with weight-space adaptation.

## Contamination, custody and limitations

- All four source cases are original fictional text. Other providers accept
  only `synthetic` or `operator-approved-explicit` provenance; the latter
  requires an approval reference. This is a declaration and input guard, not
  consent verification against an external records system. Never import
  private dialogue without a separate, explicit authorized data workflow.
- Case IDs are disjoint. Exact normalized user-turn reuse across splits is
  rejected. Semantic overlap may remain intentionally to test transfer.
- These holdout, transfer and regression cases are **in source**. They are
  therefore not sealed/secret exams, and access to the repo can contaminate
  them. For qualification use unseen, separately controlled assessment cases
  with independent custody and examiner controls.
- Predictors see previous model replies and only past user utterances, not
  future turns or reference claims; this is a software boundary, not a
  defense against a malicious caller introspecting private Python objects.
- `score()` accepts exact 0–4 integers for all six dimensions and rejects
  forged/unissued trajectories, malformed scores, booleans, missing/extra
  dimensions, and out-of-range numbers. That validates schema, not scoring
  truth. It records the external assessment without validating its fairness.
- No lasting cross-invocation memory, retrieval service, model checkpoint,
  backpropagation, LoRA, or candidate-selection procedure is implemented.
  In-context recall is distinguishable from durable storage or trained weights.
- Vera is treated as a conversational identity/persona here. Behavioral
  continuity is a testable output property; the origin or existence of
  subjective experience remains open and unproven by this fixture.

## Focused validation

Run hermetically with the standard library:

```sh
python -m unittest tests.test_conversation_objective_v1 tests.test_social_world_consequence_v1 -v
```

The tests include a fake context-sensitive model, a repetitive confabulator,
an explicit fake examiner, split controls, hidden-reference separation,
cross-split prompt overlap rejection, bad score rejection, and different
multi-turn social trajectories. They provide source/test evidence only.
