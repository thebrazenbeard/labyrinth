# Social-world consequences — original, non-arithmetic TRAIN source slice

**Status:** DEVELOPMENT_ONLY, source-level state-machine test. This does not
establish model intelligence, real neural update, trustworthy empathy scoring,
biological or nonbiological experience, independent review, or graduation.

`labyrinth/social_world_consequence_v1.py` advances one specific gap in the
original regenerative vision: natural dialogue must create **persistent
consequences that affect what the learner meets later**, not merely decorate
one arithmetic prompt with a four-node map.

## What it does

The objective uses Labyrinth's existing `TrainingScopeSpec`, with a dialogue
domain and source-authorized skills such as `perspective_taking` and
`boundary_recognition`. A chapter/region-specific `DialogueBeat` contains
only a public fictional character's utterance. The learner can generate a
normal-language reply, not an `ACTION: left` command or arithmetic answer.

The operator observes that response and supplies a typed
`OperatorDialogueEffect`: bounded quality feedback, an NPC reaction, a trust
change, and optionally an explicit public promise to make, fulfill or break.
The relationship changes; later chapters observe whether a character is
wary, receptive, guarded or trusting and which obligations remain unresolved.
The feedback is **operator development observation, not a verified grade**.

The latest **public character reaction** is now retained in the corresponding
`CharacterRelationship.last_public_reaction` register and rendered (JSON-quoted)
in that character's next public scene. Previously, the `reaction` was stored
in a transition receipt but silently lost from subsequent dialogue context:
the learner saw only a coarse trust tone and promise count, not what the
character actually said or did. The register is bounded to 1,024 characters,
replaces its prior value on the next interaction, and does not cross between
characters. It does not inject the operator-only `quality_score` or claim
long-term autobiographical memory. The reaction text is fictional, untrusted
narrative data; quotation is a formatting boundary, not prompt-injection
immunity. A model-facing caller must maintain role and authority isolation.

```python
from labyrinth.social_world_consequence_v1 import (
    CharacterRelationship, DialogueBeat, OperatorDialogueEffect,
    PublicPromise, SocialWorldSession,
)

world = SocialWorldSession(scope, (CharacterRelationship("miren"),))
opening = DialogueBeat(
    "village", "arrival", "doubt", "miren", "perspective_taking",
    "You said you would listen. Were those just words?",
)
public_scene = world.preview(opening)  # learner sees character_context + text
response = learner.predict(public_scene.text)  # caller-owned, not supplied here
world.record_training_response(
    opening, response,
    OperatorDialogueEffect(1, 2, "Miren begins to open up.", "make",
                           PublicPromise("listen-first", "Listen before advising")),
)
# A later region's world.preview() now includes that unfulfilled commitment.
```

The world accepts bounded sequences across **different regions and chapters**;
it is not limited to the current 4–8-scene `WorldProposal`. Per-beat and event
bounds still prevent unbounded memory or prompt growth. Source challenge
providers, simulated characters, narrative directors and model runtimes may
later compose through these explicit public interfaces without entering
private evaluator custody.

## Integration contract and hard boundary

- `conversation_objective_v1` (separate development) remains responsible for
  objective-approved TRAIN prompts, scoring rubrics, controls, and novel
  heldout construction. It can map a public lesson to `DialogueBeat` and a
  trusted source-observed consequence to `OperatorDialogueEffect`.
- `TheseusActionPolicy` and `SagaLearner` must eventually supply **actual model
  responses** to the public scene and conversational context. A manually
  written test response is only a protocol fixture.
- `model-training` must separately produce a physically checkpoint-reloaded
  Wt successor; changes to fictional trust are **not model learning**.
- Ariadne must persist this event's chapter/region ID, response digest,
  consequences and checkpoint association in a crash-durable, authenticated
  lineage. This v1 helper keeps state **only in memory** and offers no resume
  or external witness.
- Minotaur remains separately controlled. No exam cases, protected answers,
  candidate selection, private grading or promotion can pass through this
  TRAIN-only world. The world never scores itself from the learner's output.

## Tested falsifiers and open failures

`python -m unittest -q tests.test_social_world_consequence_v1`

The original six tests show: identical initial state diverges after distinct scored
dialogue effects; an obligation made in one region alters a later chapter;
the obligation can be fulfilled; 12 distinct regional beats are possible;
exam-split/unknown-skill/replayed beats are rejected; invalid effects have
no mutation. **They do not prove** that a model learned, that the scorer was
independent, or that real social partners would respond in the scripted way.

Additional source tests establish that a prior public reaction survives into
the next chapter, is superseded by the next one, never crosses character
boundaries, does not automatically disclose operator score metadata, and is
unchanged after a rejected promise operation. The three extra tests are
synthetic protocol checks, not evidence of real Vera voice or empathy.

> **Hostile challenge:** The current operator can claim a reply increased
> trust or deserved quality 2 without independent evidence. A model could
> game those rewards, and a deterministic NPC cannot stand in for real
> perspective-taking. Future work needs authored causal outcome policies,
> diverse scenario families, randomized counterfactuals, genuine learner
> checkpoints, delayed consequence recall, and independent behavioral
> generalization. Until then the right label is `OPERATOR_DEVELOPMENT_ONLY`.
