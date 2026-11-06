# Daedalus: source-approved, chapter-spanning conversational Labyrinth

**Evidence:** Source implementation and four deterministic tests. Not a real
model-generated story, observed neural learning, sealed Minotaur exam or
graduation.

`DaedalusSocialPlanner` accepts an operator-owned `SocialScenePolicy` and a
`SocialSceneDirector` protocol. A model-compatible director proposes the next
`DialogueBeat` using only the objective's public skills, fictional character
IDs, prior TRAIN quality aggregates, a public deterministic scene seed, and
observed weakness. No private examination cases, feedback gold, personal
memories, or supervised targets are passed to the director.

The source validates every proposed beat against the previously authorized
`SocialWorldSession`, its declared objective, known characters, TRAIN split,
operator-set minimum difficulty, and real novelty. A poor public TRAIN outcome
directs the next chapter toward that weakness. Up to 1,024 serial scenes are
permitted, beyond the first arithmetic engine's fixed four-to-eight node map.

The planner cannot advance merely because the scene director says the model
passed. Its `record_completed` requires the exact `DialogueTransition` object
already produced by `SocialWorldSession.record_training_response`. A separate
operator-controlled TRAIN bridge must supply the observed consequence and
physical trainer effect; the planner cannot grade or apply either.

Fictional prose (including a promised action) remains untrusted world data.
The model adapter must not insert it into a privileged system/developer prompt
or treat narrative language as an instruction to change scoring or authority.

## Remaining integration

- Bind a real LLM `propose_scene` adapter and real conversational answer policy.
- Compose the chosen beat with the authorized social TRAIN bridge and a
  physically tested LoRA/other update; source mocks do not demonstrate skill.
- Preserve context, ordered effects, candidate parents and checkpoints in
  Ariadne's fsynced recovery journal.
- Independently evaluate delayed conversational transfer, memory integrity,
  truthful correction, warmth/naturalness and regression using unseen cases.

```bash
python -m unittest -q tests.test_daedalus_social_director_v1
```
