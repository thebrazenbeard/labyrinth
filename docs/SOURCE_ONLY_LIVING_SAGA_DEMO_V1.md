# Two-cycle Living Labyrinth demo

Run from the repo root on Linux/WSL (Ariadne's fsynced ledger requires it):

```bash
python -m labyrinth.source_only_saga_demo_v1
python -m unittest -q tests.test_source_only_saga_demo_v1
```

This self-contained demo creates a fresh temporary directory and deletes the
fixture files on normal exit. It does not use cloud compute, contact a model
endpoint, load a trained model, perform SGD, modify production checkpoints, or
operate VN1/Lappy. Everything called a model, trainer, examiner or operator
in this file is **explicitly scripted synthetic fixture behavior**.

## What actually happens

1. A source-owned maze is generated with public seed 71. A **fixture action
   agent** chooses each typed action from current public observation and a
   map it incrementally discovers; it has no prewritten room route. It must
   collect Ariadne's thread and seals, confront the Minotaur, and retrace the
   path back to the Atrium. A seeded second maze has different paths and more
   seals. Tests exercise 12 different seed/difficulty pairs.
2. A **fixture dialogue model** authors new dialogue about the current room.
   The local `DynamicSceneAuthority` and operator callback bind those words
   to the exact current living-world state, source objective, character and
   TRAIN split. No private examiner material is passed into that writer.
3. Each of two synthetic sibling candidates passes a cross-source Ariadne
   full-state parent preflight, then consumes its operator-approved scene.
   Two **fixture trainers** separately copy a small synthetic safetensors
   genome and edit adapter/optimizer bytes. Real file hashes, fsynced
   training intents and physically inventoried successor manifests are
   checked. **Byte changes are not model learning.**
4. The first maze's action agent finishes its public quest. A source-only
   population selector compares both children to an unchanged control and
   parent using *fabricated for demonstration* metrics. The schema requires
   an `Examination` record, but those records are synthetic, **not genuinely
   independent examiner outputs**. One synthetic winner is selected, and
   Ariadne binds its successor full-state manifest.
5. A separate caller-supplied DEVELOPMENT verdict starts a harder successor
   world carrying a short public memento, not neural memory. The model action
   fixture explores that new world and wins again. The result reports actual
   world ancestry hashes, action/outcome traces, selected checkpoint hashes,
   fake-selection labels and Ariadne's ready state.

## Honest integration boundary

The demo composes `LivingWorld`, `LivingSceneDirector`, the new dynamic TRAIN
authorization seam, `SocialExperienceTrainingBridge`, Ariadne's full-state
physical ledger, and `prepare_population`/`select_generation`. It does **not**
modify or call the existing `CampaignOrchestrator.run_generation`, whose exact
predefined-beat whitelist remains in force. It does not run a genuine
Minotaur, an externally sealed holdout, a protected TrainingDay host launcher,
or actual neural inference. Production adoption needs independent model
callbacks, authenticated human operator effects, runtime host permissions,
physical cross-host locking/recovery, and separate real examiner evidence.

> **Hostile challenge:** The synthetic selector accepts syntactically valid
> examination records even when their purported independent producer is a
> local fixture. This run is a deterministic contract/regression test, not
> evidence of independent Minotaur quality or an actual training improvement.

> **Hostile challenge:** The fixture action planner reads public difficulty
> to know how many seals exist. This is environment-rule knowledge, not private
> scoring gold; a genuine model still needs to learn or infer strategies.

This is original code inspired at a conceptual level by
[`microsoft/TextWorld`](https://github.com/microsoft/TextWorld). No TextWorld
implementation has been copied.
