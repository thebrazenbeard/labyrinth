# Reverified Labyrinth chronicle and public discovery map

The existing source-only saga is **not** a neural model running the maze.
`FictionalActionModel` is a deterministic map-discovery algorithm;
`FictionalDialogueModel` uses a scripted context-aware sentence template;
the trainer edits small toy safetensors bytes without SGD; and the purported
examinations are fabricated development fixtures. The world **does** enforce
real typed actions and consequences, source-owned seeded paths, Ariadne
lineage, and full-state fixture checkpoint inventory.

This small replay tool lets a person inspect what actually happened, without
claiming any of those scripted components were genuine model inference.

## Commands (Linux/WSL; no paid services)

```bash
# Generate a fresh disposable fixture saga, reexecute public actions,
# and print an original Markdown chronicle with Mermaid maps.
python -m labyrinth.saga_replay_v1

# Print a verifiable machine-readable public replay instead.
python -m labyrinth.saga_replay_v1 --format json

# If the existing saga JSON was previously redirected to a file:
python -m labyrinth.saga_replay_v1 --input /path/to/saga-report.json

# Human steps through reverified public narrative one event at a time.
python -m labyrinth.saga_replay_v1 --interactive

python -m unittest -q tests.test_saga_replay_v1
```

The `--input` form does not retrain or write checkpoints: it reads a bounded
source report and **reexecutes** every public `WorldAction`, compares each
recorded result to the real `LivingWorld` transition, verifies the final
world/state hash for both generations, and verifies the declared public
world-ancestry/memento relationship. It rejects changed actions, outcomes,
world hashes, completion claims and inflated evidence-class labels.

The rendered Mermaid maps contain only **traversed** public rooms/edges.
They are reconstructed from verified events, not a hidden examiner map.
Narrative sentences are original prose generated from each checked action and
world consequence. The source report retained only training scene hashes,
not dialogue words, so this tool does **not** invent a verbatim exchange.

> **Limit:** Public source replay is repeatable, not proof of neural model
> ability. A model could be plugged into `ModelWorldActor` later, but none is
> called here. A compact, loopback-only real-model adapter with strict public
> JSON and a bounded retry budget is a separate integration step.

> **Limit:** The replay does not reopen temporary physical training files
> after a saga run; it reports Ariadne and candidate digest fields that the
> synthetic run produced. Persistent forensic custody is a separate job.
