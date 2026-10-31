# Labyrinth ⇄ Rezon — untrusted text-advice projection (source-only)

This experimental adapter consumes Rezon's **actual**
`CollectiveSnapshot.to_dict()` serialized shape and reproduces the exact
`CollectiveStore._snapshot()` input digest preimage. It is deliberately
located in `experiments/`; the training runtime does **not** import it.

Exact source identities examined:

| Repository | Draft PR | Exact Git head | Source tree |
| --- | --- | --- | --- |
| Labyrinth | #62 | `8e4615c715569ed4c35cf081b1a1acc13fd4f044` | `1d92e0f1ee313bcb901cc5073999af2d4a9d9495` |
| Rezon | #144 | `7dcb8b09c61c98c255e57ac0217250d821dfdd04` | `c8a3ef109649c04c08e8893968de967875ff03b7` |

The adapter accepts only a complete, current Rezon collective episode whose
participant, answer/review, synthesis-status and source digest fields match
the supplied exact pins. Historical syntheses may remain in the Rezon source
history but are excluded from the current advisory projection. It can hand
**unverified text suggestions** to a
human/development review queue. It never converts suggestions into labels,
optimizer steps, checkpoint ancestry, examinations, effect authority or
Minotaur graduation.

```bash
python -m unittest discover -s tests -p 'test_rezon_collective_projection_source.py' -v
```

> A caller able to rewrite the entire Rezon snapshot and its own expected pins
> can produce a new self-consistent digest. This adapter detects internal
> inconsistency, not authenticated source custody. It must not be connected to
> a protected examination, trainer update loop or host authorization path.

> Rezon `collective.py` executes independent whole-model **text** completions
> and model-written peer reviews. This is neither sharded single-model tensor
> execution nor distributed KV/gradient training. No claim of shared neural
> learning survives this interoperability test.

No host process, GPU, model checkpoint, credential or final exam was accessed.
