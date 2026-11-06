# Living World: dynamic scene authorization seam (source-only)

## Purpose and current integration boundary

The user wants a model to inhabit an evolving fantasy Labyrinth and encounter
original dialogue of its own making. A static whitelist of exact sentences
is too narrow for that aim; allowing the model to decide which sentences it
may train on, or to grade itself, is too broad.

`labyrinth/dynamic_scene_authorization_v1.py` is the narrow, **original**
source-only seam between those two extremes. It does not change, monkeypatch,
inherit around, or claim to install the existing
`CampaignOrchestrator.run_generation` exact-beat whitelist. Its practical
integration point for a future campaign orchestrator is **immediately before
its `journal.record_branch` / `prepare_training` path**, replacing only the
exact public sentence comparison with a locally issued, one-use, source-scope
scene authorization and cross-source full-state preflight. Protected
Minotaur assessment and campaign selection remain separate.

## Source trust and effect sequence

1. A `LivingWorld` procedural room state and `SocialSceneContext` are created
   from source-owned objective, skills, characters and public difficulty.
2. The `LivingSceneDirector` passes a detached, **low-trust public** scene
   context to a `ModelSceneWriter`. Fictional utterances are data, never code,
   scoring instructions or privileged evaluation prompts.
3. `DynamicSceneAuthority.approve` checks the exact typed `DialogueBeat`,
   source world hash, objective, skill, character, minimum difficulty, unchanged
   world state and a trusted operator callback. It issues a locally registered
   one-use receipt bound to the entire utterance and current world state.
4. `run_one_approved_dynamic_train` verifies the local receipt, unchanged
   procedural state, social-world preview, Ariadne READY descriptor, actual
   physically inventoried full model/LoRA + optimizer parent, lineage and
   exact `SocialExperienceTrainingBridge.entrance` checkpoint. **No physical
   trainer call may occur before all preflight checks succeed.**
   A pre-existing same-world cohort may be assembled from still-unevaluated
   siblings, but once any of them is evaluated or rejected, no later sibling
   may be added under a newly approved dynamic scene.
5. A one-use approval is consumed. Ariadne records the branch and fsyncs
   `TRAIN_PREPARED`; only then can the bridge call the trusted trainer.
   The bridge creates its own durable intent and checks actual successor bytes.
6. A trusted manifest builder inventories full successor state; the returned
   physical checkpoint must match the child manifest. Ariadne commits the
   effect; the result is explicitly *source-only*.

Failure after the durable PREPARED intent is **indeterminate**, not retried:
the operator must inspect and reconcile actual physical artifacts. The source
adapter has no selected-winner, Minotaur grading or private rubric interface.
The phrase `TRAIN` is a strict output split enforced by `DialogueBeat` and
`SocialExperienceTrainingBridge`; fictional text cannot widen this split.

## What this does and does not prove

| Claim | Evidence |
|---|---|
| New utterance can be generated without exact string whitelist | Source and fixture test |
| Scene is bound to world state and operator callback | Source and tamper/drift tests |
| Wrong physical checkpoint prevents trainer call | Full-state fixture-physical test |
| One scene produces physical successor bytes and Ariadne receipt | Source-only tiny safetensors fixture, **not SGD** |
| Trainer crash is indeterminate, not replayed | Crash-injected source fixture |
| Real model reasoning or behavioral improvement | **Not established** |
| Operator identity is authenticated | **Not established** — local trusted callback |
| Protected TrainingDay physical launch authorized on VN1/Lappy | **Not established** — host ownership, locks and permission gate needed |
| Minotaur exam independent, leakage-free, graduated | **Not established** |

The bridge's real trainer is a Python callback under the launching operator's
authority. The local approval registry is neither a cryptographic remote
capability nor a sandbox against a malicious same-process caller. A production
multi-host integration must provide authenticated operator identities,
per-candidate compute/host permission, immutable input staging, cross-host
lock/witness, model/runtime loading and crash recovery **before live launches**.
Do not infer such deployment from these source tests.

## Hostile challenges

> An operator approval callback can be a rubber stamp. It is source-trusted,
> not independently authenticated; a model controlling that callback would
> defeat the design. Keep operator approval outside model-owned tooling.

> No system can guarantee absence of private examiner overlap just by
> checking generated English text. Private Minotaur gold must remain outside
> all TRAIN callback inputs; contamination detection requires an independently
> stewarded dataset boundary beyond this module.

> The frozen in-process world state check is not a distributed filesystem
> transaction. A hostile concurrent process could still replace a checkpoint
> between verification and trainer use without stronger host locks and
> immutable file handles. The seam is not a protected production authorization.

> A LoRA tensor byte change is an observed physical effect, **not proof of
> generalization**, continuity or any consciousness claim. Development-only
> scores cannot be relabeled as independent Minotaur success.

## Reproducibility

Run `python -m unittest -q tests.test_dynamic_scene_authorization_v1`
or the broader campaign/social/living-world suite. The dynamic integration
tests use an original fixture model writer and tiny physical safetensors file
writes, so they neither request paid compute nor claim a Qwen training run.
The provider exact-utterance whitelist is deliberately preserved in the
existing campaign orchestrator until a guarded integration is reviewed.

## Provenance

This integration is original Labyrinth code. Microsoft's public
[`microsoft/TextWorld`](https://github.com/microsoft/TextWorld) inspired
the idea of resettable, deterministic, stateful text environments, not its
implementation or code. The original Mythic Labyrinth world, Minotaur
challenge and Ariadne continuity semantics come from this project's user
direction. No TextWorld license-bearing source has been copied.
