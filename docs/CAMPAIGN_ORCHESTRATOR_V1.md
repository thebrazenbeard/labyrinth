# Campaign orchestrator v1: source-only, social, consequential, recoverable

`labyrinth.campaign_orchestrator_v1` connects five distinct development
contracts without inventing a real model or independent blind examination:

1. `ConversationObjectiveProvider.public_train_beats` permits **TRAIN-only**
   Vera dialogue with no hidden reference claims in predictor input.
2. `DaedalusSocialPlanner` approves a scene proposed by an installed operator
   director. Its `SocialSceneContext` uses public prior-world feedback only.
3. `SocialExperienceTrainingBridge` presents that scene as untrusted user data,
   gets a bounded learner response and an operator-owned TRAIN assessment,
   fsyncs a beat intent, invokes a trusted physical trainer, records real bytes
   and applies consequent character trust/promise changes to the next scene.
4. `AriadneJournal` records a **separate, outer** TRAIN_PREPARED effect permit
   before the bridge reaches its trainer callback. After TRAIN, it checks the
   complete physically captured candidate state and commits the candidate.
5. `evolutionary_population_v1` constructs an unchanged control plus two or
   more sibling manifests, and accepts **caller-supplied** development-only
   `Examination` records for comparative Minotaur selection. Only then does
   Ariadne persist rejected/selected fates and the selected social world.

The runnable example is the fully executable
`tests/test_campaign_orchestrator_v1.py`. It uses fictional approved social
cases, a scripted scene director, a scripted dialogue responder, an operator
assessment, actual writes of tiny syntactically valid HF safetensors fixtures,
and clearly named untrusted **fixture** examination records. These are *not*
model inference, SGD, proof that any optimizer was loaded, private case
evaluation, authentic Qwen weights, or independent Minotaur generalization.

## Public API and irreversible-effect order

Create a fresh `AriadneJournal`, a complete W0 manifest, a compatible
`TrainingScopeSpec`, a `ConversationObjectiveProvider`, and initial fictional
`CharacterRelationship` state. Construct `CampaignOrchestrator` once; it
records a physical W0 `GENESIS` and source-scoped world. Then:

```python
pending = orchestrator.run_generation(
    launches=(candidate_left, candidate_right),
    control_manifest_path=unchanged_parent_control_manifest,
    evaluation_plan=operator_predeclared_plan,
)
# Stop here for separately obtained, caller-owned development records.
result = orchestrator.finalize_generation(pending, external_development_examinations)
```

Each `CandidateLaunch` holds a unique ID, TRAIN case/character/region/chapter,
installed scene director, responder, operator, physical trainer, trusted
`manifest_builder`, fresh bridge journal directory, trainer ID and beat count.
The installed Python adapters are **trusted operator code**, never code emitted
by the learner or supplied in a fictional character utterance. A model-backed
director may propose only one of the public TRAIN beats independently
whitelisted by the operator's conversation provider. The response and
operator feedback never receive evaluation gold.

For each sibling:

| Boundary | Durable proof class |
| --- | --- |
| Ariadne `BRANCH_EXPERIENCE` | Declared operator-selected public TRAIN path |
| Ariadne `TRAIN_PREPARED` | Fsynced intent and one-use same-process effect permit |
| Bridge `beat-N.intent.json` | Fsynced possible-effect intent before trainer callback |
| Trainer writes a successor | Actual local file mutation, **not** neural training proof |
| Bridge `beat-N.commit.json` | File hash and public character effect only |
| Complete HF genome capture | Base/tokenizer/config/adapter/optimizer/scheduler/RNG/cursor bytes |
| Ariadne `TRAIN_COMMITTED` | Complete manifest readback linked to live effect permit |
| Population selection | Source-only comparison to examined parent and control |
| Ariadne SELECTED/rejected | Durable selection chain, no independent fitness authority |

The orchestrator additionally verifies that the bridge's actual final
checkpoint file paths and SHA-256 values appear *inside* the child's full
genome inventory. Thus an unrelated mock checkpoint cannot be paired with a
different claimed HF candidate manifest. A candidate's pre-update optimizer
and actual SGD remain unauthenticated despite this consistency check.

For this narrow, source-only bridge, the selected adapter (or model weight
file) and optimizer must each be represented by one physical file; the full
genome manifest still covers all base shards and other state. A real sharded
Qwen/optimizer loader, complete process-death restore, and model inference
driver are **not implemented**. The orchestrator does not open, download,
execute or hot-swap arbitrary models.

`AriadneJournal` uses Unix `fcntl` file locking and POSIX fsync/link semantics.
This composition is **Linux/WSL-only**; native Windows has no qualified
journal backend and must hard-hold. A Windows `D:\VERA` project root alone
does not establish cross-platform runtime durability.

## Two generations, world continuity, and failure behavior

The first fixture generation forks W0 into `g1_left` and `g1_right`, applies
consequential public Vera TRAIN, and selects the right sibling from externally
*supplied source-only fixture metrics*. The next generation forks from that
exact selected complete state and inherits only the winner's character
history. The test reverses the selected side in generation two, verifies the
new social context, checks complete file ancestry, and fresh-process reopens
the Ariadne event chain. The maze cycle index, development selection ordinal,
and physical manifest capture depth are independent notions.

> **Hostile falsifier:** The `Examination` type and sealed cohort commitments
> are local caller-provided strings and metrics. A test can fabricate them.
> The orchestrator itself never manufactures examiner records or calls a
> trainer to grade itself; source consistency is not authenticated independent
> evaluation. Two fixture selections cannot establish neural improvement.

When a trainer throws after writing physical bytes, Ariadne remains in
`INDETERMINATE_TRAINER_EFFECT`, and the bridge retains a pending fsynced beat
intent. Neither journal automatically retries or replays the potentially
consumed effect. The orchestrator holds a blocked state on any partial failure
and provides **no automatic process-death TRAIN resume**. An examiner failure
likewise prevents selection; no failed examiner is converted into a passing
result. If no candidate passes, the population retains the old parent, while
no Ariadne `CANDIDATE_SELECTED` is emitted; the next attempt remains on that
parent rather than pretending to advance neural generations.
The operator may separately record an approved harder world/curriculum in
Ariadne before retry, but this module does not invent harder scenes: the
provider's allowed TRAIN beats and a stronger director policy must be
approved coherently first.

All outcomes are labeled
`SOURCE_ONLY_SOCIAL_CAMPAIGN_NO_NEURAL_OR_INDEPENDENT_EXAM_PROOF`.
Model loading, optimizer restore, authenticated trainer effects, operator-held
randomized blind exams, anti-rollback witnesses, and formal graduation remain
separate unsatisfied gates. No paid service or private gold is used.
