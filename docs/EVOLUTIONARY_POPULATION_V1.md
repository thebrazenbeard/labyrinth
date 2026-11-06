# Evolutionary population v1: development-only, source-first

`labyrinth.evolutionary_population_v1` implements a **comparative selection
policy**, not neural training or a genuine independent model examination. It
connects two independently useful components:

1. `campaign_full_state_v1` physically inventories the full HF safetensors
   base, tokenizer/config, adapter (for LoRA), optimizer, scheduler, RNG and
   data cursor. The write-once manifests have path-independent content state
   hashes, ancestry claims and fresh-process file readback.
2. An operator supplies an *untrusted local fixture* of independent-examiner
   measurements. The selector enforces a predeclared common suite, sealed
   cohort commitment, matching state hashes, examined parent and unchanged
   control, and multiobjective development gates.

## Candidate lineage contract

Prepare with `prepare_population(development_generation=...,
origin_manifest_path=..., parent_manifest_path=..., control_manifest_path=...,
candidates=(candidate_a, candidate_b), plan=...)`. Each `Candidate` supplies:

- Distinct bounded candidate ID, actual full-state manifest path and explicit
  `MutationKind.ADAPTER_DELTA` or `MutationKind.FULL_MODEL_EDIT`. The third enum
  value, `PROMPT_MEMORY`, deliberately fails the neural mutation contract.
- A trainer ID, a unique SHA-256 receipt *claim*, the SHA-256 of the claimed
  **pre-update parent state**, and
  `deterministic_seed(parent_state_sha256, development_generation, candidate_id)`.
  The deterministic seed controls only experimental candidate construction,
  **never** private exam case assignment or hidden scoring material.

There must be at least two non-identical siblings. Each manifest is separately
fresh-process rehashed against the **same parent**, W0 origin, matching HF
mutation family, and physically distinct changed LoRA adapter or full weights.
For LoRA, the entire base/tokenizer/config must be unchanged; for full-model
edits, tokenizer/config metadata must remain unchanged. The control is a
separate manifest of the unchanged complete parent state. No network/model
calls and no `eval`/`exec` of candidate Python occur.

**Material limitation:** A file manifest plus a string called
`trainer_receipt_sha256` does not authenticate a trainer, assert its initial
optimizer/RNG load, or prove gradient steps. A child can claim to descend
from W0 without actually starting there. This contract intentionally keeps
`qualification="SOURCE_ONLY_UNAUTHENTICATED_EXAMINER_FIXTURE"`.

## Comparative examination and developmental selection

The operator prepares an `EvaluationPlan` with an independent examiner ID,
suite SHA, sealed-case commitment, preregistration SHA, and operator-private
cohort assignment commitment. The cohort policy is exactly
`MATCHED_RANDOMIZED_PRIVATE_EXAMINER_ONLY`: matched difficulty with randomized
private assignment across siblings and controls, unseen to candidate/trainer.
The public deterministic Minotaur development seed is **not** such a blind
cohort. Repeated public seed evaluations risk benchmark overfit.

`select_generation(population, examinations)` requires one `Examination` for
the **parent**, one for the physically unchanged **control**, and one for each
candidate, each binding the same predeclared plan and the expected physical
full-state hash. Examiner run receipt claims must be distinct. The examiner
producer role must be `INDEPENDENT_DEV_EXAMINER`, distinct from all trainer
IDs; a learner or trainer claiming its own score is rejected.

Every `Metrics` vector contains bounded finite proportions:

| Objective | Selection default |
| --- | --- |
| Holdout | At least 0.01 above *both* parent and control |
| Transfer | At least 0.01 above *both* parent and control |
| Identity | At least 0.95 and no greater than 0.01 below stronger baseline |
| Retention | At least 0.95 and no greater than 0.01 below stronger baseline |
| Regression rate | At most 0.05, and no worse than *either* baseline |

No private cases, gold answers, prompts or case IDs are inputs to this module;
only operator-owned summary measurements and commitments cross the boundary.
Passed candidates are ranked deterministically by worst-gain margin, combined
holdout/transfer, combined identity/retention, regression and stable candidate
ID. If no candidate passes **all** gates, the parent physical manifest is
retained without promoted mutation. A winner advances the development
generation; the next population must be physically forked from that selected
state, not a blend of siblings or the W0 default.

`verify_selected_entry(population, decision, next_entry_path)` delegates to
`campaign_full_state_v1.verify_generation_ancestry`, which rehashes all sibling
bytes, selected champion and copied next entry in fresh Python interpreters.
It requires **exact full-state content identity**, including optimizer,
scheduler, RNG and cursor, from winner to new entry. The manifest's
`generation_index` is an ancestry/copy depth; it is **not** a Labyrinth maze
cycle number and need not equal `development_generation`.

## What the evidence does **not** establish

> **Hostile falsifier:** Any caller can manufacture the local `Examination`
> dataclasses, fake the independent producer role, claim a private randomized
> cohort, or choose flattering scores. Source checks can reject inconsistent
> claims, not authenticate independent blind evaluation. An honest status is
> `SOURCE_ONLY_SELECTION_POLICY_NOT_NEURAL_EVIDENCE` even after two selected
> generations and green tests.

To qualify actual evolutionary learning, a distinct trusted host must supply
an authenticated trainer pre-update state receipt, true model/optimizer
process-death restoration, physically verified candidate post-update bytes,
fresh *model inference* readback, and an operator-controlled sealed matched
holdout/transfer/identity/retention/regression examiner whose answers were
never trainer-visible. A signed graduation gate remains entirely outside this
module. A mocked saga `COMPLETED` WAL or local Python fixture is not proof.

The population does not claim pretrained Qwen provenance, host-qualified Qwen
loading, neural parameter updates, neural improvement, independent evaluator
authentication, life/agency, or production rollout. No paid services used.
