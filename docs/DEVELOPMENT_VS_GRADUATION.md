# Development versus Graduation — Labyrinth

**Status:** Proposed source-level policy and acceptance contract. This document does not certify a deployed learner, an independent examination, or a graduation decision.

Labyrinth is intended as a regenerative developmental journey, not merely a benchmark generator. Its Greek-myth structure supplies an operational vocabulary: Daedalus builds the maze, Theseus develops through it, Ariadne preserves continuity, and the Minotaur tests the learned capability. **The maze resets. The learner does not.** The metaphor never substitutes for an implementation or evidence.

## Two distinct lanes

| Lane | Permitted work | Evidence label | Review requirement |
| --- | --- | --- | --- |
| **Local development** | Author objectives and challenge generators; run seeded toy experiments, controls, neural-training prototypes, regression tests and internal reviews; inspect and resume checkpoints; rebuild a maze after failure. | `DEV_ONLY` | Normal project tests and internal review are useful. An external independent reviewer, sealed holdout custodian, or external monotonic witness is **not a prerequisite** to run another local experiment. |
| **Blinded qualification and graduation** | Make an evidence-backed claim that a specific learner gained a specific transferable capability and may be promoted. | `QUALIFICATION_CANDIDATE`, then scoped `GRADUATED` only on a positive decision | Requires independently controlled novel examination, preregistered rubric, contamination checks, causal model-update evidence, external/rollback-resistant custody, reviewer provenance and an explicit decision. |

An unavailable independent examiner or witness **holds qualification/promotion**, not ordinary developer work. A local experiment must never quietly inherit a graduation label. Likewise, a second agent or model reviewing the same accessible artifacts may supply useful **internal model review**, but is not external custody or independently blinded evidence by itself.

## Seven-stage epic acceptance contract

1. **The Entrant's Oath — declare a capability.** Record a typed learning objective, supported domain, constituent skills, scope exclusions, success rubric, baselines and controls. Reject or explicitly classify unsupported goals as design proposals. A text declaration alone does not create an evaluator.
2. **Daedalus's Blueprint — build a relevant maze.** Generate progressively challenging, objective-linked training tasks with reproducible seeds, difficulty levels and coverage. Keep developer training and tuning partitions distinct from any sealed qualification set; prove generator outputs are well formed and actually scoreable.
3. **Theseus's Traverse — establish baseline and adapt.** Measure the learner before training, execute intended interventions and control arms, and retain scored before/after traces. For neural-learning claims, demonstrate that the identified model/checkpoint was actually loaded, updated and evaluated; a toy optimizer or changed metadata is labelled as such.
4. **Ariadne's Thread — retain the learner's continuity.** Write a reconstructible, append-oriented lineage for each stage: objective/rubric revision, generator revision, model/base/adapter identity and hashes, challenge seed and partition, weakness signature, learning intervention, evaluations, predecessor/successor checkpoint references and recovery instructions. The thread survives maze regeneration and excludes sealed exam answers. Its custody and integrity level are explicitly labelled; an ordinary local log is not an external immutable witness.
5. **The Broken Passage — regenerate after failure.** Diagnose which capability failed; preserve the learner's legitimate checkpoint and Ariadne lineage; rebuild a fresh, appropriately targeted *training* maze under bounded attempts. Verify that regenerated training examples differ as intended. Retire or quarantine contaminated qualification material. Avoid infinite re-runs that merely optimize against a familiar exam.
6. **The Minotaur's Center — attempt a blinded challenge.** When seeking qualification, freeze candidate model and rubric; let an independent custodian expose unseen tasks only for the planned examination. Assess transfer, adversarial robustness, retention and contamination under the objective's declared scope. A deterministic reroll, an internal adversarial reviewer or a privately repeated set is not independently novel evidence.
7. **The Exit — decide graduation or return.** Preserve all required receipts and decide only the narrowly tested capability. A passing score without proven causal adaptation, independent novelty, reproducible provenance or rollback-resistant authority yields `NOT_ESTABLISHED`/promotion hold, not `GRADUATED`. The learner can return to local development with the diagnosis and thread intact.

## Lifecycle and promotion rules

`OBJECTIVE_DECLARED → MAZE_READY → BASELINED → DEVELOPING ↔ REGENERATING → QUALIFICATION_CANDIDATE → {GRADUATED | NOT_ESTABLISHED}`

The qualification branch is optional: developers may iterate within `DEVELOPING` and `REGENERATING` without requesting promotion. A failed or unavailable formal gate returns a **scoped assessment**, not a general prohibition on development. A graduation receipt must bind the exact learner checkpoint, objective and rubric, unseen evaluation set custody, decision-maker, time, controls, acceptance thresholds and external integrity anchor. Any later learner revision starts a new qualification candidate; it does not inherit the previous outcome.

Minimum claim vocabulary:

| Claim | Minimum evidence | Must not be inferred from |
| --- | --- | --- |
| `IMPLEMENTED` | Exact source plus a reproducible passing test for the claimed path | README prose or a proposed interface |
| `TOY_EFFECT_OBSERVED` | Actual toy-run receipt, baseline, controls and measured outcomes | Test count alone |
| `NEURAL_UPDATE_OBSERVED` | Identified loaded weights/adapter, training step and before/after state plus evaluation | Changed checkpoint name, hash, or a toy numeric learner |
| `TRANSFER_OBSERVED` | Held-out domain-appropriate scoring with contamination and baseline controls | Improved in-distribution training score |
| `GRADUATED` | All relevant formal gates, independently blinded custody and rollback-resistant authority | Internal model consensus or green CI |

## Explicit falsifiers and known boundaries

> **Hostile challenge:** If a journey records changing checkpoint identifiers without loading and updating the claimed model, it demonstrates a *workflow transition*, not model learning.

> **Hostile challenge:** If failure regeneration gives the candidate another chance against the same leaked or predictable examination, it measures test familiarity, not independent generalization.

> **Hostile challenge:** If the final result is guarded only by rollback-susceptible local SQLite/HMAC state, a restored historical database can replay a prior standing. That custody cannot establish irreversible graduation authority.

> **Hostile challenge:** If a free-text goal such as empathy has no goal-specific generator, scored rubric and actual learner intervention, Labyrinth must report it as unsupported rather than treating the words as completed capability.

The existing offline arithmetic, operation-chain, evidence-priority and program-reasoning work supports meaningful developer experiments. Its toy parameter changes, identity-only handoffs and local custody do **not** automatically prove Vera's neural training or independent certification. Preserve these distinctions in every UI, JSON receipt, PR review and public claim.

No step here authorizes paid compute, shared-service restarts, examiner impersonation, secrets extraction, or bypassing independent gates. Patrick's **$0 budget** holds unless he explicitly approves spending.

## Related source contracts

- [Regenerative journey vision](REGENERATIVE_LABYRINTH_VISION_V1.md)
- [Experimental controls and pre-registration](EXPERIMENT_PROTOCOL.md)
- [Minotaur gate custody](MINOTAUR_GATE_CUSTODY_MAIN.md)
- [Public inference limits](MINOTAUR_PUBLIC_INFERENCE_MAIN.md)

This document is a specification for acceptance and evidence classification; source implementation, focused tests, runtime checks and independent review must each be reported separately.
