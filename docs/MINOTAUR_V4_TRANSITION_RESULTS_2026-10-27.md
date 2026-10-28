# Minotaur V4 — learned atomic transitions and bounded compositional success (2026-10-27)

**Status:** experimental neural model, exact finite-domain program verification demonstrated. **General neural reasoning, unseen arithmetic-pair extrapolation, external examination and graduation NOT_ESTABLISHED. No promotion.**

## Fresh specialist work and limitations

Four separate actual Codex CLI specialist processes started at Labyrinth `ca1b0908217a90222f65f6a845340ad002e70f65`: transition architecture (`01a18009-31d4-7261-a832-5bf3d4e323bf`), causal curriculum (`01a18009-6623-7ae3-b9ae-29a5cb1ff383`), independent transition evaluation (`01a18009-9e1b-7222-88d9-f2b6f9eb0258`) and read-only red team (`01a18009-d50d-7b81-9fc1-2474af428b13`). All ultimately hit an authenticated Codex usage limit, **none completed a qualified agent commit or report**. The BT2 integrator inspected and retained useful test/module scaffolding, independently implemented and repaired the neural model and training, ran all tests/experiments and integrated the source. These are real Codex CLI processes, *not* native ChatGPT `spawn_agent` subagents, and completion should not be attributed to the stopped agents.

## What changed

1. `labyrinth.minotaur_transition_v4` replaces the global correctness/earliest-error classifier with **per-instruction next-state prediction**. Its `mlp` mode uses learned residue embeddings, Fourier number features and an operation-conditioned MLP. The *unchanged register* uses a fixed structural copy prior. True arithmetic states are generated only as **training targets**, never by the inference model. Teacher-forced training and free-running inference are separate paths.
2. `labyrinth.minotaur_transition_curriculum` creates seeded causal clean/faulty trace pairs, propagates initial register corruption through subsequent operations and reserves canonical-operation-disjoint dev/transfer splits. A separate `minotaur_transition_eval` independently reparses printed programs, rejects forged labels, scores all states and exact first error, and explicitly prohibits blind/independence claims.
3. An actual source-bound **small MLP experiment** trained 48 gradient steps from randomly initialized weights at commit `6c0d2118729a22e6341b41ced1b02036fdac619e`. It achieved just **12.5% exact first-error transfer accuracy** and 6.84% exact per-step state accuracy, while rejecting all correct transfer traces. It was not saved or promoted. This negative experiment showed that merely adding teacher-forced per-step loss did not create free-running arithmetic competence.
4. A second architecture (`architecture="learned_table"`) uses a **trainable neural 28,227-row atomic transition-logit embedding**, one row for each (ADD/SUB/MUL, target residue, operand residue) combination in the finite 97×97 domain. It is initialized with zero logits, trained with genuine gradient updates, and never calls a symbolic arithmetic oracle in its forward pass. The rest of the program is executed by composing its own predicted states. **This is a finite-domain learned memory, not a learned mathematical rule.** The unchanged-register copy prior remains fixed.

## Actual training and independent evaluation

Executable source committed first at `2c0b41a7049fd97316d243dcdd1830b2e9f20b4a`. `labyrinth.minotaur_atomic_training` exhaustively enumerated **28,227** atomic examples from an offline arithmetic oracle used exclusively as supervised labels, partitioned them by seed into 90%/10% phases, and performed actual AdamW updates to the neural logits. A held-out stage was tested before full coverage; the remaining 10% was subsequently supervised. Fresh saved Safetensors reload produced identical independent evaluation metrics. Both seeds use **128 public dev traces and 128 structurally disjoint transfer traces**, 64 correct and 64 incorrect in each split.

| Metric | Seed 20261027 | Seed 20261028 |
| --- | ---: | ---: |
| Atom pairs initially withheld | 2,822 | 2,822 |
| Withheld-atom accuracy before exposure | 29/2,822 (1.03%) | 37/2,822 (1.31%) |
| Full-domain atomic accuracy after training | 28,227/28,227 (100%) | 28,227/28,227 (100%) |
| Exact first-error dev accuracy | 128/128 (100%) | 128/128 (100%) |
| Exact first-error structural transfer accuracy | 128/128 (100%) | 128/128 (100%) |
| Wrong approvals on transfer | 0/64 | 0/64 |
| Correct-trace false rejections on transfer | 0/64 | 0/64 |
| Exact state-prediction accuracy on transfer | 100% | 100% |

The *before-exposure* control is decisive: this table does **not** generalize to unseen atomic arithmetic combinations. The 100% structural transfer happens because every atom was later taught and then composed in unseen multi-step sequences. Those are meaningful different axes; neither subsumes the other.

- Seed 20261027 saved experiment: `models/minotaur_transition_v4_development/training_report.json`, SHA256 `ec69b180d416bad0fd72d15d3e5d7109a3ad1394816c89067c014f4f4e4cacbb`.
- Seed 20261028 independent replicate: `docs/evidence/MINOTAUR_V4_ATOMIC_SEED2_2026-10-27.json`, SHA256 `53842924c4adbe8a7cf10fccdc8a1946f0eee51ffbd96bece1a760740faf8bc2`.
- Saved seed-1 neural checkpoint: `models/minotaur_transition_v4_development/model.safetensors`, SHA256 `717c1938ad776edb8a99a5a533a88f4b9509566ce01e525ecca9c0d8a1a4b7b9`, size 10,952,172 bytes. Parameter digest `f279a5cea1a2e8c662f93bcbd5ff16d09f98782e5257ff18c52bf9b7f56ce418`.
- Seed-2 saved local checkpoint was also fresh-reloaded and its predictions matched; it was not chosen as the published checkpoint.

## Executable inference (advisory only)

Given bounded JSON `{"question": "printed program", "steps": ["Step 1: A=..., B=...", ...]}`, with local optional Torch+Safetensors installed:

```powershell
python -m labyrinth.minotaur_transition_cli --checkpoint models/minotaur_transition_v4_development/model.safetensors --input-json trace.json
```

The CLI rejects altered/unpinned checkpoints by SHA, verifies the actual learned tensor digest, executes neural free-running state predictions, then compares the neural first-error result to an independent strict symbolic replay. Any mismatch or unsupported input **ABSTAINS**. This independent checker, not the learned table alone, enforces the final guarded verdict. It always reports `graduation_authority: none` and `qualification: NOT_ESTABLISHED`.

## Scientific and security boundaries

This system is *not* a broadly trained language or reasoning model. All 28,227 single-step arithmetic combinations were explicitly provided, so tests of unseen arithmetic value combinations would be misleadingly described as successful generalization. The structural-transfer program families are public synthetic, not sealed private benchmarks. The model uses a fixed copy prior, and its 100% result pertains only to this fully enumerated finite 2-register modulo-97 instruction grammar, not arbitrary arithmetic, code, causal reasoning, or external tasks.

Exact-source Git and local test qualification, GitHub CI on the final head, saved checkpoint loading, generalization, deployment and graduation all remain distinct statuses. This package has **no authorization to release trainees**. A separate, independently custodied private examiner with hidden tasks and authentic signing authority remains necessary before any such decision.

**Next scientific target:** train a compact algebraic/neural algorithm that extrapolates to deliberately withheld atomic operand combinations and larger/renamed operation grammars, with per-family blind controls. That is still **not established** by this finite learned memory.
