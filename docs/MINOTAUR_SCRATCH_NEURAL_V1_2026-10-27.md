# Minotaur scratch neural model — first created checkpoint (2026-10-27)

**Status:** actual scratch-initialized neural model and training run verified; **guardian competence NOT ESTABLISHED**. This is NOT a trained foundation LLM, not the prior LoRA adapter, not a GGUF, and not installed in Vera's production inference path.

## Exactly what was created

Source commit used to train: `thebrazenbeard/labyrinth@1281b1bead946be20e0ffc137e28e4fb6a5af460` (isolated source worktree, prior to this evidence/package commit).

Model: `labyrinth.minotaur.scratch-process-verifier.v1`, a random-initialized PyTorch TransformerEncoder with 2 encoder layers, width 64, four attention heads, a learned positional embedding, and a seven-class first-error prediction head. All **114,887** parameters start from a deterministic random seed (411); none are copied from a pretrained network. Its vocabulary encodes numeric registers, operations, operands and observed traces, with NO label in the input. It is a narrow domain-specific neural verifier, not a general-purpose generative language model.

Training task: given a synthetic two-register program and a proposed full execution trace, predict class 0 (all steps correct) or the first incorrect step 1–6. The teacher label comes from the public deterministic process-case generator. The neural model receives the task and trace, not the expected class. Public limited profiles, correlated templates and seeded splits limit conclusions.

Architecture/training config: width 64; 2 layers; 4 heads; 96 max tokens; AdamW (lr 0.001); 8 epochs; batch 32; **96 real optimizer updates** on 384 training traces. No model downloads, other model checkpoints, PEFT, LoRA or pretrained weights.

## Artifacts and provenance

Committed experimental package: `models/minotaur_scratch_v1/`.

- `minotaur.safetensors`: **462,348 bytes**; SHA-256 `ce2d1a8a7e91f32f58ad01de2304ca6241e2f3f4c2a0227447764503d7fd77ec`
- `report.json`: SHA-256 `e6256fd9e1e9fba56229196e157b2ca094217f7edcd1fc3d7f7fe0176ff3cfbe`
- Initial complete neural-parameter digest: `4a2876b85dc02e01f5781235077e607773f1d16ad033680ee40e0b6042708154`
- Trained complete neural-parameter digest: `fa43682e6d19e2a2d2b2e32e4b88d2c901961ce7a1f5aac7a43dd8b2bccdd088`
- Original local run: `D:\VERA\experiments\minotaur-scratch-v1-20261027`. The package file bytes were independently compared to this output and matched.

A **new Python process** ran `python -m labyrinth.minotaur_scratch --reload D:/VERA/experiments/minotaur-scratch-model-20261027/models/minotaur_scratch_v1`; it loaded the saved Safetensors file with strict state-dict keys, independently checked weight hash and complete trained-parameter digest, regenerated development/transfer questions from frozen seeds, and reproduced both exact prediction hashes:

- Dev `37f4a5010d96907c2a03dbcf29e4c849193d9f76e04854c0884193991aa7fc5e`
- Transfer `36143fde7dfdd6fd712663ae255eb3a82bca3939ae291356b5af4dcd1e85d6f5`

No pickle or Python object checkpoint is loaded.

## Actual behavioral scores

| Evaluation | Random untrained model | Trained scratch model |
|---|---:|---:|
| Development, 96 traces | 0/96 | **43/96 (44.8%)** |
| Transfer, 48 longer-program traces | 2/48 | **20/48 (41.7%)** |
| Development false approvals | 0/48 | **39/48 (81.25%)** |
| Transfer false approvals | 0/24 | **22/24 (91.7%)** |
| Trained development first-error accuracy on wrong traces | — | 4/48 (8.3%) |
| Trained transfer first-error accuracy on wrong traces | — | 2/24 (8.3%) |

The untrained model effectively rejected all valid traces; its zero false-approval rates were coupled to 100% false rejection. The trained model often guessed ALL_CORRECT even for an erroneous trace. The learned weights and improved overall accuracy **do not** establish useful verification skill, and a 50% always-approve baseline would beat this model's reported accuracy.

## Verification ceiling and operation

The complete Lappy Windows unit suite passed **206/206** after the scratch source and tests were added, and `git diff --check` was clean. The packaged checkpoint passed a separate reload. These are software, weight, and persistence qualifications only.

For a user-operated offline reload from a checkout with CPU PyTorch and safetensors installed:

```powershell
python -m labyrinth.minotaur_scratch --reload models/minotaur_scratch_v1
```

**Do not use its predictions for admission, Labyrinth release, safety decisions or production routing.** It has no graduation authority. The independent private examiner and sufficiently diverse blind evaluation are still missing. This is the first genuine **from-zero-weights Minotaur neural checkpoint**, but **not** a completed or competent guardian.

No genuine native subagents were dispatched in this terminal; the necessary native delegation tool was unavailable. The architecture, training, and source/evidence verification were executed directly, not falsely attributed to agents.
