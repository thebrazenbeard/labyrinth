# Minotaur V5 — Compact algebraic neural learning from incomplete examples

**Date:** 2026-10-28. **Qualification: RESEARCH ONLY / NOT_ESTABLISHED.** Labyrinth Draft PR #20; no release, production promotion or graduation authorization.

## Actual fresh specialist delegation

Five independent authenticated **Codex CLI** sessions were launched in separate worktrees from exact source `e9efd41f8ee2e25cfb5535b605371ad1801fcee6`: architecture `01a18208-2c3b-7892-82bb-43a4699013b0`, split and curriculum `01a18208-5ef0-7252-ad7e-79c1932c30e1`, neural training `01a18208-92ce-78b1-a54f-0320a5e09d5e`, independent evaluator `01a18208-c7c2-76d3-b60f-cdf9114cb15f`, and read-only scientific/security red team `01a18208-fd14-7000-977d-586a499993ce`. They are genuine separate **Codex processes**, not native ChatGPT child agents. All worked from isolated source and produced independent code, adversarial tests or read-only review. The linked-worktree sandbox prevented some agents from writing Git's external index; lead independently ran Torch-enabled tests, integrated and committed accepted source. No made-up agent commits or private reviews.

## Source-level replacements

`labyrinth.minotaur_algebraic_v5` is a compact, seeded **32,258-parameter** operation-conditioned network built from residue embeddings, cyclic Fourier input features, low-rank bilinear factors and an output classifier. Forward computes no arithmetic label and has **no (op, x, y) indexed table**, unlike V4's 2,738,019-param, all-atoms supervised embedding table. A model-specific commutativity prior is applied to ADD/MUL, not SUB. The Fourier prior is engineered; the model does not discover number representation independently.

`labyrinth.minotaur_algebraic_data_v5` exhaustively creates all 28,227 modulo-97 atomic examples but trains with only an **80% subset**. Opposite operand orderings are grouped into the same split to prevent `ADD(a,b)` /`ADD(b,a)` leakage, identities are checked independently, and train/validation/holdout commitments are recorded. Separate `value` and `diagonal` challenges reserve all pairs containing one residue or all same-operand pairs respectively. A split-allocation parity bug found by the tests was corrected. `minotaur_algebraic_eval_v5` independently recomputes every claimed label before scoring and refuses malformed IDs, false keys, or invalid outputs; its program evaluator does not pass any symbolic oracle into the model callback. These are **public, deterministic development tests**, not a privately held blind examination.

`minotaur_algebraic_training_v5` performs actual CPU AdamW weight updates using input tensors containing **only op, x and y**, with the target label used exclusively for cross-entropy. The held-out partition is not used by the optimizer or epoch selection. The low-rank direct-neural model was tested as an initial negative control.

## Direct learned operator (no algorithm supplied)

A precommitted, 800-update pure-neural experiment on source `236ae3daa95be879fe4cd60232481974304edfb5`, with independent seed replication:

| Public unseen atom test, each operation n=941 | Seed 20261028 | Seed 20261029 |
| --- | ---: | ---: |
| ADD | **941/941 = 100%** | **941/941 = 100%** |
| SUB | 471/941 = 50.1% | 469/941 = 49.8% |
| MUL | 40/941 = 4.3% | 38/941 = 4.0% |
| Total | 1452/2823 = 51.4% | 1448/2823 = 51.3% |

The randomly initialized model scored ~0.85% on validation before training. Thus ADD genuinely generalizes across withheld atomic operand pairs using a shared small network, **but direct neural SUB/MUL did not**. There was no successful pure-neural all-operation qualification.

Evidence: `docs/evidence/MINOTAUR_V5_PURE_NEURAL_SEED1_2026-10-28.json` (SHA-256 `807664231937a96bf484440c0fe7984931fd1c966e04683b7fe48a4342b70c70`), and corresponding `SEED2` (SHA-256 `2eb2af1e9d47b938390e49501644cf3a94baad1d9798025a7684d4c1d6e86313`).

## A working compact neuro-symbolic composition architecture

`labyrinth.minotaur_algebraic_composer_v5` trains the *same* 32,258-param network **on ADD only** (7,527 of 9,409 possible addition input pairs for the ordinary seed). For other operations it supplies **explicit program control laws**:

- `SUB(x,y)` is implemented by neural `ADD(x, inverse_mod97(y))`. The inverse operation is an engineered exact algebraic prior, not a learned subtraction algorithm.
- `MUL(x,y)` is implemented by seven rounds of binary double-and-accumulate using repeated predictions from the **neural ADD** network, not a direct multiplication oracle or an exhaustive atomic lookup.

This is a **neuro-symbolic rule composition design**. The network learned a transferable finite-domain addition mapping; the *architecture provided* the algorithms for subtraction and multiplication. It is false to describe those algorithms as discoveries of the network.

The precommitted and actually executed composition experiment source is `22e71c35c946b4abfe67d9068bd8eb66a9ecb4b7`. Two distinct random initializations and two stronger holdout challenges, each with 1,200 AdamW updates, produced:

| Independent public challenge | Test cases | Correct | Wrong approvals |
| --- | ---: | ---: | ---: |
| Random pair split, seed 20261028 | 2,823 | **2,823** | n/a; atomic value task |
| Random pair split, seed 20261029 | 2,823 | **2,823** | n/a |
| Withheld residue-value family, seed 20261030 | 2,823 | **2,823** | n/a |
| Withheld diagonal pairs, seed 20261031 | 2,829 | **2,829** | n/a |

Each number reflects arithmetic atomic output correctness, *not* a signed approval or a private reasoning examination. Test pairs are grouped under operand reversal, and no IID confidence claim is made. The holdout-value challenge excludes every training pair containing its reserved value.

## Actual saved model and repeatability

Experimental checkpoint: `models/minotaur_algebraic_v5_development/model.safetensors`, 148,936 bytes, SHA-256 `ae92b4e847a8c952d6672a14d9064d0e73df185822da2a042e98c3ce59f55c57`. The seed-1 report `training_report.json`, SHA-256 `b43c43f84e59171bf18fb35107e67f14f76fc06c694fcf3e2bd383fd448b68a6`, contains split membership commitments, ADD-only supervision counts, trained weight digests and independent held-out scoring. Fresh Safetensors reload returned **identical predictions**, and source `minotaur_algebraic_checkpoint_v5.py` pins the exact file hash, refuses altered files and performs optional CPU-Torch inference.

The other seed and two challenge reports are independently versioned under `docs/evidence/MINOTAUR_V5_*.json`. Python tests include a broken-neural-addition negative control: the composed result becomes wrong when the learned primitive is wrong. The public multi-step rollout uses a strict independent replayer and disjoint canonical structures, with no training labels or gold intermediate states passed into the model callback.

## Red-team and release boundaries

The read-only red team confirmed the original V4's exhaustive-table limitation, symbolic-checker authority, paired-trace pseudoreplication and lack of external examiner custody. It required withheld atomic pairs, macro/per-op scoring, architecture/source identity, frozen/control models, and genuine independent private tests before promoting reasoning claims. The V5 implementation addresses **held-out atomic pair learning and full-source reproducibility**, but the independent private examination, ability to learn multiplication without its supplied algorithm, new modulus, and arbitrary domains are still **NOT_ESTABLISHED**.

**V4 remains available; V5 is an experimental alternative, not a silently installed production replacement.** No model may graduate a trainee on public synthetic evidence, and no private-exam credential has been created.

## Next falsifiable scientific objective

Train a parameter-sharing algebraic latent model to discover multiplication and subtraction *without* given inverse/binary-double-and-add control flow, then independently qualify its accuracy on reserved residue-value families, new prime moduli and held-out grammar changes. The earlier direct-neural negative results remain the baseline.
