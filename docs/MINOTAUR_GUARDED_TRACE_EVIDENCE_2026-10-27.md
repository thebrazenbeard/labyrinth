# Minotaur E-022 — bounded neuro-symbolic reliability qualification (2026-10-27)

**Subject:** `thebrazenbeard/labyrinth@9dd67e553bde530cd37069dd754357dfcd26056e` (source clean when the measurement ran). This is **NOT** general reasoning competence, independent final-exam evidence, or an authorized graduation process.

## Repaired defect

The randomly initialized first Minotaur checkpoint under `models/minotaur_scratch_v1` learned unreliable predictions. The initial 8-epoch run scored 43/96 dev and 20/48 longer-program transfer, with **39/48** and **22/24** false approvals of incorrect traces. An extended 24-epoch / 384-optimizer-step run on 512 training traces **decreased** dev accuracy to 29.2% and transfer accuracy to 30.2%, with false approvals **67.7% / 64.6%** respectively. Extended original local report SHA-256: `a4739ecff531b8979f443b9fcb8feb5aab2c3babe425d3d1746e222bf2c9e807`; extended model weights were not promoted or committed.

Increasing compute alone did not cure the bad approval behavior. Instead of giving this unreliable neural model the release keys, we added `labyrinth/minotaur_guarded.py`, an **independent strict modular-arithmetic interpreter** for a bounded two-register program grammar (2–6 operations, modulo 97). It explicitly computes the first erroneous state, independently of `guardian_benchmark`'s answer keys. The actual saved scratch model is also loaded via Safetensors, with its manifest and trained tensor digest checked. Only matching neural and checker results are reported as a supported domain-specific decision. Unsupported/malformed input, untrusted JSON, missing/neural disagreement, or a changed model yields ABSTAIN/refusal. **Every response says `qualification: NOT_ESTABLISHED`, `graduation_authority: none`.**

This is a *neuro-symbolic bounded checker*, not newly gained neural mathematical understanding. The deterministic program interpreter is responsible for safety within its narrowly supported language.

## Real checkpoint and fresh evaluation

- Trained scratch checkpoint `models/minotaur_scratch_v1/minotaur.safetensors`: SHA-256 `ce2d1a8a7e91f32f58ad01de2304ca6241e2f3f4c2a0227447764503d7fd77ec`.
- Complete trained neural-parameter digest `fa43682e6d19e2a2d2b2e32e4b88d2c901961ce7a1f5aac7a43dd8b2bccdd088`.
- New **non-training** seeded development families: `501`, `911`, `2027`; 32 cases per each of three slices and two trace variants per seed, hence 192 traces/seed, 576 total. The verifier and the public oracle agreed on the gold label in all examples.
- Exact result JSON: `docs/evidence/MINOTAUR_GUARDED_EVAL_2026-10-27.json` (the copied evidence artifact SHA-256 `697415e49273403296bb00320896b5c09ebe3d6818e0da583d3c3c4fba390089`).

| Seed | Total | Incorrect | Supported decisions | Abstentions | False approvals | Gold-checker disagreement |
|---|---:|---:|---:|---:|---:|---:|
| 501 | 192 | 96 | 79 | 113 | 0 | 0 |
| 911 | 192 | 96 | 83 | 109 | 0 | 0 |
| 2027 | 192 | 96 | 80 | 112 | 0 | 0 |
| **Total** | **576** | **288** | **242 (42.0%)** | **334 (58.0%)** | **0 observed** | **0** |

All 242 non-abstained decisions matched the known task's deterministic oracle; this is **expected by construction because the independent interpreter must agree**. The non-abstention rate is modest, and the raw neural critic remains poor. Examples are public, templated and correlated, so neither a risk-confidence bound nor generalization is inferred from the zero observed false approvals. Missing uncertainty or disagreement is not silently counted as a correct verdict.

## Executable interface

Using local Python with CPU PyTorch and `safetensors` installed, supply a JSON file containing exactly `{"question": "...printed register program...", "steps": ["Step 1: A=..., B=...", ...]}`:

```powershell
python -m labyrinth.minotaur_guarded --checkpoint models/minotaur_scratch_v1 --input-json trace.json
```

The command checks duplicate JSON keys, file type, file size, strict grammar, the original model's sealed Safetensors digest, and reloaded checkpoint predictions, then runs the neural prediction and an independent arithmetic check. No network, remote model inference, executable checkpoint deserialization, or production route is used. Unsupported user data results in refusal/abstention; no automatic Labyrinth graduation.

## Code qualification versus behavioral qualification

The focused independent-checker suite includes public seeded traces, first-error mutations, malformed/injected statements, class imbalance and non-master-releasing behavior. The full Python source regression and GitHub Actions statuses are distinct: tests can verify that the code *does what it claims* in this toy domain, not that the system has mastered the art of judging trainees.

Real native subagents were unavailable in this chat. Changes were executed directly in an isolated Git worktree. An independent hosted red-team review and an isolated private examination signer were **not** performed. The existing Draft PR remains unmerged and no production runtime was changed.

## Next work

1. Develop broader independently checked problem families not exactly soluble by this one hard-coded interpreter.
2. Produce a stronger scratch-trained neural ranker/step verifier with independent heldout families and statistically defensible controls; prevent shortcut/label leakage.
3. Maintain the boundary: only an independent, privately owned evaluator with authenticated custody and predetermined policy may eventually approve graduation. An advisory neuro-symbolic trace verdict is not a graduation receipt.
