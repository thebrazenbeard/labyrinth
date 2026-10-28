# BT2 Minotaur V2 — genuine specialist dispatch and integration (2026-10-27)

## Execution identity and authority

Five original Codex CLI attempts failed immediately because Lappy's configured `gpt-6.1-sol` was unsupported for its ChatGPT account. The explicit `gpt-5.3-codex` and `gpt-6-luna` probes also failed. The local authenticated Codex model cache exposed `gpt-5.6-sol` and `gpt-5.6-terra`; retries using these names **actually executed** agent commands.

These are **six real, separate Codex CLI specialist sessions on Lappy**, not native ChatGPT collaboration.spawn_agent calls. All work began from `thebrazenbeard/labyrinth@0ef9ba4512b799a5c7e9abe4aa32e4858f9f49f5` in independent linked Git worktrees. Agents were sandboxed, directed not to push or merge, and not granted production or independent-graduation rights. Git metadata was outside their writable sandbox, so several failed to commit. The BT2 lead independently verified, committed and cherry-picked accepted work; no agent claimed unsupported Git success. One reviewer was read-only.

| Role | Real Codex session ID | Source branch commit (lead-committed where necessary) | Result |
| --- | --- | --- | --- |
| Neural microfit | `01a17fc1-ff7e-7cc1-aaa8-9235bc65b729` | `9cb44b1f02871577fb6c8bbd6acc211e6ff8ff32` | 32/32 development **training** memorization in 68 AdamW steps, loss 1.992 → 0.0479, positive finite gradients; unqualified |
| Scratch architecture V2 | `01a17fc2-340f-7781-8677-c78926eedb4b` | `7cda186989ae731a9a75ba65eba029f83dd8f955` | New randomly initialized separate correctness / first-error heads, ordered-state encoding, masked-padding tests and actual local PyTorch gradient smoke; no V2 trained checkpoint or generalization established |
| Additional curriculum | `01a17fc2-69d3-7042-acb2-476caf397ded` | `f71b215218e95c099994e01597be20dc79f213ae` | New bounded propositional-logic proof/entailment domain and independent test-only postfix oracle; finite public structures, not a blind private exam |
| Independent examiner | `01a17fc2-a085-7032-b058-0fb250044602` | `8fbdf95b9f1bd4d2c98423bad1bac17b7b063fb1` | Explicit signed score-provenance field, reject public advisory/missing provenance; **signature is still only an assertion**, not genuine independent custody |
| Hostile red team | `01a17fc1-9b66-7493-89b9-f4f084f44efb` | Read-only / no commit | Confirmed high neural false-approval rates; four structural templates per split and predictable +1 corruption; neural path skipped in old CI; signer can fabricate HMAC receipts; scratch report hashes self-attested |
| Neural CI qualification | `01a17fc6-a9d2-7bd2-ad66-434e773d9980` | `b774e47051e7f67e321bc390d1f1019a8fb51995` | Dedicated Linux Python 3.12 workflow exercises the actual committed scratch checkpoint and guarded inference using pinned top-level CPU Torch+Safetensors; transitives deliberately resolved (not individually locked) |

## Integrated source and qualification boundaries

Integration worktree `bt2/minotaur-v2-integration-20261027` cherry-picked each accepted commit (new hashes differ because of different parents). The source-only CI / local test suite is *not* a neural mastery test. The exact 32-case overfit result proves trainability, **not transfer**. The proposed V2 model has not demonstrated improvement over V1 on a large blind holdout. The propositional dataset remains templated/public and may contain format shortcuts. The bounded symbolic checker, not neural competence, accounts for guarded V1 reliability.

The examiner provenance field makes receipt downgrade explicit but **does not verify honesty** of the signer or implement a real independent exam. A compromised or dishonest key holder can still claim private provenance. Promotion to production or trainee graduation is therefore prohibited until an independently governed evaluator actually owns hidden questions, checkpoint execution, oracle decisions and signing credentials.

The new optional-dependency workflow is separate from the standard dependency-free Python suite. It will only qualify **source execution/checkpoint compatibility** when its exact GitHub Actions run passes. CI must never be reported as success before workflow readback.

No Git main merge, live training installation, background deployment, or graduation effect is implied by this package.
