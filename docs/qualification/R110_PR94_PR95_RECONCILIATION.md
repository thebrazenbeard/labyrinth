# R110 Labyrinth PR #94 / #95 source-contract reconciliation

**Status: INTEGRATION_REVIEW — source-only; no physical-training, merge-to-main,
deployment, examination, or graduation authority.** This reconciliation uses
[#94](https://github.com/thebrazenbeard/labyrinth/pull/94) as the upstream
source stack and incorporates the independent
[#95](https://github.com/thebrazenbeard/labyrinth/pull/95) research changes.
The parent branches diverged at #90; a textual merge alone could not preserve
their audit, roster, and callback contracts.

## Three intentionally distinct HF evidence contracts

| Source mode | Entry point and report | Required consistency boundary | Qualification limit |
| --- | --- | --- | --- |
| Ordinary HF | `run_pilot(...)`; `labyrinth.hf-lora-pilot.v3-exposure.v1` | Strict-by-default `audit_hf_report`, exact `optimizer_exposure_recording` marker and all three arms' ordered in-loop `visited_exposures` | Internal self-report only, not optimizer or external custody attestation |
| R85 staged Minotaur | `run_pilot(minotaur_roster=...)`, `minotaur_staged_train`, `snapshot_staged_roster` / `replay_staged_roster` | Reconstituted bounded TRAIN prompts, operator-private gold, exact visit order, public staged manifest; audited **`INCOMPLETE`** because source custody is operator-held | Diagnostic-only reload may proceed after narrow consistency verification; never promotes `INCOMPLETE` to `PASS` |
| R105 adaptive Minotaur | `run_pilot(adaptive_train_roster=..., allow_unattested_adaptive_train=True)`, `labyrinth.hf-lora-adaptive-source.v1` | Replays actual feedback-conditioned runner, policy route, room identities, gold, and schedule; three-arm in-loop exposure receipts and separate PASS-only reload audit | Explicit opt-in, source-only cycle-zero arithmetic pilot; not pretrained Qwen training |

The two Minotaur roster arguments are mutually exclusive and cannot be
combined with a predecessor/resume path. Cross-injecting an adaptive manifest
into a strict/staged report, or a staged source into an adaptive report, must
fail closed. Historical v2/v3 audits require `allow_legacy=True`; they are
never silently upgraded to the v3-exposure or adaptive evidence contract.
`reload_evaluate` likewise refuses historical v2/v3 reports by default.
An explicit `allow_legacy_reload=True` invokes a *compatibility audit* and
may proceed only on its `PASS` result; it does not convert historical evidence
into current strict exposure or training authorization.

## Source protections retained across both branches

- **#94 deterministic Minotaur novelty:** `journey_policy_runner.py` and
  `taskmaster_routing.py` can redraw a duplicate or equivalent TRAIN chamber
  deterministically with a bounded retry budget. R110 additionally screens
  all three constituent arithmetic atoms of a contrast TRAIN challenge
  against earlier TRAIN chambers, including staged roster export/replay; a
  400-corridor availability regression covers weak and mixed feedback. Its
  arithmetic oracle `_source_arithmetic` must remain available from
  `minotaur_hf_roster_bridge.py`; removing it breaks the runner. The independent
  R85 answerless roster-export module and its exact ordered optimizer visits
  remain separate from R105's feedback-bearing adaptive source manifest.
- **#95 R109 three-atom heldout semantics:** a contrast question
  `A=a+b; B=a-b; A-B` contains the source computations `a+b`, `a-b`, and
  `b+b`. `arithmetic_semantic_atoms` must screen public adaptive diagnostics
  against all three; matching just `b+b` misses underlying source reuse.
  Source-level checks cannot establish a general theorem of mathematical
  equivalence or independent evaluation custody.
- **#95 Q6 source immutability:** all four admitted toy-study cohorts have
  primitive snapshots rechecked around callbacks, scoring, checkpoint and
  report publication. This guards local post-admission mutation, not a
  separately owned historical exposure register.
- **CI union:** mandatory pinned neural CPU tests must keep both R80/R84/R85
  strict/staged source cases and R105 adaptive/Q6 cases, including cross-branch
  reconciliation tests. Python 3.10/3.12 and neural-CPU pull-request runs must
  assert the exact event-head SHA; wheel packaging must contain both source
  modules.

## Deliberate non-authority and next gates

Neither source hash is an external signature, proof of chronological
feedback, or observation that a neural optimizer consumed exact token IDs.
The terminal adaptive feedback bit can be changed with a recomputed self-hash
without a subsequent policy transition to constrain it. A host-owned witness
of feedback, exposure and checkpoint lineage is still required. The existing
HF pilot's nonquantized 1 GiB checkpoint ceiling does not permit training
the intended Qwen3.5-4B GGUF used with KoboldCPP, and the separately admitted
TrainingDay arithmetic-chain corpus is not interchangeable with the
Minotaur roster.

The separate physical host path is **HOLD**. At the checked head of
[model-training Draft #181](https://github.com/thebrazenbeard/model-training/pull/181),
the physical-launch guard refuses every launch; current owner-held host/GPU
lease, stable Executor/NAS route, exact model-loader/VRAM fit, fresh-load
adapter update/reload, and independent Q5–Q9 custody evidence are missing.
No Labyrinth source test or CI success overrides that guard.

## Integration acceptance record

- [ ] Reconcile and inspect the exact #94 + #95 combined Git tree and
      independently review all conflicting APIs and audit schemas.
- [ ] Run both predecessor test families and new adversarial/redraw suites
      against the same final checkout; classify failures rather than waive them.
- [ ] Run the complete local suite, package wheel and verify both module
      members, then obtain fresh exact-head hosted Python and pinned neural
      CPU success on the proposed integration branch.
- [ ] Keep the integration PR Draft until all source gates pass; separately
      clear documented owner-held physical, custody and promotion gates
      before authorizing any mainline merge, Qwen training or deployment.

The checkboxes are intentionally pending here. Final statuses must cite a
specific tested tree, command result, and CI run, not this document.
