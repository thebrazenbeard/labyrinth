# Visual retention: acquisition cost and useful non-use

Status: controlled offline numerical-policy research; no model training, runtime binding, or generalization claim. This supplements the evaluation discipline in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), not the curriculum or scoring implementation.

## Evidence and question

A preceding synthetic visual-witness study compared six fixed inspection policies on 4 workload families, 12 paired seeds and 160 image pairs per seed (46,080 policy records). It used Vision's actual sampled RGB observation and full-frame digest at source revision eda87da062a22f52dcf06cab6e17f4b55fddd4a6. The task was a verified changed-pixel witness or correctly unchanged; uncertainty remained UNKNOWN. No ground-truth feedback was passed to policies. Each case allowed 1,024 additional pixel comparisons. Separate-process continuation agreed with uninterrupted execution. The already-acquired prior was a preferred quadrant; replaying its acquisition consumed 142,035 comparisons.

The earlier comparison charged only marginal inspection work. Here the same frozen traces are re-accounted, without policy tuning, to ask whether paying for history alters the conclusion.

## Lifetime budget audit

The common per-seed budget is 163,840 comparisons (160 x 1,024). Retained variants pay ceil(142035 / share) before the first case, with hypothetical share=1,12,48 or zero marginal acquisition cost (sunk history); reset variants pay zero. In every arm, admit the next case only if its full 1,024-comparison cap can be reserved, then deduct actual recorded cost. Stop when the remaining budget is below that cap. No lookahead to actual future cost. Unattempted and abstained cases both remain unresolved; report them separately in the audit data. The allocations are hypothetical sensitivity analyses, not observed separate acquisition runs. Full-frame digest/sampling, state operations and latency are not charged in these comparison units.

Correct resolutions of 1,920 cases per workload (12 seeds):

| Workload | Row-major | Witness reset | Rotate reset | Witness retained, full acquisition | Rotate retained, full acquisition |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stable | 1155 | 1908 | 1908 | 623 | 623 |
| Abrupt change | 673 | 1044 | 1896 | 623 | 623 |
| Alternating | 578 | 950 | 372 | 183 | 60 |
| Uniform random | 677 | 611 | 603 | 112 | 120 |

All reset arms attempt all cases. Full-cost witness-retained attempts 623/623/363/326; full-cost rotate-retained attempts 623/623/297/333. For all four families, paying for a new prior per episode reduces resolutions relative to the corresponding reset procedure. This is an accounting counterexample, not an empirical population estimate.

With acquisition allocated across 12 episodes, both retained policies attempt all cases: witness-retained resolves 1920/1056/960/625, rotate-retained 1920/1908/372/618. At 48-way allocation the same resolution counts hold. Even at 48-way allocation, retained variants use more total comparisons than their respective reset variants in every family; a small accuracy benefit may still justify extra cost, depending on the task. With sunk history the marginal comparison is different. In alternating conditions rotation resolves only 372 regardless of prior, versus 960 for witness-retained; the adaptation rule itself matters.

## Reproduction and falsifiers

Input archive seat2-vision-feedback-cost-20261127.zip SHA256 c97b315b4b4eb8f9f57d129b5e8b6d688053a5e874cdb8ab55937550d14f8564. Frozen study.py SHA256 b8461b939ba128f126632ae3f7cc7630963cabad09726501c3cb25549e79a2b6. The source archive is retained in the originating research workspace, not embedded here.

The audit checked ZIP/manifest integrity, 46,080 ordered policy records against the optimized counterpart (excluding timings), 288 complete 160-case groups, valid statuses and costs, and all 96 aggregate budget results. Normal and optimized CPython executions exited zero; JSON output matched byte-for-byte. Original audit.py SHA256 80a3a6f91b8c9aaa5abe546be46790922826777f8de744d6f88bb72384406303; result JSON SHA256 024533ede2779efa483be462b7c7fad744bab601649cea71896f373b9e651eb2. The audit is retrospective and not a blind independent experiment. Prior acquisition cost is attributed to the preceding replay. It excludes common observation and storage cost, and uses authored workload families.

Reproduce budget accounting from the archived traces: group by workload, seed, policy; iterate recorded case order, starting from the budget minus applicable prior allocation; reserve the full case cap before admitting a case; subtract its recorded cost; count WITNESS and UNCHANGED as correct based on the original evaluator's reported zero-error checks, UNKNOWN as abstention, remaining cases as unattempted. Do not count skipped cases as correct or use future costs to decide admission.

## Disposition

Do not install a universal retention/adaptation policy on this evidence. Prefer already-existing procedures only where their applicability and incremental value are supported. Compare actual acquisition, reuse, revision, abstention, and replacement costs. A retained artifact can remain available without receiving preferential selection. Missing evidence is not forgetting; retirement from active selection is not deletion. Independently authored changed workloads and an end-to-end resource accounting boundary are necessary before consumer adoption.

No model-weight update, private interaction data, reinforcement modification, or operational learning state is implied. This document adds no new runtime mechanism.
