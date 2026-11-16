# Wave 84 — Reject reused evaluation mechanisms at policy freeze (source-only, 2026-11-16)

## Exact subject and threat
- Canonical `thebrazenbeard/labyrinth`; isolated source branch `fix/minotaur-cross-slice-family-wave84-20261116` created from exact PR #265 head `fbb6fddc31a06bd1ec663122c6248dac2fec73fd` (PR #265 is DRAFT/OPEN and hosted Python+CPU SUCCESS at Wave84 start).
- The v2 `GraduationPolicy.independent_semantic_registry` is supposed to precommit normalized task-mechanic equivalence classes, not cases/prompt aliases. Existing `decide_graduation` already rejected reused semantic families across heldout/transfer slices and across attempts, but `GraduationPolicy.__post_init__` accepted the impossible roster. Invalid campaigns could thus be frozen and executed only to be disqualified later.
- **Hostile falsifier:** signed synthetic receipt with 16 identical semantic-family IDs in both heldout and transfer evaluations but distinct case IDs, prompt digests and apparently perfect candidate scores was **BLOCKED**, not graduated. This disproved an initial overbroad bypass suspicion. It is not evidence of a current graduation bypass or a real neural model.

## Reproducible RED → GREEN
- Parent baseline `python -m unittest -q tests.test_graduation_gate tests.test_graduation_state`: 36/36 PASS.
- RED (exact `python -m unittest -v` on three new named tests): **2 FAIL, 1 PASS**, `AssertionError: ValueError not raised` for (a) same 16 evaluation families precommitted to both heldout and transfer slices; (b) repeat of a heldout semantic family across campaign attempts. A third control confirmed that multiple variant cases of one family inside one slice are allowed and conservatively counted.
- Regression: Windows 141/141 PASS, WSL 135/135 PASS, optimized WSL (python3 -O) 99/99 PASS; **375 targeted test executions**. Python compileall exit 0, git diff --check exit 0. These are source-level checks, not an independent held-out model trial.
- GREEN: track each non-training semantic family to exactly one frozen pair `(attempt, evaluation_slice)` in `GraduationPolicy.__post_init__`. Raise explicit `ValueError` if a family crosses slice or retry-attempt boundaries. Preserve within-slice variants and repeat train-only drills. Exact three-test rerun **3/3 PASS**.
- New source and tests: `labyrinth/graduation_gate.py`, `tests/test_graduation_gate.py`. No changes to signed-evidence schema or existing receipts. The source gate still expects genuinely independent off-model semantic registry custody.

## Evidence classes and limitations
- This is a **SOURCE-INSPECTION and TEST** strengthening of immutable policy admission; **not** a demonstration of actual semantic diversity, authentic model origin, independent examiner custody, or model learning.
- A dishonest custodian can still invent different semantic-family digests for identical mechanics. Cryptographic commitments authenticate label origins, not the truth of task-equivalence judgments. Issue #147 therefore remains OPEN; separate examiner principal isolation issue #146 remains OPEN.
- Existing v2 receipt-time cross-slice/cross-attempt replay guards remain active, and the externally controlled policy/registry lifecycle remains unqualified until custody is independently proven.
- No neural inference, training, GPU workload, installed service mutation, grader graduation, production deployment, merge, or paid cloud compute was performed. Worktree isolated from other workers.

## Repeatable focused command
`python -m unittest -q tests.test_graduation_gate tests.test_graduation_state tests.test_round4_learning_controls tests.test_minotaur_local_trial tests.test_minotaur_ollama_child tests.test_minotaur_candidate_bridge tests.test_minotaur_subprocess_adapter tests.test_minotaur_posix_uid_boundary tests.test_examiner_private_read_probe tests.test_adapters`

Keep as a **stacked DRAFT/HOLD** on Wave83 PR #265 until independent hostile review, #146 OS-custody probe and #147 independently custodied semantic mechanism and model-origin qualification.