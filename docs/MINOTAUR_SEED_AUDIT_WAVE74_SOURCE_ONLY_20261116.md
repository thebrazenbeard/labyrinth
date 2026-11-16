# Minotaur Wave 74 — impossible seed metadata (source only)

## Exact source and intent
- Repo: `thebrazenbeard/labyrinth`.
- Parent commit: `bd9f9f6b278423f6e979dade10491fe2d03fdb4c`, draft PR #255.
- Isolated Lappy checkout: `D:\VERA\worktrees\labyrinth-minotaur-seed-audit-wave74-20261116`.
- Scope: internal receipt/auditor consistency, not signer authority or neural inference.

## Reproduction (RED)
- Runner `run_candidate` rejects a seed outside `0 <= seed < 2**31`.
- Auditor previously accepted the same impossible metadata after an attacker
  recomputed the *unkeyed* receipt self-digest.
- Original receipt: public seed 12, one valid `wait` action, one call.
- Forged self-sealed seeds: `-1` and `2**31`, both returned audit True.
- New focused regression failed two subtests; `RED_EXIT=1`.

## Repair and verification (GREEN)
- Validate auditor seed range exactly like runner before examiner construction.
- Test asserts forged receipts fail audit without invoking examiner factory;
  legitimate receipt still passes.
- Focused regression `GREEN_EXIT=0`.
- Windows: bridge, subprocess and private-read probes **59/59 PASS**.
- WSL: bridge and subprocess **53/53 PASS**; optimized WSL `-O`
  bridge **39/39 PASS**; **151 targeted test executions** total.
- `compileall` and `git diff --check`: exit 0.
- Tests do not cover full repo, neural learning or any real model.

## Hostile limits
- The seed is still a self-reported field: no independent model-call,
  signer, source weights or seed-delivery custody is established.
- Cross-user examiner isolation #146 and independent held-out mechanism
  custody #147 remain open; no graduation, deployment or merge authorized.
