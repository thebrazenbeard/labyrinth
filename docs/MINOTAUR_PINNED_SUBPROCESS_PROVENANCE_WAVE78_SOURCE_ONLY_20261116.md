# Minotaur Wave 78 — pinned subprocess invocation

## Exact scope and source
- Canonical: `thebrazenbeard/labyrinth`; isolated Lappy worktree.
- Parent draft PR #259: `3a0d1bfc1fd4677defd546ea34f481bfea1168e4`.
- No merge, deployment, networked model inference, or host credential/ACL changes.

## Hostile RED reproducer
- For an exact `SubprocessCandidateAdapter`, `run_candidate` recorded
  `MODEL_SUBPROCESS_UNATTESTED` provenance even with the optional
  `require_process_separation` flag omitted.
- In that default path, the runner invoked `adapter.generate` dynamically.
  A class-level monkeypatch executed Python **inside the host examiner**.
- The instrumented override observed a `CandidateSession` in a live host
  stack frame (`host_reach=[True]`) and returned a forged `leak` ACTION.
- Added `test_subprocess_label_cannot_run_host_callback_without_process_only_flag`.
  Before correction: FAIL (`[True] != []`); `RED_EXIT=1`.

## Correction and GREEN
- The exact subprocess adapter now always uses the captured
  `_SUBPROCESS_GENERATE` entry point, independent of optional mode.
- The explicit flag still refuses unrelated in-process Python adapters.
- Identical focused test: `GREEN_EXIT=0`, actual child `wait` response,
  no host callback, replay auditor PASS.
- Windows bridge/process/private-read/graduation controls: 105/105 PASS.
- WSL normal bridge/process/graduation controls: 99/99 PASS.
- WSL optimized bridge/graduation: 72/72 PASS. Total 276 targeted.
- Python compileall and Git diff check: exit 0.

> Hold: The child still inherits same-user filesystem permissions.
> This removes one in-process frame-inspection route but does not satisfy
> examiner-custody issue #146. Semantic custody #147 also remains open.
> A source test cannot prove neural learning, independent review, or graduation.
