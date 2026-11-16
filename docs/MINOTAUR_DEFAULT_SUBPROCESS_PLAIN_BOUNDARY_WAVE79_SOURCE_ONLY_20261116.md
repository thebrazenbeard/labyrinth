# Minotaur Wave 79: inert process-boundary snapshots by adapter identity

## Source and threat
- Repo: `thebrazenbeard/labyrinth`, parent draft PR #260
  `cca62a3cd59f7e515559fa86c6112816c38e6467`.
- Exact `SubprocessCandidateAdapter` is subprocess-labeled even when
  `require_process_separation=False` (default).
- Pre-fix public/examiner serialization used only that optional flag:
  programmable `dict` subclasses invoked host Python `items()` callbacks,
  despite later replay requiring inert process-labeled JSON.

## Reproduced RED and corrected GREEN
- `test_subprocess_default_rejects_programmable_public_before_host_callbacks`:
  real default subprocess run failed `ValueError not raised`; RED exit 1.
- `test_subprocess_default_rejects_programmable_examiner_feedback`:
  default subprocess returned `UNKNOWN` instead of `EVALUATOR_ERROR`,
  having traversed a host callback; RED exit 1.
- Source fix: `strict_plain = separated` and pass it to all four
  public-input, initial-view, binding and feedback validation sites.
- Both identical focused regressions GREEN exit 0; callback arrays empty.
- Windows relevant bridge/process/private-read/graduation controls: 107 PASS.
- WSL normal relevant bridge/process/graduation controls: 101 PASS.
- WSL optimized bridge/graduation controls: 74 PASS. Total: 282 targeted.
- `python -m compileall -q labyrinth`, `git diff --check`: exit 0.

> HOLD: A same-user subprocess still reads examiner files if paths/ACLs permit.
> This is source-only Python callback prevention, NOT distinct OS principal,
> authenticated model execution, hidden-case custody, learning or graduation.
> Preserve issues #146 (examiner file access) and #147 (semantic custody).
