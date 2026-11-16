# Minotaur Wave 77 — action-earned escape, not claim-earned escape

## Exact source scope
- Repository: `thebrazenbeard/labyrinth`; draft and hold.
- Parent: `94fa900340f91d0fb9f00e715287a44daf1174d4` (draft PR #258).
- Isolated Lappy worktree: `D:\VERA\worktrees\labyrinth-minotaur-action-earned-escape-wave77-20261116`.
- Only runner, auditor, new adversarial test, and this evidence note changed.

## Reproduced hostile behavior (RED)
- A synthetic examiner returns `VERIFIED_WIN` and `escape_verified=True`
  from `submit(CLAIM)` without any successful escape action.
- Existing bridge credited `verified_escape=True` with one CLAIM-only turn,
  and also after a nonwinning ACTION followed by a winning CLAIM.
- Matched-pair candidate/control inherited false wins.
- Focused test: three expected `AssertionError: True is not false` failures;
  `RED_EXIT=1`.

## Smallest validated correction (GREEN)
- The final recorded turn must have disposition `ACTION` to earn escape
  credit, in both runner and replay auditor. Claim verdict remains recorded.
- Focused identical test: `GREEN_EXIT=0`; forged self-hashed claim-win denied.
- Real action-earned `WINNING_SEED_ZERO` remains credited and auditable.
- Windows 98/98 targeted tests PASS; WSL normal 92/92 PASS;
  WSL optimized 72/72 PASS. Total: 262 targeted test executions.
- Python `compileall -q labyrinth` and `git diff --check` exit 0.

## Hostile limitation
> A terminal ACTION feedback record is a necessary attribution condition,
> not independent authentication of the model, neural inference, examiner
> identity, case provenance, or learning. Issues #146 and #147 remain OPEN.
> No full repo suite, qualification, deployment, merge, or external inference.
