# Minotaur Wave 73 — one-character overflow receipt evidence (source only)

## Exact starting point
- Repository: `thebrazenbeard/labyrinth`.
- Parent: `ede4fefe8f3416fb30d7466814e525e29bc6b69b`, draft PR #254.
- Lappy isolated worktree: `D:\VERA\worktrees\labyrinth-minotaur-oversize-proof-wave73-20261116`.

## Reproduced defect (RED)
- A model response exceeding the 4,096-character limit was invalid, but the
  runner retained at most the first 4,096 characters in the receipt.
- If this prefix was a syntactically valid JSON action followed by spaces, the
  auditor parsed the truncated prefix as valid and rejected the genuine invalid-output receipt.
- New public-boundary test supplied valid JSON padded to exactly 4,096 chars
  plus `!`; RED failed on missing last character, `RED_EXIT=1`.

## Minimal change (GREEN)
- Retain at most `MAX_RESPONSE_CHARACTERS + 1` characters *only* for invalid
  string responses; parsed/valid response limit stays 4,096.
- The 4,097th character establishes a length violation without retaining an
  unbounded original. Extra-long 10,096-char input still stores just 4,097.
- Focused regression passed with `GREEN_EXIT=0`, including long-output control.

## Local verification
- Windows Python: bridge, subprocess, private-read probes **58/58 PASS**.
- WSL Linux Python: bridge and subprocess **52/52 PASS**.
- WSL optimized `python3 -O`: bridge **38/38 PASS**.
- Total: **148 targeted test executions**; compileall and diff --check exit 0.
- No full repository suite or neural learning result claimed.

## Hostile limits
- This proves conditional invalidity of a recorded prefix, not that the
  claimed model/provider/bytes are genuine or independently signed.
- Cross-user examiner isolation (#146), held-out mechanism custody (#147),
  independent signature, training generalization and graduation remain open.
- DRAFT/HOLD; no merge, model installation, deployment, protected runtime effect.
