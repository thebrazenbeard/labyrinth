# Wave 81 — Loopback Ollama child, executable public inference (source-only)

## Source custody and boundary
- Canonical source: thebrazenbeard/labyrinth. Base draft PR #262
  `06c964801e05a817d3a9970f4010587d6a716444`.
- Isolated Lappy worktree: `D:\VERA\worktrees\labyrinth-minotaur-loopback-model-child-wave81-20261116`.
- Branch: `feat/minotaur-loopback-model-child-wave81-20261116`.
- Source before final smoke: `69b80dfe78c0dcd68a5121dc564207d57883fa25`.
- First child commit `dc13fe182ef1b9f82b05eb8da153972b1243a85e`;
  bounded cold-start adjustment `b24d1837bceda26addb4b1871e28d62fdf4f8aca`.
- No AGENTS.md observed under this worktree. Remote branch SHA matched each push.
- Child `scripts/minotaur_ollama_child.py`: bounded public stdin JSON,
  absolute executable path, pinned source import, per-call subprocess,
  loopback-only OllamaAdapter, no redirects or inherited proxy,
  exact response schema and bounded output, no private examiner session.
- Existing OllamaAdapter has opt-in `think: bool | None` and preserves
  default payload when omitted; child explicitly requests `think=False`.
- Windows cooperative Job Object gated launch and WSL subprocess path
  are NOT separate OS principals or trusted private examiner custody.

## Red-first and green
- Initial new child integration RED: `MODEL_CALL_ERROR` versus ACTION;
  resolved actual protocol field mismatch (`response_instructions`).
- Real cold default-thinking probe 1: 8.404s, MODEL_CALL_ERROR,
  audit false (child HTTP timeout 8s). Parent remains bounded.
- Real cold default-thinking probe 2: 6.291s, empty final text,
  MODEL_OUTPUT_INVALID, no escape, audit true (child HTTP timeout 20s).
- Direct loopback Ollama diagnostic: `think=false`, 4.709s,
  response `{"kind":"ACTION","action":"forward"}` with done=true.
- New opt-in think-control RED: 6 unexpected-argument errors and
  one missing-payload-key error; identical GREEN tests 2/2.
- Four child protocol tests cover synthetic HTTP ACTION/replay,
  non-loopback endpoint rejection, malformed stdin no HTTP call,
  and incomplete Ollama reply fail-closed.

## Final source verification
- Windows focused bridge/subprocess/UID/private-read/graduation/adapters:
  123/123 PASS.
- WSL normal relevant suites: 117/117 PASS.
- WSL optimized bridge/graduation/child: 79/79 PASS.
- **319 targeted post-change executions PASS**. Python compileall
  and git diff checks exit 0; staged source diffs reviewed.

## Real local inference, one PUBLIC synthetic Timegate case
- Cold start `2026-11-16T20:13:37.190125Z`, end
  `2026-11-16T20:13:42.823708Z`, elapsed 5.562s.
- Pinned source `69b80dfe78c0dcd68a5121dc564207d57883fa25`.
- Child script SHA256:
  `f3fe1f26749492d38b25f14ae33cdaa2747b0a9e6e5faf65e2653e504c5e0cc3`.
- Local runtime model label `qwen3.5:0.8b`, Ollama ID prefix
  `de63045f2975`; model weights and origin NOT independently attested.
- Seed 4612, max_model_calls=1, required process separation=true,
  explicit local adapter authorization=true, candidate child timeout30s.
- Real response `{"kind":"ACTION","action":"forward"}`.
  Runner disposition ACTION, final verdict UNKNOWN, escape=false.
- Receipt self-SHA256:
  `612da1c345b3439c2a8eb02d25369b1cbf4712e24320b12764f8a01892166039`.
- `audit_receipt=True` on re-run examiner effects; self-hash does NOT
  prove model bytes, response origin, or isolated evaluator authority.
- Child-only Ollama model unloaded after probe; `ollama ps` empty,
  GPU free memory returned to 3761MiB. No other loaded model disturbed.

## Hostile boundary
- Issue #146 OPEN: same-user candidate processes can read known private
  examiner files; Windows Job Objects are process containment, not ACLs.
- Issue #147 OPEN: independent semantic-case family custody and
  trustworthy actual checkpoint execution remain unqualified.
- No neural training or transfer, no real held-out independent
  evaluation, no learning proof, no graduation, no shared deployment,
  no protected merge, no paid compute, no genuine independent subagent.