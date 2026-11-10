# Minotaur admission gate and custodial state — source integration

These are bounded, independently invocable, INACTIVE library components.

labyrinth.graduation_gate verifies policy-frozen, evaluator-signed,
per-case paired heldout/transfer evidence. It distinguishes semantic
accuracy from valid output format, rejects wrong checkpoint identities,
invalid conditions, duplicate/replayed cases and missing controls, and
accounts for rejected attempts. The evaluator signing key must be held by
a separately trusted private evaluator; this repository does not
provision or hold such a key.

The experimental v2 attestation also requires a policy-frozen
`SemanticCaseCommitment` registry **authored and held independently of the
trainee and the evaluator's unilateral control**. Each entry binds attempt,
run, slice, case ID, prompt commitment and a reviewed **mechanical equivalence
class**, not a seed, wording or rotated-map fingerprint. Every preregistered
case must appear in its attempt; post-hoc favorable-subset selection blocks.
Each heldout/transfer slice needs at least eight distinct semantic families,
and paired significance counts conservative all-correct family clusters, not
wording variants. Training, evaluation slices and later attempts cannot reuse
a semantic family. Legacy v1 signed receipts explicitly return `BLOCKED`.

These remain conditional software checks, **not an authenticated semantic
oracle**. Source code cannot establish that a custodian classified mechanics
honestly, prompts were truly unseen or reported model checkpoints ever ran.
A colluding registry issuer can still fabricate distinct labels. Independent
registry custody, actual examiner execution and operational readback remain
**UNVERIFIED**; therefore no real qualification is established here.
The added consumed-family IDs are persisted in the same HMAC-protected SQLite
state, which does **not** prevent restoration of an entire previously valid
database snapshot. Anti-rollback/anti-replay across full-state replacement
requires a host-owned monotonic witness or append-only external custody;
neither is established by these unit tests.

labyrinth.graduation_state supplies transactional SQLite record
updates with a separate custodian-owned state-MAC secret. The user or
trusted operator must initialize each campaign and keep database/key
custody independent of the trainee. The library cannot defeat rollback
of a complete old database without an external monotonic anchor.

These files intentionally have NO automatic registration, model
deployment, release trigger, and no trusted private examiner attached.
A signed SYNTHETIC test fixture can exercise a GRADUATED decision,
but that never constitutes evidence about a real pretrained model.
Public Minotaur experiment reports remain unqualified. Do not
manufacture signed attestations from their report fields.

This limited source admission establishes the mechanics and their unit
tests only. Runtime security, evaluator operation, independent data
custody, and any actual promotion remain NOT_ESTABLISHED.
