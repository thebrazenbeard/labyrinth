# Minotaur admission gate and custodial state — source integration

These are bounded, independently invocable, INACTIVE library components.

labyrinth.graduation_gate verifies policy-frozen, evaluator-signed,
per-case paired heldout/transfer evidence. It distinguishes semantic
accuracy from valid output format, rejects wrong checkpoint identities,
invalid conditions, duplicate/replayed cases and missing controls, and
accounts for rejected attempts. The evaluator signing key must be held by
a separately trusted private evaluator; this repository does not
provision or hold such a key.

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
