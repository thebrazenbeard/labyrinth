# Minotaur Wave62 — saved evaluation history exact registry match (2026-11-14)

Base: Draft PR #243 head `c8e489000babd0725e6eb72061d2deeda34e5722`.
Branch: `fix/minotaur-graduation-registry-history-20261114`.

## Hostile falsifiers (Lappy, Windows Python 3.14.5)

Wave61 rejected missing or undersized prior history, but **did not**
reject the same number of forged history elements. The public
`decide_graduation` function accepted the following right-sized, bogus
saved `GateState` collections despite a matching policy fingerprint:

1. An unknown consumed run ID.
2. An unknown final evaluation case ID.
3. An unregistered prompt commitment digest.
4. An unregistered semantic-family digest.

A pre-fix hostile test demonstrated **four subtest failures**: all four
invalid prior-success states returned `GRADUATED` instead of `BLOCKED`.
A second hostile test replaced one of the saved first-attempt prompt
commitments with an unrelated 64-character digest, retained its count,
and submitted a synthetically signed second attempt. Pre-fix return was
`GRADUATED` rather than `BLOCKED` (one assertion failure).
No real private test set, candidate training or checkpoint read occurred.

## Source-only correction

For all completed attempts (1 through `state.next_attempt - 1`),
reconstruct from the already-frozen independent semantic registry:
the exact declared run IDs and the complete union of final-slice
case IDs, prompt commitments, and semantic-family IDs. Refuse a cached
success or a later signed attempt unless the persisted consumed sets
are identical to these registry-derived sets. Exclude the predeclared
training roster from evaluation consumption. Preserve existing minimum
count/attempt consistency checks from #243.

This is a stronger necessary condition for source-level state integrity
and leaves the v2 receipt schema, signing mechanism, and policy frozen.

## Test evidence

- RED: saved-success substitutions: four assertion failures, exit 1.
- RED: poisoned-history signed second attempt: one failure, exit 1.
- GREEN: both same hostile tests exit 0; 28 graduation-gate tests pass on
  Lappy Windows Python 3.14.5.
- WSL Ubuntu26.04: 28 graduation-gate tests normal and 28 optimized PASS.
- Full-suite and exact-commit hosted CI to be attached separately.

**Limits:** A malicious party able to control the trusted registry, signer,
or state and craft fully consistent lies remains outside source protection;
honest semantic-family equivalence, OS-enforced examiner-private custody,
correct checkpoint execution and genuine neural generalization are NOT
qualified. Labyrinth issues #146/#147 stay open. Do not graduate, train,
install, merge protected work, or deploy on this test result.
