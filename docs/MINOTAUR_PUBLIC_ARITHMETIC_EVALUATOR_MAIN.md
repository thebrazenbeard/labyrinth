# Independent public arithmetic evaluator — inactive research utility

This module supplies a dependency-free, separately callable evaluator
for the bounded Minotaur modulo-97 arithmetic family (ADD/SUB/MUL).
It recomputes atomic labels from operation and operands, validates
prediction identities and label provenance, distinguishes abstention
coverage from unconditional correctness, and reports training-pair
overlaps. For six-step programs it recomputes public trajectories,
checks the claimed first-error position, and scores a separate
prediction callback that sees only the current operation and operands.

Invalid case batches refuse before callbacks are invoked. Unhashable
operations, malformed output identifiers, invalid residues, unbounded
group IDs, and mismatched labels cannot be silently scored as correct.
This evaluator contains no neural model, optimizer, checkpoint loader,
signing key, deployment trigger, or private question custody.

First-error accuracy is **not** full-state rollout accuracy. The
public generator may create structurally related cases; no IID
confidence estimate is invented. Even 100% public arithmetic scores
do not certify independent private Minotaur graduation. Runtime
authorization and Vera activation are entirely outside this module.
