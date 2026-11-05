# Public structural-transfer development fixture v1

This is a source-frozen, **public development** challenge set. It is not an
independent private Minotaur exam and must not be used to tune the final model
and later described as untouched test evidence.

- Generator: `labyrinth.maze.generate_maze`, with tier 4 supplied by
  `labyrinth.program_maze.generate_program_challenges`.
- Seed: `20261030`; split: `devstruct_v1`; tiers: operation-chain (2),
  source-priority (3), register-program (4); eight cases per tier (24 total).
- Canonical fixture payload SHA-256:
  `1ce1da3f8a6c1e2367f634986c9ecd62e442de4f91296f11826b9f9ee5c5af13`.
- Reproduction: `python -m unittest tests.test_reserved_structural_transfer -v`.
- Oracles: independent reading and execution of printed arithmetic chain,
  authoritative source priority (ignoring quoted injection), and register
  machine instructions. The test verifies generator gold against these oracles.
- Isolation: the test checks ID and prompt disjointness against the corresponding
  `train` split; register-program train structures are also disjoint from
  structural-holdout patterns. Public source alone cannot prove no future
  training run has seen the cases.

## Qualification use

Evaluate identical precommitted cases against frozen, correct-label,
zero-update, and corrupted-label arms only after train budgets and model
selection are frozen. Record the exact model checkpoint bytes/adapter digest,
evaluation process, generator commit, baseline skill retention, and seed-level
variance. A final untouched private test must be generated and controlled by
an evaluator not exposed to the trainer.

The existing four-parameter `ToyLinearAdapter` accepts only canonical tier-1
arithmetic, and correctly **rejects all 24 cases**. Thus its present
same-family improvement cannot be reported as operation-chain, source-priority,
or unseen register-program transfer. No pretrained-language-model training or
transfer has been executed as part of this fixture.
