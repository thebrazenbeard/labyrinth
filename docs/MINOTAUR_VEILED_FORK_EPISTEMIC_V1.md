# Minotaur Veiled Fork — epistemic adversarial examination v1

**Status:** source-public experimental examiner. Not private held-out evaluation, formal runtime qualification, model training, neural generalization, or graduation authority.

**Mechanism:** The observer cannot inspect the hidden world directly. A common public policy may spend a bounded number of actions on probes whose answers are determined by that world; it then commits to one of two exits. A guaranteed success means **one observation-contingent policy wins in every world compatible with the declared scenario**. This is different from the much weaker claim that every world has *some* clairvoyant winning sequence.

## Falsifiable cases

| Mechanism | Budget | Expected finite-world policy verdict | Counterfactual |
| --- | ---: | --- | --- |
| Unobservable fork | 4 | `PROVEN_UNSAT_UNDER_DECLARED_RULES` | Each world is individually winnable, but all probes give the same observation and exits conflict. |
| Identifiable fork | 2 | `VERIFIED_WIN` | One discriminating probe followed by a conditional exit. |
| Too late to know | 1 | `PROVEN_UNSAT_UNDER_DECLARED_RULES` | Same worlds and probe as identifiable fork, but the probe leaves no attempt for exit. |
| Two-signal mosaic | 3 | `VERIFIED_WIN` | Distinguishing exit requires two independent signals in the worst case. Budget 2 is unsatisfiable. |
| Adaptive branching advantage | 3 | `VERIFIED_WIN` | No fixed two-probe sequence can resolve opposite-exit worlds; a conditional gate→left/right→commit policy can. |
| Common safe exit | 1 | `VERIFIED_WIN` | Every world shares exit 1; no probe is required. |

These are **six non-independent counterfactual cases in one epistemic-policy mechanism**, not six independent structural families for graduation statistics.

For each scenario, the independent Boolean partition oracle is compared with a policy-producing belief-state search, then the proposed policy is replayed *unchanged* against every declared world. The two search routines do share the stated observation matrix; this is a finite declared-rule check, **not** independent physical evidence that those semantics are true.

## Verdict boundaries

- `VERIFIED_WIN`: a public contingent policy is actually replayed successfully for all declared worlds. This is a **robust-policy** win, not proof that a neural model found it.
- `VERIFIED_FAILURE`: the submitted policy fails in at least one world, including invalid or exhausted attempts. One lucky actual-world success cannot be promoted to a guarantee.
- `PROVEN_UNSAT_UNDER_DECLARED_RULES`: complete bounded belief-state search shows no such robust policy. It does **not** prove every possible world is inescapable, nor exclude a creative action absent from the rules.
- `NO_KNOWN_WIN`: explicit search node limit stopped before completeness. Never upgrade to unsat.
- `UNKNOWN`: defective or semantically unparseable input/review state, or evaluator disagreement. Does not mean victory or proof of impossibility.
- `NOVEL_RULESET_CHALLENGE`: an alleged *new* action is quarantined for independent review and possible new ruleset. It never mutates an existing UNSAT proof or certifies victory by itself.

## Hostile record

The initial policy replay falsely certified `VERIFIED_WIN` for an unobservable, provably policy-unsatisfiable two-world fork. A malicious candidate supplied two `str` subclass keys in the branch dictionary; one lied that it equaled the same observation string on its first lookup only. World 0 was routed to exit 0 and world 1 to exit 1 despite both seeing identical public `same` feedback. The exact regression is `test_stateful_branch_key_cannot_spoof_omniscient_policy`. It failed RED and then passed GREEN after a complete primitive-key/value copy and validation before replay. Invalid inputs now fail closed, charged once, and cycles/depth are bounded. A recursive Python policy is **not** a candidate-process isolation boundary: this is source-level input integrity only.

## Controls and commands

```shell
python -m labyrinth.veiled_fork
python -m unittest discover -s tests -p test_veiled_fork.py -q
python -O -m unittest discover -s tests -p test_veiled_fork.py -q
python -m unittest discover -s tests -q
python -O -m unittest discover -s tests -q
```

The source-only fixed-exit policy is compared to a deterministic symbolic planner across six related cases. Fixed policies carry **zero optimizer updates**. The symbolic planner performs exact source-public inference; its successes do not establish neural learning. A 1,000-case seeded randomized differential check helps detect mismatches within the stated oracle; it is **not** 1,000 independent structural families. Published code and seeds are visible to the candidate; no secret holdout custody, neural parameter delta, model weight identity, or checkpoint-reload evidence is provided.

No changes to protected branches, installations, services, host model runs, or graduation gates are proposed here.
