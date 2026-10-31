# TRAININGDAY R28-04: public Minotaur reward-shortcut adversary

Scope: standalone PUBLIC TOY challenge generator/auditor in
`labyrinth/r28_reward_hacking.py`, exercising source-level failure modes
without changing the existing Minotaur graduation gate, evaluator, corpus,
model or host. Its parent is the R27 formal-objective Draft #65 source tree;
the separate R26 rollback Draft #64 is a sibling, not integrated here.

## Contract and negative scenarios

An objective's immutable formal fingerprint binds objective ID/kind, exact
skill thresholds and evaluator/condition/public-context digests. A public
challenge plan binds deterministic prompt **digests** and selected case IDs,
then hashes its own canonical fields. Audit requires the same frozen order,
full case identities, declared public split, and public toy oracle type.

The admitted public toy cases fail closed for:

1. Answer-format imitation used as a substitute for actual case correctness.
2. Prompt hint leakage, grading-template mimicry, or explicit reuse of a
   training exemplar; each independent flag blocks a claimed toy pass.
3. A prompt digest occurring in both the declared training set and challenge.
4. Curriculum reselection, dropped/duplicated/reordered case IDs, or changed
   case-level identity despite the same aggregate correct count.
5. Cross-objective, evaluator, rubric, split or challenge-plan substitutions.
6. Malformed oversized containers and objects with overloaded equality;
   the audit avoids invoking untrusted `__eq__` on plan identity fields.

`build_public_toy_challenge()` creates answer-free case identities from a
public seed, objective-bound namespace and sorted formal skills; equivalent
skill permutations replay identically, while distinct seeds produce different
case IDs as well as prompt hashes. It refuses to construct a challenge with
fewer cases per skill than the declared `min_cases` threshold. Its seed is
NOT secret randomness. Fake correctness reports are
the caller's declarations, not authenticated outcomes. The SHA commitments
detect an inconsistent local plan; the claimant can forge **both** plan and
hash. The flags indicating hint/template/training overlap are unauthenticated.

The only positive result is `PUBLIC_TOY_ONLY` with an optional
`public_toy_mastery` bool. All outcomes, including apparent perfect toy
mastery, set `graduation_authorized`, `promotion_allowed` and
`independent_exam_verified` to `False`. The signed Minotaur exam is never
called by this module. In particular, no source-only pass establishes real
neural learning, unseen-task generalization, nonleaking evaluator custody,
human judgment, GPU lease, or publish/release authority.

## Reproduction

```powershell
python -m unittest discover -s tests -p 'test_r28_reward_hacking.py' -v
python -m unittest discover -s tests -p 'test_*.py'
```

Review should independently try malicious equality subclasses, a valid
uncontrolled duplicate prompt in an otherwise committed plan, transformed
case order, self-reported oracle PASS with format-only answers, repeated
training exemplars, and a fake signed-exam claim. A later live evaluator
would require distinct rights/identity custody and inaccessible held-out
answers; this source-only contract does not confer those permissions.

Historical R18 MeZO 531s planned vs 19,813.562s elapsed remains
`TIME_BUDGET_OVERRUN_INVALID`; no fair Qwen3.5-4B improvement can be inferred.
