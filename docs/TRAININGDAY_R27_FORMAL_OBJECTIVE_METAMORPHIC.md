# TrainingDay R27: formal objective metamorphic invariants (public-only)

This is a **source/CPU fixture**, not a pretrained-model training result,
private examination, evidence of independent evaluator custody, or graduation
authority. It builds on the R25 Labyrinth–Minotaur handoff interface without
changing that existing gate. No private exam answer, evaluator key, or protected
dataset is required or read.

`compare_public_objective_metamorphs` checks precisely what can be established
by deterministic public formal declarations:

- Reordering already-pinned **formal** skill criteria, exclusions, and public
  case records preserves a canonical contract fingerprint and reported
  *public-development practice* scores.
- Presentation wording is expressly **nonnormative**. Changing the prose label
  never proves that two natural-language objectives mean the same thing.
- Objective ID/kind, skill set, metric identifiers, integer score floors,
  assessment context, evaluator identity, and excluded scopes are normative.
  Their alteration produces `SCOPE_OR_RUBRIC_CHANGED`; malformed/ambiguous
  declarations fail closed. Expanded empathy or other new objectives never
  silently inherit a prior graduation.
- Every case has an exact ID, formal skill, correctness bit, output-format bit
  and `public_development` split. A duplicate case, changed score, reassigned
  case, training/private split, undeclared skill or subclassed record is
  rejected, even if aggregate accuracy would remain the same.
- Normative primitive fields are type-checked before canonical hashing; a
  caller-defined equality object or `str` subclass cannot impersonate a
  supported objective kind and reach JSON serialization.
- The result **always** has `graduation_authorized=false` and
  `promotion_allowed=false`, including when all public practice criteria pass.

Acceptance is intentionally narrow: this cannot determine natural-language
semantic equivalence, ontology-specific skill mastery, unbiased evaluation
cases, fresh private scores, independent evaluator keys, adapter weight
changes, or full Qwen3.5-4B behavioral improvement. Formal IDs and public
case-score claims are provided by the caller and unauthenticated. Any actual
Minotaur graduation still requires a new evaluator-custodied private exam and
separate host/rights/model qualification.

Tests: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest -q
tests.test_trainingday_objective_metamorphic`.
