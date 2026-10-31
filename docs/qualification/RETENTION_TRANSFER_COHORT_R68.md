# R68 Q6 — Fixed retention and structural-transfer source cohort

Status: **SOURCE-ONLY / UNATTESTED / NON-AUTHORIZING**.

`labyrinth/retention_transfer_cohort_r68.py` adds a deterministic manifest
contract around the existing prior-skill retention development gate
(`retention_qualification.py`) and the R57 exact fixed-sample planner
(`retention_sample_planning_r57.py`). It deliberately does **not** modify the
existing examination decision, graduation, training run, workflow, or runtime.

## What is counted

| Domain | Independent unit for sample counts | Forms/cases | Invariants |
| --- | --- | --- | --- |
| PRIOR-SKILL retention | One predeclared semantic situation for one prior skill | Exactly one BEFORE and one AFTER form | Both forms count as **one**, not two, independent binomial units. Exactly the R57 planned number of situations is mandatory for each prior skill. |
| Structural transfer | One predeclared probe of one novel target derived from a declared prior skill | Exactly one case/prompt | Exactly the fixed per-target sample count is mandatory. Targets are distinct and cannot be prior-skill IDs. |

Each record must have a unique situation ID and structure-family ID *across
both domains*, not just within a skill. Case IDs must be unique across all
BEFORE, AFTER, and transfer probes. Prompt SHA-256 commitments must be unique
across all prompts and must not overlap the declared training inventory.
Reserved structure families must not overlap the declared training families.
These checks reject obvious deduplication, leakage, or reuse, but **do not
prove** that two differently labeled situations are semantically independent.

## Deterministic source-only protocol

1. An independent owner separately fixes the training inventory, policy,
   distinct pre/post checkpoint identities, R57 loss budgets and fixed sample
   counts, retention roster, novel transfer targets, and fixed transfer counts
   *before learner outcomes can be inspected*. Source code cannot verify this
   event or its timing.
2. Construct `StructuralTransferUnit` records and call
   `build_fixed_cohort_manifest(policy, retention_roster, sample_plan,
   targets, transfer_counts, transfer_units)`. Only record IDs, structure
   families, prompt commitments, a digest of the original retention policy,
   and fixed-count parameters appear in the canonical JSON. **Raw prompts
   remain outside the manifest.**
3. Independently hold the SHA-256 of the exact manifest bytes under a real
   prior custody record. Do not treat a hash computed after seeing results as
   a valid precommit. `verify_fixed_cohort_manifest(expected_digest,
   manifest_bytes, ...)` requires the separately supplied expected digest and
   reconstructs and compares the canonical byte sequence. A self-supplied
   matching digest has no evidentiary value beyond source consistency.
4. When source identities for outcome observations become available, call
   `verify_fixed_cohort_outcome_selection(...,
   retention_situation_ids=..., transfer_case_ids=...)`. The set must contain
   *every* declared unit exactly once, with no extras, omissions, duplicates,
   or replacements. It does not inspect any pass/fail scores.
5. Actual retention scoring remains in the separately governed retention
   qualification machinery. Transfer effectiveness, generalization, and
   graduation require independent reserved examinations and authority, not
   this source manifest.

The canonical JSON contains `schema =
labyrinth.retention-transfer-fixed-source-cohort.v1`, the retention policy and
roster digest, exact R57 statistical plan parameters and sample counts, and
the exact transfer cohort. No unsigned file or request field can silently
expand the cohort at observation time without changing the digest. A policy
with an identical roster but changed margins, checkpoints, or inventory also
changes the manifest's policy digest.

## Fail-closed boundaries

- 71 or 73 situations cannot silently replace a fixed 72-unit plan, even
  where the raw 64-unit policy floor would otherwise allow the run.
- Optional stopping, choosing successful cases after evaluation, duplicate
  identities, same-family reuse across retention and transfer, normalized
  prompt collisions, or a post-commit replacement fail the local source
  verifier if compared against a genuinely prior independent digest.
- An outcome subset cannot be declared complete merely by attaching another
  identifier or reusing one from a successful case.
- Frozen R57 plans are re-derived, and mutable frozen retention policies are
  revalidated at each boundary; source review results are tuple-backed and
  cannot be mutated to claim graduation authority.
- All returned roster-digest fields and later expected observation identities
  are derived from the **already validated canonical manifest**, never by
  rereading live caller-owned retention pairs after the manifest comparison.
  Deterministic `sys.settrace` adversarial tests mutate the live input after
  the check and verify that neither a posthoc hash nor an attacker-supplied
  replacement observation ID can be admitted as part of the sealed cohort.
- This implementation is bounded: 12 prior skills maximum (retention policy),
  12 novel transfer targets, up to 512 fixed probes per target and 4096 total
  transfer probes, canonical manifest at most 2 MiB. Oversize cases HOLD.

## Explicit limitations

This is a **development source assertion**, not a physical observation. It
does not enforce who provided the precommit digest, when it was recorded,
whether cases were disclosed or tuned against, whether the training inventory
is complete, whether semantic situations are truly independent, whether the
recorded checkpoints were actually loaded, whether inference was run, or
whether an independent custodian sealed the bank. A malicious producer able
to change both the manifest and its purported external expected digest can
pass this source check; genuine custody is an external prerequisite.

Python `@dataclass(frozen=True)` is **not** a process-level immutability
boundary: a hostile party sharing the interpreter can still use
`object.__setattr__`, tracing, or other process compromise to alter live
caller objects around validation. R68's review and observation-selection
operations use the validated manifest snapshot after comparison, preventing
those known post-validation reread inconsistencies. They do not assert that
the caller's mutable inputs remained unchanged afterward, nor offer isolation
against arbitrary hostile code in the same interpreter. Independently owned
case banks, process integrity, timestamped custody, and physical evidence
remain required outside this source module.

The public structural-transfer fixtures in
`tests/test_reserved_structural_transfer.py` are development fixtures and are
**not** a sealed final exam. The code never claims independent transfer,
neural learning success, host-level authorization, private-bank access, or a
promotion/graduation decision. The review status is always
`SOURCE_ONLY_UNATTESTED_FIXED_COHORT`, and all privilege properties are false.

## Source verification

```bash
python -m unittest -q tests.test_retention_transfer_cohort_r68
python -m unittest -q tests.test_retention_qualification tests.test_retention_sample_planning_r57 tests.test_reserved_structural_transfer
```

This scope intentionally does not alter Rezon, Minotaur, production
workflows, owner leases, or deployment authority.
