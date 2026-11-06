# R95 Q6 — structural-transfer falsification and owner-controlled examination

**Disposition: `SOURCE_ONLY_Q6_SEMANTIC_INDEPENDENCE_UNPROVEN`; no Q6 PASS.**

This is a hostile source review of `labyrinth/retention_transfer_cohort_r68.py`
at `74dae964a139b54be3433e4a3e72db365d0a199d`, not an independent
model examination. No model weights, evaluator, private bank, host, GPU,
network route, or external precommit custodian were accessed.

## Observed falsifiers

The R68 manifest refuses **exact** versioned normalized-prompt commitments,
duplicate IDs, and caller-declared reused `structure_family_id` values. It
does not compute semantic or grammar equivalence, and it deliberately does
not claim to. An adversary controlling only the public *declarations* can
invent fresh family/situation/case IDs for related prompts and produce a
canonically valid manifest. This is not a SHA-256 collision.

With the repository's `_fixture()` from
`tests/test_retention_transfer_cohort_r68.py`, three probes were admitted by
`build_fixed_cohort_manifest` while using a new declared structure-family ID:

| Probe | Exposed/training structure | Nominal transfer structure | Admitted manifest SHA-256 |
| --- | --- | --- | --- |
| Punctuation paraphrase | `canonical training prompt` | `canonical training prompt!` | `9de3020fe8cd7edc3c680a2b9807d1d9a2ce47dc0e533ecd6e4f86289df2362e` |
| Public template variant | `Novel symbolic structure transfer prompt 1` (another transfer item) | `Novel symbolic structure transfer prompt 100` | `d952901f71c10199c0f0b78732afd0c5983a5925ef5c16d2289b4e3c3662f5ea` |
| Algebraic inverse | TRAIN: `Calculate 2 + 3.` (added to the declared training inventory) | `Calculate 5 - 3.` | `b84df836275708a39b6dc20a4e515ef3d6d1c7f93a04db9c8a4cdca2fc4dea84` |

For the third probe the policy was reconstructed with both the training text
and its correct normalized SHA-256, retaining the same fixed-size retention
cohort. No digest was forged. The arithmetic expressions share a simple
inverse equation, but the contract permits independently declared family
labels. All these cases are in public source/test space; none is a sealed
examination item or evidence that an actual trainee memorized the answer.

Minimal reproduction (run from repository root; public fixtures only):

```python
from dataclasses import replace
from hashlib import sha256
from tests.test_retention_transfer_cohort_r68 import _fixture
from labyrinth.retention_qualification import normalized_prompt_sha256
from labyrinth.retention_transfer_cohort_r68 import build_fixed_cohort_manifest

policy, roster, plan, targets, counts, transfer = _fixture()

def substitute(prompt, p=policy):
    changed = transfer[0]._replace(
        prompt_text=prompt,
        prompt_sha256=normalized_prompt_sha256(prompt),
    )
    return sha256(build_fixed_cohort_manifest(
        p, roster, plan, targets, counts, (changed, transfer[1])
    )).hexdigest()

print(substitute("canonical training prompt!"))
print(substitute("Novel symbolic structure transfer prompt 100"))

training_text = "Calculate 2 + 3."
policy = replace(
    policy,
    excluded_training_prompts=policy.excluded_training_prompts + (training_text,),
    excluded_training_prompt_sha256s=(
        policy.excluded_training_prompt_sha256s
        + (normalized_prompt_sha256(training_text),)
    ),
)
print(substitute("Calculate 5 - 3.", policy))
```

These examples falsify the *stronger claim* that a valid R68 manifest means
structural independence or non-exposure. They do **not** falsify its actual
source-only exact-identity contract. Importing a few arithmetic regexes or
stripping punctuation/digits from arbitrary text would reject these examples
but would not establish general semantic independence; it could also reject
genuinely unrelated tasks and create a brittle, gameable assurance.

## Separate evidence classes

- **Source inspection:** R68 counts one retention situation (paired BEFORE and
  AFTER forms) or one novel-target transfer case per statistical unit; it
  compares exact canonical manifest bytes to a supplied digest and marks
  `precommitted_before_outcomes`, `independent_case_custody_verified`,
  `model_execution_authenticated`, and `graduation_authorized` false.
- **Local tests:** the unchanged R68 retention/transfer, R57 planner, prior
  retention, and reserved-transfer test modules pass: `46 tests, OK`.
- **Adversarial public probes:** all three unsealed near-aliases above were
  accepted as expected under the documented contract; this is evidence of a
  *claim boundary*, not a test of pretrained model learning.
- **Unverified:** the completeness and authorized use of the training
  inventory, model/checkpoint loading, genuine sample independence, sealed
  outcome custody, actual transfer gain, delayed retention, and Q6 approval.

## Host-owned Q6 examination needed to prove generalization

1. **Rights and exposure custody.** An independent data owner records rights,
   consent, deduplication, purpose constraints and protected-evaluation
   exclusions. Before training, the custodian opens a monotonically ordered,
   nonrollback register of *all* previously displayed exam prompts, public
   grammar templates, semantic equation aliases, and prior study results.
   Replaying a public generator seed is exposure, not a fresh private split.
2. **Blind grammar partition.** A separate examiner authors novel operation
   compositions and task grammars unavailable to trainers and source
   developers. Allocate full related families, algebraic inverses, operand
   permutations, paraphrases, code aliases, and isomorphic template variants
   entirely to **one** of TRAIN, validation, or reserved transfer. Have an
   independent semantic reviewer check cross-family leakage without sending
   sealed prompts, answers, or mapping rules to the trainee or its developers.
3. **Freeze before execution.** The independent examiner privately fixes
   model/checkpoint identities, adapter, inference settings, baseline and
   zero-update/inverted-target controls, evaluation rubric, exclusion roster,
   Q6 fixed sample counts, thresholds, and a timestamped digest before
   outputs are visible. Retention paired situations and transfer units remain
   distinct. The examiner alone retains case/answer custody and the actual
   semantic equivalence decisions; R68 can be one low-level integrity check.
4. **Authorized physical run.** Only after exclusive host/GPU lease release
   and authenticated route-owner grant, load real pretrained weights, prove
   checkpoint and optimizer/RNG durable restart, and run identically configured
   baseline/trained/controls with authenticated execution receipts. Keep
   source reports, runtime logs, evaluator observations, and model review as
   separate evidence classes.
5. **Independent inference and retention.** Score never-trained structural
   families and compositional operations by preregistered target/stratum;
   compare against controls rather than only pre/post score. Evaluate actual
   prior skills at a separately delayed checkpoint using the exact R57
   planned paired-situation counts and adjusted deterioration bounds. Release
   aggregate observations and signed custody claims only, not a reusable
   private exam bank. Failed thresholds and missing authority are HOLD.
6. **Fresh adversarial follow-up.** An examiner who has not seen model tuning
   choices supplies a second truly new grammar bank and attempts to falsify
   apparent gains with lexical shortcuts, inverted operations, relabelings,
   alternate normalizations, near-duplicates, and prior benchmark exposure.
   Log model/provider/runtime, exact artifact head and disagreements. A
   public model-review consensus is not independent proof of learning.

**No code change recommended inside R68 from this finding.** The generic
manifest cannot enforce truth of caller-chosen semantic families without an
independent semantic owner and private custody. An unqualified source-only
limitation is safer than a cosmetic parser advertised as a generalization
gate. Q5/Q7/Q8/Q9 and merge/deployment remain separate decisions.
