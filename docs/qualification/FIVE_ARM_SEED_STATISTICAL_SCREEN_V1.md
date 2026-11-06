# Five-arm seed-paired source statistical screen v1

Status: **SOURCE-ONLY / UNATTESTED**. This is a deliberately stringent,
fixed statistical decision rule for a later, independently collected physical
five-arm MiniLM experiment. It neither collects observations nor trains or
promotes a model. It is **not** a physical Q5/Q6 qualification result.

## Prespecified estimand

For each seed, run the five registered arms with the same starting pretrained
model, trainable parameters, AdamW optimizer moments, TRAIN visit order, and
declared compute budget. Correct-label learning is compared against all four
controls: zero-update, zero-learning-rate (optimizer steps still occur),
inverted-label, and randomized-label. Cases must be paired within seed and
split, and pseudoidentifiers must not recur in a different split or seed.

The evaluator supplies **only** a SHA-256 pseudonym and five ordered booleans
indicating whether each arm answered that same case correctly. No prompts,
reference labels, case-family details, answers or sealed final data enter this
source statistical module. Source-plan digests and case-set hashes are
structural checks; independent custody of those values has **not** been proven.

## Fixed rule

- 8–64 distinct plans/seeds with different run IDs, private transfer roster
  commitments, and private final-bank commitments. Shared initial pretrained
  model/optimizer, Git source, learning-rate budget and cycle configuration.
- 20–512 unique pseudoidentified development cases and 20–512 distinct
  transfer cases per seed. All five arms answer exactly each case. All case
  pseudoidentifiers must be unique **globally**, including between seeds.
- Prior-skill retention must independently satisfy the existing source
  `audit_prior_skill_retention` screen: at least 75% previously correct and no
  more individually regressed formerly correct cases than the frozen plan
  allows. Net accuracy cannot conceal forgetting.
- Seed is the independent comparison unit, *not* each response. For each of
  eight split/control hypotheses (development and transfer × four controls),
  compare the seed-level correct-arm accuracy to control accuracy, compute a
  one-sided exact binomial sign-test tail over positive/negative differences,
  ignoring ties, and apply Holm step-down familywise α=0.05 across eight tests.
- Each hypothesis must also have at least a **10-percentage-point** mean
  correct-label advantage across its seed-level accuracy differences.
- No threshold or number of comparisons is an API parameter. Changing this
  rule requires a new protocol and a genuine independently timestamped
  precommitment before evaluation, not post-hoc selection.

Eight all-positive seeds have one-sided p=1/256 for each hypothesis, which can
pass the first Holm threshold 0.05/8. Seven wins and one loss give p=9/256,
which is insufficient at the early thresholds. This illustrates how low-power
small-seed trials cannot be promoted because an average accuracy looks good.
The test is conservative for close effects and can lack statistical power even
where real learning exists; failure to pass is **not** proof that learning did
not occur.

## Unqualified boundaries

- The case pseudonyms may differ even when underlying prompts are semantically
  identical. This source screen cannot establish truly unseen transfer or
  semantic independence, and cannot prove heldout data stayed private.
- A caller who fabricates correctness bits, case digests, model plan digests
  or retention observations may fabricate a passing screen. No actual model,
  checkpoint, evaluator, hardware, process isolation, independent grader,
  signed time or blinded final-exam bank is consulted.
- Matched source plans and a favorable sign-test result do **not** establish
  causal neural learning. Physical checkpoint hashes, real model inference,
  matched control execution and independent source-to-runtime provenance are
  separate prerequisites.
- If the exact same development/transfer cases are evaluated repeatedly under
  multiple seeds, their pseudonyms must not be relabeled to evade the global
  uniqueness check. Semantic identity still requires an external custodian.
- `screen_five_arm_seed_outcomes` always returns
  `SOURCE_ONLY_UNATTESTED_STATISTICAL_SCREEN`, with all physical and authority
  flags false, even when `source_screen_pass` is true. It cannot authorize
  training, graduation, merging, deployment, or a final examination.

The historical three-seed MiniLM pilot used repeated small public development
cases; it is **not** eight independent evaluation seeds and must not be
silently relabeled as satisfying this screen.
