# TrainingDay R29 — public-toy Minotaur objective regeneration

**Evidence class:** source implementation + deterministic CPU unit tests, **not** a physical Minotaur model run, private independent exam, authenticated rights check, or neural improvement.

## Bounded implementation

`labyrinth.r29_objective_regeneration` accepts the R27 `FormalObjectiveContract` and an exact R28 `FrozenChallengePlan`/`ChallengeObservation` failure. The local R28 audit must report an actual flagged shortcut (`PROMPT_HINT_LEAKAGE`, `GRADING_TEMPLATE_IMITATION`, or `TRAINING_EXEMPLAR_REUSE`) with valid case identities and split; a clean run or stale/inconsistent case replay is rejected.

The caller freezes public declarations for the objective fingerprint, evaluator/conditions/context, rights SHA-256, and reference SHA-256. One SHA-256 boundary commitment binds these declarations. **These are caller-supplied fingerprints**, not evidence that permission currently exists or that a reference has been independently authenticated. Rights changes require a new admission/review workflow, not silent regeneration under the old boundary.

For each accepted regeneration, deterministic domain-separated seeds generate disjoint public development and **untouched public challenge** case identities and prompt digests. The seed namespace binds the exact weakness categories **and** template-family declarations; a hint-leak failure cannot silently recycle a grading-template-attack corridor's identities. The challenge plan records all development prompt digests as excluded training hashes. Public per-skill case counts cannot fall below the frozen objective's `min_cases` or the previous failed challenge's per-skill counts. Every skill declares one development and one challenge template-family digest; the digests must be distinct across all skills and splits. This catches **declared exact-family duplication**, not semantic paraphrase leakage or dishonest digest declarations.

Every receipt commits to the objective/rights/reference boundary, exact failed source plan commitment, previous generation receipt commitment, generation index, seed, shortcut categories, template declarations, and both generated plans. A successor must consume the exact predecessor's challenge plan, prove a **new** valid shortcut failure against it, increment the generation by one, change its seed, and use template-family digests disjoint from **both** the predecessor's development and challenge families. This rejects stale receipt replay, source switching, same-generation reuse, and one-hop cross-generation template contamination within the locally committed lineage. The new public challenge is deliberately **not scored or trained on** by this code.

`verify_public_regeneration_receipt` reconstructs deterministic plans and checks local hashes and disjointness. For generation 2 or later it **requires** a separately supplied, exact committed predecessor receipt; for generation 3 or later the `predecessor_ancestors` tuple must additionally supply every preceding generation in order. Verification checks each local receipt and each transition, including template-family quarantine. Thus a self-rehashed generation-2 receipt cannot declare itself valid by omitting the previous generation, changing its committed reference, or reusing the previous challenge template family. It does not attest the truth of caller observations or SHA labels, inspect prompt/template content, or examine protected case banks.

## Security and qualification boundaries

- This is a *public synthetic curriculum identity generator*, not a production task/prompt generator or trained taskmaster. Real text, difficulty semantics, and template paraphrase disjointness require separately held challenge content and an independent evaluator.
- `graduation_authorized`, `promotion_allowed`, and `independent_exam_verified` remain `False` in the receipt and verification verdict, including on a locally valid toy receipt.
- No model weights, QLoRA/MeZO optimizer, GPU lease, private Minotaur challenges, live host, deployment or merge is touched.
- A signed/fresh rights and consent decision, sealed holdout custody, neural-update and fresh-process reload receipts, and an independently examined exact-head PR remain separate gates. This code creates no substitute proof.

## Reproduce

Run from the repository root:

```bash
python -m unittest tests.test_r29_objective_regeneration -v
python -m unittest discover -s tests -p 'test_*.py' -q
git diff --check
```

Hostile test cases cover deterministic seed/weakness/template-conditioned case IDs, exact train/challenge case and digest disjointness, no shortcut, same-family leakage (including cross-skill and cross-generation), wrong objective/rights/reference, independently supplied predecessor custody against self-rehashed receipts, tampered receipt, impossible `min_cases`, downgrade from the previous curriculum, stale generation/seed/source observations, and valid N→N+1→N+2 lineage without promotion.

Review should be **bound to the exact Git head**; head movement requires another independent hostile review. The R18 over-budget MeZO artifacts remain invalid for equal-time comparison and are unrelated to this source-only work.
