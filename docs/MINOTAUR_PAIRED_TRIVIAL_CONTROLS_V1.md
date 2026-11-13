# Wave40: matched trivial-control falsification for blind structural exams

**Source-only methodology.** This protocol compares exact responses to trivial strategies on the **same case roster**. It reports descriptive two-sided exact paired discordance probabilities; it does not certify learning, generalized reasoning, independent cases or trusted model execution.

## Why paired comparison is necessary

A candidate can be scored against 12 transition systems, but two separately reported scores do not reveal which exact cases each control got right. A claim of neural improvement requires frozen, matched cases, matched generation settings, protected answer custody and a real checkpoint/control lineage—not just a larger raw number.

This module reads the same case-source, original precommitment, candidate response file and optional pinned Wave39 local HTTP receipt for every comparison. It computes the strict candidate score using the Wave38 action replay, then derives two deterministic zero-update controls from those **same** case rows:

- **Always UNSAT:** outputs `UNSAT` for every case; succeeds precisely where exhaustive search proves no path exists
- **Fixed ABC:** outputs `PLAN: A B C` for every case; succeeds only if that exact sequence reaches the goal under the same two-phase rules and move budget

These controls are deliberately weak. Other policies, task-aware symbolic solvers or stronger baselines may do better; this module does not claim that either trivial control is the best possible model-free strategy.

It records for each baseline the paired cells: candidate-only correct (b), control-only correct (c), both correct and both wrong. It also reports the exact two-sided conditional McNemar/binomial calculation

```text
n = b + c
p_two_sided = min(1, 2 * sum(comb(n, k) for k=0..min(b,c)) / 2**n)
```

For `n=0`, report `p=1.0`. No one-sided test, multiple-comparison correction, independent-case assumption or effect-size uncertainty is silently asserted. The calculations are **descriptive on this single fixed exam**; no automatic declaration of significance or training transfer is emitted.

## Usage

Given a previously prepared source exam and actual candidate responses:

```powershell
python -m labyrinth.blind_control_comparison --private C:/exam/private.json --public C:/exam/public.json --precommit C:/exam/precommit.json --responses C:/exam/model-responses.json --expected-precommit-sha256 ORIGINAL_RETAINED_COMMITMENT_SHA --out C:/exam/paired-trivial-controls.json
```

If the responses have an original Wave39 HTTP witness, add the following before `--out`:

```text
--observation C:/exam/model-http-observation.json
--expected-observation-sha256 ORIGINAL_RETAINED_HTTP_RECEIPT_SHA
```

The output is exclusive-create and binds the original private/public case bytes, precommit SHA, exact response roster, response-file SHA and (when supplied) the original HTTP witness SHA. A changed model response, substituted case, duplicate trial, missing observation half or changed original pin fails closed.

The report always sets `neural_learning_verified=false`, `independent_custody_verified=false`, `served_weight_bytes_attested=false` and `claim_ceiling=NO_NEURAL_OR_GENERALIZATION_CERTIFICATION`.

## Observed controls on Lappy, 2026-11-13

The following are **real local model response observations and source-scored controls**, not protected held-out graduation exams. The exact source-head evidence and private artifacts remain on Lappy outside Git.

| Exam | Qwen3.5 0.8B | Always UNSAT | Fixed ABC | Candidate-only vs UNSAT | UNSAT-only vs candidate | Exact paired p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 fresh random transition systems | 1/12 | 6/12 | 0/12 | 1 | 6 | 0.125 |
| 4 caller-authored counterfactual near-twins | 1/4 | 2/4 | 1/4 | 1 | 2 | 1.0 |

The 12-case answers came from actual local Ollama CPU-only HTTP responses captured separately in Wave38, without an independently retained HTTP observation digest at the time. Its Wave40 comparison therefore **correctly reports `invocation_evidence=NOT_SUPPLIED`**. The 4-case responses were reproduced with the Wave39 observer, a local declared Ollama model digest and an original observation pin; the output reports `LOCAL_HTTP_SELF_REPORTED_NOT_ATTESTED`. Neither scorer claims to authenticate served model-weight bytes.

A `p=0.125` does **not** prove parity, equivalence, or success; it also does not meet a conventional 0.05 threshold if the test's assumptions were applicable. Its purpose here is to avoid converting `1/12 vs 6/12` into an unqualified population-level claim. The four-case comparison is too small for a broad inference. The generator intentionally balances SAT/UNSAT by construction, so an all-UNSAT 50% control is expected; it is not a natural-world prior.

## Hostile limits

- Model-source code and public transition-machine grammar are available; *new tables* do not imply an unseen semantics family
- A symbolic parser can score 100% without updating any neural parameters
- Exact-case replay cannot prove a model actually emitted the recorded text unless an independent witness preserves it; Wave39 local HTTP self-witness is still operator-controlled
- A precommit and HTTP receipt hash are not an external signature, and an Ed25519 signature on the precommit is not proof of signer separation or execution
- Bounded `1..16` cases produce noisy comparisons; no matched trained / zero-update / corrupted-label model branch or independent test custodian is established
- This protocol cannot release Qwen TrainingDay's physical-launch HOLD, automatically admit a new Minotaur rule, merge protected branches, or claim general reasoning

Reproduce:

```bash
python -m unittest discover -s tests -p 'test_blind_control_comparison.py' -q
python -O -m unittest discover -s tests -p 'test_blind_control_comparison.py' -q
python -m unittest discover -s tests -q
```
