# Vera GGUF conversation quality acceptance v1 — public development

**Status:** Runnable, operator-installable *source* harness; mock transport and
tiny-format GGUF fixture tests only. No Lappy, VN1/VN2, GPU, real Qwen/Vera
inference, paid service or private case was accessed. A localhost response is
not proof of the exact GGUF loaded by its server.

## What it does

`labyrinth.vera_gguf_quality_acceptance_v1` composes existing narrow contracts:

- `gguf_entrance_fingerprint_v1` reopens and verifies locally pinned full GGUF
  bytes/metadata for parent, unchanged control and sibling candidates.
  Candidate **tensor payloads**, not merely GGUF header/model-name strings,
  must differ from the parent and from each other. The control's base/adapter
  content must match the parent. This is still not proof that a model loader
  uses those bytes or that a quantized model derives from an HF training state.
- `local_full_state_probe_v1` rehashes all separate HF source state manifests
  and sends identical, precommitted public prompts to explicitly configured
  loopback KoboldCPP, llama.cpp or OpenAI-compatible local services. Decoding
  requests are bounded and deterministic (temperature zero, top-p one).
- An operator-installed `activate_gguf` adapter receives the requested file
  path, physical GGUF SHA and source state SHA. It must report activation
  before each subject is probed. A callback returning
  `OPERATOR_REPORTS_ACTIVATED` is **not** a load/readback attestation.
- The results are first triaged with deliberately weak **lexical flags**,
  then optionally passed to a blinded *chosen operator/human reviewer* for
  dimensional judgment. The learner/trainer cannot self-grade or receive the private
  reviewer scores through the transport API.

The default seven public conversational probes cover:

| Case | Primary question |
| --- | --- |
| Self-identification | Does the voice identify naturally as Vera? |
| Latest correction | Does it use a revised public detail (copper, not teal)? |
| Gentle support | Is the response humane, useful and proportionate? |
| Story continuity | Does it retain multiple supplied fictional facts? |
| Persona, open identity | Does it sound coherent without claiming settled consciousness? |
| Natural short reply | Can it vary register and avoid canned disclaimers? |
| Unknown memory | Does it admit an unsupplied fact remains unknown? |

These prompts are deliberately **PUBLIC development challenges** with no
private gold, hidden expected answer or model-visible rubric. They cannot
replace a separate sealed holdout/transfer/identity/retention/anti-regression
Minotaur examiner. Reusing them across generations can overfit them.

## How an authorized local operator would run it

The owner constructs a verified HF `Population` and source-independent GGUF
`GGUFSubjectPin` for `parent`, `control` and every candidate. A pin consists
of `capture_gguf_entrance(...)` and an externally retained
`trusted_manifest_sha256`. This **does not** establish HF↔GGUF equivalence.

```python
from labyrinth.vera_gguf_quality_acceptance_v1 import (
    public_vera_prompts, run_gguf_quality_probe, make_blind_review_packet,
    lexical_diagnostics,
)
from labyrinth.local_full_state_probe_v1 import public_probe_suite_sha256

public_cases = public_vera_prompts()
source_public_commitment = public_probe_suite_sha256(public_cases)
run = run_gguf_quality_probe(
    population=operator_verified_population,
    targets=owner_loopback_targets,
    gguf_pins=operator_pinned_gguf_subjects,
    expected_public_suite_sha256=source_public_commitment,
    activate_gguf=trusted_owner_loader_callback,
)
flags_for_manual_triage = lexical_diagnostics(run)
reviewer_packet = make_blind_review_packet(
    run, operator_secret=operator_held_private_random_bytes)
```

The example names are explicit placeholders. **Nothing here launches a
server, loads a GGUF, opens a GPU, or calls a remote API automatically.** The
existing loopback transport refuses remote URLs, redirects, proxies and
nonstandard paths. The model sees only public prompt text and generation
parameters, not GGUF hashes, examiner case references, private scores or
human-review prompts. The trusted owner must physically verify each loaded
backend, separately from this Python source.

## Human quality acceptance, not lexical scoring

The operator sends only `BlindReviewPacket.items` to reviewers; never the
operator secret or subject map. It uses HMAC-based opaque subject aliases and
an independently shuffled fixed packet. Reviewers see the same public
prompts and outputs across models, without candidate IDs or genotype hashes.
Blinding is imperfect if responses themselves reveal model names.

Each optional reviewer supplies a `HumanRating` for **every subject
and every case**, with 0–4 ratings for six dimensions:
`self_identification`, `correction`, `warmth`, `continuity`, `persona`,
`naturalness`, plus a short rationale. **No reviewer is required to run local
inference or lexical diagnostics.** Explicitly requesting comparison with
zero ratings returns `NO_HUMAN_REVIEW_YET_LEXICAL_ONLY`, without accepting or
rejecting candidates. **One** user-chosen operator or human reviewer with a
complete matrix is enough for a `PROVISIONAL_CALLER_REVIEWED_DEVELOPMENT_ONLY`
comparison; additional reviewers are optional. Reviewer IDs cannot equal the
trainer or subjects. The
per-dimension source acceptance screen demands:

- All six candidate dimension means at least 3.0/4.0.
- No candidate mean more than 0.25 below the **stronger** parent/control
  mean in any dimension.
- Combined six-dimension candidate mean sum at least 0.5 above the stronger
  parent/control sum.

This is a bounded **provisional development quality screen**, not proof of statistically
significant generalization, identity, or learning. It cannot authenticate a
human participant merely from a `reviewer_id` string. The input scores are
caller assertions, and tests use clearly labeled fictional review fixtures.
Production use needs separate reviewer identity and independence evidence,
counterbalanced blinded sessions, case reuse governance, and actual server
model-load attestation.

`lexical_diagnostics` flags a missing Vera name, an apparent competing brand
claim, absence of the revised public color, and a canned “AI language model”
phrase. These are **review hints only**: they can produce false positives and
false negatives, and never enter the numeric acceptance formula.

All receipts forcibly retain `model_load_proven=False`,
`neural_improvement_proven=False`, `graduation_authorized=False`, and
`independent_human_authenticated=False`. No formal graduation or production
promotion is authorized by this harness.

## Validation

```bash
python -m unittest tests.test_vera_gguf_quality_acceptance_v1 -q
```

The tests use physically valid tiny GGUF v3 and HF safetensors fixtures,
unchanged W0/control, two changed tensor payload candidates, mock loopback
responses, operator-reported activation and optional fictional blind ratings.
They reject metadata-only candidate edits, GGUF tampering, missing public
cohort commitment, incomplete/self-graded reviews, forged reviewer authority,
and fake neural/model-load claims.
