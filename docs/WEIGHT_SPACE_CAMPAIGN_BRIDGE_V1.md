# Source-only Ariadne/Minotaur weight-space generation bridge

**Status:** tested physical integration of `weight_space_spark_v1` with the
existing Ariadne WAL and evolutionary population seam. No model was loaded,
run, examined, trained by optimizer, or shown to improve. This bridge accepts
only W0 (`generation_index == 0`) and constructs one W1 generation.

`labyrinth.weight_space_campaign_bridge_v1.run_weight_space_generation`:

1. Validates canonical, unchanged, complete W0 and a new campaign destination.
   Refuses physical W0/control/sibling copies beyond the explicit disk
   budget (default 512 MiB, absolute maximum 4 GiB for this source pilot).
2. Calls an **operator-installed Python proposal callback** once per candidate
   with only `PublicProposalContext(candidate_id, W0 SHA, public TRAIN source SHA,
   development generation)`. Model providers may implement that interface, but
   their code runs under the caller's permissions. It is **not a sandbox**.
3. Validates all proposals and physical edit policy before publishing effects.
   No Minotaur examination plan, private cases, grades or answers are sent to
   the proposal callback.
4. Makes a **distinct physical negative control** containing W0's exact base,
   tokenizer/config, optimizer, scheduler, RNG and cursor bytes, with a
   separately verified full-state manifest.
5. Creates a real `AriadneJournal` (Linux/WSL only), commits GENESIS, and for
   each of 2–8 sibling proposals appends BRANCH_EXPERIENCE,
   fsync'd TRAIN_PREPARED, physically edits a *new* candidate genome through
   `weight_space_spark_v1`, then appends TRAIN_COMMITTED and a causal-*agnostic*
   source-only receipt. Failed or interrupted effects stay INDETERMINATE;
   partial candidate directories are never silently promoted or erased.
6. Uses `evolutionary_population_v1.prepare_population` to bind the genuine
   physical W0/control/W1 manifests to the caller's predeclared evaluation
   plan. No private examinations, model winner selection, graduation or
   deployment are executed.
7. Fresh-process rehashes the full lineage and returns the journal head SHA,
   source-only candidate receipts, and `Population` for an external Minotaur.

The bridge does **not** invoke `select_generation`: candidate creation needs
neither a compulsory outside review nor fabricated independent metrics.
`select_generation` remains a separate qualification mechanism when an actual
examiner is present. A caller-provided plan is *not* proof the examiner exists,
ran or is independent.

## Source-only test

`python -m unittest -q tests.test_weight_space_campaign_bridge_v1`

Tests create real tiny local safetensors W0, two distinct direct-edit W1
siblings, a byte-identical separate W0 negative control, a 7-event real
Ariadne chain, and a physically reverified Minotaur population. Negative
tests reject unbound TRAIN source provenance *before* any output, preserve
W0, and leave an intended `INDETERMINATE_TRAINER_EFFECT` when a malformed
edit fails after durable PREPARED intent.

## The next actual experimental work

- Implement a trusted, resource-bounded local proposal adapter that performs
  actual model inference and logs its provider/runtime/version; do not mistake
  this callback shape for proof of model authorship.
- Capture exact public observational/behavioral failure traces, derive the
  numeric direction with a testable causal hypothesis, and establish an
  architecture-aware BF16/FP16/multi-shard implementation before using real
  Qwen or Vera weights.
- Feed untouched parent, *physical* control and all sibling artifacts to a
  genuine hidden/cohort-balanced heldout Minotaur; measure transfer, dialogue,
  identity, retention and regressions. Train proposal agent never sees gold.
- Only after a legitimate selection, use Ariadne to verify the champion state
  and hand it to an independently qualified next-generation loader. This
  module does **not** synthesize a restored optimizer/cursor or evolving world.

Keep zero-budget processing on authorized local hosts; preserve the untouched
W0. Published research inspiration:
[Takehi et al., *Sparks of Weight-Space Programming*, November 2 2026](https://rikiyat.github.io/blog/weight-space-programming/),
[authors' public experimental logs](https://github.com/RikiyaT/lm_understand-logs).
