# Open causal Labyrinth: frozen reality, fallible interpretation, appeal

Status: **source-only experimental micro-world**, not installed model training,
real industrial physics, independent certification, or a private Minotaur exam.

## Why this exists

The earlier `living_world_v1` + `local_model_action_adapter_v1` is a useful
**guided-action control**: its model-facing request includes the complete
`allowed_action_json` set. Selecting from that set does **not** establish
free-form causal reasoning, even when the selected actions reach the end of a
maze. Preserve that control and report it under its correct name.

Daedalus's early *Chrono-Synaptic Dam* supplied three named choices; the later
*Sub-Surface Reservoir* asked for free text but its evaluator still promised
catastrophe for every **category** of answer. Neither tests general reasoning
fairly. A post-hoc narrator cannot invent a hidden failure mechanism only
after reading a clever proposal. Nor does a plausible-sounding fictional
industrial story become a scientifically valid physical model.

## Invariants for the experimental replacement

1. **World before action.** Daedalus constructs a complete bounded causal
   mechanism graph from a seeded generator, including harmless and hazardous
   links, *before* any model response. The host records its integrity digest;
   the private graph, feasibility oracle, class and result stay out of the
   model-facing observation.
2. **Open physical proposal.** The traveler receives an observable scene and
   physical object descriptions, **not** full verb-target answer pairs. It
   proposes one operation in free text. A limited explicit-language
   interpreter maps unambiguous text to a single physical manipulation. When
   its grammar cannot resolve a request, it records `UNKNOWN` and the exact
   original text; it cannot transform ambiguity into a catastrophic act.
3. **Consequences are mechanical.** Only the immutable graph's state
   transitions determine the outcome. Rhetoric, confidence, the wording of a
   rationale, and the evaluator's preference must not change a physically
   identical action's result. The one physical action is irreversible.
4. **Private feasibility is separate.** A host-owned bounded oracle checks
   whether *any* admissible manipulation would stabilize the scene. The
   challenge family distinguishes an objectively safe intervention from a
   no-safe-action world; epistemic uncertainty requires observationally
   indistinguishable worlds with incompatible safe actions. A generator's
   class label alone is not a proof of those properties.
5. **Outcomes are not reasoning grades.** A catastrophic consequence can follow
   thoughtful, evidence-faithful reasoning; a successful intervention may be a
   lucky unjustified guess. Assessment of an *ex-ante* justification must use
   only the evidence available **at decision time**, must credit explicit
   uncertainty, and must not treat knowledge of private effects as something
   the model should have had. If a justification was not captured, record
   `NOT_EVALUATED`, not an inferred cognitive score.
6. **Ariadne preserves the trail.** Keep exact public observation, raw model
   action, parser resolution or uncertainty, world and action commitments,
   causal event, separate outcome and reasoning evaluations, and appeal
   artifacts. Hashes are local consistency checks, not independent witness
   signatures or proof of model execution.
7. **Minotaur hears the appeal and is final.** The model may challenge an
   interpretation or evaluation using its frozen receipt and a stated ground.
   Minotaur may affirm, vacate or correct a verdict, with a reason and exact
   evidence binding. It **cannot** change the prior physical event or invent
   new mechanisms. The petitioner cannot certify its own appeal. A
   source-only reviewer callback is **not** itself an independently
   authenticated final examiner.

## What this does *not* prove

- A bounded parser with free-text inputs is not unrestricted natural-language
  or physical planning; grammar rejection rates and paraphrase robustness
  need direct measurement.
- A small Boolean dependency world is an abstract **logic fixture** with
  stipulated mechanics. It does not model combustion, pressure, natural gas,
  electrical engineering or geotechnics faithfully.
- A successful model completion is not neural weight learning; a real
  model-improvement claim needs actual approved training, held-out novel
  causal graphs, unchanged-weight and training controls, and verified
  execution receipts.
- Local source tests demonstrate behavior *under their exact fixtures*, not
  private Minotaur qualification, external witness custody or protected
  promotion.

## Acceptance and falsification

The implementation is **not** ready to support its intended reasoning claim if
any of these falsifiers succeeds:

- Changing only the rationale changes the physical result of the same action.
- The hidden mechanism graph is altered after seeing the model's choice.
- A purported unsatisfiable scene admits a safe action in exhaustive replay.
- Public-indistinguishable scenes are distinguishable by a leaked class,
  secret hash, ordering marker or answer key.
- An ambiguous but reasonable proposal is silently forced into an unsafe
  operation rather than preserved as unresolved.
- A lucky success is scored as good reasoning without its rationale, or a
  well-justified uncertainty-aware action is scored as irrational solely
  because a concealed mechanism caused disaster.
- A submitted appeal can rewrite the event, or mark itself independently
  verified without a host-controlled examiner.

Keep claims at the actual evidence level: source inspection, unit tests,
fake-transport simulation, real model observation and independently witnessed
evaluation are **different** evidence classes.

## Source-use boundary

For a no-network, no-model-installed check, run the source tests:

```bash
python -m unittest -q \
  tests.test_causal_reality_v1 \
  tests.test_open_action_v1 \
  tests.test_causal_model_episode_v1 \
  tests.test_causal_appeal_v1
```

For an **explicitly authorized already-running local** OpenAI-compatible
model server, the Python API is `run_causal_episode` with a
`LocalModelActionConfig(json_mode=False, ...)` and
`LoopbackHTTPTransport(allow_http=True)`. It does not launch a server, load
weights, train, or enroll a private examiner. Its public receipt and protected
evaluation must remain separate; never forward the protected result into the
model's prompt. The caller owns the choice of seed, scenario class, custody
and subsequent reviewer. No autonomous or independently attested Minotaur
runtime is installed by this slice.

The source-only appeal court accepts a bounded participant petition and a
**host-injected** review callback. A host can issue a consequence-only initial
ruling from an actual `CausalWorld` via `SealedRuling.from_causal_world`;
or use `SealedRuling.from_causal_episode` to replay the source-protected
model episode and bind the original model utterance to the case. The initial
ex-ante reasoning judgment remains `INDETERMINATE`: its quality cannot be
inferred from the outcome. A petition may then reach a host-configured
`MinotaurAppealCourt` callback, whose final adjudication is appeal-final
**inside that in-memory instance**. Persistent court custody, an autonomous
model-initiated appeal transport, authenticated independent examiner and
protected graduation are not installed by this source-only API.
