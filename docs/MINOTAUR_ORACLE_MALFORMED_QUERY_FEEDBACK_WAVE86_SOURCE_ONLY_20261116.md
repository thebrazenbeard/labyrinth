# Minotaur Wave86: malformed Oracle query stays replayable

Date: 2026-11-16. Scope: draft stacked Labyrinth source change, not main.
Parent: PR #266 `9a5193446980034878bc697ff59aa65c70f2a52e`.
Target: `labyrinth/minotaur_candidate_bridge.py` and its real Oracle Chamber regression.
Context: open custody blocker #146, semantic provenance blocker #147.

## Hostile source defect

Oracle Chamber publicly declares malformed questions consume one of five queries.
Its `CandidateSession.ask("query")` returns public feedback with
`answers=[{"question":"INVALID","answer":null}]`, `remaining_queries=4`,
`verdict="UNKNOWN"`. The bridge formerly allowed only binary 0/1 at
`answers[index].answer`; it rejected the public null and misclassified
the entire valid evaluator response as `EVALUATOR_ERROR` without replay.

The boundary change admits null solely in the exact nested public answer
position; root or other nested `answer` values, nonbinary integers, strings
and programmable objects remain blocked. The public contract, not private
Case metadata, defines the allowed value.

## Red-first / green evidence

- New exact regression `CandidateBridgeTests.test_oracle_invalid_query_null_is_public_feedback_not_evaluator_failure`.
- RED with original source: `AssertionError: 'EVALUATOR_ERROR' != 'UNKNOWN'`, exit 1.
- GREEN same focused test: 1/1, exit 0. Malformed question charged and
  `UNKNOWN` recorded; `audit_receipt` replay true; negative fields rejected.
- Windows targeted suite with bridge, subprocess, private-read, graduation:
  106 tests, 0 failures, exit 0.
## Local-model diagnostic (not an attestation)

- Lappy Windows RTX 3050 4 GiB; local Ollama service already running.
  No new model/download/server/deployment, no paid compute.
- Installed local tag `qwen3.5:0.8b`; `OllamaAdapter` invoked
  `http://127.0.0.1:11434` with `think=False`, seed 987654, call cap 1,
  output cap 180 tokens, timeout 45 seconds; examiner Oracle sealed seed 7.
- Before: source checkout PR266, model text
  `{"kind":"ACTION","action":"query"}`,
  `EVALUATOR_ERROR`, replay false, elapsed 25.323 s (cold run),
  self-hash `41bfcb8af1722b53fae41ec74ec603816543f01f899b1275478e4cb26749e477`.
- After: Wave86 source, same model text and spec, `ACTION`,
  `UNKNOWN`, replay true, elapsed 1.228 s (warm run),
  self-hash `f4fc1b3a63fcb305fc9668fab4a7db00f42cd19d29baf3c1777eea76e7691cc3`.
- These are local tool outputs and self-hashed receipts, not separately
  authenticated execution or evidence the model solved the chamber.
- `MODEL_ADAPTER_UNATTESTED`; caller-authorized in-process Ollama
  transport is unsuitable for trusted private examiner custody.
- The installed local model produced an invalid query. The fix preserves
  this informative failure; it does not improve the model's reasoning.
- `#146` denied-read OS principal probe and `#147` independent mechanism
  evidence are still missing. Keep draft/hold. No merge, learning claim,
  actual training, release, private graduation or qualification.

## Final local verification

- Full exact Windows repository command:
  `python -m unittest discover -s tests -v`; 1324 tests,
  6 skipped, no errors or failures, exit 0, 427.701 seconds.
  Log SHA-256: `B99A59538DC415DEFD53C65BD229C8FEDF7B47655256500125024881B0F3114B`.
- WSL Python targeted bridge/subprocess/graduation suite: 100/100 PASS.
- WSL Python `-O` targeted bridge/graduation suite: 85/85 PASS.
- Hosted Python and Minotaur neural CPU CI on original Wave86
  `34b6c10fd2248f7f73e0980be767f4899471f52c` both COMPLETED SUCCESS,
  read back before Wave87.

## Wave87 follow-up: public action/claim decision contract

Follow-up source change on the same DRAFT PR #267 (not a new release).
The first live Ollama qwen3.5:0.8b probe under the earlier source emitted
`{"kind":"ACTION","action":"query"}`, an invalid Oracle question.
The prior bridge repaired its null feedback. The public prompt still
misstated that *every* query is binary and gave no concrete JSON claim forms.

Two red-first targeted failures established missing protocol guidance and
the danger of incomplete `{...}` output examples; both were corrected.
An intermediate new `decision_protocol` public field broke two hostile
schema tests out of 113; this interface-changing design was **retired**.
Final design preserves EXACTLY the five public keys: prompt, questions,
remaining_queries, submissions, answers. Valid literals and example claims
are appended to the existing prompt; generic bridge instructions no longer
invite copying the invalid word "query" as an action. Examples deliberately
contain no private Case identity or pre-solved universal policy.

Final focused corrected-schema test GREEN, 1/1 exit 0.
Expanded Windows bridge, process, Oracle, cross-family, Ollama child and
round3 integration suite: 124 tests, 0 failures, 36.536 s, exit 0.
`python -m compileall -q` on three edited Python files exit 0;
`git diff --check` exit 0. Exact three-file source diff inspected.
No full repo Windows regression rerun for this *new* commit; prior full
1324 tests applies only to original PR267 SHA. Hosted checks must rerun.
No live neural-model probe was repeated for Wave87 due Lappy resource
pressure (measured RAM intermittently ~0.7-3.1 GiB free). A protocol fix
is not model intelligence, independent identity, custody (#146), semantic
independence (#147), learning, or graduation. Keep DRAFT/HOLD.
