# Wave45 — JSON-constrained output is not model reasoning

Status: source-public, CPU-only **development experiment**, not neural training, independent examiner custody, loaded-weight attestation, or generalization.

## Question and protocol

The existing finite-transition blind exam asks for PLAN with up to three action letters or UNSAT, exactly. Could JSON output constraints fix small-model reply-format failures *and* produce more correct mechanical solutions?

The new labyrinth.json_enum_observer module uses the ORIGINAL unchanged source-public question prompts and asks Ollama to return a JSON object with exactly one answer property. The public format enum contains UNSAT and **every** action trace of length 1, 2 or 3 (40 possible answers for a three-move world). This enum contains losing plans as well as winning plans and provides NO hidden answer key. Boundaries above three moves are rejected before any model network access.

A strict parser rejects duplicate JSON keys, unknown fields, invalid answer strings, incomplete/token-cap generations, wrong model tags, mid-exam advertised digest changes, missing original receipt pins, response-file tampering, and non-loopback endpoints. The original raw model JSON string, its SHA, the extracted answer SHA, public-prompt SHA and syntax-schema SHA are retained for exact replay. The observation is deliberately DIFFERENT from original V2 free-text model observations: schema labyrinth.blind-json-enum-local-http.v1 and evidence LOCAL_HTTP_JSON_ENUM_SELF_REPORTED_NOT_ATTESTED. The existing blind scorer routes this exact schema to a separate verifier and never silently credits JSON formatting as ordinary free-text output or authenticated model weights.

Run with an already-authorized existing local Ollama service, e.g.:

    python -m labyrinth.json_enum_observer --public C:/exam/public.json --responses-out C:/exam/json-responses.json --observation-out C:/exam/json-observation.json --endpoint http://127.0.0.1:11580 --model qwen3:4b-instruct --timeout 90 --max-tokens 128

Use the existing blind_control_comparison with the exact original precommit, private/public cases, output responses, JSON observation file, and its original retained observation SHA. The *model runner* receives only the public question file, not private cases or examiner keys.

## Actual observed Lappy inference, 2026-11-14

All comparisons below reused FOUR earlier examiner-authored causal near-twins from the same original source. The original private exam remains on Lappy under previously observed weak permissions. This is a DEVELOPMENT set with compromised independent custody, not a protected blind test; the first public case had also been tested once in an exploratory JSON-mode diagnostic. No population-level conclusion follows.

- Original public-file SHA-256: bb64c0a1984377beb4ad4d6b0598fbb53668977a752d5aee1d6449c6af26f0d3
- Original private-exam source precommit SHA: 79511e606a58099435ef79f37f3207946caeb29b13ee8dfd26fdb8f2fc33ba3a

| Candidate or control | Correct / 4 | Evidence class |
|---|---:|---|
| Qwen3.5 0.8B, original free text (previous wave) | 1 | actual local, self-reported HTTP |
| Qwen2.5 Coder 1.5B q8, original free text | 1 | actual local, self-reported HTTP |
| Qwen3 4B Instruct, JSON-enum constrained | 0 | actual local, self-reported HTTP |
| Always answer UNSAT | 2 | trivial zero-weight baseline |
| Enumerate all public-rule transitions | 4 | deterministic zero-neural solver |

The Qwen2.5 Coder 1.5B returned an identical plan on all four cases, despite differing world mechanics. Exact per-case answers remain off Git. Its completed V2 local HTTP receipt SHA was 064ca5105e75f3594025cd26c88e576616a3768bee6cea4394e3aecd26c5e186, with nine advertised model tag snapshots and declared model digest 00f372390c1dac8c6d0cca549d683475d5854c0f0d77d5f5368e0ffe4724fe8b. Exact paired score: 1/4 versus all-UNSAT 2/4, descriptive two-sided exact p=1.0.

A 96-token original-protocol attempt with Qwen3 4B returned too-long text and done_reason=length. The original V2 observer correctly rejected it; **there is no admissible four-case V2 result for that attempt**. In an exploratory single-public-case test, JSON constraints instead produced syntactically valid output in 17 generated tokens. The separate full JSON-enum run then completed all four questions with nine model-tag snapshots. Its four decoded answer strings and raw JSON replies are retained in the original off-Git evidence files rather than being published as per-case exam data. The original action replay found **ZERO winning/correct answers**, despite valid JSON. Qwen3 4B JSON observation SHA 1f991dac831a178484a73d2bc0be33b8f8e7aaaed5a2809af33f3cac9207c9e2; declared model tag digest 0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0. Exact paired score receipt 2201c027e9b55215797d7591532b5aebc9342b743aa3af6321356ce72f8449f0; strict score receipt 244b59e60a816ce71e941b2f0ae4ab6c62d7499e977e5ab33704e7f5ee09ac83. Compared with always UNSAT 2/4, exact two-sided paired p=0.5, NOT a generalization or statistical confidence claim.

A separate manually written *public-only* exhaustive enumerator ran over those same transition tables, without opening the private-case file during response generation. It produced a four-response roster retained **outside Git**; its exact correct plans and per-case SAT/UNSAT labels are intentionally **not published in this document**. The original private scorer later verified **4/4**. Candidate response SHA b11543a063af018a7cddaac287c1b80d4fc328f3cfb08df2a74a238bfb9887bf; paired source-only receipt 1d94857fa4287275ea7e054224aee9de0949e9bb98c7bec9889bbd85d786ebcb. Its evidence remains invocation_evidence=NOT_SUPPLIED and neural_learning_verified=false, because it is not a neural model. This is proof that the public mechanics can be solved algorithmically, not proof a neural candidate learned how.

Original files/observations are retained outside Git at D:\VERA\experiments\minotaur-wave45-20261114. An isolated CPU-only cloud-disabled Ollama service at 127.0.0.1:11580 was stopped after the experiments. The owned child PID 80196 is gone and port 11580 is closed, with no shared service touched. No weight updates, training launch, paid cloud or protected-main changes occurred.

## Hostile result

> SURVIVES: constrained JSON produces syntactically valid, raw-byte-verifiable model response records and preserves original source/private and HTTP hash pins across scoring.

> FAILS: the proposition that formatting controls by themselves improve neural mechanical reasoning on these four reused examples. Observed constrained 4B scored 0/4.

> UNKNOWN: independently custodied examiner, a true denied read by a separately authenticated candidate principal, trusted loaded-weight identity, neural generalization and optimizer-trained checkpoint. An operator-controlled HTTP server can fabricate its own transcript and receipt unless the original evidence is independently held.

A symbolic solver may serve as an explicitly identified **game mechanics fallback**, but its correctness must not be presented as neural cognition. The next genuine neural milestone requires a trained, weight-pinned candidate and separately controlled held-out tests. The official TrainingDay physical-launch HOLD remains unchanged.

Run focused tests:

    python -m unittest discover -s tests -p test_json_enum_observer.py -q
    python -O -m unittest discover -s tests -p test_json_enum_observer.py -q
    python -m unittest discover -s tests -p test_local_neural_observer.py -q
    python -m unittest discover -s tests -q
