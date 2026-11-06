# A model actually walking the public Labyrinth

`python -m labyrinth.local_model_world_runner_v1` is the **opt-in inference
episode runner** connecting the source-owned `LivingWorld` to an already-running
OpenAI-compatible `/v1/chat/completions` service on literal loopback. It does
not start a server, download a model, load or edit weights, train, evaluate a
candidate, invoke the protected host broker, or graduate a model.

On a machine where the operator has already started a local inference service:

```shell
python -m labyrinth.local_model_world_runner_v1 \
  --allow-local-inference --model your-local-model \
  --port 11434 --json-mode --reasoning-effort none \
  --max-completion-tokens 128 --max-actions 48 --timeout 20
```

Replace the example model name with the actual model registered in your local
server. Some OpenAI-compatible implementations do not accept `--json-mode` or
`--reasoning-effort`; omit those flags for those servers. This adapter never
assumes that a model name proves which GGUF or LoRA weights were loaded.

The runner prints a JSON artifact containing:

- Every original public world observation and the resulting source-owned event;
- World definition and final state SHA-256 identifiers;
- A canonical SHA-256 of the public trace for reproducible comparison (not a
  cryptographic signature or a model identity attestation);
- The count of model completion requests, any action rejection reasons, and an
  explicit stopping condition.

`PUBLIC_QUEST_COMPLETED` means the *public maze* was solved. It does **not**
mean that Minotaur's private examination passed or neural learning occurred.
`MODEL_ACTION_REJECTED` has CLI exit code 2 and leaves the failed action's world
state unchanged. `ACTION_BUDGET_EXHAUSTED` has exit code 1 and preserves the
partial trace; a completed public quest has exit code 0.

All live HTTP requires explicit opt-in; the outbound transport rejects proxies,
redirects, remote hosts and oversized messages. It passes only observations,
never hidden gold, checkpoint paths or protected authority. The default tests
are fake-transport source tests, not live-model qualification.

VN1's loopback is *VN1*, not Lappy. Remote inferencing over a verified bridge
requires its own authenticated architecture and operator-controlled deployment;
this CLI intentionally does not turn a private service into a LAN listener.
