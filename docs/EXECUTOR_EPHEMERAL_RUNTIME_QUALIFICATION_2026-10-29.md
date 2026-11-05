# Labyrinth — Executor VN1 ephemeral runtime qualification (2026-10-29 UTC)

Status: **EPHEMERAL_SOURCE_RUNTIME_PROBED; PERSISTENT_INSTALLATION_NOT_ESTABLISHED**.

This is a strictly scoped runtime experiment, not a training report, permanent deployment, baseline admission, or daemon restart qualification. The plan/negative-control hypothesis predated these observations in `HOSTILE_RUNTIME_RESEARCH_LOG_2026-10-28.md`. The second model comparison is exploratory.

## Instance identification and custody

- Executor device: an authorized Windows test node (VN1), with user-operated Ollama at loopback `127.0.0.1:11434`.
- An initial read-only check of the host's user-scoped Python 3.13 interpreter found no importable `labyrinth` package. A previously checked local repository path was absent.
- Before staging, a uniquely named test checkout inside the host's user TEMP directory did not exist. With `GIT_TERMINAL_PROMPT=0`, a shallow clone of the public canonical repository into this isolated TEMP directory completed with exit 0 and `git rev-parse HEAD` = `39a95ac1d12cce372de980b8cad2bc41016e1705`.
- The live Labyrinth JSON report bound source commit `39a95ac1d12cce372de980b8cad2bc41016e1705` and `source_tree_sha256=6750d7e1946b07a47132cb7c39c1130bac30586d34db5aeef3071356796c8dbf` with `clean=true`.
- No global package installation, daemon restart, service reconfiguration, model pull, paid API call, or persistent startup registration occurred. The TEMP checkout was deliberately retained for inspectability; its later retention and availability are not guaranteed.

## Command and decisive behavior

From the exact TEMP repository directory, using the Python 3.13 executable above:

```powershell
python.exe -m labyrinth run --adapter ollama --model gemma3:1b --seed 2026 --tiers 1 --count 1 --timeout-seconds 25
```

Exit 0, `labyrinth.report.v1`, one tier-1 task, `challenge_id=eval-t1-0-b3ba0625d2bab362`, `challenge_commitment_sha256=8e369f5583a8cbabba41d4551b54c6c6a3be67eed443e28162c571e1fb95c77b`. The held task was `Calculate 270 - 832` with a required `ANSWER: <integer>` line; its evaluator key is `-562`. The real Ollama model emitted `ANSWER: -552` (response SHA-256 `e9a4d2ce711cc870f4ae2ed9e142c82fe9425c234159fd1fdbc5033c2187f579`). The scorer reported `accuracy=0.0`, `format_adherence=1.0`, reason `incorrect`. This proves model generation reached the Labyrinth evaluator and that incorrect arithmetic failed, not that reasoning training succeeded.

Ollama `/api/tags` identified this model as `gemma3:1b`, digest `97558784bdc9175075a72655bf360b15da17df6b3b9e5182af9f8c0c8e58ba7c`, size `806067407` bytes. The Labyrinth report itself had `adapter.weight_digest=null`; the tag digest is a separately observed runtime identity, not a signed model provenance claim.

## Controls and fresh process

In the same TEMP source checkout, this single-process Python control harness ran without files:

```python
from labyrinth.maze import generate_maze
from labyrinth.runner import run_maze
from labyrinth.adapters import ReplayAdapter
c = generate_maze(2026, tiers=(1,), count=1, split="eval")[0]
for label, response in (("positive", "ANSWER: " + c.answer), ("malformed", c.answer)):
    result = run_maze(ReplayAdapter({c.prompt: response}, model_id="replay:" + label),
                      seed=2026, count=1, tiers=(1,), split="eval")
    print(label, result["evaluation"]["accuracy"],
          result["evaluation"]["format_adherence"],
          result["evaluation"]["trials"][0]["scoring_reason"])
```

Observed: positive `1.0 1.0 correct`; malformed `0.0 0.0 invalid_format`. These are controlled replay-evaluator checks **on the VN1 staged runtime**, not evidence of the Ollama model producing either control response.

A **new Executor/Python CLI process** repeated the identical `gemma3:1b` command with the same source commit, challenge commitment, response SHA-256, `ANSWER: -552`, accuracy `0.0`, and `incorrect` reason. That proves repeatability across **client-process starts only**. Ollama daemon restart, hardware reboot, and weight-change persistence were not exercised.

The VN1 staged checkout ran `python.exe -m unittest discover -s tests -p test_cli.py -q` (21 passed, exit 0) and `python.exe -m unittest discover -s tests -p test_adapters.py -q` (11 passed, exit 0).

## Exploratory second installed model

Same CLI arguments except `--model qwen2.5-coder:1.5b-instruct-q3_K_M` produced `ANSWER: -562`, accuracy `1.0`, format `1.0`, reason `correct`, response SHA-256 `9a22eeda86a6851889935383798cc6d788eb53c85ecfc666af22381ae9a49de7`. `/api/tags` digest `3100989a4f7e43898b10279a6e5f0fd6064d7e8f742c6bf62fd4ded723a33d98`, size `824192265` bytes. **One problem is not a model ranking or statistically meaningful improvement.** No weights were updated.

## Gates still open

1. Persistent installed Labyrinth package/service identity and governed promotion from an ephemeral tested checkout.
2. A preregistered multi-domain held-out batch with negative and zero-update controls on actual pretrained-model weights, not one exploratory arithmetic case.
3. Authenticated/signed model provenance and model weight digest in the report (currently `null`).
4. Ollama daemon restart/reboot persistence and independent qualified reviewer sign-off.

No main branch, service, model weights, or existing user source directory was changed by this experiment.

## Planned additional exploratory E4 (frozen before run)

Question: Does the exploratory Qwen candidate maintain exact response formatting and correctness when one freshly reserved case is drawn from each of Labyrinth's four distinct tier types, including operation chains, source priority, and program structure?

Method: run **one** new source-generated case per tier at `seed=3719`, split `eval`, `--adapter ollama --model qwen2.5-coder:1.5b-instruct-q3_K_M --timeout-seconds 25`, save no report file, capture the JSON summary and each `scoring_reason`. This E4 is *exploratory*, not a sealed benchmark, because the task generator is public and the seed was not externally sequestered. Hypothesis: the candidate's exact-format adherence may exceed its correctness on multi-step/structural tasks; a four-for-four exact result would falsify this specific prediction for this tiny sample. Stop after the four tasks or first adapter timeout. No tuning or repeated selection on this seed.

### E4 observation (after preregistration)

Exactly one combined command executed on VN1, exit 0:

```powershell
python.exe -m labyrinth run --adapter ollama --model qwen2.5-coder:1.5b-instruct-q3_K_M --seed 3719 --tiers 1,2,3,4 --count 1 --timeout-seconds 25
```

Source commit remained `39a95ac1d12cce372de980b8cad2bc41016e1705`, `challenge_commitment_sha256=0582a2d159ed02782d670a69654fc1c362f5c74855958fad4a9cdddd751cb418`. Aggregate `accuracy=0.25`, `format_adherence=0.25` across four tasks:

| Tier | Response preview | Scorer reason | Response SHA-256 |
| --- | --- | --- | --- |
| 1 arithmetic | `119` | `invalid_format` | `3038bfb575bee6a0e61945eff8784835bb2c720634e42734678c083994b7f018` |
| 2 operation chain | `14` | `invalid_format` | `8527a891e224136950ff32ca212b45bc93f69fbb801c3b1ebedac52775f99e61` |
| 3 source priority | `ANSWER: 84464` | `correct` | `f92361e6b347045eecbc9ac9c5e1001a7c2785824a4093bf201a62524ae12256` |
| 4 program structure | `5` | `invalid_format` | `ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d` |

This exploratory run falsifies neither broad model competency nor architecture, but exposes an immediately actionable **format-adherence bottleneck** even where the raw scalar might be correct. The scorer appropriately failed closed; changing its strictness after seeing these samples would invalidate the negative control. The preceding single-case perfect Qwen result must not be promoted into a general performance claim.

### Independent E4 repeat and variance

An independent reviewer repeated the same Qwen model, source revision, seed, and challenge commitment. It reported **accuracy `0.25`, format adherence `0.50`**, not the original `0.25`. In particular, Tier 2 differed: the first run emitted raw `14` (`invalid_format`); the independent repeat emitted `ANSWER: 1` (`incorrect`). Other trial details were not supplied in this report. These are two observations with a genuine disagreement, not a stable per-trial outcome.

Source inspection at `labyrinth/runner.py` shows a challenge-derived `model_seed` is passed to `adapter.generate`; `labyrinth/adapters.py` transmits it as Ollama generation option `seed`, with temperature `0`. This is a requested seed, **not** a determinism guarantee. The cause of the repeat divergence was not isolated (model runtime, backend, or other execution state remain possible), and the earlier exact Gemma response hash across two client starts is only a pair of matching observations, not proof of general repeat determinism.
