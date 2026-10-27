# Minotaur external benchmark adapter (2026-10-27)

## Provenance and permitted use

The following are **external primary sources**, not bundled test data:

1. [Qwen/ProcessBench official dataset](https://huggingface.co/datasets/Qwen/ProcessBench) and [evaluation implementation](https://github.com/QwenLM/ProcessBench/blob/main/code/run_eval.py). Dataset license declared **Apache-2.0**. The official four splits are `gsm8k`, `math`, `olympiadbench`, and `omnimath`. Records contain `id`, `generator`, `problem`, `steps`, `final_answer_correct`, `label`. In the official evaluator, `label=-1` denotes all steps correct, and `label>=0` is the **zero-based** first wrong paragraph index.
2. [allenai/reward-bench-2 official dataset](https://huggingface.co/datasets/allenai/reward-bench-2). Its chosen/rejected **ranking** objective differs from ProcessBench's first-error localization. The card declares **ODC-BY** and third-party model-output rights; this adapter deliberately does not copy or re-license it. No RewardBench 2 result is claimed.
3. [Helff et al., *LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking*](https://www.dfki.de/en/web/research/projects-and-publications/publication/17274), 2026. Its Isomorphic Perturbation Testing (IPT) tests whether logical equivalents defeat shortcut strategies. The optional pair metric in this module is merely **inspired by** IPT; it does not implement the full paper protocol.

## What this integration actually implements

- `labyrinth.processbench_adapter.load_processbench_jsonl`: strict, bounded, **offline** JSONL parser (4 MiB or 1024 records per invocation), not a downloader. Duplicate JSON properties, duplicate IDs, invalid labels, invalid booleans, empty steps, malformed UTF-8 and incompatible splits fail closed. Each dataset row must have exactly the official six fields.
- `labyrinth.processbench_adapter.evaluate_processbench`: invokes a reviewer on `(problem, steps)` only. No gold label or correctness flag is passed in the review call. It demands `ALL_CORRECT` or `ERROR_STEP:N`, where **N is one-based**, and translates official `label=0` into `ERROR_STEP:1`. Reports class accuracies, their harmonic mean (the official paper's F1-like aggregate), coverage and false approvals/rejections. Unrecognized outputs count wrong, including abstentions.
- `labyrinth.processbench_adapter.evaluate_isomorphic_pairs`: caller supplies pairs of purportedly logically equivalent but textually distinct instances with the same first-error gold index. Reports valid-pair coverage, verdict consistency, both-correct rate, and one-sided failure rate. **The caller's isomorphism assertion is not proven.** Constant wrong answers can have 100% consistency but 0% correctness.

All evaluation outputs carry `reasoning_qualification=NOT_ESTABLISHED`, and the ProcessBench adapter reports `evidence_status=UNVERIFIED_CALLER_SUPPLIED_DATA`. This is intentional. Parsing a JSONL file whose fields resemble ProcessBench is not evidence that it is a complete, unmodified official split, and performance on any publicly accessible set is not a sequestered graduation exam.

## Reproduction

From the Labyrinth repository root, without downloading any dependencies or datasets:

```bash
python -m unittest tests.test_processbench_adapter -v
python -m unittest discover -s tests -v
```

In an externally orchestrated experiment, obtain an authorized ProcessBench JSONL file, record its pinned dataset revision and SHA-256, register train/evaluation split separation, and load its **bytes**:

```python
from pathlib import Path
from labyrinth.processbench_adapter import load_processbench_jsonl, evaluate_processbench

payload = Path("/path/to/authorized-processbench.jsonl").read_bytes()
cases = load_processbench_jsonl(payload, split="gsm8k")
report = evaluate_processbench(cases, reviewer=my_frozen_model_reviewer)
```

`my_frozen_model_reviewer(question, steps)` must be supplied by the experimenter. It should run a pinned model at fixed decoding configuration, log its own prompts/response/latency with access controls, and return the strict verdict string. A **trained** critic must be evaluated with the same protocol and a predeclared held-out split, with contamination checks and negative controls.

## Audit status and limitations

| Evidence class | Status |
|---|---|
| Primary external source review | Completed for schemas, labels, and caveats above |
| Source implementation | Implemented as an opt-in Labyrinth module |
| Unit/fixture test | Executed; synthetic invented fixtures only |
| Official Qwen/ProcessBench input | **NOT DOWNLOADED / NOT EXECUTED** |
| RewardBench 2 | Schema researched; **NOT IMPLEMENTED / NOT EXECUTED** |
| Neural model inference on benchmark | **NOT EXECUTED** |
| Comparative reasoning improvement | **NOT ESTABLISHED** |
| Real isomorphism proof / published IPT reproduction | **NOT ESTABLISHED** |

The existing `guardian_benchmark.py` is deliberately generated from public register-program templates. Repeated correctness on those tasks, or invariance across superficial renamings, could reflect memorization or shortcuts. Genuine generalization requires independent task structures, held-out generation families, and clean baseline/zero-update/corrupted-label controls, not merely a high consistency figure.

> **Hostile challenge:** Since the benchmark schema and validation logic are public, a candidate could learn the response format, per-split distribution, or even the answer keys from training contamination. High ProcessBench agreement is not sufficient evidence of general reasoning, much less justification to release a model from Labyrinth. Use privately custodied test problems for the final graduation authority.
