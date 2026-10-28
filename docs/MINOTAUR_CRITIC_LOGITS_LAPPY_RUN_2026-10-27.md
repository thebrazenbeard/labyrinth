# Minotaur correctness critic v3 logit diagnosis — E-021 (2026-10-27)

This is a live CPU pretrained LoRA result and fresh-process reload, NOT evidence of improved reasoning or an independent qualification.

## Execution binding

- Source at run: 7cd2c982d9dbdb231ef900b5ebf812204181c6a3; source tree SHA-256 9834ffae2909d8e864d340e0fa5c3d213e3680e0f2b99523a7a3e51fa69eefd4.
- Preexisting Open-Reasoner-Zero-0.5B local pretrained checkpoint, Windows CPU PyTorch 2.14.1 / Transformers 5.14.1 / PEFT 0.19.1.
- CriticConfig with 2 balanced correct/incorrect optimizer steps; 4 training forward/backward passes per arm, identical frozen/zero-update/corrupted-label controls and evaluation prompts. Two development problems per tier, total 8 development verdicts; 4 structural-transfer verdicts.
- Candidate and corrupted-label adapter hashes changed; zero-update retained its original tensor values. No model files were downloaded or production components changed.

## Results

| Arm | Dev accuracy (8) | Transfer accuracy (4) | Dev false approvals (4 incorrect) | Mean Yes-minus-No logit on dev incorrect | Mean logit on dev correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 5/8 | 2/4 | 3/4 | +0.08474 | +0.21982 |
| Zero-update | 5/8 | 2/4 | 3/4 | +0.08474 | +0.21982 |
| Candidate | 4/8 | 2/4 | 4/4 | +3.16508 | +3.30248 |
| Corrupted labels | 4/8 | 2/4 | 4/4 | +5.05580 | +5.22926 |

Training substantially moved both classes toward Yes, with no useful separation and no improvement over controls. The corrupted-label arm was even more Yes-biased, so weight changes and reduced training losses are insufficient evidence of verifier learning. These small, public, structurally limited test suites cannot establish generalization.

## Real fresh-process reload and regression repair

The first v3 run at prior source f7c7037ecbfa0a5820e8e65393a8aedd39a8f123 produced the same negative behavior but its raw-logit exact-float reload comparison failed even though all verdicts and decision metrics matched. Direct diagnostic re-scoring observed CPU logit differences between processes of approximately 0.000002 and 0.000021. A regression test was first RED; source 7cd2c98 fixes reload to require exact classification metrics, verdict digests, labels and identities, while allowing at most 0.00005 absolute difference in diagnostic floating-point logits/summaries. Larger margin discrepancies and changed verdicts remain blocked.

The source 7cd2c98 experiment then trained the real model anew, saved its adapter, and reloaded in separate process. Training PID 7708; reload PID 13308; different_process_id=true; same_predictions_and_metrics=true. Both processes exited successfully.

- Training report SHA-256: 2d3a824fb53a261bf423801ba74561895e4101cc039599ccb7bc21942feca4bd
- Fresh reload report SHA-256: 5a5cc6089d4f7edaf40e6e078a6c6a43048b8eac5bb3f46cf320fdc14a90e727
- Candidate adapter safetensors SHA-256: f7ed2c78ec1a15c872433ca66a1bcd58a8c3abb5d39dab9ad225ef174949b34c
- Candidate LoRA tensor-value SHA-256: 5589e1f09592d0b9b182659b56f36ad423b98f541bcdc1346708d5ce7cf7b631

All binary artifacts and reports are retained locally in isolated research directories, not Git. The controlled execution and reload are observable, but no real private examiner, independent custody, neural competence or production graduation is established.
