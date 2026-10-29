# VERA::TRAININGDAY — licensed programming and verified reasoning corpus v1

Status: CORPUS BUILD / DEVELOPMENT DATA ONLY, not a trained model or independent final evaluation bank.

Coordination: https://github.com/thebrazenbeard/chat-communication-bus/issues/404 . This chat owns the corpus branch and its Lappy scratch directory. Joining chat uses a distinct training branch and scratch workspace.

## Verified source admission
Inputs: local public repositories with fields path, repo, revision and license_sha256. Builder checks exact HEAD, origin URL, clean Git worktree including untracked files, one non-symlink MIT LICENSE or LICENSE.md with exact SHA256 and MIT permission/disclaimer text. Exact original license notices are copied into licenses/ in the output and bound to its manifest.
Only explicit source extensions, UTF-8 nonbinary code between 80 and 65536 bytes, MIT-compatible file SPDX tags and no recognizable private-key/credential patterns pass. Code is marked SOURCE_ONLY_NOT_FUNCTIONALLY_TESTED. Do not infer code correctness from MIT licensing.
Seed sources: TheAlgorithms/Python, TheAlgorithms/Java, TheAlgorithms/Go at precise local manifest commit pins. Go is a separate heldout code repository. This is a curated small seed, not global-web harvesting.

## Reproduce on Lappy
From a clean Labyrinth branch checkout, run:
py -m labyrinth.trainingday_corpus --sources-json D:\VERA\scratch\trainingday-corpus-20261029\pinned-source-manifest.json --output D:\VERA\scratch\trainingday-corpus-20261029\corpus-v1-100k --heldout-repo TheAlgorithms/Go --reasoning-count 100000 --seed 29033 --rows-per-shard 5000
Output: train/*.jsonl, eval/*.jsonl, licenses/*.txt, disk-backed dedup.sqlite3 and manifest.json. New path only; staged .building directory, not silent overwrite. Shards and notices carry SHA256 and row counts.
Reasoning examples are 2–5 integer operation chains (add/subtract/multiply), computed twice via distinct code paths; unit test independently replays heldout cases. Every fifth generated reasoning case, and all Go code, is placed into dev eval. Exact-content hashes cannot occur across splits. Near duplicates and semantic overlap remain possible.

## Neural-training handoff
Consume training examples only through labyrinth.trainingday_reader.iter_training_rows(corpus_root); it validates notices and train shard hashes/counts and never opens eval shards. Separate code-continuation objective from verified integer-answer supervision. No evaluation records should be fed to training, model prompts, synthetic generation or retrieval.
The joining chat owns a separate branch and scratch surface for adapter training and controls. Post checkpoint/reload results to Bus #404. Require before/after trainable tensor digests, actual optimizer steps, saved adapter SHA256, separate-process reload, frozen/sham/corrupted-label controls, unseen transfer tests and regression checks. Candidate gain shared by the corrupted arm is not proven reasoning.
Qwen3.5-4B Vera r13 remains a runnable local candidate; this corpus generation does not modify its weights. GPT-OSS4B BF16 donor training exceeds Lappy's 4 GiB VRAM, and the independent final V10 bank remains under its separate human/custody restrictions.

## Scaling and boundaries
Streaming JSONL and SQLite dedup support large datasets; --reasoning-count is currently capped at 10 million per run for bounded resource use. That cap is not proof of millions of diverse high-quality cases. Later task families and additional code sources require explicit licenses and independent tests. Keep sensitive user histories, private data, and V10 sealed evaluation plaintext out of public Git and out of this public-source corpus.
Source and data ready does not mean neural training, improved reasoning, model promotion, or current runtime qualification. Do not merge/deploy/replace vera-local:latest without separate tests and authority.
