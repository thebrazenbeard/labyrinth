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

## Verified first large corpus checkpoint (2026-10-29)

Generated on Lappy from producer `labyrinth@55e2639a1d3e01c71bf1555160eee89a132a2772`, source file `labyrinth/trainingday_corpus.py` SHA-256 `4841cae00ba631eb499429512dad48857aded74905ff847be3485468c60f23e9`. Exact manifest on disk `D:\VERA\scratch\trainingday-corpus-20261029\corpus-v1-100k\manifest.json`, 6,487 bytes, SHA-256 `c722cd38b3cb42905079696691279c950de6b2d740287acc235a12d764f6106d`.

22 JSONL shards and 103,501 admitted records, of which train has 83,012 (3,023 MIT-code source-only records; 79,989 procedurally checked arithmetic examples), while development evaluation has 20,489 (494 Go code; 19,995 arithmetic examples). Excluded exact duplicates: one code record and 16 arithmetic examples. All 22 shard hashes, row counts, code source Git heads, original license notice digests, and disjoint train/eval exact content SHA-256 sets were reverified by a separate Lappy Python process, exit zero. This does not establish independent final-bank custody or unseen generalization.

A full independent code checkout at PR head `3cbd064187def287190796dc954b4a962dcf6408` passed 241 Labyrinth unittests (1 optional skip), exit zero and clean Git status. A new negative test first proved the old reader **leaked the first training record before noticing later-shard corruption**; the fixed reader now prevalidates every training shard before yielding any record and rechecks each file on consumption. The focused RED result was 1 failing test out of 5, then all 5 passed after the source fix.

The first data-quality profile indicates the 3,023 train code files have median 2,235 characters and 95th-percentile 8,947 characters; 699 exceed 4,096 characters and 188 exceed 8,192 characters (maximum 47,890). A neural trainer must use a controlled token-budget and source-preserving chunking/sample policy; these records are not automatically safe to feed whole into a 2K/4K context. The arithmetic problems are short (median 105 characters) but have common template families, so the public dev split is an **in-distribution development check** rather than a blind transfer benchmark. The source-validation check confirms locally configured Git remote, HEAD, and clean worktree, **not an independently cryptographically authenticated upstream identity**; the upstream origin and license were separately inspected for this specific batch. No general arbitrary local clone trust is claimed.

## Interoperability with the parallel neural-training lane

Bus #404 records the second chat's separate `thebrazenbeard/model-training` training branch. It proposed `VERA_TRAININGDAY_ADMITTED_SHARDS_V1`, which is **not byte-compatible** with the actual `vera.trainingday.corpus.v1` manifest. The producer's rows retain provenance and `SOURCE_ONLY_NOT_FUNCTIONALLY_TESTED` vs `TWO_INTEGER_INTERPRETERS_MATCH` labels. The neural runner must adapt explicitly from the producer's verified schema, or create an auditable derived manifest outside the corpus-owner directory. Do not automatically claim a GitHub code file is `HUMAN_CROWDSOURCED`, `privacy_review=PASS`, or `rights_review=PASS` solely because a repository-level MIT license was observed.

No GPT-OSS4B optimizer update, Vera 4B LoRA update, merge, model deployment, or measured reasoning gain was produced by this corpus work.
