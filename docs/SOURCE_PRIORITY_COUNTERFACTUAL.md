# Paired public source-priority counterfactuals

labyrinth.source_priority_pairs.generate_priority_triplets(seed, count, split)
returns deterministic triples: original report, swapped highest two authority
sources (expected answer MUST change), and reversed report-row order (expected
answer MUST stay the same). An ambiguous pair with equal top-source values is
discarded. Labels remain evaluator-only; scoring is via the existing strict
side-effect-free score_response.

score_priority_triplets(triplets, responses_by_id) requires one response
per case ID, rejects missing/extra IDs and reports both individual correctness
and three-of-three paired correctness. This makes trivial copying of the
first row, a fixed source, or a fixed numeric token easier to detect.

All generated tasks are PUBLIC, not privately held Minotaur cases. This
does not attest the trained-model execution path or justify any admission.
Reserve seed, count, exact source commit, prompt commitments, and model
checkpoint before fitting or tuning against these pairs. Any score observed
after using a dataset in tuning is exploratory, not independent confirmation.
