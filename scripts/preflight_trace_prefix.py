"""Read-only tokenizer boundary and target-contract preflight on a training host."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

_script_directory = Path(__file__).resolve().parent
if (_script_directory.parent / "labyrinth").is_dir():
    # Running from a source checkout: imports use the installed package layout.
    sys.path.insert(0, str(_script_directory.parent))
    from labyrinth.neural_cpu_pilot import read_training, supervised_target, training_prefix
else:
    # Copied into an isolated training-host directory with two standalone modules.
    sys.path.insert(0, str(_script_directory))
    from neural_cpu_pilot import read_training, supervised_target, training_prefix

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--train-data", required=True)
a = p.parse_args()
from transformers import AutoTokenizer
data = read_training(Path(a.train_data))
tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True, trust_remote_code=False)
lengths = []
for step, item in enumerate(data):
    prefix = training_prefix(item, policy="trace")
    suffix = supervised_target(item, policy="trace", arm="trained", step=step)
    pids = tok(prefix, add_special_tokens=False).input_ids
    ids = tok(prefix + suffix + (tok.eos_token or ""), add_special_tokens=False).input_ids
    if ids[:len(pids)] != pids or len(ids) > 384:
        raise ValueError("prefix mismatch or token limit at " + item["case_id"])
    lengths.append(len(ids) - len(pids))
print(json.dumps({"cases": len(data), "min_target_tokens": min(lengths),
                  "max_target_tokens": max(lengths),
                  "train_data_sha256": hashlib.sha256(Path(a.train_data).read_bytes()).hexdigest()}))
