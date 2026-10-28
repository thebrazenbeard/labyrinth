"""Stage an immutable, hash-checked pretrained checkpoint for the HF pilot.

The network fetch is explicit and isolated from the training process. Only a
small allowlist of data/config files is copied; no downloaded Python code runs.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil


DONOR = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
WEIGHTS_SHA256 = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
MODEL_FILES = (
    "config.json", "generation_config.json", "model.safetensors",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
)
REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_snapshot(snapshot: Path, destination: Path) -> dict[str, str]:
    """Copy fixed checkpoint files into a *new* tree with no cache symlinks."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination already exists; refusing checkpoint overwrite")
    if not destination.parent.is_dir():
        raise ValueError("checkpoint parent directory must already exist")
    for name in REQUIRED_FILES:
        if not (snapshot / name).is_file():
            raise ValueError(f"missing required checkpoint file: {name}")
    sources = {name: snapshot / name for name in MODEL_FILES
               if (snapshot / name).is_file()}
    for name, source in sources.items():
        if source.stat().st_size > (1_073_741_824 if name == "model.safetensors" else 268_435_456):
            raise ValueError(f"checkpoint file exceeds the byte budget: {name}")
    if sha256(snapshot / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("pretrained weights differ from the pinned SHA-256")

    destination.mkdir(mode=0o700)
    manifest: dict[str, str] = {}
    for name, source in sources.items():
        copied = destination / name
        shutil.copyfile(source, copied)
        manifest[name] = sha256(copied)
        if manifest[name] != sha256(source) or copied.is_symlink():
            raise RuntimeError(f"checkpoint file changed while staging: {name}")
    if manifest["model.safetensors"] != WEIGHTS_SHA256:
        raise RuntimeError("pretrained weights changed while staging")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage one verified public pretrained model")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=DONOR, revision=REVISION, allow_patterns=list(MODEL_FILES),
        max_workers=2,
    ))
    manifest = stage_snapshot(snapshot, args.destination)
    print(f"staged {DONOR}@{REVISION}: {len(manifest)} files; pinned safetensors SHA-256 verified")


if __name__ == "__main__":
    main()
