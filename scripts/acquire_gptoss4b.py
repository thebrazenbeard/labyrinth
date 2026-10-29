"""Opt-in acquisition of Vera's *inference-only* GPT-OSS 4.2B candidate.

No neural training, model selection, or deployment is performed by this script.
The URL is pinned to an immutable Hugging Face commit. It does not import or
execute any code from the model repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone
from urllib.request import Request, urlopen


REPO = "cyberjunkee/gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts-Q4_K_M-GGUF"
REVISION = "d3825cfdc1c250b75a2485e9054162bf237fdd01"
FILENAME = "gpt-oss-4.2b-specialized-all-pruned-moe-only-4-experts-q4_k_m.gguf"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"
EXPECTED_BYTES = 3_233_019_552  # Hugging Face's versioned Xet pointer.
EXPECTED_SHA256 = "38b0279e4294beb54f97ac23051674174b34b2acd6c5e6532860c72298789586"
MAX_BYTES = 4_000_000_000  # Patrick's explicit GGUF file-size ceiling.
RESERVE_BYTES = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight(directory: Path) -> tuple[Path, Path, Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("destination directory must already exist and not be a symlink")
    target = directory / FILENAME
    partial = directory / (FILENAME + ".part")
    receipt = directory / (FILENAME + ".receipt.json")
    for path in (target, partial, receipt):
        if path.is_symlink():
            raise ValueError(f"refusing symlink: {path}")
    if target.exists() or receipt.exists():
        raise FileExistsError("candidate or receipt already exists; refusing overwrite")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > MAX_BYTES:
        raise ValueError("partial file already exceeds the 4 GB ceiling")
    required = (MAX_BYTES - offset) + RESERVE_BYTES
    free = shutil.disk_usage(directory).free
    if free < required:
        raise OSError(f"insufficient free space: {free} bytes; need {required} bytes")
    return target, partial, receipt


def download(directory: Path, *, opener=urlopen, expected_bytes: int = EXPECTED_BYTES,
             expected_sha256: str = EXPECTED_SHA256,
             max_bytes: int = MAX_BYTES) -> dict[str, object]:
    """Resume only when server proves the requested offset; never overwrite."""
    target, partial, receipt = _preflight(directory)
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Vera-Labyrinth-Candidate/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with opener(Request(URL, headers=headers), timeout=60) as response:
        status = getattr(response, "status", None)
        if offset:
            content_range = response.headers.get("Content-Range", "")
            if status != 206 or not content_range.startswith(f"bytes {offset}-"):
                raise ValueError("server did not prove resumable range; partial retained")
        elif status != 200:
            raise ValueError(f"expected HTTP 200, got {status}")
        with partial.open("ab" if partial.exists() else "xb") as stream:
            total = offset
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("model exceeds the 4 GB file ceiling; partial retained")
                stream.write(chunk)

    size = partial.stat().st_size
    if size != expected_bytes or size > max_bytes:
        raise ValueError("downloaded size disagrees with pinned Xet pointer; partial retained")
    with partial.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("download is not a GGUF file; partial retained")
    digest = _sha256(partial)
    if digest.lower() != expected_sha256.lower():
        raise ValueError("download SHA-256 disagrees with pinned Xet pointer; partial retained")
    record = {
        "candidate": "gpt-oss-4.2b-Q4_K_M",
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_url": URL,
        "filename": FILENAME,
        "bytes": size,
        "sha256_verified": digest,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "downloaded-gguf-only; untrained; unqualified",
    }
    # Hard-link promotion fails rather than overwriting if another process
    # creates the target. It keeps the validated bytes on the same volume.
    os.link(partial, target)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
    partial.unlink()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path(r"D:\VERA\Models"))
    parser.add_argument("--download", action="store_true",
                        help="Opt in to downloading the pinned 3.23 GB GGUF")
    args = parser.parse_args()
    target, partial, _ = _preflight(args.dest)
    if not args.download:
        print(json.dumps({"stage": "preflight-only", "model": REPO,
                          "revision": REVISION, "target": str(target),
                          "resumable_partial_bytes": partial.stat().st_size if partial.exists() else 0,
                          "training_started": False}, indent=2))
        return 0
    print(json.dumps(download(args.dest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
