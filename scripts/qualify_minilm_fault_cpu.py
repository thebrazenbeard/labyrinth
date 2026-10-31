"""Opt-in MiniLM/LoRA CPU process-kill recovery development experiment.

Four fresh processes: uninterrupted reference, full-state checkpoint writer,
a resumed learner killed after its first post-checkpoint AdamW update, and a
fresh learner restored from the same checkpoint. The parent *kills* only the
worker it started, after observing a flushed post-update marker.

This tests process termination and checkpoint replay, not power-loss storage,
signed Minotaur-event replay, blind evaluation, or graduation authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from scripts.qualify_minilm_cpu import MODEL_ID, validate_fixtures
from scripts.qualify_minilm_resume_cpu import (
    _neural_stage, _read_json, _sha_bytes, _write_json, checkpoint_exposures,
    plan_output, verify_reference_resume,
)


SCHEMA = "labyrinth.minilm-process-kill-resume-development.v1"
MARKER_SCHEMA = "labyrinth.minilm-killed-after-adamw-update.v1"
MAX_WAIT_SECONDS = 120


def _snapshot_digest(root: Path) -> str:
    """Byte commitment to both durable checkpoint files, before and after kill."""
    snapshot = root / "snapshot"
    if root.is_symlink() or snapshot.is_symlink():
        raise ValueError("linked checkpoint directory")
    result = hashlib.sha256()
    for name in ("manifest.json", "state.safetensors"):
        path = snapshot / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing or linked checkpoint file")
        result.update(name.encode("ascii") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
    return result.hexdigest()


def _crash_after_first_optimizer_step(root: Path, seed: int,
                                      steps: int, split_at: int) -> None:
    """Run real pretrained MiniLM continuation; pause after a changed AdamW step."""
    import torch

    saved_receipt = _read_json(root / "checkpoint_receipt.json")
    original_step = torch.optim.AdamW.step
    calls = 0

    def step_then_pause(optimizer: Any, closure: Any = None) -> Any:
        nonlocal calls
        calls += 1
        if calls != 1:
            raise RuntimeError("fault worker progressed past the intended kill point")
        parameters = [(f"group{i}-param{j}", p)
                      for i, group in enumerate(optimizer.param_groups)
                      for j, p in enumerate(group["params"])]
        before = _sha_bytes(parameters)
        original_step(optimizer, closure=closure)
        after = _sha_bytes(parameters)
        if before == after:
            raise RuntimeError("fault-injected optimizer step did not change tensors")
        _write_json(root / "kill_ready.pending.json", {
            "schema": MARKER_SCHEMA, "pid": os.getpid(),
            "run_id": root.name, "seed": seed, "steps": steps,
            "split_at": split_at, "post_checkpoint_update_index": 1,
            "checkpoint_sha256": saved_receipt["checkpoint_sha256"],
            "pre_update_sha256": before, "post_update_sha256": after,
        })
        # A complete, fsynced marker becomes visible in one rename. The parent
        # must not mistake a partially serialized file for a finished update.
        os.replace(root / "kill_ready.pending.json", root / "kill_ready.json")
        # The parent normally kills this exact process. If the controlling
        # process disappears, a bounded watchdog exits without completing or
        # writing a false result; it cannot leave a training process forever.
        deadline = time.monotonic() + MAX_WAIT_SECONDS + 30
        while time.monotonic() < deadline:
            time.sleep(0.1)
        os._exit(97)

    torch.optim.AdamW.step = step_then_pause
    _neural_stage(root, "resume", seed, steps, split_at)
    raise RuntimeError("fault worker unexpectedly completed without parent kill")


def verify_forced_kill(marker: dict[str, Any], *, root_name: str,
                       expected_checkpoint_sha256: str, worker_pid: int,
                       other_pids: tuple[int, ...], returncode: int,
                       seed: int, steps: int, split_at: int) -> None:
    """Check the kill was in a distinct process, after a real changed step."""
    if (type(marker) is not dict or set(marker) != {
            "schema", "pid", "run_id", "seed", "steps", "split_at",
            "post_checkpoint_update_index", "checkpoint_sha256",
            "pre_update_sha256", "post_update_sha256"}):
        raise ValueError("missing or malformed post-update marker")
    if (marker["schema"] != MARKER_SCHEMA or marker["run_id"] != root_name
            or marker["checkpoint_sha256"] != expected_checkpoint_sha256
            or marker["seed"] != seed or marker["steps"] != steps
            or marker["split_at"] != split_at
            or type(marker["post_checkpoint_update_index"]) is not int
            or marker["post_checkpoint_update_index"] != 1):
        raise ValueError("marker is not bound to this checkpoint and fault point")
    if (type(worker_pid) is not int or type(marker["pid"]) is not int
            or marker["pid"] != worker_pid or worker_pid in other_pids
            or len(set(other_pids)) != len(other_pids)):
        raise ValueError("four-process isolation is not established")
    for key in ("pre_update_sha256", "post_update_sha256"):
        value = marker[key]
        if (type(value) is not str or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("invalid changed-parameter commitment")
    if marker["pre_update_sha256"] == marker["post_update_sha256"]:
        raise ValueError("post-checkpoint learner did not change parameters")
    if type(returncode) is not int or returncode == 0:
        raise ValueError("forced worker termination was not established")


def _stage_command(root: Path, seed: int, steps: int,
                   split_at: int, stage: str) -> list[str]:
    return [sys.executable, "-m", "scripts.qualify_minilm_fault_cpu",
            "--run-root", str(root), "--seed", str(seed),
            "--steps", str(steps), "--split-at", str(split_at),
            "--internal", "--stage", stage]


def wait_for_marker_and_kill(root: Path, worker: subprocess.Popen[str],
                             *, timeout_seconds: float = MAX_WAIT_SECONDS,
                             ) -> tuple[dict[str, Any], int]:
    """Kill exactly our subprocess only after its fsynced post-update marker.

    Do not conflate this with training verification: the caller must authenticate
    the marker's checkpoint/PID binding and compare final neural tensors.
    """
    marker: dict[str, Any] | None = None
    try:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if (root / "kill_ready.json").exists():
                marker = _read_json(root / "kill_ready.json")
                break
            if worker.poll() is not None:
                raise RuntimeError("kill worker exited before changed-step marker")
            time.sleep(0.1)
        if marker is None:
            raise RuntimeError("kill worker never reached changed-step marker")
        if worker.poll() is not None:
            raise RuntimeError("kill worker exited without parent termination")
        worker.kill()
        return marker, worker.wait(timeout=15)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=15)


def _orchestrate(root: Path, seed: int, steps: int, split_at: int) -> dict[str, Any]:
    root.mkdir(mode=0o700)
    repo_root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"})
    for stage in ("reference", "checkpoint"):
        with (root / f"{stage}.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                _stage_command(root, seed, steps, split_at, stage),
                cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=600, check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{stage} failed; inspect its isolated log")
    checkpoint = _read_json(root / "checkpoint_receipt.json")
    baseline_digest = _snapshot_digest(root)
    reference = _read_json(root / "reference.json")

    worker_pid = -1
    with (root / "kill_worker.log").open("x", encoding="utf-8") as log:
        worker = subprocess.Popen(
            _stage_command(root, seed, steps, split_at, "kill_worker"),
            cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        worker_pid = worker.pid
        marker, returncode = wait_for_marker_and_kill(root, worker)
    verify_forced_kill(marker, root_name=root.name,
                       expected_checkpoint_sha256=checkpoint["checkpoint_sha256"],
                       worker_pid=worker_pid,
                       other_pids=(reference["process_pid"], checkpoint["process_pid"]),
                       returncode=returncode, seed=seed, steps=steps,
                       split_at=split_at)
    if (root / "resumed.json").exists():
        raise RuntimeError("terminated worker incorrectly published completed output")
    if _snapshot_digest(root) != baseline_digest:
        raise RuntimeError("fault worker mutated its source checkpoint")
    with (root / "resume.log").open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            _stage_command(root, seed, steps, split_at, "resume"),
            cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=600, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError("post-kill resume failed; inspect isolated resume.log")
    resumed = _read_json(root / "resumed.json")
    verify_reference_resume(reference, resumed, checkpoint)
    if worker_pid == resumed["process_pid"]:
        raise RuntimeError("resumed learner reused terminated worker PID")
    if _snapshot_digest(root) != baseline_digest:
        raise RuntimeError("resume mutated its source checkpoint")
    report = {
        "schema": SCHEMA, "model_id": MODEL_ID, "seed": seed,
        "steps": steps, "split_at": split_at,
        "fresh_processes": 4, "killed_worker_pid": worker_pid,
        "kill_returncode": returncode, "forced_process_kill": True,
        "discarded_post_checkpoint_update": True,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_bytes_sha256": baseline_digest,
        "initial_trainable_sha256": reference["initial_digest"],
        "final_trainable_sha256": reference["final_trainable_digest"],
        "initial_lora_sha256": reference["initial_lora_digest"],
        "final_lora_sha256": reference["final_lora_digest"],
        "heldout_correct": reference["heldout_correct"],
        "transfer_correct": reference["transfer_correct"],
        "exact_resume_match": True, "graduation_authorized": False,
        "qualification": "PUBLIC_DEVELOPMENT_PRETRAINED_PROCESS_KILL_ONLY",
        "signed_event_journal_integrated": False,
        "durable_power_loss_tested": False,
        "independent_exam_custody": False,
    }
    _write_json(root / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent")
    parser.add_argument("--run-name")
    parser.add_argument("--run-root")
    parser.add_argument("--seed", type=int, default=20261033)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--split-at", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--stage", choices=(
        "reference", "checkpoint", "kill_worker", "resume"))
    args = parser.parse_args(argv)
    try:
        validate_fixtures()
        checkpoint_exposures(args.steps, args.split_at)
        if args.internal:
            if not args.stage or not args.run_root:
                raise ValueError("internal stage requires stage and run root")
            root = Path(args.run_root).resolve(strict=True)
            if not root.is_dir() or root.is_symlink():
                raise ValueError("internal root missing or symlink")
            if args.stage == "kill_worker":
                _crash_after_first_optimizer_step(root, args.seed,
                                                  args.steps, args.split_at)
            else:
                _neural_stage(root, args.stage, args.seed,
                              args.steps, args.split_at)
            return 0
        if args.stage or args.run_root or not args.output_parent or not args.run_name:
            raise ValueError("external invocation requires output parent and run name")
        root = plan_output(args.output_parent, args.run_name)
        if not args.execute:
            print(json.dumps({"schema": SCHEMA, "execute": False,
                              "run_root": str(root), "model_id": MODEL_ID,
                              "steps": args.steps, "split_at": args.split_at,
                              "graduation_authorized": False}, sort_keys=True))
            return 0
        report = _orchestrate(root, args.seed, args.steps, args.split_at)
        print(json.dumps({"run_root": str(root), **report}, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"preflight or execution failure: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
