"""Custodian-side, synthetic-only executable examiner seam.

This is a protocol integration harness, NOT an independent neural evaluator.
Only an operator outside the trainee trust boundary may supply private cases,
trusted arm artifacts, a fresh campaign policy/state, and the HMAC key. The
caller must separately establish checkpoint provenance, training budgets,
genuinely unseen independent problems, source pinning, and key custody. This
module cannot establish any of those facts, so refuses non-synthetic attestation.
Never put this module's case objects or signing key in trainee storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .graduation_gate import GateState, GraduationPolicy

SYNTHETIC_SCHEMA = "labyrinth.minotaur-synthetic-exam-attestation.v1"
_SYNTHETIC_DOMAIN = b"labyrinth.minotaur.synthetic-exam.v1\x00"


_FINAL = re.compile(r"ANSWER: (-?[0-9]+)\Z")
_SEMANTIC = re.compile(r"(?<![0-9])-?[0-9]+(?![0-9])")
_ARMS = ("candidate", "baseline", "zero_update", "corrupted")


@dataclass(frozen=True)
class ExamCase:
    id: str
    prompt: str
    answer: str  # private oracle; never serialized into the receipt or child stdin


@dataclass(frozen=True)
class Arm:
    checkpoint: Path
    command: tuple[str, ...]  # receives checkpoint path as argv[1], prompt on stdin
    training_steps: int


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _score(response: str, gold: str) -> tuple[bool, bool]:
    formatted = _FINAL.fullmatch(response) is not None
    # A malformed final line may contain a recognizable answer, so semantic
    # correctness is separately scored; formatting alone grants no gain.
    numbers = _SEMANTIC.findall(response)
    semantic = len(numbers) == 1 and int(numbers[0]) == int(gold)
    return semantic, formatted


def run_exam(*, policy: GraduationPolicy, state: GateState,
             arms: dict[str, Arm], cases: dict[str, list[ExamCase]],
             training_cases: list[ExamCase], key: bytes | None, run_id: str,
             evaluator_sha256: str, conditions_sha256: str,
             synthetic: bool = False) -> tuple[bytes, str]:
    """Run four fresh subprocesses per case, then sign oracle-derived scores.

    Child stdin contains exactly the prompt; child stdout is treated as plain
    answer text, never as a claim of correctness. stderr is never propagated.
    A fresh process is spawned per prediction; this does not prove that an
    external model adapter actually reloaded its weights.
    """
    if not synthetic:
        raise ValueError("only synthetic protocol attestation is implemented")
    if type(key) is not bytes or len(key) < 32:
        raise ValueError("independent custodian signing key required")
    if not isinstance(policy, GraduationPolicy) or not isinstance(state, GateState):
        raise ValueError("policy and state required")
    if state.next_attempt < 1 or set(arms) != set(_ARMS) or set(cases) != set(policy.required_slices):
        raise ValueError("incomplete arm or slice protocol")
    if (evaluator_sha256 != policy.expected_evaluator_sha256
            or conditions_sha256 != policy.expected_conditions_sha256):
        raise ValueError("unrecognized evaluator or conditions pin")
    if type(run_id) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", run_id):
        raise ValueError("invalid run ID")
    if not training_cases or any(not isinstance(c, ExamCase) for c in training_cases):
        raise ValueError("training custody list required")
    all_cases = training_cases + [c for group in cases.values() for c in group]
    if (any(not isinstance(c, ExamCase) or not c.id or not c.prompt
            or not re.fullmatch(r"-?[0-9]+", c.answer) for c in all_cases)
            or len({c.id for c in all_cases}) != len(all_cases)
            or len({c.prompt for c in all_cases}) != len(all_cases)
            or any(len(group) < policy.min_cases_per_slice for group in cases.values())):
        raise ValueError("missing oracle, overlap, or insufficient private cases")
    checkpoints = {}
    steps = {}
    for name in _ARMS:
        arm = arms[name]
        if (not isinstance(arm, Arm) or not arm.command or
                type(arm.training_steps) is not int or arm.training_steps < 0):
            raise ValueError("invalid arm")
        checkpoints[name] = _sha(arm.checkpoint.read_bytes())
        steps[name] = arm.training_steps
    if (checkpoints["candidate"] != policy.trainee_sha256
            or checkpoints["baseline"] != policy.base_sha256
            or len(set(checkpoints.values())) != 4
            or steps["baseline"] != 0 or steps["candidate"] == 0
            or steps["candidate"] != steps["zero_update"]
            or steps["candidate"] != steps["corrupted"]):
        raise ValueError("checkpoint identity or arm budget mismatch")

    groups = {}
    for name, group in cases.items():
        trials = []
        for case in group:
            trial = {"case_id": case.id, "prompt_commitment": _sha(case.prompt.encode("utf-8"))}
            for arm_name in _ARMS:
                arm = arms[arm_name]
                try:
                    child = subprocess.run(
                        [*arm.command, str(arm.checkpoint.resolve())],
                        input=case.prompt, text=True, capture_output=True,
                        timeout=10, check=True, encoding="utf-8")
                except (OSError, subprocess.SubprocessError) as exc:
                    raise RuntimeError(f"{arm_name} subprocess failed; no attestation") from None
                response = child.stdout
                if len(response.encode("utf-8")) > 8192:
                    raise RuntimeError("oversized adapter response; no attestation")
                semantic, formatted = _score(response.removesuffix("\n"), case.answer)
                trial[f"{arm_name}_semantic_correct"] = semantic
                trial[f"{arm_name}_format"] = formatted
                trial[f"{arm_name}_correct"] = semantic and formatted
                trial[f"{arm_name}_response_sha256"] = _sha(response.encode("utf-8"))
            trials.append(trial)
        groups[name] = trials
    suite = [{"slice": name, "id": c.id, "prompt_sha256": _sha(c.prompt.encode())}
             for name, group in cases.items() for c in group]
    receipt = {
        "schema": SYNTHETIC_SCHEMA, "objective_id": policy.objective_id,
        "trainee_sha256": policy.trainee_sha256, "base_sha256": policy.base_sha256,
        "session_nonce": policy.session_nonce, "run_id": run_id,
        "attempt": state.next_attempt, "evaluator_sha256": evaluator_sha256,
        "conditions_sha256": conditions_sha256,
        "suite_sha256": _sha(json.dumps(suite, sort_keys=True).encode()),
        "fresh_process_reload": False, "independent_private_oracle": False,
        "training_case_ids": [c.id for c in training_cases],
        "training_prompt_commitments": [_sha(c.prompt.encode()) for c in training_cases],
        "arm_checkpoint_sha256s": checkpoints, "arm_training_steps": steps,
        "slices": groups,
    }
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return raw, hmac.new(key, _SYNTHETIC_DOMAIN + raw, hashlib.sha256).hexdigest()


def main() -> int:
    """Private operator CLI: JSON config path and output prefix, never gold on stdout.

    The key is supplied as 64+ hexadecimal characters in EXAMINER_HMAC_KEY_HEX.
    This CLI intentionally supports only the explicitly synthetic protocol.
    """
    parser = argparse.ArgumentParser(description="Synthetic-only private examiner harness")
    parser.add_argument("config", type=Path, help="custodian-only JSON configuration")
    parser.add_argument("output_prefix", type=Path, help="new receipt file prefix")
    args = parser.parse_args()
    try:
        key = bytes.fromhex(os.environ.get("EXAMINER_HMAC_KEY_HEX", ""))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        policy = GraduationPolicy(**config["policy"])
        arms = {name: Arm(Path(spec["checkpoint"]), tuple(spec["command"]),
                          spec["training_steps"]) for name, spec in config["arms"].items()}
        cases = {name: [ExamCase(**case) for case in group]
                 for name, group in config["cases"].items()}
        training = [ExamCase(**case) for case in config["training_cases"]]
        raw, sig = run_exam(
            policy=policy, state=GateState(next_attempt=config["attempt"]),
            arms=arms, cases=cases, training_cases=training, key=key,
            run_id=config["run_id"], evaluator_sha256=config["evaluator_sha256"],
            conditions_sha256=config["conditions_sha256"], synthetic=config["synthetic"])
        # Exclusive creation prevents silently overwriting custody evidence.
        receipt_path = args.output_prefix.with_suffix(".receipt.json")
        signature_path = args.output_prefix.with_suffix(".signature.txt")
        if receipt_path.exists() or signature_path.exists():
            raise ValueError("output already exists")
        with receipt_path.open("xb") as stream:
            stream.write(raw)
        with signature_path.open("x", encoding="ascii") as stream:
            stream.write(sig + "\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        # No private prompts, gold, subprocess stderr, or key material printed.
        print("exam blocked; no valid attestation produced", file=sys.stderr)
        return 1
    print("synthetic-only attestation written; NOT real model graduation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
