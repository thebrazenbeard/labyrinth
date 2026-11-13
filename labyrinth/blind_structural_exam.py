"""Answer-withheld, evaluator-side finite-mechanics examination.

Fresh action tables may be authored after a model checkpoint freezes. Merely
implementing this protocol does not secure case custody, attest model outputs,
or prove that a neural candidate reasoned about unfamiliar mechanics.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import random
import re
import secrets

from .live_model_probe import (_canonical, _file_bytes, _load, _receipt,
                               _write_once, ID, SHA)

PUBLIC_SCHEMA = "labyrinth.blind-mechanics-public.v1"
PRIVATE_SCHEMA = "labyrinth.blind-mechanics-private.v1"
COMMIT_SCHEMA = "labyrinth.blind-mechanics-precommit.v1"
RESPONSE_SCHEMA = "labyrinth.blind-mechanics-responses.v1"
SCORE_SCHEMA = "labyrinth.blind-mechanics-score.v1"
ACTIONS = ("A", "B", "C")
MAX_BYTES = 128 * 1024
PLAN_RE = re.compile(r"PLAN: [ABC](?: [ABC]){0,5}\Z")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _world(world: object) -> dict:
    """Accept arbitrary examiner-authored transition tables, not just seeds."""
    if type(world) is not dict or set(world) != {
            "state_count", "start", "target", "max_moves", "action_tables"}:
        raise ValueError("invalid structural mechanic fields")
    n = world["state_count"]
    start, target, bound = world["start"], world["target"], world["max_moves"]
    if (type(n) is not int or not 4 <= n <= 16
            or type(start) is not int or not 0 <= start < n
            or type(target) is not int or not 0 <= target < n
            or start == target or type(bound) is not int or not 1 <= bound <= 6):
        raise ValueError("invalid structural mechanic state, goal or move limit")
    table = world["action_tables"]
    if type(table) is not dict or set(table) != set(ACTIONS):
        raise ValueError("structural mechanic must have exactly three actions")
    for action in ACTIONS:
        phases = table[action]
        if type(phases) is not list or len(phases) != 2:
            raise ValueError("structural mechanic needs two temporal phases")
        for transitions in phases:
            if (type(transitions) is not list or len(transitions) != n
                    or any(type(x) is not int or not -1 <= x < n
                           for x in transitions)):
                raise ValueError("invalid structural mechanic transition")
    return deepcopy(world)


def _replay(world: dict, actions: tuple[str, ...] | list[str]) -> bool:
    state = world["start"]
    if not 1 <= len(actions) <= world["max_moves"]:
        return False
    for turn, action in enumerate(actions):
        if action not in ACTIONS:
            return False
        state = world["action_tables"][action][turn % 2][state]
        if state == -1:
            return False
    return state == world["target"]


def _bfs_witness(world: dict) -> list[str] | None:
    """Shortest-policy breadth-first symbolic reference."""
    q = deque([(world["start"], 0, ())])
    visited = {(world["start"], 0)}
    while q:
        state, turn, path = q.popleft()
        if turn >= world["max_moves"]:
            continue
        for action in ACTIONS:
            next_state = world["action_tables"][action][turn % 2][state]
            if next_state == -1:
                continue
            witness = path + (action,)
            if next_state == world["target"]:
                return list(witness)
            key = (next_state, turn + 1)
            if key not in visited:
                visited.add(key)
                q.append((next_state, turn + 1, witness))
    return None


def _enumerated_witness(world: dict) -> list[str] | None:
    """Independent exhaustive enumeration, not calling the BFS step function."""
    for size in range(1, world["max_moves"] + 1):
        for word in product(ACTIONS, repeat=size):
            current = world["start"]
            for step, symbol in enumerate(word):
                current = world["action_tables"][symbol][step & 1][current]
                if current < 0:
                    break
            else:
                if current == world["target"]:
                    return list(word)
    return None


def _oracle(world: dict) -> list[str] | None:
    breadth = _bfs_witness(world)
    brute = _enumerated_witness(world)
    if (breadth is None) != (brute is None) or (
            breadth is not None and len(breadth) != len(brute)):
        raise ValueError("dual structural mechanics oracles disagree")
    if breadth is not None and not _replay(world, breadth):
        raise ValueError("structural mechanic oracle proposed invalid witness")
    return breadth


def _random_world(rng) -> dict:
    n = rng.randrange(10, 16)
    start = rng.randrange(n)
    target = rng.randrange(n - 1)
    if target >= start:
        target += 1
    return {"state_count": n, "start": start, "target": target,
            "max_moves": 3,
            "action_tables": {
                action: [[rng.randrange(-1, n) for _ in range(n)]
                         for _ in range(2)]
                for action in ACTIONS}}


def _prompt(world: dict) -> str:
    return (
        "This is a new finite transition system, not a named Minotaur case. "
        "Its mechanics are the action_tables below. Start at the stated state. "
        "At move number 0 use phase 0; at move number 1 use phase 1; then alternate "
        "phase = move number modulo 2. Choose one action A, B, or C per move. "
        "Each row action_tables[action][phase][state] gives the next state; -1 "
        "blocks that move. Reach the target state within max_moves. "
        "Return exactly PLAN: A B C (one to max_moves space-separated action "
        "names) for a winning trace, or UNSAT if no trace can reach the goal. "
        "Do not answer with a case label or an unsupported verdict.\n"
        + json.dumps(world, sort_keys=True, separators=(",", ":")))


def _public_trial(trial: dict) -> dict:
    return {"trial_id": trial["trial_id"],
            "world": deepcopy(trial["world"]), "prompt": _prompt(trial["world"])}


def prepare_exam(private_path, public_path, precommit_path, *,
                 count=8, rng=None, worlds=None) -> dict:
    """Prepare caller-held answer key and a separate candidate-facing bundle."""
    locations = [Path(private_path), Path(public_path), Path(precommit_path)]
    if len({str(p.absolute()) for p in locations}) != 3 or any(p.exists() for p in locations):
        raise ValueError("exam preparation requires three distinct fresh output paths")
    if worlds is None:
        if type(count) is not int or not 2 <= count <= 16 or count % 2:
            raise ValueError("generated exam count must be an even 2..16")
    elif (type(worlds) is not list or not 1 <= len(worlds) <= 16):
        raise ValueError("external structural worlds require 1..16 cases")
    rng = rng if rng is not None else secrets.SystemRandom()
    exam_id = f"blind-{rng.getrandbits(128):032x}"
    seen = set()
    trials = []
    n = count if worlds is None else len(worlds)
    for index in range(n):
        if worlds is None:
            want_win = index % 2 == 0
            for _ in range(2048):
                world = _random_world(rng)
                witness = _oracle(world)
                if (witness is not None) == want_win:
                    break
            else:
                raise ValueError("could not produce balanced structural mechanics")
        else:
            world = _world(worlds[index])
            witness = _oracle(world)
        if worlds is None:
            world = _world(world)
        for _ in range(16):
            trial_id = f"blind-{rng.getrandbits(96):024x}"
            if trial_id not in seen:
                break
        else:
            raise ValueError("fresh trial identity collision")
        seen.add(trial_id)
        trials.append({"trial_id": trial_id, "world": world,
                       "oracle_witness": witness, "satisfiable": witness is not None})
    private = {"schema": PRIVATE_SCHEMA, "exam_id": exam_id,
               "source_class": ("RANDOM_SOURCE_DEFINED_NOT_INDEPENDENT" if worlds is None
                                else "CALLER_AUTHORED_TABLES_NOT_AUTHENTICATED"),
               "trials": trials}
    public = {"schema": PUBLIC_SCHEMA, "exam_id": exam_id,
              "trials": [_public_trial(case) for case in trials]}
    _write_once(private_path, private)
    _write_once(public_path, public)
    raw_private = _file_bytes(private_path, MAX_BYTES)
    raw_public = _file_bytes(public_path, MAX_BYTES)
    plan = {
        "schema": COMMIT_SCHEMA, "exam_id": exam_id, "case_count": n,
        "private_sha256": _sha(raw_private), "public_sha256": _sha(raw_public),
        "status": "NOT_EXECUTED",
        "custody_class": "CALLER_CONTROLLED_UNATTESTED",
        "independent_custody_verified": False}
    return _write_once(precommit_path, _receipt(plan))


def _read_bundle(private_path, public_path, precommit_path,
                 expected_precommit_sha256):
    if (type(expected_precommit_sha256) is not str
            or SHA.fullmatch(expected_precommit_sha256) is None):
        raise ValueError("expected original precommit pin must be lowercase SHA256")
    raw_private = _file_bytes(private_path, MAX_BYTES)
    raw_public = _file_bytes(public_path, MAX_BYTES)
    plan = _load(_file_bytes(precommit_path, MAX_BYTES))
    expected = {"schema", "exam_id", "case_count", "private_sha256",
                "public_sha256", "status", "custody_class",
                "independent_custody_verified", "receipt_sha256"}
    if (type(plan) is not dict or set(plan) != expected
            or plan["schema"] != COMMIT_SCHEMA
            or plan["status"] != "NOT_EXECUTED"
            or plan["custody_class"] != "CALLER_CONTROLLED_UNATTESTED"
            or plan["independent_custody_verified"] is not False
            or type(plan["receipt_sha256"]) is not str
            or SHA.fullmatch(plan["receipt_sha256"]) is None
            or _sha(_canonical({k: v for k, v in plan.items()
                                if k != "receipt_sha256"})) != plan["receipt_sha256"]
            or plan["receipt_sha256"] != expected_precommit_sha256):
        raise ValueError("exam precommit pin or custody schema mismatch")
    if plan["private_sha256"] != _sha(raw_private) or plan["public_sha256"] != _sha(raw_public):
        raise ValueError("exam private/public source differs from original precommit")
    private, public = _load(raw_private), _load(raw_public)
    if (type(private) is not dict
            or set(private) != {"schema", "exam_id", "source_class", "trials"}
            or private["schema"] != PRIVATE_SCHEMA
            or private["source_class"] not in (
                "RANDOM_SOURCE_DEFINED_NOT_INDEPENDENT",
                "CALLER_AUTHORED_TABLES_NOT_AUTHENTICATED")
            or type(public) is not dict or set(public) != {"schema", "exam_id", "trials"}
            or public["schema"] != PUBLIC_SCHEMA
            or type(private["exam_id"]) is not str or
            ID.fullmatch(private["exam_id"]) is None
            or private["exam_id"] != public["exam_id"] or
            private["exam_id"] != plan["exam_id"]
            or type(private["trials"]) is not list or not 1 <= len(private["trials"]) <= 16
            or type(public["trials"]) is not list
            or len(private["trials"]) != len(public["trials"])
            or plan["case_count"] != len(private["trials"])):
        raise ValueError("invalid exam source schemas or case roster")
    ids = set()
    for original, advertised in zip(private["trials"], public["trials"]):
        if (type(original) is not dict
                or set(original) != {"trial_id", "world", "oracle_witness", "satisfiable"}
                or type(original["trial_id"]) is not str
                or ID.fullmatch(original["trial_id"]) is None
                or original["trial_id"] in ids):
            raise ValueError("invalid exam private case identity")
        ids.add(original["trial_id"])
        world = _world(original["world"])
        witness = _oracle(world)
        if (type(original["satisfiable"]) is not bool
                or original["satisfiable"] is not (witness is not None)
                or original["oracle_witness"] != witness
                or type(advertised) is not dict
                or advertised != _public_trial(original)):
            raise ValueError("exam public mechanics or private witness inconsistent")
    return private, plan


def _response_correct(world: dict, text: str, witness: list[str] | None):
    if text == "UNSAT":
        return witness is None, False, witness is None
    if PLAN_RE.fullmatch(text):
        actions = tuple(text[6:].split(" "))
        return _replay(world, actions), _replay(world, actions), False
    return False, False, False


def score_exam(private_path, public_path, precommit_path, responses_path, out_path, *,
               expected_precommit_sha256,
               examiner_attestation_path=None, examiner_public_key_path=None,
               expected_examiner_key_sha256=None, examiner_exam_id=None,
               observation_path=None, expected_observation_sha256=None) -> dict:
    """Score caller-supplied text, never attesting it came from a real model."""
    if Path(out_path).exists():
        raise FileExistsError("exam score output already exists")
    private, plan = _read_bundle(private_path, public_path, precommit_path,
                                 expected_precommit_sha256)
    sig = (examiner_attestation_path, examiner_public_key_path,
           expected_examiner_key_sha256, examiner_exam_id)
    signed = None
    if any(item is not None for item in sig):
        if not all(item is not None for item in sig):
            raise ValueError("incomplete examiner signed attestation arguments")
        from .examiner_signature import verify_examiner_attestation
        signed = verify_examiner_attestation(
            examiner_attestation_path, examiner_public_key_path,
            expected_key_sha256=expected_examiner_key_sha256,
            expected_exam_id=examiner_exam_id,
            expected_precommit_sha256=plan["receipt_sha256"])
        if signed["exam_id"] != private["exam_id"]:
            raise ValueError("signed examiner exam id differs from actual private exam")
    if (observation_path is None) != (expected_observation_sha256 is None):
        raise ValueError("local HTTP observation needs the original receipt pin")
    raw_responses = _file_bytes(responses_path, MAX_BYTES)
    verified_observation = None
    if observation_path is not None:
        from .local_neural_observer import verify_local_observation
        raw_public = _file_bytes(public_path, MAX_BYTES)
        if _sha(raw_public) != plan["public_sha256"]:
            raise ValueError("public source differs from precommit during observation replay")
        verified_observation = verify_local_observation(
            observation_path, expected_observation_sha256,
            raw_public, raw_responses)
    response = _load(raw_responses)
    if (type(response) is not dict
            or set(response) != {"schema", "exam_id", "responses"}
            or response["schema"] != RESPONSE_SCHEMA
            or response["exam_id"] != private["exam_id"]
            or type(response["responses"]) is not list
            or len(response["responses"]) != len(private["trials"])):
        raise ValueError("candidate response roster is invalid")
    correct = verified_plans = verified_unsat = 0
    row_results = []
    for item, trial in zip(response["responses"], private["trials"]):
        if (type(item) is not dict or set(item) != {"trial_id", "response"}
                or item["trial_id"] != trial["trial_id"]
                or type(item["response"]) is not str or
                len(item["response"]) > 128):
            raise ValueError("candidate response roster has duplicate, misplaced or invalid trial")
        passed, plan_win, unsat = _response_correct(
            trial["world"], item["response"], trial["oracle_witness"])
        correct += int(passed)
        verified_plans += int(plan_win)
        verified_unsat += int(unsat)
        row_results.append({"trial_id": trial["trial_id"], "correct": passed,
                            "plan_win": plan_win, "unsat_proven": unsat,
                            "response_sha256": _sha(item["response"].encode("utf-8"))})
    output = {
        "schema": SCORE_SCHEMA,
        "status": "SOURCE_ONLY_RESPONSES_NOT_MODEL_ATTESTED",
        "exam_id": private["exam_id"], "case_count": len(private["trials"]),
        "correct": correct, "verified_plans": verified_plans,
        "verified_unsat": verified_unsat,
        "dual_oracles_agree": True, "score_evidence": "SOURCE_ONLY_LOCAL_REPLAY",
        "original_precommit_sha256": plan["receipt_sha256"],
        "responses_file_sha256": _sha(raw_responses),
        "invocation_evidence": (verified_observation["evidence"]
                                 if verified_observation else "NOT_SUPPLIED"),
        "observed_model_digest": (verified_observation["declared_model_digest"]
                                  if verified_observation else None),
        "observed_model_tag": (verified_observation["model_tag"]
                               if verified_observation else None),
        "observation_receipt_sha256": (verified_observation["receipt_sha256"]
                                       if verified_observation else None),
        "examiner_signature_evidence": (signed["evidence"] if signed else "NOT_SUPPLIED"),
        "examiner_key_sha256": (signed["examiner_key_sha256"] if signed else None),
        "neural_learning_verified": False, "independent_custody_verified": False,
        "cases": row_results,
        "limitations": [
            "Candidate responses are caller-supplied; model execution is not attested.",
            "The private-case file and external digest may share a controller.",
            "Source-defined generated tables do not establish hidden family novelty.",
            "Signature authentication does not prove independent examiner custody.",
            "A locally pinned HTTP receipt cannot authenticate actual loaded model weights.",
        ],
    }
    return _write_once(out_path, _receipt(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prepare = sub.add_parser("prepare", help="generate private rules and public trial bundle")
    prepare.add_argument("--out-private", required=True)
    prepare.add_argument("--out-public", required=True)
    prepare.add_argument("--out-precommit", required=True)
    prepare.add_argument("--count", type=int, default=8)
    prepare.add_argument("--worlds-file", help="examiner-authored list of novel world tables")
    score = sub.add_parser("score", help="strictly replay candidate plans and claimed UNSAT")
    score.add_argument("--private", required=True)
    score.add_argument("--public", required=True)
    score.add_argument("--precommit", required=True)
    score.add_argument("--responses", required=True)
    score.add_argument("--expected-precommit-sha256", required=True)
    score.add_argument("--observation", help="write-once loopback model HTTP receipt")
    score.add_argument("--expected-observation-sha256",
                       help="separately retained original model HTTP receipt SHA256")
    score.add_argument("--examiner-attestation")
    score.add_argument("--examiner-public-key")
    score.add_argument("--expected-examiner-key-sha256")
    score.add_argument("--examiner-exam-id")
    score.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        worlds = _load(_file_bytes(args.worlds_file, MAX_BYTES)) if args.worlds_file else None
        result = prepare_exam(args.out_private, args.out_public,
                              args.out_precommit, count=args.count, worlds=worlds)
        status = "LOCAL_SOURCE_PREPARED_NOT_INDEPENDENT"
    else:
        result = score_exam(args.private, args.public, args.precommit,
                            args.responses, args.out,
                            expected_precommit_sha256=args.expected_precommit_sha256,
                            examiner_attestation_path=args.examiner_attestation,
                            examiner_public_key_path=args.examiner_public_key,
                            expected_examiner_key_sha256=args.expected_examiner_key_sha256,
                            examiner_exam_id=args.examiner_exam_id,
                            observation_path=args.observation,
                            expected_observation_sha256=args.expected_observation_sha256)
        status = result["status"]
    print(json.dumps({"status": status, "exam_id": result["exam_id"],
                      "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
