"""Command-line interface for offline mazes and controlled prompt experiments."""

from __future__ import annotations

import argparse
import json
import sys

from .adapters import AdapterError, load_adapter
from .runner import (
    build_training_artifact,
    load_json,
    run_experiment,
    run_maze,
    save_json_new,
)


def _tiers(raw: str) -> tuple[int, ...]:
    try:
        tiers = tuple(int(token.strip()) for token in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tiers must be comma-separated integers") from exc
    if not tiers or len(tiers) != len(set(tiers)) or any(t not in (1, 2, 3) for t in tiers):
        raise argparse.ArgumentTypeError("tiers must be unique values drawn from 1,2,3")
    return tiers


def _positive(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="labyrinth", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    def shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--seed", type=int, default=2026)
        subparser.add_argument("--tiers", type=_tiers, default=(1, 2, 3))
        subparser.add_argument("--adapter", choices=("replay", "ollama"), default="replay")
        subparser.add_argument("--replay-file", metavar="JSON", help="Exact prompt-to-answer JSON mapping")
        subparser.add_argument("--model", help="Required Ollama model identifier")
        subparser.add_argument("--base-url", default="http://127.0.0.1:11434")
        subparser.add_argument("--timeout-seconds", type=float, default=30.0)
        subparser.add_argument("--allow-remote-ollama", action="store_true")
        subparser.add_argument("--report", metavar="JSON", help="New report path; existing files are never overwritten")

    run = subcommands.add_parser("run", help="Generate and score a deterministic maze")
    shared(run)
    run.add_argument("--count", type=_positive, default=3, help="Tasks PER tier (default: 3)")
    run.add_argument("--split", default="eval", help="Short split identifier (default: eval)")

    experiment = subcommands.add_parser("experiment", help="Compare four held-out in-context conditions")
    shared(experiment)
    experiment.add_argument("--train-count", type=_positive, default=3, help="Training examples PER tier")
    experiment.add_argument("--eval-count", type=_positive, default=3, help="Held-out tasks PER tier")
    artifacts = experiment.add_mutually_exclusive_group()
    artifacts.add_argument("--artifact-in", metavar="JSON", help="Reuse a saved demonstrations artifact")
    artifacts.add_argument("--artifact-out", metavar="JSON", help="Persist a new demonstrations artifact")
    return parser


def _load_model(args: argparse.Namespace):
    if args.adapter == "ollama":
        if not args.model:
            raise ValueError("--model is required for the Ollama adapter")
        if args.replay_file:
            raise ValueError("--replay-file cannot be used with Ollama")
        return load_adapter(
            "ollama", model=args.model, base_url=args.base_url,
            timeout_seconds=args.timeout_seconds, allow_remote=args.allow_remote_ollama,
        )
    if args.model or args.allow_remote_ollama:
        raise ValueError("Ollama flags require --adapter ollama")
    if args.replay_file:
        responses = load_json(args.replay_file)
        if not isinstance(responses, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in responses.items()
        ):
            raise ValueError("replay file must contain a JSON string-to-string object")
        return load_adapter("replay", responses=responses)
    # Reproducible negative-control model: always responds with the same value.
    # It is explicitly not a reasoning model or a trainer.
    return load_adapter("replay", responses={}, default="ANSWER: 0", model_id="replay:constant-zero")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        adapter = _load_model(args)
        if args.command == "run":
            report = run_maze(adapter, seed=args.seed, count=args.count, tiers=args.tiers, split=args.split)
        else:
            artifact = (
                load_json(args.artifact_in) if args.artifact_in else
                build_training_artifact(
                    seed=args.seed, count=args.train_count, tiers=args.tiers,
                    adapter=adapter,
                )
            )
            report = run_experiment(
                lambda: _load_model(args), seed=args.seed, eval_count=args.eval_count,
                tiers=args.tiers, artifact=artifact,
            )
            if args.artifact_out:
                save_json_new(args.artifact_out, artifact)
        if args.report:
            save_json_new(args.report, report)
            print(f"Wrote reproducible report to {args.report}", file=sys.stderr)
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except (ValueError, TypeError, OSError, json.JSONDecodeError, AdapterError) as exc:
        print(f"labyrinth: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
