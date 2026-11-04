"""Development-only entry point for package-private MiniLM multi-seed probe."""

if __name__ == "__main__":
    from labyrinth.maze_multiseed_protocol_cpu_experiment import main

    raise SystemExit(main())
else:
    import importlib
    import sys

    sys.modules[__name__] = importlib.import_module(
        "labyrinth.maze_multiseed_protocol_cpu_experiment"
    )
