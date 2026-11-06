"""Development-only compatibility entry point for packaged Maze/MiniLM.

The source implementation now lives under ``labyrinth`` so installed clients
need not import a global, generic ``scripts`` namespace. Running this module
as ``-m`` preserves the single canonical TrainingCycle/TrainingExample class
identity expected by the neural source-custody contracts.
"""

if __name__ == "__main__":
    import runpy

    runpy.run_module(
        "labyrinth.maze_conditioned_minilm_cpu_experiment",
        run_name="__main__", alter_sys=True,
    )
else:
    import importlib
    import sys

    sys.modules[__name__] = importlib.import_module(
        "labyrinth.maze_conditioned_minilm_cpu_experiment"
    )
