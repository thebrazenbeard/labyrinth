"""Reproducible fixed-weight ISA proof and bounded CPU/memory measurements."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import tempfile
from time import perf_counter_ns, process_time_ns
import tracemalloc

from labyrinth.fixed_weight_alu import (
    EncodedProgram, HostSupervisor, compile_stepper, load_stepper, opcode_reference,
)


def run() -> None:
    parser = ArgumentParser()
    parser.add_argument("--write", type=Path, help="Optional persistent compiled weight JSON")
    args = parser.parse_args()

    tracemalloc.start()
    with tempfile.TemporaryDirectory() as directory:
        path = args.write or Path(directory) / "compiled-neural-weights.json"
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
        compiled = compile_stepper()
        compiled.save(path)
        exact_weight_bytes = path.stat().st_size
        compiled_hash = compiled.sha256()
        frozen_bytes = path.read_bytes()
        weights = load_stepper(path)

        for bits in product((0, 1), repeat=5):
            if weights.forward(bits) != opcode_reference(bits):
                raise AssertionError(f'neural ISA mismatch at bits={bits}')

        program1 = EncodedProgram("counter", ((0, 1, 1),) * 3)
        program2 = EncodedProgram("xor-then-add", ((1, 0, 0), (1, 0, 1), (0, 1, 1)))
        # New program comes into existence AFTER weights were saved and reloaded.
        postfreeze_program = EncodedProgram("postfreeze-new-code", (
            (0, 1, 1), (1, 0, 1), (0, 1, 1), (1, 0, 1)))
        programs = (program1, program2, postfreeze_program)
        expected_states = ((1, 1), (0, 0), (0, 1))
        for program, expected in zip(programs, expected_states):
            session = HostSupervisor.start(weights, program)
            result = session.run()
            if result != expected:
                raise AssertionError(f'program {program.name} expected {expected}, got {result}')
            print(f"program={program.name} word_count={len(program.words)} final_state={result} "
                  f"program_sha256={program.sha256()}")

        if weights.sha256() != compiled_hash:
            raise AssertionError('frozen neural weight hash mismatch')
        if frozen_bytes != path.read_bytes():
            raise AssertionError('frozen neural weight bytes changed during execution')
        print(f"exhaustive_domain=32/32 neural_weights_sha256={weights.sha256()}")
        print(f"architecture=5->32->3 parameter_count=291 weight_file_bytes={exact_weight_bytes}")

        # Contextual local measurements only, not portable speed comparisons.
        sample = (0, 1, 1, 0, 0)
        count = 10_000
        for _ in range(100):
            weights.forward(sample)
        cpu_start, wall_start = process_time_ns(), perf_counter_ns()
        for _ in range(count):
            weights.forward(sample)
        elapsed_cpu_ns = process_time_ns() - cpu_start
        elapsed_wall_ns = perf_counter_ns() - wall_start
        _current_alloc, peak_alloc = tracemalloc.get_traced_memory()
        print(f"local_cpu_ms_per_10000_steps={elapsed_cpu_ns / 1_000_000:.3f} "
              f"local_wall_ms_per_10000_steps={elapsed_wall_ns / 1_000_000:.3f}")
        print(f"tracemalloc_python_peak_bytes={peak_alloc} "
              "(Python allocator peak, NOT full process RSS or GPU VRAM)")

        if args.write:
            print(f"saved_weights={path}")
    tracemalloc.stop()


if __name__ == "__main__":
    run()
