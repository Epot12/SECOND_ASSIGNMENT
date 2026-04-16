import os
import sys
import subprocess
from pathlib import Path


def run_benchmark(target_script: Path, mode: str):
    print("\n" + "=" * 80)
    print(f"Start of automatic execution: MODE [{mode.upper()}]")
    print("=" * 80 + "\n")

    # using sys.executable to ensure that the subprocess uses the SAME
    # Python interpreter (e.g. python3.13t No-GIL) with which this orchestrator has been launched.
    command = [sys.executable, str(target_script), "--mode", mode]

    try:
        # executing command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Benchmark for mode '{mode}' failed with code {e.returncode}.")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n[WARNING] Execution interrupted by the user while in mode: '{mode}'.")
        sys.exit(1)


def main():
    current_dir = Path(__file__).parent.resolve()

    benchmark_script = current_dir / "striped_bench.py"

    if not benchmark_script.exists():
        print(f"[ERROR] File {benchmark_script.name} not found in folder {current_dir}")
        sys.exit(1)

    print("=================================================================")
    print("Starting Execution")
    print("=================================================================")

    # executing
    modes_to_run = ["synthetic", "real"]

    for mode in modes_to_run:
        run_benchmark(benchmark_script, mode)

    print("\n" + "=" * 80)
    print("All benchmarks have been completed successfully")
    print("JSON files have been saved in current directory")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()