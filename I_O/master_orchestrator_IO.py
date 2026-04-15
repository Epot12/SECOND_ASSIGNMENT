import os
import sys
import subprocess
import json
import re
from pathlib import Path


def run_and_capture(command: list[str], description: str, cwd: Path) -> str:
    print(f"\n[IO-ORCHESTRATOR] {description}")

    try:

        result = subprocess.run(command, check=True, cwd=cwd, capture_output=True, text=True)

        print(result.stdout)

        return result.stdout

    except subprocess.CalledProcessError as e:

        print(f"\n[CRITICAL ERROR] Execution failed: {' '.join(command)}")

        print(f"Error Output:\n{e.stderr}")

        sys.exit(1)


def parse_io_metrics(stdout: str) -> dict:
    """Extracts metrics using Regex on output printed by workers."""

    metrics = {}

    # Search throughput (catch all occurrences)

    all_th_matches = re.findall(r"\[METRIC\] Throughput:\s+([\d,]+)\s+items/second", stdout)

    if len(all_th_matches) >= 2:
        metrics['Sequential_Throughput'] = int(all_th_matches[0].replace(',', ''))

        metrics['Async_Throughput'] = int(all_th_matches[1].replace(',', ''))

    # searching speedup

    speedup_match = re.search(r"achieved a ([\d.]+)x speedup", stdout)

    if speedup_match:

        metrics['Speedup'] = float(speedup_match.group(1))

    else:

        metrics['Speedup'] = 1.0  # Fallback if times are comparable

    return metrics


def main():
    current_dir = Path(__file__).parent.resolve()  # I_O fold

    base_dir = current_dir.parent  # SECOND_ASSIGNMENT root

    # 1. Resolving Python Interpreters

    is_windows = os.name == 'nt'

    bin_dir = "Scripts" if is_windows else "bin"

    exe = ".exe" if is_windows else ""

    python_gil = base_dir / ".venv-gil" / bin_dir / f"python{exe}"

    python_nogil_t = base_dir / ".venv-nogil" / bin_dir / f"python3.13t{exe}"

    python_nogil_std = base_dir / ".venv-nogil" / bin_dir / f"python{exe}"

    python_nogil = python_nogil_t if python_nogil_t.exists() else python_nogil_std

    # 2. Parametric script path

    script_gil = current_dir / "main_IO_bench_GIL.py"

    script_nogil = current_dir / "main_IO_bench_NO_GIL.py"

    if not script_gil.exists() or not script_nogil.exists():
        print("[CRITICAL ERROR] I/O Benchmark scripts not found!")

        sys.exit(1)

    io_results = {}


    # synthetic data execution


    out = run_and_capture([str(python_gil), str(script_gil), "--mode", "synthetic"], "Running Synth GIL IPC",
                          current_dir)

    io_results["Synth_GIL_IPC"] = parse_io_metrics(out)

    out = run_and_capture([str(python_nogil), str(script_nogil), "--mode", "synthetic"], "Running Synth NO-GIL Threads",
                          current_dir)

    io_results["Synth_NOGIL"] = parse_io_metrics(out)

    # real data execution

    out = run_and_capture([str(python_gil), str(script_gil), "--mode", "real"], "Running Real GIL IPC", current_dir)

    io_results["Real_GIL_IPC"] = parse_io_metrics(out)

    out = run_and_capture([str(python_nogil), str(script_nogil), "--mode", "real"], "Running Real NO-GIL Threads",
                          current_dir)

    io_results["Real_NOGIL"] = parse_io_metrics(out)

    # 3. JSON export of results

    out_json = current_dir / "telemetry_io.json"

    with open(out_json, 'w') as f:
        json.dump(io_results, f, indent=4)

    print(f"\n[IO-ORCHESTRATOR] All benchmarks completed. Data saved to {out_json.name}")


if __name__ == "__main__":
    main()