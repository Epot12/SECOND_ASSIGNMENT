import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command: list[str], description: str, cwd: str):
    print(f"\n[SUB-ORCHESTRATOR] {description}")
    try:
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError:
        print(f"\n[CRITICAL ERROR] Execution failed: {' '.join(command)}")
        sys.exit(1)

def print_report(gil_data, nogil_data, mode_name):
    print("\n" + "=" * 85)
    print(f"SCIENTIFIC BENCHMARK REPORT: {mode_name.upper()} WORKLOAD")
    print("=" * 85)
    all_results = {**gil_data, **nogil_data}

    print(f"{'Architecture':<25} | {'Wall: Ins (s)':<13} | {'Wall: Query':<11} | {'CPU: Ins (s)':<12} | {'CPU: Query':<12}")
    print("-" * 85)
    seq_tot = all_results['Sequential']['ins'] + all_results['Sequential']['read']

    for name, metrics in all_results.items():
        cpu_ins = metrics.get('ins_cpu', 0.0)
        cpu_read = metrics.get('read_cpu', 0.0)
        print(f"{name:<25} | {metrics['ins']:<13.2f} | {metrics['read']:<11.2f} | {cpu_ins:<12.2f} | {cpu_read:<12.2f}")

    print("\n" + "=" * 85)
    print("MICRO-ARCHITECTURAL SPEEDUP ANALYSIS (vs Sequential)")
    print("=" * 85)
    for name, metrics in all_results.items():
        if name == 'Sequential': continue
        tot = metrics['ins'] + metrics['read']
        speedup = seq_tot / tot if tot > 0 else 0
        print(f" -> {name:<22} : {speedup:.2f}x faster globally (Wall-Clock)")

    print("=================================================================\n")

def main():
    current_dir = Path(__file__).parent.resolve() # BENCHMARKS
    base_dir = current_dir.parent                 # SECOND_ASSIGNMENT

    # finding executables
    is_windows = os.name == 'nt'
    bin_dir = "Scripts" if is_windows else "bin"
    exe = ".exe" if is_windows else ""

    python_gil = base_dir / ".venv-gil" / bin_dir / f"python{exe}"
    python_nogil_t = base_dir / ".venv-nogil" / bin_dir / f"python3.13t{exe}"
    python_nogil_std = base_dir / ".venv-nogil" / bin_dir / f"python{exe}"
    python_nogil = python_nogil_t if python_nogil_t.exists() else python_nogil_std

    script_gil = current_dir / "bench_gil.py"
    script_nogil = current_dir / "bench_nogil.py"


    # Execution

    run_command([str(python_gil), str(script_gil), "--mode", "synthetic"], "Running GIL Worker (Synthetic)", current_dir)
    run_command([str(python_nogil), str(script_nogil), "--mode", "synthetic"], "Running No-GIL Worker (Synthetic)", current_dir)

    with open(current_dir / 'telemetry_gil_synthetic.json', 'r') as f: gil_synth = json.load(f)
    with open(current_dir / 'telemetry_nogil_synthetic.json', 'r') as f: nogil_synth = json.load(f)
    print_report(gil_synth, nogil_synth, "Synthetic")


    # Real execution

    run_command([str(python_gil), str(script_gil), "--mode", "real"], "Running GIL Worker (Real)", current_dir)
    run_command([str(python_nogil), str(script_nogil), "--mode", "real"], "Running No-GIL Worker (Real)", current_dir)

    with open(current_dir / 'telemetry_gil_real.json', 'r') as f: gil_real = json.load(f)
    with open(current_dir / 'telemetry_nogil_real.json', 'r') as f: nogil_real = json.load(f)
    print_report(gil_real, nogil_real, "Real Data")

    # json are left for general main

if __name__ == "__main__":
    main()