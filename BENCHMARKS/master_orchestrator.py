import os
import sys
import subprocess
import json


def run_command(command: list[str], description: str):
    print(f"\n[ORCHESTRATOR] {description}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(f"\n[CRITICAL ERROR] Execution failed: {' '.join(command)}")
        sys.exit(1)


def print_report(gil_data, nogil_data):
    print("\n" + "=" * 65)
    print("SCIENTIFIC BENCHMARK REPORT: SCALABLE BLOOM FILTERS")
    print("=" * 65)

    all_results = {**gil_data, **nogil_data}

    print(f"{'Architecture':<25} | {'Insertion (s)':<13} | {'Query (s)':<11} | {'Total (s)':<9}")
    print("-" * 65)

    seq_tot = all_results['Sequential']['ins'] + all_results['Sequential']['read']

    for name, metrics in all_results.items():
        tot = metrics['ins'] + metrics['read']
        print(f"{name:<25} | {metrics['ins']:<13.2f} | {metrics['read']:<11.2f} | {tot:<9.2f}")

    print("\n" + "=" * 65)
    print("MICRO-ARCHITECTURAL SPEEDUP ANALYSIS (vs Sequential)")
    print("=" * 65)

    for name, metrics in all_results.items():
        if name == 'Sequential': continue
        tot = metrics['ins'] + metrics['read']
        speedup = seq_tot / tot if tot > 0 else 0
        print(f" -> {name:<22} : {speedup:.2f}x faster globally")

    # Deep architectural insights
    sota_ins = all_results['SotaIPC']['ins']
    thread_ins = all_results['NativeThreads']['ins']

    print("\n[RESEARCH INSIGHTS]")
    print(f"1. IPC vs No-GIL Memory Access:")
    if thread_ins < sota_ins:
        print(
            f"   Native Free-Threading is {(sota_ins / thread_ins):.2f}x faster in allocation than IPC Shared Memory.")
    else:
        print(
            f"   Multiprocessing IPC outperforms No-GIL threading by {(thread_ins / sota_ins):.2f}x in this workload.")
    print("=================================================================\n")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench_dir = os.path.join(base_dir, "BENCHMARKS")

    # 1. SETUP ENVIRONMENTS (OS-Agnostic with uv)
    run_command(["uv", "venv", ".venv-gil", "--python", "3.12"], "Setting up GIL Environment...")
    run_command(["uv", "pip", "install", "--python", ".venv-gil", "mmh3", "joblib"], "Installing GIL dependencies...")

    run_command(["uv", "python", "install", "3.13t"], "Fetching Free-Threaded Python...")
    run_command(["uv", "venv", ".venv-nogil", "--python", "3.13t"], "Setting up No-GIL Environment...")
    run_command(["uv", "pip", "install", "--python", ".venv-nogil", "mmh3", "joblib"],
                "Installing No-GIL dependencies...")

    # 2. RESOLVE EXECUTABLES DYNAMICALLY (Windows vs POSIX)
    is_windows = os.name == 'nt'
    bin_dir = "Scripts" if is_windows else "bin"
    exe = ".exe" if is_windows else ""

    python_gil = os.path.join(base_dir, ".venv-gil", bin_dir, f"python{exe}")
    python_nogil = os.path.join(base_dir, ".venv-nogil", bin_dir, f"python{exe}")

    script_gil = os.path.join(bench_dir, "bench_gil.py")
    script_nogil = os.path.join(bench_dir, "bench_nogil.py")

    # 3. RUN TELEMETRY WORKERS
    run_command([python_gil, script_gil], "Executing GIL Workloads (Tests 1-4)...")
    run_command([python_nogil, script_nogil], "Executing No-GIL Workloads (Test 5)...")

    # 4. AGGREGATE TELEMETRY
    gil_json = os.path.join(bench_dir, 'telemetry_gil.json')
    nogil_json = os.path.join(bench_dir, 'telemetry_nogil.json')

    with open(gil_json, 'r') as f: gil_data = json.load(f)
    with open(nogil_json, 'r') as f: nogil_data = json.load(f)

    # 5. GENERATE FINAL REPORT
    print_report(gil_data, nogil_data)

    # Clean up telemetry files (optional but clean)
    os.remove(gil_json)
    os.remove(nogil_json)


if __name__ == "__main__":
    main()