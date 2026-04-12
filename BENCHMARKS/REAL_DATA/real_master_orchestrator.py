import os
import sys
import subprocess
import json


def run_command(command: list[str], description: str, cwd: str):
    print(f"\n[ORCHESTRATOR] {description}")
    try:
        # cwd enforces execution in project root
        subprocess.run(command, check=True, cwd=cwd)
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


def check_file_exists(filepath: str, role: str):
    """Checks if the file exists and stops the program with a clear error if it is missing."""
    if not os.path.exists(filepath):
        print(f"\n[DIAGNOSTIC ERROR] Fundamental file missing: ({role})!")
        print(f"Searching here: {filepath}")
        sys.exit(1)


def main():
    # current_dir = REAL_DATA
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # bench_dir = BENCHMARKS
    bench_dir = os.path.dirname(current_dir)
    # base_dir = SECOND_ASSIGNMENT (Root del progetto)
    base_dir = os.path.dirname(bench_dir)

    print(f"[SYSTEM] Project Root Directory: {base_dir}")

    # SETUP GIL ENVIRONMENT
    path_gil_venv = os.path.join(base_dir, ".venv-gil")
    if not os.path.exists(path_gil_venv):
        run_command(["uv", "venv", ".venv-gil", "--python", "3.12"],
                    "Creating GIL Environment...", base_dir)
    else:
        print("\n[ORCHESTRATOR] GIL Environment already exists. Skipping creation.")

    run_command(["uv", "pip", "install", "--python", ".venv-gil", "mmh3", "joblib", "bitarray"], "Ensuring GIL dependencies...", base_dir)

    # SETUP NO-GIL ENVIRONMENT

    run_command(["uv", "python", "install", "3.13t"],
                "Checking/Fetching Free-Threaded Python...", base_dir)

    path_nogil_venv = os.path.join(base_dir, ".venv-nogil")
    if not os.path.exists(path_nogil_venv):
        run_command(["uv", "venv", ".venv-nogil", "--python", "3.13t"],
                    "Creating No-GIL Environment...", base_dir)
    else:
        print("\n[ORCHESTRATOR] No-GIL Environment already exists. Skipping creation.")

    run_command(["uv", "pip", "install", "--python", ".venv-nogil", "mmh3", "joblib", "bitarray"],
                "Ensuring No-GIL dependencies...", base_dir)

    # solving executables and script
    is_windows = os.name == 'nt'
    bin_dir = "Scripts" if is_windows else "bin"
    exe = ".exe" if is_windows else ""

    # Python GIL path
    python_gil = os.path.join(base_dir, ".venv-gil", bin_dir, f"python{exe}")

    # Python No-GIL path
    python_nogil_std = os.path.join(base_dir, ".venv-nogil", bin_dir, f"python{exe}")
    python_nogil_t = os.path.join(base_dir, ".venv-nogil", bin_dir, f"python3.13t{exe}")
    python_nogil = python_nogil_t if os.path.exists(python_nogil_t) else python_nogil_std

    # benchmark scripts paths
    script_gil = os.path.join(current_dir, "real_bench_gil.py")
    script_nogil = os.path.join(current_dir, "real_bench_nogil.py")

    #check
    check_file_exists(python_gil, "Python GIL Interpreter")
    check_file_exists(script_gil, "Benchmark GIL Script")
    check_file_exists(python_nogil, "Python No-GIL Interpreter")
    check_file_exists(script_nogil, "Benchmark No-GIL Script")

    # execution
    run_command([python_gil, script_gil], "Executing GIL Workloads (Tests 1-4)...", base_dir)
    run_command([python_nogil, script_nogil], "Executing No-GIL Workloads (Test 5)...", base_dir)

    # REPORTING
    gil_json = os.path.join(current_dir, 'telemetry_gil.json')
    nogil_json = os.path.join(current_dir, 'telemetry_nogil.json')

    check_file_exists(gil_json, "Telemetry JSON (GIL)")
    check_file_exists(nogil_json, "Telemetry JSON (No-GIL)")

    with open(gil_json, 'r') as f:
        gil_data = json.load(f)
    with open(nogil_json, 'r') as f:
        nogil_data = json.load(f)

    print_report(gil_data, nogil_data)


    os.remove(gil_json)
    os.remove(nogil_json)

if __name__ == "__main__":
    main()