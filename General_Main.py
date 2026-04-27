import json
import time
import argparse
from utils.utilities import *
from utils.plot_functions import *
import os
import sys
import subprocess
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# GENERAL MAIN: THE GRAND ORCHESTRATOR
#this file is the entry point of the entire project


print("[SYSTEM] Forcing deterministic behavior (PYTHONHASHSEED=0)...")
os.environ["PYTHONHASHSEED"] = "0"
print("[SYSTEM] Setting Orchestrator Flag for sub-processes...")
os.environ["IS_ORCHESTRATOR"] = "1"

DATASET_FILE = "cdx-00000"
EXPECTED_SHA256 = "3D9F2F2BAEFF3DB20262B6E5580A8BA34CECBD3742D0C898B484D5ACF5C476B1"

# Graphics settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['pdf.fonttype'] = 42  # Ensures fonts are embedded in the PDF
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")


def run_sub_process(command: list[str], description: str, cwd: Path):
    """Helper to run sub-orchestrators and stream their output in real-time."""
    print(f"\n{'=' * 70}")
    print(f"[GRAND ORCHESTRATOR] {description}")
    print(f"{'=' * 70}")
    try:
        # the colored/formatted output flows directly into the terminal
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError:
        print(f"\n[CRITICAL ERROR] Pipeline halted during: {description}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Grand Orchestrator: Pipeline Executive")
    parser.add_argument("--all", action="store_true", help="Executes all the sections")
    parser.add_argument("--base", action="store_true", help="Runs Basic Benchmarks")
    parser.add_argument("--io", action="store_true", help="Performs I/O Stress Tests")
    parser.add_argument("--scaling", action="store_true", help="Performs Scaling Tests")
    parser.add_argument("--stripe", action="store_true", help="Performs Tests on Stripe Strategies")

    args = parser.parse_args()

    run_base = args.all or args.base
    run_io = args.all or args.io
    run_scaling = args.all or args.scaling
    run_stripe = args.all or args.stripe

    if not any([run_base, run_io, run_scaling, run_stripe]):
        print("\n[!] NO SECTIONS SPECIFIED. Use --all or specific flags (e.g. --base --scaling).")
        parser.print_help()
        sys.exit(0)
    root_dir = Path(__file__).parent.resolve()
    data_dir = root_dir / "DATA"
    outputs_dir = root_dir / "Outputs"
    plots_dir = outputs_dir / "Plots"
    tables_dir = outputs_dir / "Tables"
    bench_dir = root_dir / "BENCHMARKS"
    utils_dir = root_dir/"utils"
    io_dir = root_dir / "I_O"
    stripe_dir = root_dir / "MULTITHREAD" / "stripe_strategy"

    print("=" * 80)
    print(" " * 25 + "PIPELINE EXECUTIVE" + " " * 25)
    print("=" * 80)

    # 1. Setup Directories
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SYSTEM] Output directories verified at: {outputs_dir.name}")

    # 2. Dataset Verification
    dataset_path = data_dir / DATASET_FILE
    if not verify_dataset_integrity(dataset_path, EXPECTED_SHA256):
        print("\n[HALT] Dataset integrity check failed. Please ensure the file is correctly")
        print("       downloaded from CC-MAIN-2024-10 and extracted in the DATA folder.")
        sys.exit(1)

    # --- 3. Virtual Environments Setup ---
    print("\n[SYSTEM] Verifying Virtual Environments...")

    core_packages_gil = ["mmh3", "joblib", "loky", "bitarray", "numpy", "seaborn", "matplotlib", "pandas"]
    core_packages_nogil = ["mmh3", "joblib", "loky", "bitarray", "numpy", "pandas"]

    # Gestione ambiente GIL
    path_gil_venv = root_dir / ".venv-gil"
    if not path_gil_venv.exists():
        print("[SYSTEM] Creating GIL Environment...")
        subprocess.run(["uv", "venv", ".venv-gil", "--python", "3.12"], cwd=root_dir, check=True)
        subprocess.run(["uv", "pip", "install", "--python", ".venv-gil"] + core_packages_gil, cwd=root_dir,
                           check=True)

    # Gestione ambiente No-GIL
    path_nogil_venv = root_dir / ".venv-nogil"

    if not path_nogil_venv.exists():
        print("[SYSTEM] Creating No-GIL Environment (Isolated Build)...")

        # --- SANIFICAZIONE AGGRESSIVA ---
        essential_vars = ["SYSTEMROOT", "PATH", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TMP", "TEMP", "USERNAME",
                              "COMSPEC"]
        clean_env = {k: v for k, v in os.environ.items() if k.upper() in essential_vars}

        parent_python_dir = str(Path(sys.executable).parent).lower()
        clean_env["PATH"] = os.pathsep.join(
                [p for p in clean_env.get("PATH", "").split(os.pathsep) if parent_python_dir not in p.lower()]
            )

        clean_env["PYTHONPATH"] = ""
        clean_env["PYTHONHOME"] = ""
        clean_env["PYTHONNOUSERSITE"] = "1"
        clean_env["VIRTUAL_ENV"] = ""
        # --------------------------------

        # Creazione venv con isolamento totale
        subprocess.run(["uv", "venv", ".venv-nogil", "--python", "3.13t"], cwd=root_dir, check=True, env=clean_env)

        # Installazione pacchetti
        print("[SYSTEM] Installing No-GIL dependencies...")
        subprocess.run(["uv", "pip", "install", "--python", ".venv-nogil"] + core_packages_nogil, cwd=root_dir,
                           check=True, env=clean_env)

    else:
        # Questo else ora è correttamente collegato a "if not path_nogil_venv.exists()"
        print("[SYSTEM] No-GIL Environment verified.")

    # Resolving Interpreters
    is_windows = os.name == 'nt'
    bin_dir = "Scripts" if is_windows else "bin"
    exe = ".exe" if is_windows else ""

    python_gil = path_gil_venv / bin_dir / f"python{exe}"
    python_nogil_t = path_nogil_venv / bin_dir / f"python3.13t{exe}"
    python_nogil_std = path_nogil_venv / bin_dir / f"python{exe}"
    python_nogil = python_nogil_t if python_nogil_t.exists() else python_nogil_std


    # PHASE 1: TASK DELEGATION

    # A. Base Architecture Benchmarks
    if run_base:
        master_orch = bench_dir / "master_orchestrator.py"
        if master_orch.exists():
            run_sub_process([str(python_gil), str(master_orch)], "DELEGATING TO MASTER ORCHESTRATOR", root_dir)
        else:
            print(f"[CRITICAL] Master Orchestrator not found at {master_orch}")
            sys.exit(1)
    else:
        print("\n[SKIP] Base Architecture Benchmarks skipped.")


    # B. I/O Stress Tests
    if run_io:
        io_orch = io_dir / "master_orchestrator_IO.py"
        if io_orch.exists():
            run_sub_process([str(python_gil), str(io_orch)], "DELEGATING TO I/O ORCHESTRATOR", root_dir)
        else:
            print(f"[CRITICAL] I/O Orchestrator not found at {io_orch}")
            sys.exit(1)
    else:
        print("\n[SKIP] I/O Stress Tests skipped.")

    # C. Scaling Tests (Amdahl's Law)
    if run_scaling:
        script_scal = bench_dir / "bench_script.py"
        if script_scal.exists():
            run_sub_process([str(python_nogil), str(script_scal)], "RUNNING SCALING LAWS ANALYSIS", root_dir)
        else:
            print(f"[WARNING] Scaling Script not found at {script_scal}. Skipping.")
    else:
        print("\n[SKIP] Scaling Laws Analysis skipped.")

    if run_stripe:
        stripe_orch = stripe_dir / "stripe_orchestrator.py"
        if stripe_orch.exists():
            # using No GIL python
            run_sub_process([str(python_nogil), str(stripe_orch)], "RUNNING STRIPE STRATEGIES ANALYSIS", root_dir)
        else:
            print(f"[WARNING] Stripe Orchestrator not found at {stripe_orch}. Skipping.")
    else:
        print("\n[SKIP] Stripe Strategies Analysis skipped.")

    # PHASE 2: DATA AGGREGATION

    print("\n[DATA EXPORT] Aggregating telemetry from all subsystems...")

    def load_json_safe(path: Path) -> dict:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    # Load Base Benchmarks
    synth_gil = load_json_safe(utils_dir / "telemetry_gil_synthetic.json")
    synth_nogil = load_json_safe(utils_dir / "telemetry_nogil_synthetic.json")
    real_gil = load_json_safe(utils_dir / "telemetry_gil_real.json")
    real_nogil = load_json_safe(utils_dir / "telemetry_nogil_real.json")

    # Load I/O and Scaling
    io_results = load_json_safe(io_dir / "telemetry_io.json")
    scal_results = load_json_safe(bench_dir / "telemetry_bench_script.json")

    stripe_synth = load_json_safe(stripe_dir / "telemetry_patterns_synthetic.json")
    stripe_real = load_json_safe(stripe_dir / "telemetry_patterns_real.json")

    synth_all = {**synth_gil, **synth_nogil, **stripe_synth}
    real_all = {**real_gil, **real_nogil, **stripe_real}

    aggregated_tables = {
        "Synthetic_Workloads": synth_all,
        "Real_Workloads": real_all,
        "IO_Stress_Tests": io_results,
        "Scaling_Laws": scal_results
    }

    tables_file = tables_dir / f"Aggregated_Benchmark_Tables_{TIMESTAMP}.json"
    with open(tables_file, 'w') as f:
        json.dump(aggregated_tables, f, indent=4)
    print(f"[DATA EXPORT] Aggregated Master Table saved to: {tables_file.name}")


    # PHASE 3: GARBAGE COLLECTION (CLEANUP)

    print("\n[CLEANUP] Removing intermediate JSON telemetry files...")

    temp_files = [
        utils_dir / "telemetry_gil_synthetic.json",
        utils_dir / "telemetry_nogil_synthetic.json",
        utils_dir / "telemetry_gil_real.json",
        utils_dir / "telemetry_nogil_real.json",
        bench_dir / "telemetry_bench_script.json",
        io_dir / "telemetry_io.json",
        stripe_dir / "telemetry_patterns_synthetic.json",
        stripe_dir / "telemetry_patterns_real.json"
    ]

    cleanup_count = 0
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()
            cleanup_count += 1

    print(f"[CLEANUP] Workspace is clean. Removed {cleanup_count} temporary files.")

    # PHASE 4: PLOT GENERATION
    if synth_all or real_all or io_results:
        generate_plots(synth_all, real_all, io_results, plots_dir, TIMESTAMP)
    else:
        print("\n[SKIP] Bypassing Base/IO plot generation (no data available).")

    if scal_results:
        # Amdahl (Speedup + Wall Clock Time)
        if "Amdahl" in scal_results:
            plot_amdahl_scaling(scal_results["Amdahl"], plots_dir, TIMESTAMP)

            scientific_data = {}
            base_seq_time = None

            for arch, core_data in scal_results["Amdahl"].items():
                times = [core_data[str(c)]["mean"] for c in sorted(core_data.keys(), key=int)]
                if arch == "Amdahl_Sequential":
                    base_seq_time = times[0]
                else:
                    scientific_data[arch] = times

            if base_seq_time is not None:
                generate_execution_time_plot(
                    scientific_data,
                    base_seq_time,
                    output_path=plots_dir / f"Fig3b_Wall_Clock_Time_Scaling_{TIMESTAMP}.pdf"
                )

        # Gustafson (Weak Scaling)
        if "Gustafson" in scal_results:
            plot_gustafson_scaling(scal_results["Gustafson"], plots_dir, TIMESTAMP)

        # Granularity (Chunk Size)
        if "Granularity" in scal_results:
            plot_chunk_optimization(scal_results["Granularity"], plots_dir, TIMESTAMP)

    print("\n" + "=" * 80)
    print(" ORCHESTRATION COMPLETE. ALL DATA SECURED. PDFS GENERATED.")
    print("=" * 80)


if __name__ == "__main__":
    main()