import os
import json
import seaborn as sns
from utils.utilities import *


# GENERAL MAIN TO AUTOMATE ALL THE PIPELINE OF EXPERIMENTS

DATASET_FILE = "cdx-00000"
EXPECTED_SHA256 = "3D9F2F2BAEFF3DB20262B6E5580A8BA34CECBD3742D0C898B484D5ACF5C476B1"

# Graphics settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['pdf.fonttype'] = 42  # Ensures fonts are embedded in the PDF


# ORCHESTRATION LOGIC

def main():
    root_dir = Path(__file__).parent.resolve()
    data_dir = root_dir / "DATA"
    outputs_dir = root_dir / "Outputs"
    plots_dir = outputs_dir / "Plots"
    tables_dir = outputs_dir / "Tables"

    print("=" * 70)
    print(" GENERAL MAIN ")
    print("=" * 70)

    # 1. Setup Directories
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SYSTEM] Output directories verified at: {outputs_dir}")

    # 2. Dataset Verification
    dataset_path = data_dir / DATASET_FILE
    if not verify_dataset_integrity(dataset_path, EXPECTED_SHA256):
        print("\n[HALT] Dataset integrity check failed. Please ensure the file is correctly")
        print("       downloaded from CC-MAIN-2024-10 and extracted in the DATA folder.")
        sys.exit(1)

    # 3. Execution of Base Benchmarks
    python_gil = root_dir / ".venv-gil" / ("Scripts" if os.name == 'nt' else "bin") / (
        "python.exe" if os.name == 'nt' else "python")
    python_nogil = root_dir / ".venv-nogil" / ("Scripts" if os.name == 'nt' else "bin") / (
        "python3.13t.exe" if os.name == 'nt' else "python3.13t")
    if not python_nogil.exists():
        python_nogil = root_dir / ".venv-nogil" / ("Scripts" if os.name == 'nt' else "bin") / (
            "python.exe" if os.name == 'nt' else "python")

    run_and_capture([str(python_gil), str(root_dir / "BENCHMARKS" / "SYNTHETIC_DATA" / "master_orchestrator.py")],
                    "Running Synthetic Orchestrator...", root_dir)

    run_and_capture([str(python_gil), str(root_dir / "BENCHMARKS" / "REAL_DATA" / "real_master_orchestrator.py")],
                    "Running Real Data Orchestrator...", root_dir)

    # 4. Execution of I/O Stress Tests

    io_scripts = {
        "Synth_GIL_IPC": root_dir / "I_O" / "main_IO_bench_GIL.py",
        "Synth_NOGIL": root_dir / "I_O" / "main_IO_bench_NO_GIL.py",
        "Real_GIL_IPC": root_dir / "I_O" / "Real_main_IO_bench_GIL.py",
        "Real_NOGIL": root_dir / "I_O" / "Real_main_IO_bench_NO_GIL.py"
    }

    io_results = {}
    for name, script_path in io_scripts.items():
        if script_path.exists():
            interpreter = str(python_nogil) if "NO_GIL" in name else str(python_gil)
            stdout = run_and_capture([interpreter, str(script_path)], f"Running I/O Test: {name}", root_dir)
            io_results[name] = parse_io_metrics(stdout)
        else:
            print(f"[WARNING] I/O Script not found: {script_path.name}. Skipping.")

    # 4.5 Execution of Scaling Tests (Amdahl's Law etc)
    print("\n[ORCHESTRATOR] Initiating Amdahl's Law Scaling Analysis...")
    script_scal = root_dir / "BENCHMARKS" / "bench_script.py"
    scal_results = {}

    if script_scal.exists():
        # Running the script using python_nogil
        run_and_capture([str(python_nogil), str(script_scal)], "Running scaling Benchmarks...", root_dir)


        scal_json = root_dir / "BENCHMARKS" / "telemetry_bench_script.json"
        if scal_json.exists():
            with open(scal_json, 'r') as f:
                scal_results = json.load(f)
            os.remove(scal_json)  # cleaning temporary file
    else:
        print(f"[WARNING] Amdahl Script not found: {script_scal.name}. Skipping.")

    # 5. Data Aggregation and Storage (For LaTeX/Word Tables)
    print("\n[DATA EXPORT] Aggregating telemetry for Tables...")

    with open(root_dir / "BENCHMARKS" / "SYNTHETIC_DATA" / "telemetry_gil.json", 'r') as f:
        synth_gil = json.load(f)
    with open(root_dir / "BENCHMARKS" / "SYNTHETIC_DATA" / "telemetry_nogil.json", 'r') as f:
        synth_nogil = json.load(f)

    with open(root_dir / "BENCHMARKS" / "REAL_DATA" / "telemetry_gil.json", 'r') as f:
        real_gil = json.load(f)
    with open(root_dir / "BENCHMARKS" / "REAL_DATA" / "telemetry_nogil.json", 'r') as f:
        real_nogil = json.load(f)

    synth_all = {**synth_gil, **synth_nogil}
    real_all = {**real_gil, **real_nogil}

    aggregated_tables = {
        "Synthetic_Workloads": synth_all,
        "Real_Workloads": real_all,
        "IO_Stress_Tests": io_results,
        "Scaling_Laws": scal_results
    }

    tables_file = tables_dir / "Aggregated_Benchmark_Tables.json"
    with open(tables_file, 'w') as f:
        json.dump(aggregated_tables, f, indent=4)
    print(f"[DATA EXPORT] Raw table data saved to {tables_file}")

    # 6. Plot Generation
    generate_plots(synth_all, real_all, io_results, plots_dir)


    if scal_results:
        plot_amdahl_scaling(scal_results, plots_dir)

    print("\n" + "=" * 70)
    print(" ORCHESTRATION COMPLETE. ALL DATA SECURED.")
    print("=" * 70)


if __name__ == "__main__":
    main()