import json
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utilities import *
from utils.plot_functions import *

# GENERAL MAIN: THE GRAND ORCHESTRATOR
#this file is the entry point of the entire project


print("[SYSTEM] Forcing deterministic behavior (PYTHONHASHSEED=0)...")
os.environ["PYTHONHASHSEED"] = "0"

DATASET_FILE = "cdx-00000"
EXPECTED_SHA256 = "3D9F2F2BAEFF3DB20262B6E5580A8BA34CECBD3742D0C898B484D5ACF5C476B1"

# Graphics settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['pdf.fonttype'] = 42  # Ensures fonts are embedded in the PDF


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
    root_dir = Path(__file__).parent.resolve()
    data_dir = root_dir / "DATA"
    outputs_dir = root_dir / "Outputs"
    plots_dir = outputs_dir / "Plots"
    tables_dir = outputs_dir / "Tables"
    bench_dir = root_dir / "BENCHMARKS"
    io_dir = root_dir / "I_O"

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

    # 3. Virtual Environments Setup
    print("\n[SYSTEM] Verifying Virtual Environments...")
    path_gil_venv = root_dir / ".venv-gil"
    if not path_gil_venv.exists():
        print("[SYSTEM] Creating GIL Environment...")
        subprocess.run(["uv", "venv", ".venv-gil", "--python", "3.12"], cwd=root_dir, check=True)
        subprocess.run(["uv", "pip", "install", "--python", ".venv-gil", "mmh3", "joblib", "loky", "bitarray"], cwd=root_dir,
                       check=True)

    path_nogil_venv = root_dir / ".venv-nogil"
    if not path_nogil_venv.exists():
        print("[SYSTEM] Creating No-GIL Environment...")
        subprocess.run(["uv", "venv", ".venv-nogil", "--python", "3.13t"], cwd=root_dir, check=True)
        subprocess.run(["uv", "pip", "install", "--python", ".venv-nogil", "mmh3", "joblib", "loky", "bitarray", "numpy"],
                       cwd=root_dir, check=True)

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
    master_orch = bench_dir / "master_orchestrator.py"
    if master_orch.exists():
        run_sub_process([str(python_gil), str(master_orch)], "DELEGATING TO MASTER ORCHESTRATOR", root_dir)
    else:
        print(f"[CRITICAL] Master Orchestrator not found at {master_orch}")
        sys.exit(1)

    # B. I/O Stress Tests
    io_orch = io_dir / "master_orchestrator_IO.py"
    if io_orch.exists():
        run_sub_process([str(python_gil), str(io_orch)], "DELEGATING TO I/O ORCHESTRATOR", root_dir)
    else:
        print(f"[CRITICAL] I/O Orchestrator not found at {io_orch}")
        sys.exit(1)

    # C. Scaling Tests (Amdahl's Law)
    script_scal = bench_dir / "bench_script.py"
    if script_scal.exists():
        run_sub_process([str(python_nogil), str(script_scal)], "RUNNING SCALING LAWS ANALYSIS", root_dir)
    else:
        print(f"[WARNING] Scaling Script not found at {script_scal}. Skipping.")

    # PHASE 2: DATA AGGREGATION

    print("\n[DATA EXPORT] Aggregating telemetry from all subsystems...")

    def load_json_safe(path: Path) -> dict:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    # Load Base Benchmarks
    synth_gil = load_json_safe(bench_dir / "telemetry_gil_synthetic.json")
    synth_nogil = load_json_safe(bench_dir / "telemetry_nogil_synthetic.json")
    real_gil = load_json_safe(bench_dir / "telemetry_gil_real.json")
    real_nogil = load_json_safe(bench_dir / "telemetry_nogil_real.json")

    # Load I/O and Scaling
    io_results = load_json_safe(io_dir / "telemetry_io.json")
    scal_results = load_json_safe(bench_dir / "telemetry_bench_script.json")

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
    print(f"[DATA EXPORT] Aggregated Master Table saved to: {tables_file.name}")


    # PHASE 3: GARBAGE COLLECTION (CLEANUP)

    print("\n[CLEANUP] Removing intermediate JSON telemetry files...")

    temp_files = [
        bench_dir / "telemetry_gil_synthetic.json",
        bench_dir / "telemetry_nogil_synthetic.json",
        bench_dir / "telemetry_gil_real.json",
        bench_dir / "telemetry_nogil_real.json",
        bench_dir / "telemetry_bench_script.json",
        io_dir / "telemetry_io.json"
    ]

    cleanup_count = 0
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()
            cleanup_count += 1

    print(f"[CLEANUP] Workspace is clean. Removed {cleanup_count} temporary files.")


    # PHASE 4: PLOT GENERATION

    generate_plots(synth_all, real_all, io_results, plots_dir)

    if scal_results:
        if "Amdahl" in scal_results:
            plot_amdahl_scaling(scal_results["Amdahl"], plots_dir)
        if "Gustafson" in scal_results:
            plot_gustafson_scaling(scal_results["Gustafson"], plots_dir)
        if "Granularity" in scal_results:
            plot_chunk_optimization(scal_results["Granularity"], plots_dir)

    print("\n" + "=" * 80)
    print(" ORCHESTRATION COMPLETE. ALL DATA SECURED. PDFS GENERATED.")
    print("=" * 80)


if __name__ == "__main__":
    main()