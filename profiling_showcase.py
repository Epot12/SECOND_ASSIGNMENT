import os
import sys
import cProfile
import pstats
import time
import subprocess
from pathlib import Path

# paths setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Imports
from SEQUENTIAL.ScalableBloomFilter import ScalableBloomFilter as Sequential
from PARALLEL.PermPoolBloom import PermPoolScalableBloomFilter as MultiprocessingIPC
from MULTITHREAD.stripe_strategy.StripedBloomFilterSoA_opt import StripedBloomFilterSoA as MultithreadingSoA

# HACK FOR LINE PROFILER
import builtins

if 'profile' not in builtins.__dict__:
    def profile(func): return func


    builtins.profile = profile


# Parameters
ITEMS_COUNT = 300_000
DATASET = [f"profiling_item_{i}" for i in range(ITEMS_COUNT)]
INIT_CAP = 100_000
FPR = 0.01


def run_sequential():
    print("\n--- Running Sequential ---")
    bf = Sequential(initial_capacity=INIT_CAP, target_fp_rate=FPR)
    for item in DATASET:
        bf.add(item)


def run_multiprocessing():
    print("\n--- Running Multiprocessing (IPC Shared Memory) ---")
    with MultiprocessingIPC(initial_capacity=INIT_CAP, target_fp_rate=FPR, num_processes=4) as bf:
        bf.add_batch(DATASET)


def run_multithreading():
    print("\n--- Running Multithreading (SoA SIMD No-GIL) ---")
    with MultithreadingSoA(initial_capacity=INIT_CAP, target_fp_rate=FPR, num_threads=4) as bf:
        bf.add_batch(DATASET)


def profile_function(func, name, use_cprofile=True):
    """
    Executes the function. Use cProfile if use_cprofile=True,
    otherwise it does a pure time benchmark
    """
    start_time = time.perf_counter()

    if use_cprofile:
        profiler = cProfile.Profile()
        profiler.enable()

        func()

        profiler.disable()
        end_time = time.perf_counter()

        # saving results
        stats_file = f"{name}_profiler.pstat"
        with open(stats_file, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('tottime').print_stats(20)

        print(f"[COMPLETED] {name} in {end_time - start_time:.2f}s. Stats saved to {stats_file}")

    else:
        # execution without profiler
        func()
        end_time = time.perf_counter()
        print(f"[COMPLETED] {name} in {end_time - start_time:.2f}s. (Pure Wall-Clock Time, No Profiler overhead)")


if __name__ == "__main__":
    # WORKER MODE
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "sequential":
            profile_function(run_sequential, "1_Sequential", use_cprofile=True)
        elif mode == "multiprocessing":
            profile_function(run_multiprocessing, "2_Multiprocessing", use_cprofile=True)
        elif mode == "multithreading":
            profile_function(run_multithreading, "3_Multithreading_SoA", use_cprofile=False)
        sys.exit(0)

    # MASTER MODE
    print("=" * 60)
    print(" BLOOM FILTER PROFILING SHOWCASE (AUTOMATED ORCHESTRATOR)")
    print("=" * 60)

    root_dir = Path(current_dir)

    # 1. Resolving Interpreters
    is_windows = os.name == 'nt'
    bin_dir = "Scripts" if is_windows else "bin"
    exe = ".exe" if is_windows else ""

    python_gil = root_dir / ".venv-gil" / bin_dir / f"python{exe}"
    python_nogil_t = root_dir / ".venv-nogil" / bin_dir / f"python3.13t{exe}"
    python_nogil_std = root_dir / ".venv-nogil" / bin_dir / f"python{exe}"
    python_nogil = python_nogil_t if python_nogil_t.exists() else python_nogil_std

    # check that environments exist
    if not python_gil.exists() or not python_nogil.exists():
        print("[CRITICAL] Python interpreters could not be found. Make sure venvs exist.")
        sys.exit(1)

    this_script = str(Path(__file__).resolve())

    # 2. launching the three profiles by delegating to the respective interpreters
    print("\n[ORCHESTRATOR] Starting Sequential Profiling (GIL)...")
    subprocess.run([str(python_gil), this_script, "sequential"], check=True)

    print("\n[ORCHESTRATOR] Starting Multiprocessing Profiling (GIL)...")
    subprocess.run([str(python_gil), this_script, "multiprocessing"], check=True)

    print("\n[ORCHESTRATOR] Starting Multithreading Profiling (NO-GIL)...")
    subprocess.run([str(python_nogil), this_script, "multithreading"], check=True)

    print("\n" + "=" * 60)
    print("[SUCCESS] cProfile analysis complete. Check the .pstat files.")
    print("=" * 60)