import os
import sys
import cProfile
import pstats
import time

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


def profile_function(func, name):
    """Runs cProfile on the passed function and saves the results to a .pstat file"""
    profiler = cProfile.Profile()

    start_time = time.perf_counter()
    profiler.enable()

    func()

    profiler.disable()
    end_time = time.perf_counter()

    # Saves .pstat file
    stats_file = f"{name}_profiler.pstat"
    with open(stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('tottime').print_stats(20)

    print(f"[COMPLETED] {name} in {end_time - start_time:.2f}s. Stats saved to {stats_file}")


if __name__ == "__main__":
    print("=" * 60)
    print(" BLOOM FILTER PROFILING SHOWCASE")
    print("=" * 60)

    # Execution and Profiling with cProfile (Macro-Profiling)
    profile_function(run_sequential, "1_Sequential")
    profile_function(run_multiprocessing, "2_Multiprocessing")
    profile_function(run_multithreading, "3_Multithreading_SoA")

    print("\n[SUCCESS] cProfile analysis complete. Check the .pstat files.")