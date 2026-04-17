import sys
import os
import json
import multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Importing engine functions
from utils.scaling_engine import run_strong_scaling, run_weak_scaling, run_chunk_optimization

from MULTITHREAD.MapReduceScalBloom import ThreadedScalableBloomFilter as MapReduceNoGIL
from MULTITHREAD.ThreadedScalBloomFilter import ThreadedScalableBloomFilter as MultiThreading
from MULTITHREAD.stripe_strategy.StripedBloomFilter import StripedBloomFilter
from MULTITHREAD.stripe_strategy.StripedBloomFilterColMajor import StripedBloomFilterColMajor as StripedBloomColMajor
from MULTITHREAD.stripe_strategy.StripedBloomFilterSoA import StripedBloomFilterSoA as StripedBloomSoA
from MULTITHREAD.stripe_strategy.StripedBloomFilterSoA_opt import StripedBloomFilterSoA as StripedBloomSoA_opt
from PARALLEL.PermPoolBloom import PermPoolScalableBloomFilter as SotaIPC
from PARALLEL.ScalJobLib import JoblibScalableBloomFilter as JoblibBloom
from PARALLEL.ScalMultProcBloomOpt import ParallelScalableBloomFilter as ScalParBloomOpt

def run():
    print("\n[SCALING WORKER] Starting Full Scaling and Profiling Benchmarks...")

    max_cores = 8  # logical cores
    fixed_size = 3_000_000
    items_per_worker = 500_000 # For Gustafson
    chunk_sizes = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000] # For Chunk Profiling

    # Dictionary structured to contain the 3 experiments
    final_telemetry = {
        "Amdahl": {},
        "Gustafson": {},
        "Granularity": {}
    }


    # 1. AMDAHL'S LAW (Strong Scaling)

    print("\n--- Running Amdahl's Law ---")
    final_telemetry["Amdahl"]["Amdahl_IPC"] = run_strong_scaling(SotaIPC, max_cores, fixed_size)
    final_telemetry["Amdahl"]["Amdahl_Joblib"] = run_strong_scaling(JoblibBloom, max_cores, fixed_size)
    final_telemetry["Amdahl"]["Amdahl_ScalParOpt"] = run_strong_scaling(ScalParBloomOpt, max_cores, fixed_size)
    final_telemetry["Amdahl"]["Amdahl_NoGIL_Map_Red"] = run_strong_scaling(MapReduceNoGIL, max_cores, fixed_size)
    final_telemetry["Amdahl"]["Amdahl_NoGIL_Mul_Thr"] = run_strong_scaling(MultiThreading, max_cores, fixed_size)
    final_telemetry["Amdahl"]["AmdahlStripedBloom"] = run_strong_scaling(StripedBloomFilter, max_cores, fixed_size)
    final_telemetry["Amdahl"]["AmdahlStripedColMajor"] = run_strong_scaling(StripedBloomColMajor, max_cores, fixed_size)
    final_telemetry["Amdahl"]["AmdahlStripedFilterSoA"] = run_strong_scaling(StripedBloomSoA, max_cores, fixed_size)
    final_telemetry["Amdahl"]["AmdahlStripedFilterSoAOpt"] = run_strong_scaling(StripedBloomSoA_opt, max_cores, fixed_size)

    # 2. GUSTAFSON'S LAW (Weak Scaling)

    print("\n--- Running Gustafson's Law ---")
    final_telemetry["Gustafson"]["Gustafson_NoGIL_Map_Red"] = run_weak_scaling(MapReduceNoGIL, max_cores, items_per_worker)
    final_telemetry["Gustafson"]["Gustafson_NoGIL_Mul_Thr"] = run_weak_scaling(MultiThreading, max_cores, items_per_worker)
    final_telemetry["Gustafson"]["Gustafson_IPC"] = run_weak_scaling(SotaIPC, max_cores, items_per_worker)


    # 3. GRANULARITY PROFILING (Chunk Size)

    print("\n--- Running Granularity Profiling ---")
    # testing the granularity only on the best architecture (Map-Reduce NoGIL) using the maximum physical cores (4)
    final_telemetry["Granularity"] = run_chunk_optimization(MapReduceNoGIL, 4, fixed_size, chunk_sizes)

    # saving JSON
    out_path = os.path.join(current_dir, 'telemetry_bench_script.json')
    with open(out_path, 'w') as f:
        json.dump(final_telemetry, f, indent=4)

    print(f"\n[SCALING WORKER] All scaling telemetry saved to {out_path}")

if __name__ == "__main__":
    mp.freeze_support()
    run()