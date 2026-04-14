import sys
import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# importing scaling engine and filters
from utils.scaling_engine import run_strong_scaling

from MULTITHREAD.MapReduceScalBloom import ThreadedScalableBloomFilter as MapReduceNoGIL
from PARALLEL.PermPoolBloom import PermPoolScalableBloomFilter as SotaIPC
from MULTITHREAD.ThreadedScalBloomFilter import ThreadedScalableBloomFilter as MultiThreading

def run():
    print("\n[AMDAHL WORKER] Starting Strong Scaling Benchmarks...")

    max_cores = 8  # logical cores
    fixed_size = 3_000_000

    results = {}

    # 1. Test No-GIL Map Reduce
    print("Testing No-GIL Map Reduce Architecture...")
    results["Amdahl_NoGIL_Map_Red"] = run_strong_scaling(MapReduceNoGIL, max_cores, fixed_size)

    # 2. Test No-GIL Multi Threading without Map Reduce
    print("Testing No-GIL Multi Threading Architecture...")
    results["Amdahl_NoGIL_Mul_Thr"] = run_strong_scaling(MultiThreading, max_cores, fixed_size)

    # 3. Test IPC Multiprocessing
    print("Testing IPC Architecture...")
    results["Amdahl_IPC"] = run_strong_scaling(SotaIPC, max_cores, fixed_size)

    # saving JSON file for Grand Master
    out_path = os.path.join(current_dir, 'telemetry_bench_script.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"[AMDAHL WORKER] Telemetry saved to {out_path}")


if __name__ == "__main__":
    run()