import os
import sys
import time
import json
import gc
import argparse
from typing import Dict, Any
from utils.stats_engine import *


# SETUP PATHS and IMPORTS

current_dir = os.path.dirname(os.path.abspath(__file__))
multithread_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(multithread_dir)

sys.path.insert(0, root_dir)

from utils.utilities import load_data

# Importing 3 architectural variables
from MULTITHREAD.stripe_strategy.StripedBloomFilter import StripedBloomFilter as StripedBloomBase
from MULTITHREAD.stripe_strategy.StripedBloomFilterColMajor import StripedBloomFilterColMajor as StripedBloomColMajor
from MULTITHREAD.stripe_strategy.StripedBloomFilterSoA_opt import StripedBloomFilterSoA as StripedBloomSoA_opt


# CONSTANTS and SCIENTIFIC PARAMETERS

WARMUP_RUNS = 2
MEASURE_RUNS = 5
TOTAL_RUNS = WARMUP_RUNS + MEASURE_RUNS


def run_benchmarks(mode: str):
    print("=" * 110)
    print(f"BENCHMARKING ORCHESTRATOR (Mode: {mode.upper()})")
    print(f"   Config: {WARMUP_RUNS} Warmup Runs, {MEASURE_RUNS} Measurement Runs")
    print("=" * 110)

    # parameters
    cap, ins, tst, fpr = 100_000, 3_000_000, 500_000, 0.05
    present_items, absent_items = load_data(mode, ins, tst)

    telemetry_data: Dict[str, Any] = {}

    architectures = [
        ("Base_AoS_RowMajor", StripedBloomBase),
        ("Test_ColMajor_Query", StripedBloomColMajor)
        #("Test_SoA_Insert_opt", StripedBloomSoA_opt)
    ]

    for name, ArchClass in architectures:
        print(f"\n Testing Architecture: [{name}]")

        # Raw data arrays
        ins_wall_raw, read_wall_raw = [], []
        ins_cpu_raw, read_cpu_raw = [], []

        for run_idx in range(TOTAL_RUNS):
            is_warmup = run_idx < WARMUP_RUNS
            run_type = "Warmup" if is_warmup else f"Measure {run_idx - WARMUP_RUNS + 1}/{MEASURE_RUNS}"
            print(f"  {run_type}...", end="", flush=True)

            with ArchClass(cap, fpr) as s:

                gc.disable()

                t0_wall, t0_cpu = time.perf_counter(), time.process_time()
                s.add_batch(present_items)
                t1_wall, t1_cpu = time.perf_counter(), time.process_time()

                s.contains_batch(present_items)
                s.contains_batch(absent_items)
                t2_wall, t2_cpu = time.perf_counter(), time.process_time()

                gc.enable()

                if not is_warmup:
                    ins_wall_raw.append(t1_wall - t0_wall)
                    read_wall_raw.append(t2_wall - t1_wall)
                    ins_cpu_raw.append(t1_cpu - t0_cpu)
                    read_cpu_raw.append(t2_cpu - t1_cpu)
                    print(f" Done (Ins Wall: {t1_wall - t0_wall:.2f}s | CPU: {t1_cpu - t0_cpu:.2f}s)")
                else:
                    print(" Done (Ignored for stats)")

        # computing statistics
        telemetry_data[name] = {
            "metrics": {
                "insert_wall": compute_statistics(ins_wall_raw),
                "query_wall": compute_statistics(read_wall_raw),
                "insert_cpu": compute_statistics(ins_cpu_raw),
                "query_cpu": compute_statistics(read_cpu_raw)
            },
            # saving raw data
            "raw_data": {
                "insert_wall": ins_wall_raw,
                "query_wall": read_wall_raw,
                "insert_cpu": ins_cpu_raw,
                "query_cpu": read_cpu_raw
            }
        }

    # exporting JSON and printing report
    out_path = os.path.join(current_dir, f'telemetry_patterns_{mode}.json')
    with open(out_path, 'w') as f:
        json.dump(telemetry_data, f, indent=4)

    print("\n" + "=" * 115)
    print(f" SUMMARY REPORT ({mode.upper()})")
    print("=" * 115)
    print(
        f"{'Architecture':<25} | {'Wall Ins (Mean ± Std)':<20} | {'Wall Qry (Mean ± Std)':<20} | {'CPU Ins (Mean ± Std)':<20} | {'CPU Qry (Mean ± Std)':<20}")
    print("-" * 115)

    for name, data in telemetry_data.items():
        w_ins = data['metrics']['insert_wall']
        w_qry = data['metrics']['query_wall']
        c_ins = data['metrics']['insert_cpu']
        c_qry = data['metrics']['query_cpu']

        w_ins_str = f"{w_ins['mean']:.2f}s ± {w_ins['ci_95_margin']:.2f}"
        w_qry_str = f"{w_qry['mean']:.2f}s ± {w_qry['ci_95_margin']:.2f}"
        c_ins_str = f"{c_ins['mean']:.2f}s ± {c_ins['ci_95_margin']:.2f}"
        c_qry_str = f"{c_qry['mean']:.2f}s ± {c_qry['ci_95_margin']:.2f}"

        print(f"{name:<25} | {w_ins_str:<20} | {w_qry_str:<20} | {c_ins_str:<20} | {c_qry_str:<20}")

    print(f"\n[INFO] Detailed scientific telemetry saved to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scientific Micro-Benchmark for Memory Patterns")
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True)
    args = parser.parse_args()

    run_benchmarks(args.mode)