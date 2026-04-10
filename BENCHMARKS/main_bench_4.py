import os, sys
import time
import multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from SEQUENTIAL.ScalableBloomFilter import *

from PARALLEL.ScalMultProcBloom import *

from PARALLEL.PermPoolBloom import PermPoolScalableBloomFilter


def run_benchmark():
    # BENCHMARK PARAMETERS
    initial_capacity = 100_000
    elements_to_insert = 3_000_000
    elements_to_test_missing = 500_000
    target_p_rate = 0.05

    print("==========================================================")
    print("   BENCHMARK: EVOLUTION OF SCALABLE BLOOM FILTERS")
    print("==========================================================")

    print("\n[PHASE 0] Synthetic Data Generation (Shared Memory Space)...")
    start_gen = time.perf_counter()
    present_items = [f"IN_{i}" for i in range(elements_to_insert)]
    absent_items = [f"OUT_{i}" for i in range(elements_to_test_missing)]
    print(f"Generated {len(present_items) + len(absent_items)} elements in {time.perf_counter() - start_gen:.2f} s.")


    # TEST 1: SEQUENTIAL BASELINE

    print("\n" + "=" * 58)
    print(" TEST 1: SEQUENTIAL ARCHITECTURE (Baseline) ")
    print("=" * 58)

    seq_sbf = ScalableBloomFilter(initial_capacity, target_p_rate)

    start_seq_ins = time.perf_counter()
    for item in present_items:
        seq_sbf.add(item)
    seq_ins_time = time.perf_counter() - start_seq_ins

    start_seq_read = time.perf_counter()
    seq_fn = sum(1 for item in present_items if item not in seq_sbf)
    seq_fp = sum(1 for item in absent_items if item in seq_sbf)
    seq_read_time = time.perf_counter() - start_seq_read

    print(f" -> Insertion Time: {seq_ins_time:.2f} s.")
    print(f" -> Read Time:      {seq_read_time:.2f} s.")
    print(f" -> FP Rate:        {(seq_fp / len(absent_items)) * 100:.2f}%")


    # TEST 2: PARALLEL ARCHITECTURE (Restart Pool)

    print("\n" + "=" * 58)
    print(" TEST 2: PARALLEL ARCHITECTURE (Restart Pool) ")
    print("=" * 58)

    par1_sbf = ParallelScalableBloomFilter(initial_capacity, target_p_rate)

    start_par1_ins = time.perf_counter()
    par1_sbf.add_batch(present_items)
    par1_ins_time = time.perf_counter() - start_par1_ins

    start_par1_read = time.perf_counter()
    par1_fn = par1_sbf.contains_batch(present_items).count(False)
    par1_fp = par1_sbf.contains_batch(absent_items).count(True)
    par1_read_time = time.perf_counter() - start_par1_read

    print(f" -> Insertion Time: {par1_ins_time:.2f} s.")
    print(f" -> Read Time:      {par1_read_time:.2f} s.")
    print(f" -> FP Rate:        {(par1_fp / len(absent_items)) * 100:.2f}%")


    # TEST 3: PARALLEL ARCHITECTURE (Persistent Pool)

    print("\n" + "=" * 58)
    print(" TEST 3: PARALLEL ARCHITECTURE (Persistent Pool) ")
    print("=" * 58)

    with PermPoolScalableBloomFilter(initial_capacity, target_p_rate) as sota_sbf:
        start_sota_ins = time.perf_counter()
        sota_sbf.add_batch(present_items)
        sota_ins_time = time.perf_counter() - start_sota_ins

        start_sota_read = time.perf_counter()
        sota_fn = sota_sbf.contains_batch(present_items).count(False)
        sota_fp = sota_sbf.contains_batch(absent_items).count(True)
        sota_read_time = time.perf_counter() - start_sota_read

        print(f" -> Insertion Time: {sota_ins_time:.2f} s.")
        print(f" -> Read Time:      {sota_read_time:.2f} s.")
        print(f" -> FP Rate:        {(sota_fp / len(absent_items)) * 100:.2f}%")


    # GROUND TRUTH VERIFICATION

    # Mathematical safety assertions to guarantee algorithm integrity
    assert seq_fn == 0 and par1_fn == 0 and sota_fn == 0, "CRITICAL ERROR: False Negatives detected!"


    # SPEEDUP ANALYSIS


    # Total Execution Time Calculation
    seq_tot = seq_ins_time + seq_read_time
    par1_tot = par1_ins_time + par1_read_time
    sota_tot = sota_ins_time + sota_read_time

    # Speedup Calculation relative to the Sequential Baseline
    par1_speedup = seq_tot / par1_tot if par1_tot > 0 else 0
    sota_speedup = seq_tot / sota_tot if sota_tot > 0 else 0

    # Micro-Architectural Optimization Calculation (SOTA vs Pool Restart)
    sota_vs_par1_ins = par1_ins_time / sota_ins_time if sota_ins_time > 0 else 0

    print("\n" + "=" * 58)
    print(" FINAL ARCHITECTURAL VERDICT and SPEEDUP ANALYSIS ")
    print("=" * 58)

    print(f"TOTAL EXECUTION TIMES:")
    print(f" - Sequential Baseline:    {seq_tot:.2f} s")
    print(f" - Parallel (Lazy Restart):{par1_tot:.2f} s")
    print(f" - SOTA (Persistent Pool): {sota_tot:.2f} s\n")

    print(f"GLOBAL SPEEDUP (vs Sequential):")
    print(f" - Parallel (Lazy Restart): {par1_speedup:.2f}x faster")
    print(f" - SOTA (Persistent Pool):  {sota_speedup:.2f}x faster\n")

    print(f"MICRO-ARCHITECTURAL GAIN (SOTA vs Lazy Restart):")
    print(f" - Insertion phase is {sota_vs_par1_ins:.2f}x faster due to zero-overhead memory allocation.")
    print("-" * 58)


if __name__ == "__main__":
    # Standard multiprocessing safeguard for Windows environments
    mp.freeze_support()
    run_benchmark()