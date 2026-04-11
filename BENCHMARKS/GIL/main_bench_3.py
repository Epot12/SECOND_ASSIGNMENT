import time
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from PARALLEL.ScalMultProcBloom import *
from SEQUENTIAL.ScalableBloomFilter import *
from PARALLEL.MultiProcBloom import *

def run_master_benchmark():

    initial_capacity = 100_000
    elements_to_insert = 3_000_000
    elements_to_test_missing = 1_000_000
    target_p_rate = 0.05

    print("==========================================================")
    print("   BENCHMARK: SEQUENTIAL VS PARALLEL PIPELINE    ")
    print("==========================================================")
    print(f"Global Target FP Rate: {target_p_rate * 100:.2f}%")
    print(f"Initial Capacity:      {initial_capacity}")
    print(f"Total Insertions:      {elements_to_insert}")
    print(f"Total Read Queries:    {elements_to_insert + elements_to_test_missing}")

    # data generation
    print("\n[PHASE 1] Synthetic Data Generation (Shared for both tests)")
    start_gen = time.perf_counter()
    present_items = [f"IN_{i}" for i in range(elements_to_insert)]
    absent_items = [f"OUT_{i}" for i in range(elements_to_test_missing)]
    gen_time = time.perf_counter() - start_gen
    print(f"Generated {len(present_items) + len(absent_items)} elements in {gen_time:.2f} s.")


    # SEQUENTIAL BASELINE

    print("\n" + "=" * 58)
    print(" TEST 1: SEQUENTIAL SCALABLE BLOOM FILTER (Baseline) ")
    print("=" * 58)

    seq_sbf = ScalableBloomFilter(initial_capacity, target_p_rate)

    print("[Sequential] Dynamic Insertion Phase...")
    start_seq_ins = time.perf_counter()
    for item in present_items:
        seq_sbf.add(item)
    seq_ins_time = time.perf_counter() - start_seq_ins
    print(f" -> Insertion Time: {seq_ins_time:.2f} s.")

    print("[Sequential] Read Operations (Verifying Ground Truth)...")
    start_seq_read = time.perf_counter()

    # False Negatives
    seq_fn = sum(1 for item in present_items if item not in seq_sbf)
    # False Positives
    seq_fp = sum(1 for item in absent_items if item in seq_sbf)

    seq_read_time = time.perf_counter() - start_seq_read
    seq_total_time = seq_ins_time + seq_read_time

    print(f" -> Read Time:      {seq_read_time:.2f} s.")
    print(f" -> TOTAL TIME:     {seq_total_time:.2f} s.")
    print(f" -> FP Rate:        {(seq_fp / len(absent_items)) * 100:.2f}%")


    # PARALLEL ARCHITECTURE

    print("\n" + "=" * 58)
    print(" TEST 2: PARALLEL SCALABLE BLOOM FILTER (Optimized) ")
    print("=" * 58)

    par_sbf = ParallelScalableBloomFilter(initial_capacity, target_p_rate)

    print("[Parallel] Dynamic Insertion Phase...")
    start_par_ins = time.perf_counter()
    par_sbf.add_batch(present_items)
    par_ins_time = time.perf_counter() - start_par_ins
    print(f" -> Insertion Time: {par_ins_time:.2f} s.")

    print("[Parallel] Read Operations (Verifying Ground Truth)...")
    start_par_read = time.perf_counter()

    # False Negatives
    fn_results = par_sbf.contains_batch(present_items)
    par_fn = fn_results.count(False)

    # False Positives
    fp_results = par_sbf.contains_batch(absent_items)
    par_fp = fp_results.count(True)

    par_read_time = time.perf_counter() - start_par_read
    par_total_time = par_ins_time + par_read_time

    print(f" -> Read Time:      {par_read_time:.2f} s.")
    print(f" -> TOTAL TIME:     {par_total_time:.2f} s.")
    print(f" -> FP Rate:        {(par_fp / len(absent_items)) * 100:.2f}%")

    # Error checking matematico
    assert seq_fn == 0 and par_fn == 0, "CRITICAL: False negatives detected!"


    # SPEEDUP

    print("\n" + "=" * 58)
    print(" ARCHITECTURAL VERDICT and SPEEDUP ANALYSIS ")
    print("=" * 58)

    ins_speedup = seq_ins_time / par_ins_time if par_ins_time > 0 else 0
    read_speedup = seq_read_time / par_read_time if par_read_time > 0 else 0
    total_speedup = seq_total_time / par_total_time if par_total_time > 0 else 0

    print(f"1. Insertion Speedup: {ins_speedup:.2f}x faster")
    print(f"2. Query Speedup:     {read_speedup:.2f}x faster")
    print(f"3. GLOBAL SPEEDUP:    {total_speedup:.2f}x faster")

    print("-" * 58)
    if total_speedup > 1:
        print(f"CONCLUSION: The Parallel architecture successfully bypassed the")
        print(f"Python GIL, processing {elements_to_insert} elements in {par_total_time:.2f}s")
        print(f"compared to the {seq_total_time:.2f}s of the sequential baseline.")
    else:
        print("CONCLUSION: Overhead exceeded parallelization benefits on this hardware.")


if __name__ == "__main__":
    mp.freeze_support()
    run_master_benchmark()