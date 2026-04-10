import time
import multiprocessing as mp
# Ensure correct imports based on your file structure
from BloomFilter import ScalableBloomFilter
from IndustrialScalableBloomFilter import IndustrialScalableBloomFilter


def run_industrial_benchmark():
    initial_capacity = 100_000
    elements_to_insert = 1_500_000
    elements_to_test_missing = 500_000
    target_p_rate = 0.05

    print("==========================================================")
    print("   SOTA BENCHMARK: SEQUENTIAL VS INDUSTRIAL PIPELINE      ")
    print("==========================================================")

    print("\n[PHASE 1] Synthetic Data Generation...")
    start_gen = time.perf_counter()
    present_items = [f"IN_{i}" for i in range(elements_to_insert)]
    absent_items = [f"OUT_{i}" for i in range(elements_to_test_missing)]
    print(f"Generated {len(present_items) + len(absent_items)} elements in {time.perf_counter() - start_gen:.2f} s.")

    # ==========================================================
    # TEST 1: SEQUENTIAL BASELINE
    # ==========================================================
    print("\n" + "=" * 58)
    print(" TEST 1: SEQUENTIAL SCALABLE BLOOM FILTER ")
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

    # ==========================================================
    # TEST 2: INDUSTRIAL ARCHITECTURE (SOTA April 2026)
    # ==========================================================
    print("\n" + "=" * 58)
    print(" TEST 2: INDUSTRIAL SCALABLE BLOOM FILTER ")
    print("=" * 58)

    # CRITICAL: Using the 'with' context manager to guarantee RAM Cleanup
    with IndustrialScalableBloomFilter(initial_capacity, target_p_rate) as par_sbf:
        start_par_ins = time.perf_counter()
        par_sbf.add_batch(present_items)
        par_ins_time = time.perf_counter() - start_par_ins

        start_par_read = time.perf_counter()
        par_fn = par_sbf.contains_batch(present_items).count(False)
        par_fp = par_sbf.contains_batch(absent_items).count(True)
        par_read_time = time.perf_counter() - start_par_read

        print(f" -> Insertion Time: {par_ins_time:.2f} s.")
        print(f" -> Read Time:      {par_read_time:.2f} s.")

    # ==========================================================
    # ARCHITECTURAL VERDICT
    # ==========================================================
    ins_speedup = seq_ins_time / par_ins_time if par_ins_time > 0 else 0
    read_speedup = seq_read_time / par_read_time if par_read_time > 0 else 0
    total_speedup = (seq_ins_time + seq_read_time) / (par_ins_time + par_read_time)

    print("\n" + "=" * 58)
    print(" ARCHITECTURAL VERDICT & SPEEDUP ANALYSIS ")
    print("=" * 58)
    print(f"1. Insertion Speedup: {ins_speedup:.2f}x faster")
    print(f"2. Query Speedup:     {read_speedup:.2f}x faster")
    print(f"3. GLOBAL SPEEDUP:    {total_speedup:.2f}x faster")
    print("-" * 58)


if __name__ == "__main__":
    mp.freeze_support()
    run_industrial_benchmark()