import time
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from PARALLEL.ScalMultProcBloom import *


def run_parallel_scalable_test():

    # Calibrated to execute in approximately 10 seconds on modern hardware
    initial_capacity = 100_000
    elements_to_insert = 1_500_000
    elements_to_test_missing = 500_000
    target_p_rate = 0.05

    print("=== STRESS TEST: PARALLEL SCALABLE BLOOM FILTER ===")
    print(f"Global Target FP Rate: {target_p_rate * 100:.2f}%")
    print(f"Initial Capacity (Layer 0): {initial_capacity}")
    print(f"Elements to Insert: {elements_to_insert}")

    # Initialization of the Parallel version
    # (Upon initialization, the hardware calibration of the Auto-Tuner will be triggered)
    sbf = ParallelScalableBloomFilter(initial_capacity, target_p_rate)

    print("\n[1/3] Synthetic Data Generation")
    start_gen = time.perf_counter()
    # Utilizing short strings to optimize memory allocation and processing speed
    present_items = [f"IN_{i}" for i in range(elements_to_insert)]
    absent_items = [f"OUT_{i}" for i in range(elements_to_test_missing)]
    gen_time = time.perf_counter() - start_gen
    print(f"Generated {elements_to_insert + elements_to_test_missing} elements in {gen_time:.2f} s.")

    print("\n[2/3] Dynamic Insertion Phase")
    start_insert = time.perf_counter()

    # ARCHITECTURAL DIFFERENCE: We pass the entire list in a single batch.
    # The Master process handles partitioning and delegating chunks to the Workers.
    sbf.add_batch(present_items)

    insertion_time = time.perf_counter() - start_insert
    print(f"Insertion completed in {insertion_time:.2f} s.")
    # Using 'bitmaps' instead of 'filters' to reflect the shared memory architecture
    print(f"Total number of generated layers: {len(sbf.bitmaps)}")
    print(f"Total elements successfully stored: {sbf.total_elements_count()}")

    print("\n[3/3] Ground Truth Verification (Read Operations and Statistics)")
    start_test = time.perf_counter()

    # False Negative Testing
    sample_present = present_items

    # Utilizing contains_batch, which returns a list of booleans
    fn_results = sbf.contains_batch(sample_present)
    # Count the occurrences of 'False' to detect any false negatives
    false_negatives = fn_results.count(False)

    assert false_negatives == 0, f"CRITICAL ERROR: Detected {false_negatives} false negatives!"
    print(f"False Negatives (out of {len(sample_present)} tested): 0 (Test Passed)")

    # False Positive Testing
    fp_results = sbf.contains_batch(absent_items)
    # Count the occurrences of 'True' to calculate the empirical false positive rate
    false_positives = fp_results.count(True)

    empirical_fp_rate = false_positives / len(absent_items)
    test_time = time.perf_counter() - start_test

    total_time = gen_time + insertion_time + test_time

    print("\n=== FINAL RESULTS ===")
    print(f"Testing Time (Read operations): {test_time:.2f} s.")
    print(f"TOTAL Execution Time:           {total_time:.2f} s.")
    print("-" * 40)
    print(f"Detected False Positives:       {false_positives} out of {len(absent_items)}")
    print(f"Target Theoretical FP Rate:     {target_p_rate:.4f} ({target_p_rate * 100:.2f}%)")
    print(f"Measured Empirical FP Rate:     {empirical_fp_rate:.4f} ({empirical_fp_rate * 100:.2f}%)")

    # Verifying if the algorithm maintained its mathematical guarantee
    if empirical_fp_rate <= target_p_rate or abs(empirical_fp_rate - target_p_rate) < 0.005:
        print("\n SUCCESS: The Scalable Bloom Filter successfully bounded the error rate across millions of entries.")
    else:
        print("\n ANOMALY: The empirical error rate significantly deviates from theoretical expectations.")


if __name__ == "__main__":
    # Standard multiprocessing safeguard for Windows environments
    mp.freeze_support()
    run_parallel_scalable_test()