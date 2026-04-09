import time
from ScalableBloomFilter import ScalableBloomFilter


def run_scalable_test():
    # Setup calibrated to execute in approximately 10 seconds on modern hardware
    initial_capacity = 100_000
    elements_to_insert = 1_500_000  # 1.5 million elements to adequately stress the dynamic expansion
    elements_to_test_missing = 500_000  # Half a million to accurately measure the False Positive rate
    target_p_rate = 0.05

    print("=== STRESS TEST: SCALABLE BLOOM FILTER ===")
    print(f"Global Target FP Rate: {target_p_rate * 100:.2f}%")
    print(f"Initial Capacity (Layer 0): {initial_capacity}")
    print(f"Elements to Insert: {elements_to_insert}")

    # Initialize the composite (nested) scalable filter
    sbf = ScalableBloomFilter(initial_capacity, target_p_rate)

    print("\n[1/3] Synthetic Data Generation")
    start_gen = time.time()
    # Utilizing short strings to optimize memory allocation and processing speed
    present_items = [f"IN_{i}" for i in range(elements_to_insert)]
    absent_items = [f"OUT_{i}" for i in range(elements_to_test_missing)]
    gen_time = time.time() - start_gen
    print(f"Generated {elements_to_insert + elements_to_test_missing} elements in {gen_time:.2f} s.")

    print("\n[2/3] Dynamic Insertion Phase")
    start_insert = time.time()
    for item in present_items:
        sbf.add(item)
    insertion_time = time.time() - start_insert
    print(f"Insertion completed in {insertion_time:.2f} s.")
    print(f"Total number of generated layers: {len(sbf.filters)}")
    print(f"Total elements successfully stored: {sbf.total_elements_count()}")

    print("\n[3/3] Ground Truth Verification (Read Operations and Statistics)")
    start_test = time.time()

    # False Negative Testing (conducted on a 200k sample to balance execution time)
    sample_present = present_items[:200_000]
    false_negatives = sum(1 for item in sample_present if item not in sbf)
    assert false_negatives == 0, f"CRITICAL ERROR: Detected {false_negatives} false negatives!"
    print(f"False Negatives (out of {len(sample_present)} tested): 0 (Test Passed)")

    # False Positive Testing (conducted across all 500,000 absent elements)
    false_positives = sum(1 for item in absent_items if item in sbf)
    empirical_fp_rate = false_positives / len(absent_items)
    test_time = time.time() - start_test

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
    run_scalable_test()