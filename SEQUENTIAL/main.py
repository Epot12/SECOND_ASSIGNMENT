import time
from SEQUENTIAL.BloomFilter import *

def run_synthetic_test():
    filter_capacity = 50_000
    elements_to_insert = 50_000  # number of elements to insert
    p_rate = 0.05  # expected false positive rate

    print(f"Test initialization")
    print(f"Target: n={filter_capacity}, False Positive Rate={p_rate * 100}%")
    bf = BloomFilter(filter_capacity, p_rate)
    print(f"Calculated parameters: m={bf.m} bit, k={bf.k} hash functions")

    print("\n Generation of synthetic data")
    # Data to be inserted
    present_items = [f"item_IN_{i}" for i in range(elements_to_insert)]

    # Data not to be inserted
    absent_items = [f"item_OUT_{i}" for i in range(elements_to_insert)]
    print(f"Generated {len(present_items)} elements to insert and {len(absent_items)} elements not to be inserted.")

    print("\n Data entry")
    start_time = time.time()
    try:
        for item in present_items:
            bf.add(item)
    except ValueError as e:
        print(f"\n[WARNING] Insertion interrupted due to saturation: {e}")
    insertion_time = time.time() - start_time
    print(f"Insertion completed in {insertion_time:.2f} s.")
    print(f"Current fill ratio: {bf.get_fill_ratio() * 100:.2f}% bits are 1.")

    print("\n Test Ground Truth")

    # False Negative Check
    false_negatives = 0
    for item in present_items:
        if item not in bf:
            false_negatives += 1

    print(f"False Negatives (should be 0): {false_negatives}")
    assert false_negatives == 0, "ERROR: the filter has lost some data"

    # False positives check
    false_positives = 0
    for item in absent_items:
        if item in bf:
            false_positives += 1

    empirical_fp_rate = false_positives / len(absent_items)
    theoretical_actual_fp = bf.get_actual_fp_rate()

    print(f"\n--- Final False Positive Results ---")
    print(f"Missing elements tested:          {len(absent_items)}")
    print(f"False Positives detected:           {false_positives}")
    print(f"Initial Theoretical PF Rate (p):     {p_rate:.4f} ({p_rate * 100:.2f}%)")
    print(f"Current Theoretical PF Rate:          {theoretical_actual_fp:.4f} ({theoretical_actual_fp * 100:.2f}%)")
    print(f"Empirical PF Rate (Measured):      {empirical_fp_rate:.4f} ({empirical_fp_rate * 100:.2f}%)")

    if abs(empirical_fp_rate - p_rate) < 0.02:
        print("\n TEST PASSED: The empirical rate is in line with the expectation")
    else:
        print("\n ANOMALY: The measured rate deviates too much from the theory")


if __name__ == "__main__":
    run_synthetic_test()