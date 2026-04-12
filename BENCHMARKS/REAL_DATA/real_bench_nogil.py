import os, sys, time, json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from MULTITHREAD.ThreadedScalBloomFilter import ThreadedScalableBloomFilter as ThreadBloom


def run():
    print("\n[NO-GIL WORKER] Starting NO-GIL Benchmarks (Tests 1 to 4)...")

    # Identical parameters for fairness
    cap, ins, tst, fpr = 100_000, 3_000_000, 500_000, 0.05

    print(f"[SYSTEM] Pre-loading {ins + tst:,} REAL items into RAM. Please wait...")

    # building the path to real data
    data_path = os.path.join(parent_dir, "DATA", "common_crawl_FULL.txt")

    # reading in RAM before starting measuring time
    all_real_items = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for _ in range(ins + tst):
                line = f.readline()
                if not line:
                    break  # end of file
                all_real_items.append(line.strip())
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found in: {data_path}")
        sys.exit(1)

    # dividing data
    present_items = all_real_items[:ins]
    absent_items = all_real_items[ins:ins + tst]

    print("[SYSTEM] Data loaded successfully. I/O is completely isolated.")
    print("------------------------------------------------------------")

    results = {}

    # TEST 5: MULTITHREAD NATIVE
    with ThreadBloom(cap, fpr) as s5:
        t0 = time.perf_counter()
        s5.add_batch(present_items)
        t1 = time.perf_counter()
        s5.contains_batch(present_items)
        s5.contains_batch(absent_items)
        t2 = time.perf_counter()
        results['NativeThreads'] = {'ins': t1 - t0, 'read': t2 - t1}

    # exporting measurements
    out_path = os.path.join(current_dir, 'telemetry_nogil.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"[NOGIL-WORKER] Telemetry saved to {out_path}")


if __name__ == "__main__":
    run()