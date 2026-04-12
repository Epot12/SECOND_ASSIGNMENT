import os, sys, time, json
import multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from SEQUENTIAL.ScalableBloomFilter import ScalableBloomFilter as SeqBloom
from PARALLEL.ScalMultProcBloom import ParallelScalableBloomFilter as OnDemandBloom
from PARALLEL.ScalMultProcBloomOpt import ParallelScalableBloomFilter as LazyBloom
from PARALLEL.PermPoolBloom import PermPoolScalableBloomFilter as SotaBloom


def run():
    print("\n[GIL-WORKER] Starting GIL Benchmarks (Tests 1 to 4)...")

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

    # TEST 1: SEQUENTIAL
    s1 = SeqBloom(cap, fpr)
    t0 = time.perf_counter()
    for item in present_items: s1.add(item)
    t1 = time.perf_counter()
    for item in present_items: _ = item in s1
    for item in absent_items: _ = item in s1
    t2 = time.perf_counter()
    results['Sequential'] = {'ins': t1 - t0, 'read': t2 - t1}

    # TEST 2: ON-DEMAND
    s2 = OnDemandBloom(cap, fpr)
    t0 = time.perf_counter()
    s2.add_batch(present_items)
    t1 = time.perf_counter()
    s2.contains_batch(present_items)
    s2.contains_batch(absent_items)
    t2 = time.perf_counter()
    results['OnDemand'] = {'ins': t1 - t0, 'read': t2 - t1}

    # TEST 3: LAZY RESTART
    with LazyBloom(cap, fpr) as s3:
        t0 = time.perf_counter()
        s3.add_batch(present_items)
        t1 = time.perf_counter()
        s3.contains_batch(present_items)
        s3.contains_batch(absent_items)
        t2 = time.perf_counter()
        results['LazyRestart'] = {'ins': t1 - t0, 'read': t2 - t1}

    # TEST 4: SOTA IPC
    with SotaBloom(cap, fpr) as s4:
        t0 = time.perf_counter()
        s4.add_batch(present_items)
        t1 = time.perf_counter()
        s4.contains_batch(present_items)
        s4.contains_batch(absent_items)
        t2 = time.perf_counter()
        results['SotaIPC'] = {'ins': t1 - t0, 'read': t2 - t1}

    # exporting measurements
    out_path = os.path.join(current_dir, 'telemetry_gil.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"[GIL-WORKER] Telemetry saved to {out_path}")


if __name__ == "__main__":
    mp.freeze_support()
    run()