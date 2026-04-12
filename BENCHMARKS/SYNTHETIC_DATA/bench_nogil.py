import os, sys, time, json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from MULTITHREAD.ThreadedScalBloomFilter import ThreadedScalableBloomFilter as ThreadBloom
from MULTITHREAD.MapReduceScalBloom import ThreadedScalableBloomFilter as MapReduceBloom


def run():
    print("\n[NOGIL-WORKER] Starting No-GIL Benchmarks (Test 5)...")

    # Identical parameters for fairness
    cap, ins, tst, fpr = 100_000, 3_000_000, 500_000, 0.05

    present_items = [f"IN_{i}" for i in range(ins)]
    absent_items = [f"OUT_{i}" for i in range(tst)]

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

    del s5

    with MapReduceBloom(cap, fpr) as s6:
        t0 = time.perf_counter()
        s6.add_batch(present_items)
        t1 = time.perf_counter()

        s6.contains_batch(present_items)
        s6.contains_batch(absent_items)
        t2 = time.perf_counter()

        results['MapReduceVectorized'] = {'ins': t1 - t0, 'read': t2 - t1}


    # exporting measurements
    out_path = os.path.join(current_dir, 'telemetry_nogil.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"[NOGIL-WORKER] Telemetry saved to {out_path}")


if __name__ == "__main__":
    run()