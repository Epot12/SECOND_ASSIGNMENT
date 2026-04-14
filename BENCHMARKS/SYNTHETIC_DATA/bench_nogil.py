import os, sys, time, json
import gc
from utils.stats_engine import compute_statistics, TOTAL_RUNS, WARMUP_RUNS

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

    print(" -> Running Test 5: Native Threads Architecture...")
    ins_times, read_times = [], []
    ins_cpu_times, read_cpu_times = [], []
    for run_idx in range(TOTAL_RUNS):
        # TEST 5: MULTITHREAD NATIVE
        with ThreadBloom(cap, fpr) as s5:
            gc.disable()
            t0 = time.perf_counter()
            t0_cpu = time.process_time()
            s5.add_batch(present_items)
            t1 = time.perf_counter()
            t1_cpu = time.process_time()
            s5.contains_batch(present_items)
            s5.contains_batch(absent_items)
            t2 = time.perf_counter()
            t2_cpu = time.process_time()
            gc.enable()

            if run_idx >= WARMUP_RUNS:
                ins_times.append(t1 - t0)
                read_times.append(t2 - t1)
                ins_cpu_times.append(t1_cpu - t0_cpu)
                read_cpu_times.append(t2_cpu - t1_cpu)

    stat_ins = compute_statistics(ins_times)
    stat_read = compute_statistics(read_times)

    stat_ins_cpu = compute_statistics(ins_cpu_times)
    stat_read_cpu = compute_statistics(read_cpu_times)

    results['NativeThreads'] = {
        'ins': stat_ins['mean'],
        'read': stat_read['mean'],
        'ins_cpu': stat_ins_cpu['mean'],
        'read_cpu': stat_read_cpu['mean']
    }

    print(f"    [+] Wall-Clock -> Ins: {stat_ins['mean']}s | Query: {stat_read['mean']}s")
    print(f"    [+] CPU Time   -> Ins: {stat_ins_cpu['mean']}s | Query: {stat_read_cpu['mean']}s")
    del s5

    print(" -> Running Test 6: Map-Reduce Vectorized Architecture...")
    ins_times, read_times = [], []
    ins_cpu_times, read_cpu_times = [], []
    for run_idx in range(TOTAL_RUNS):
        with MapReduceBloom(cap, fpr) as s6:
            gc.disable()
            t0 = time.perf_counter()
            t0_cpu = time.process_time()
            s6.add_batch(present_items)
            t1 = time.perf_counter()
            t1_cpu = time.process_time()

            s6.contains_batch(present_items)
            s6.contains_batch(absent_items)
            t2 = time.perf_counter()
            t2_cpu = time.process_time()
            gc.enable()

            if run_idx >= WARMUP_RUNS:
                ins_times.append(t1 - t0)
                read_times.append(t2 - t1)
                ins_cpu_times.append(t1_cpu - t0_cpu)
                read_cpu_times.append(t2_cpu - t1_cpu)

    stat_ins = compute_statistics(ins_times)
    stat_read = compute_statistics(read_times)

    stat_ins_cpu = compute_statistics(ins_cpu_times)
    stat_read_cpu = compute_statistics(read_cpu_times)

    results['MapReduceVectorized'] = {
        'ins': stat_ins['mean'],
        'read': stat_read['mean'],
        'ins_cpu': stat_ins_cpu['mean'],
        'read_cpu': stat_read_cpu['mean']
    }

    print(f"    [+] Wall-Clock -> Ins: {stat_ins['mean']}s | Query: {stat_read['mean']}s")
    print(f"    [+] CPU Time   -> Ins: {stat_ins_cpu['mean']}s | Query: {stat_read_cpu['mean']}s")

    # exporting measurements
    out_path = os.path.join(current_dir, 'telemetry_nogil.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n[NOGIL-WORKER] Telemetry saved to {out_path}")


if __name__ == "__main__":
    run()