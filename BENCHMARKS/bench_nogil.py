import time, json
import gc
import argparse
from utils.stats_engine import compute_statistics, TOTAL_RUNS, WARMUP_RUNS
from utils.utilities import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from MULTITHREAD.ThreadedScalBloomFilter import ThreadedScalableBloomFilter as ThreadBloom
from MULTITHREAD.MapReduceScalBloom import ThreadedScalableBloomFilter as MapReduceBloom
from MULTITHREAD.stripe_strategy.StripedBloomFilter import StripedBloomFilter as StripedBloom


def run(mode):
    print(f"\n[NOGIL-WORKER] Starting No-GIL Benchmarks (Mode: {mode})...")
    cap, ins, tst, fpr = 100_000, 3_000_000, 500_000, 0.05
    present_items, absent_items = load_data(mode, ins, tst)
    results = {}

    architectures = [("NativeThreads", ThreadBloom), ("MapReduceVectorized", MapReduceBloom),
                     ("StripedVectorized", StripedBloom)]

    for name, ArchClass in architectures:
        print(f" -> Running Test: {name}...")
        ins_times, read_times = [], []
        ins_cpu_times, read_cpu_times = [], []

        for run_idx in range(TOTAL_RUNS):
            with ArchClass(cap, fpr) as s:
                gc.disable()
                t0, t0c = time.perf_counter(), time.process_time()
                s.add_batch(present_items)
                t1, t1c = time.perf_counter(), time.process_time()
                s.contains_batch(present_items)
                s.contains_batch(absent_items)
                t2, t2c = time.perf_counter(), time.process_time()
                gc.enable()

                if run_idx >= WARMUP_RUNS:
                    ins_times.append(t1 - t0)
                    read_times.append(t2 - t1)
                    ins_cpu_times.append(t1c - t0c)
                    read_cpu_times.append(t2c - t1c)

        si, sr = compute_statistics(ins_times), compute_statistics(read_times)
        sic, src = compute_statistics(ins_cpu_times), compute_statistics(read_cpu_times)
        results[name] = {'ins': si['mean'], 'read': sr['mean'], 'ins_cpu': sic['mean'], 'read_cpu': src['mean']}

    out_path = os.path.join(current_dir, f'telemetry_nogil_{mode}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True)
    args = parser.parse_args()
    run(args.mode)