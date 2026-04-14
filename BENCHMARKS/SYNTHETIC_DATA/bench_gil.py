import os, sys, time, json
import multiprocessing as mp
import gc
from utils.stats_engine import *


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

    present_items = [f"IN_{i}" for i in range(ins)]
    absent_items = [f"OUT_{i}" for i in range(tst)]

    results = {}
    # TEST 1: SEQUENTIAL
    print(" -> Running Test 1: Sequential Architecture...")
    ins_times, read_times = [], []
    ins_cpu_times, read_cpu_times = [], []
    for run_idx in range(TOTAL_RUNS):
        s1 = SeqBloom(cap, fpr)
        gc.disable()
        t0 = time.perf_counter()
        t0_cpu = time.process_time()
        for item in present_items: s1.add(item)
        t1 = time.perf_counter()
        t1_cpu = time.process_time()
        for item in present_items: _ = item in s1
        for item in absent_items: _ = item in s1
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

    results['Sequential'] = {
        'ins': stat_ins['mean'],
        'read': stat_read['mean'],
        'ins_cpu': stat_ins_cpu['mean'],
        'read_cpu': stat_read_cpu['mean']
    }

    print(f"    [+] Wall-Clock -> Ins: {stat_ins['mean']}s | Query: {stat_read['mean']}s")
    print(f"    [+] CPU Time   -> Ins: {stat_ins_cpu['mean']}s | Query: {stat_read_cpu['mean']}s")


    # TEST 2: ON-DEMAND
    print(" -> Running Test 2: On-Demand Multiprocessing...")
    ins_times, read_times = [], []
    ins_cpu_times, read_cpu_times = [], []
    for run_idx in range(TOTAL_RUNS):
        s2 = OnDemandBloom(cap, fpr)
        gc.disable()
        t0 = time.perf_counter()
        t0_cpu = time.process_time()
        s2.add_batch(present_items)
        t1 = time.perf_counter()
        t1_cpu = time.process_time()
        s2.contains_batch(present_items)
        s2.contains_batch(absent_items)
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

    results['OnDemand'] = {
        'ins': stat_ins['mean'],
        'read': stat_read['mean'],
        'ins_cpu': stat_ins_cpu['mean'],
        'read_cpu': stat_read_cpu['mean']
    }

    print(f"    [+] Wall-Clock -> Ins: {stat_ins['mean']}s | Query: {stat_read['mean']}s")
    print(f"    [+] CPU Time   -> Ins: {stat_ins_cpu['mean']}s | Query: {stat_read_cpu['mean']}s")

    # TEST 3: LAZY RESTART
    print(" -> Running Test 3: Lazy Restart Architecture...")
    ins_times, read_times = [], []
    ins_cpu_times, read_cpu_times = [], []
    for run_idx in range(TOTAL_RUNS):
        with LazyBloom(cap, fpr) as s3:
            gc.disable()
            t0 = time.perf_counter()
            t0_cpu = time.process_time()
            s3.add_batch(present_items)
            t1 = time.perf_counter()
            t1_cpu = time.process_time()
            s3.contains_batch(present_items)
            s3.contains_batch(absent_items)
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

    results['LazyRestart'] = {
        'ins': stat_ins['mean'],
        'read': stat_read['mean'],
        'ins_cpu': stat_ins_cpu['mean'],
        'read_cpu': stat_read_cpu['mean']
    }

    print(f"    [+] Wall-Clock -> Ins: {stat_ins['mean']}s | Query: {stat_read['mean']}s")
    print(f"    [+] CPU Time   -> Ins: {stat_ins_cpu['mean']}s | Query: {stat_read_cpu['mean']}s")

    # TEST 4: SOTA IPC
    print(" -> Running Test 4: SOTA IPC (Persistent Pool)...")
    ins_times, read_times = [], []
    ins_cpu_times, read_cpu_times = [], []
    for run_idx in range(TOTAL_RUNS):
        with SotaBloom(cap, fpr) as s4:
            gc.disable()
            t0 = time.perf_counter()
            t0_cpu = time.process_time()
            s4.add_batch(present_items)
            t1 = time.perf_counter()
            t1_cpu = time.process_time()
            s4.contains_batch(present_items)
            s4.contains_batch(absent_items)
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

    results['SotaIPC'] = {
        'ins': stat_ins['mean'],
        'read': stat_read['mean'],
        'ins_cpu': stat_ins_cpu['mean'],
        'read_cpu': stat_read_cpu['mean']
    }

    print(f"    [+] Wall-Clock -> Ins: {stat_ins['mean']}s | Query: {stat_read['mean']}s")
    print(f"    [+] CPU Time   -> Ins: {stat_ins_cpu['mean']}s | Query: {stat_read_cpu['mean']}s")


    # exporting measurements
    out_path = os.path.join(current_dir, 'telemetry_gil.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n[GIL-WORKER] Telemetry saved to {out_path}")


if __name__ == "__main__":
    mp.freeze_support()
    run()