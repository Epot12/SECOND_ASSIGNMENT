# BENCHMARK: SEQUENTIAL VS PARALLEL (NON-SCALABLE)

import time
import multiprocessing as mp
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from SEQUENTIAL.BloomFilter import *
from PARALLEL.MultiProcBloom import *

def run_performance_comparison():
    elements_to_insert = 1_500_000
    elements_to_query = 500_000
    target_p_rate = 0.05

    print("=========================================================")
    print("BENCHMARK: SEQUENTIAL VS PARALLEL (NON-SCALABLE)")
    print("=========================================================\n")


    #PREPARATION PHASE
    print("[*] Generating dataset ")
    present_items = [f"IN_{i}" for i in range(elements_to_insert)]
    absent_items = [f"OUT_{i}" for i in range(elements_to_query)]
    print(f"[*] Dataset generated: {elements_to_insert} Insertions, {elements_to_query} Reads.\n")

    # TESTING SEQUENTIAL VERSION

    print("[*] Executing sequential Bloom Filter\n ")
    seq_bf = BloomFilter(elements_to_insert, target_p_rate)

    # timing insertion
    t0_seq_add = time.perf_counter()
    for item in present_items:
        seq_bf.add(item)
    t1_seq_add = time.perf_counter()
    seq_insert_time = t1_seq_add - t0_seq_add

    # timing reading
    t0_seq_chk = time.perf_counter()
    _ = [item in seq_bf for item in absent_items]
    t1_seq_chk = time.perf_counter()
    seq_query_time = t1_seq_chk - t0_seq_chk

    #TESTING PARALLEL VERSION

    cores = mp.cpu_count()
    print(f"[*] Executing parallel Bloom Filter ({cores} Core)...")
    par_bf = ParallelBloomFilter(elements_to_insert, target_p_rate)

    # timing insertion
    t0_par_add = time.perf_counter()
    par_bf.add_batch(present_items)
    t1_par_add = time.perf_counter()
    par_insert_time = t1_par_add - t0_par_add

    # timing reading
    t0_par_chk = time.perf_counter()
    _ = par_bf.contains_batch(absent_items)
    t1_par_chk = time.perf_counter()
    par_query_time = t1_par_chk - t0_par_chk

    # RESULTS

    speedup_insert = seq_insert_time / par_insert_time
    speedup_query = seq_query_time / par_query_time

    total_seq = seq_insert_time + seq_query_time
    total_par = par_insert_time + par_query_time
    speedup_total = total_seq / total_par

    print("\n" + "=" * 70)
    print(f" RISULTATI DEL BENCHMARK (Hardware: {cores} Core Logici)")
    print("=" * 70)
    print(f"{'Fase':<15} | {'Sequenziale':<15} | {'Parallelo':<15} | {'Speedup':<10}")
    print("-" * 70)
    print(f"{'Inserimento':<15} | {seq_insert_time:>12.4f} s | {par_insert_time:>12.4f} s | {speedup_insert:>8.2f}x")
    print(f"{'Lettura':<15} | {seq_query_time:>12.4f} s | {par_query_time:>12.4f} s | {speedup_query:>8.2f}x")
    print("-" * 70)
    print(f"{'TOTALE':<15} | {total_seq:>12.4f} s | {total_par:>12.4f} s | {speedup_total:>8.2f}x")
    print("=" * 70 + "\n")


if __name__ == '__main__':

    run_performance_comparison()