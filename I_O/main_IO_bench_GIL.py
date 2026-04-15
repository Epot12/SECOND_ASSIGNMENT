import asyncio
import time
import os
import sys
import multiprocessing as mp
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from SEQ_I_O.seq_stream import StreamProcessor
from ASYNC_I_O.async_stream import AsyncParallelStreamProcessor
from PARALLEL.PermPoolBloom import PermPoolScalableBloomFilter
from DATA_MANAGEMENT.data_loader import mmap_url_stream


# TRAFFIC GENERATOR (NETWORK SIMULATION)
def network_stream(total_items):
    for i in range(total_items):
        yield f"USER_SESSION_{i % 10_000}_{i}"

        # Network latency simulation
        if i % 2000 == 0:
            time.sleep(0.001)


# parametric STRESS TEST ROUTINE
async def run_stress_test(mode: str):
    BATCH_SIZE = 50_000  # Large enough to justify IPC overhead

    # 1. dynamic parameter setup
    TOTAL_ITEMS = 5_000_000
    if mode == "synthetic":
        # generates a new stream for every call
        stream_factory = lambda: network_stream(TOTAL_ITEMS)
        mode_name = "SYNTHETIC (Network Simulation)"
    else:
        DATA_PATH = os.path.join(parent_dir, "DATA", "common_crawl_FULL.txt")
        if not os.path.exists(DATA_PATH):
            print(f"\n[CRITICAL ERROR] Real dataset not found at: {DATA_PATH}")
            sys.exit(1)
        # generates a new memory-mapped stream at each invocation
        stream_factory = lambda: mmap_url_stream(DATA_PATH, TOTAL_ITEMS)
        mode_name = "REAL DATA (Common Crawl Memory-Mapped)"

    print("=" * 60)
    print(f"[INFO] INITIATING STRESS TEST: {mode_name}")
    print(f"[INFO] Elements to process: {TOTAL_ITEMS:,}")
    print(f"[INFO] Batch Size Configuration: {BATCH_SIZE:,}")
    print("=" * 60)

    # TEST 1: SEQUENTIAL STREAM PROCESSOR (Stop-and-Wait)
    print("\n[TEST 1] Executing Sequential Orchestration (Stop-and-Wait)...")

    with PermPoolScalableBloomFilter(initial_capacity=200_000, target_fp_rate=0.01) as bf_seq:
        seq_processor = StreamProcessor(bf_seq, batch_size=BATCH_SIZE)

        # Wall-Clock and CPU Time
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        # calling factory to generate a new stream
        await seq_processor.ingest(stream_factory())

        seq_total_time = time.perf_counter() - start_time
        seq_cpu_time = time.process_time() - start_cpu
        seq_throughput = TOTAL_ITEMS / seq_total_time

    print(f"[COMPLETED] Sequential execution time: {seq_total_time:.2f} seconds (Wall-Clock)")
    print(f"            Sequential CPU time:       {seq_cpu_time:.2f} seconds")
    print(f"[METRIC] Throughput: {seq_throughput:,.0f} items/second")

    # TEST 2: ASYNC PARALLEL PROCESSOR (State-of-the-Art Producer-Consumer)
    print("\n[TEST 2] Executing Asynchronous Parallel Orchestration (Producer-Consumer)...")

    with PermPoolScalableBloomFilter(initial_capacity=200_000, target_fp_rate=0.01) as bf_async:
        async_processor = AsyncParallelStreamProcessor(bf_async, batch_size=BATCH_SIZE)

        # Wall-Clock and CPU Time
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        # calling factory to generate a new stream
        await async_processor.run_stream(stream_factory())

        async_total_time = time.perf_counter() - start_time
        async_cpu_time = time.process_time() - start_cpu
        async_throughput = TOTAL_ITEMS / async_total_time

    print(f"[COMPLETED] Asynchronous execution time: {async_total_time:.2f} seconds (Wall-Clock)")
    print(f"            Asynchronous CPU time:       {async_cpu_time:.2f} seconds")
    print(f"[METRIC] Throughput: {async_throughput:,.0f} items/second")

    # FINAL RESULTS AND SPEEDUP ANALYSIS
    print("\n" + "=" * 60)
    print("FINAL COMPARATIVE RESULTS")
    print("=" * 60)

    if async_total_time < seq_total_time:
        speedup = seq_total_time / async_total_time
        print(f"[ANALYSIS] The Asynchronous implementation achieved a {speedup:.2f}x speedup.")
        print("Rationale: Network ingestion (I/O) and hash computation (CPU via Multiprocessing)")
        print("occurred with perfect temporal overlap. By delegating the CPU-bound workload to a")
        print("Persistent Process Pool via Shared Memory, the system successfully bypassed the GIL,")
        print("maximizing both I/O throughput and multi-core resource utilization.")
    else:
        print("[ANALYSIS] Execution times are comparable. This outcome may indicate an absolute")
        print("CPU bottleneck or substantial IPC (Inter-Process Communication) overhead that")
        print("offsets the benefits of asynchronous orchestration.")


if __name__ == "__main__":
    # parametrization from CLI
    parser = argparse.ArgumentParser(description="I/O Stream Stress Test")
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True, help="Mode: 'synthetic' or 'real'")
    args = parser.parse_args()

    mp.freeze_support()
    asyncio.run(run_stress_test(args.mode))