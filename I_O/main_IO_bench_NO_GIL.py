import asyncio
import time
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from MULTITHREAD.ThreadedScalBloomFilter import ThreadedScalableBloomFilter
from SEQ_I_O.seq_stream import StreamProcessor
from ASYNC_I_O.async_stream import AsyncParallelStreamProcessor
from DATA_MANAGEMENT.data_loader import mmap_url_stream


# TRAFFIC GENERATOR (NETWORK SIMULATION)
def network_stream(total_items):
    for i in range(total_items):
        yield f"USER_SESSION_{i % 10_000}_{i}"

        # Network latency simulation
        if i % 2000 == 0:
            time.sleep(0.001)


# PARAMETRIC STRESS TEST ROUTINE
async def run_stress_test(mode: str):
    TOTAL_ITEMS = 5_000_000  # Elements to impose significant load on RAM and CPU
    BATCH_SIZE = 50_000  # Large enough to justify the overhead of Thread dispatching

    # 1. DYNAMIC PARAMETER SETUP BASED ON MODE
    if mode == "synthetic":
        INIT_CAP = 200_000
        # Factory Pattern: generates a new stream for every call
        stream_factory = lambda: network_stream(TOTAL_ITEMS)
        mode_name = "SYNTHETIC (Network Simulation)"
    else:
        INIT_CAP = 50_000
        DATA_PATH = os.path.join(parent_dir, "DATA", "common_crawl_FULL.txt")
        if not os.path.exists(DATA_PATH):
            print(f"\n[CRITICAL ERROR] Real dataset not found at: {DATA_PATH}")
            sys.exit(1)
        # Factory Pattern: generates a new memory-mapped stream at each invocation
        stream_factory = lambda: mmap_url_stream(DATA_PATH, TOTAL_ITEMS)
        mode_name = "REAL DATA (Common Crawl Memory-Mapped)"

    print("=" * 70)
    print(f"[INFO] INITIATING STRESS TEST: {mode_name}")
    print(f"[INFO] Elements to process: {TOTAL_ITEMS:,}")
    print(f"[INFO] Batch Size: {BATCH_SIZE:,} | Initial Capacity: {INIT_CAP:,}")
    print("=" * 70)

    # TEST 1: SEQUENTIAL STREAM PROCESSOR (Stop-and-Wait)
    print("\n[TEST 1] Executing Sequential Orchestration (Stop-and-Wait)...")

    # The context manager ensures proper initialization and teardown of thread pools
    with ThreadedScalableBloomFilter(initial_capacity=INIT_CAP, target_fp_rate=0.01) as bf_seq:
        seq_processor = StreamProcessor(bf_seq, batch_size=BATCH_SIZE)

        # Dual-Clock Measurement
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        # Calling factory to generate a fresh stream
        await seq_processor.ingest(stream_factory())

        seq_total_time = time.perf_counter() - start_time
        seq_cpu_time = time.process_time() - start_cpu
        seq_throughput = TOTAL_ITEMS / seq_total_time

    print(f"[COMPLETED] Sequential execution time: {seq_total_time:.2f} seconds (Wall-Clock)")
    print(f"            Sequential CPU time:       {seq_cpu_time:.2f} seconds")
    print(f"[METRIC] Throughput: {seq_throughput:,.0f} items/second")

    # TEST 2: ASYNC PARALLEL PROCESSOR
    print("\n[TEST 2] Executing Asynchronous Parallel Orchestration (Producer-Consumer)...")

    with ThreadedScalableBloomFilter(initial_capacity=INIT_CAP, target_fp_rate=0.01) as bf_async:
        async_processor = AsyncParallelStreamProcessor(bf_async, batch_size=BATCH_SIZE)

        # Dual-Clock Measurement
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        # ingestion: network stream operates continuously;
        # computational processing is strictly delegated to background threads.
        await async_processor.run_stream(stream_factory())

        async_total_time = time.perf_counter() - start_time
        async_cpu_time = time.process_time() - start_cpu
        async_throughput = TOTAL_ITEMS / async_total_time

    print(f"[COMPLETED] Asynchronous execution time: {async_total_time:.2f} seconds (Wall-Clock)")
    print(f"            Asynchronous CPU time:       {async_cpu_time:.2f} seconds")
    print(f"[METRIC] Throughput: {async_throughput:,.0f} items/second")

    # FINAL RESULTS AND SPEEDUP ANALYSIS
    print("\n" + "=" * 70)
    print("FINAL COMPARATIVE RESULTS")
    print("=" * 70)

    if async_total_time < seq_total_time:
        speedup = seq_total_time / async_total_time
        print(f"[ANALYSIS] The Asynchronous implementation achieved a {speedup:.2f}x speedup.")
        print("Rationale: Network ingestion (I/O) and hash computation (CPU via multithreading)")
        print("occurred with perfect temporal overlap, maximizing system resource utilization.")
    else:
        print("[ANALYSIS] Execution times are comparable. This outcome indicates either an absolute")
        print("CPU bottleneck or the presence of an active Global Interpreter Lock (GIL),")
        print("which is typical in standard Python environments (<= 3.12).")


if __name__ == "__main__":
    # CLI parametrization
    parser = argparse.ArgumentParser(description="Multithreaded I/O Stream Stress Test")
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True,
                        help="Select the data source mode: 'synthetic' or 'real'")
    args = parser.parse_args()

    # Initialize the primary Event Loop passing the CLI argument
    asyncio.run(run_stress_test(args.mode))