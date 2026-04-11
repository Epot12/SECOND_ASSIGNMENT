import asyncio
import time
import os, sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from MULTITHREAD.ThreadedScalBloomFilter import ThreadedScalableBloomFilter
from SEQ_I_O.seq_stream import *
from ASYNC_I_O.async_stream import *


# TRAFFIC GENERATOR (NETWORK SIMULATION)
async def network_stream(total_items):
    """
    Simulates a high-speed network log stream.
    The brief sleep command emulates the latency associated with packet arrival,
    effectively modeling an I/O-bound operation.
    """
    for i in range(total_items):
        yield f"USER_SESSION_{i % 10_000}_{i}"

        # Simulates network latency: yields execution for 1 millisecond every 2000 elements.
        # This minimal latency is crucial to evaluate the asynchronous overlap between I/O and CPU.
        if i % 2000 == 0:
            await asyncio.sleep(0.001)


# STRESS TEST ROUTINE
async def run_stress_test():
    TOTAL_ITEMS = 3_000_000  # One million elements to impose significant load on RAM and CPU
    BATCH_SIZE = 50_000  # Large enough to justify the overhead of No-GIL Thread dispatching

    print("=" * 60)
    print(f"[INFO] INITIATING STRESS TEST: {TOTAL_ITEMS} elements")
    print(f"[INFO] Batch Size Configuration: {BATCH_SIZE}")
    print("=" * 60)


    # TEST 1: SEQUENTIAL STREAM PROCESSOR (Stop-and-Wait)

    print("\n[TEST 1] Executing Sequential Orchestration (Stop-and-Wait)...")

    # The context manager ensures proper initialization and teardown of thread pools
    with ThreadedScalableBloomFilter(initial_capacity=200_000, target_fp_rate=0.01) as bf_seq:
        seq_processor = StreamProcessor(bf_seq, batch_size=BATCH_SIZE)

        start_time = time.perf_counter()

        # Blocking ingestion: execution halts every 50k elements for processing
        await seq_processor.ingest(network_stream(TOTAL_ITEMS))

        seq_total_time = time.perf_counter() - start_time
        seq_throughput = TOTAL_ITEMS / seq_total_time

    print(f"[COMPLETED] Sequential execution time: {seq_total_time:.2f} seconds")
    print(f"[METRIC] Throughput: {seq_throughput:,.0f} items/second")


    # TEST 2: ASYNC PARALLEL PROCESSOR

    print("\n[TEST 2] Executing Asynchronous Parallel Orchestration (Producer-Consumer)...")

    with ThreadedScalableBloomFilter(initial_capacity=200_000, target_fp_rate=0.01) as bf_async:
        async_processor = AsyncParallelStreamProcessor(bf_async, batch_size=BATCH_SIZE)

        start_time = time.perf_counter()

        # ingestion: network stream operates continuously;
        # computational processing is strictly delegated to background threads.
        await async_processor.run_stream(network_stream(TOTAL_ITEMS))

        async_total_time = time.perf_counter() - start_time
        async_throughput = TOTAL_ITEMS / async_total_time

    print(f"[COMPLETED] Asynchronous execution time: {async_total_time:.2f} seconds")
    print(f"[METRIC] Throughput: {async_throughput:,.0f} items/second")


    # FINAL RESULTS AND SPEEDUP ANALYSIS

    print("\n" + "=" * 60)
    print("FINAL COMPARATIVE RESULTS")
    print("=" * 60)

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
    # Initialize the primary Event Loop
    asyncio.run(run_stress_test())