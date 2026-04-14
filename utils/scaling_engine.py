import time
import gc



# scaling engine - Amdahl, Gustafson and Granularity Profiler


def run_strong_scaling(BloomClass, max_workers: int, total_items: int, fpr: float = 0.05):
    """
    AMDAHL'S LAW (Strong Scaling)
    The problem (total_items) is fixed. We vary the number of cores (from 1 to max_workers).
    It measures how quickly we can solve the same problem by adding hardware.
    """
    print(f"\n[STRONG SCALING] Fixed Problem Size: {total_items:,} items. Testing 1 to {max_workers} threads...")

    # 1. Pre-allocation in RAM to measure pure CPU only
    dataset = [f"STRONG_{i}" for i in range(total_items)]
    results = {}

    for workers in range(1, max_workers + 1):
        # disabling Garbage Collector to avoid interruptions during measuring
        gc.disable()

        # initializing the filter by explicitly passing the number of threads/processes
        with BloomClass(initial_capacity=100_000, target_fp_rate=fpr, num_threads=workers) as bf:
            t0 = time.perf_counter()
            bf.add_batch(dataset)  # Insertion of the fixed dataset
            t1 = time.perf_counter()

            elapsed = t1 - t0
            results[workers] = elapsed
            print(f"  -> Workers/Cores: {workers} | Time: {elapsed:.4f}s")

        gc.enable()

    return results


def run_weak_scaling(BloomClass, max_workers: int, items_per_worker: int, fpr: float = 0.05):
    """
    GUSTAFSON'S LAW (Weak Scaling)
    The load per worker is FIXED. If the cores are doubled, then the data are doubled.
    """
    max_total_items = items_per_worker * max_workers
    print(f"\n[WEAK SCALING] Fixed Load Per Worker: {items_per_worker:,}. Max Total Size: {max_total_items:,}...")

    full_dataset = [f"WEAK_{i}" for i in range(max_total_items)]
    results = {}

    for workers in range(1, max_workers + 1):
        current_workload = items_per_worker * workers
        dataset_slice = full_dataset[:current_workload]

        gc.disable()
        with BloomClass(initial_capacity=100_000, target_fp_rate=fpr, num_threads=workers) as bf:
            t0 = time.perf_counter()
            bf.add_batch(dataset_slice)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            results[workers] = {
                "workload": current_workload,
                "time": elapsed
            }
            print(f"  -> Workers: {workers} | Workload: {current_workload:,} | Time: {elapsed:.4f}s")

        gc.enable()

    return results


def run_chunk_optimization(BloomClass, num_workers: int, total_items: int, chunk_sizes: list, fpr: float = 0.05):
    """
    GRANULARITY PROFILING
    Test to find the optimal micro-batch size.
    """
    print(f"\n[CHUNK OPTIMIZATION] Cores: {num_workers} | Items: {total_items:,}. Testing chunk sizes...")

    dataset = [f"CHUNK_{i}" for i in range(total_items)]
    results = {}

    for chunk_size in chunk_sizes:
        gc.disable()

        with BloomClass(initial_capacity=100_000, target_fp_rate=fpr, num_threads=num_workers) as bf:
            bf.min_chunk_size = chunk_size

            t0 = time.perf_counter()
            bf.add_batch(dataset)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            results[chunk_size] = elapsed
            print(f"  -> Chunk Size: {chunk_size:^10} | Time: {elapsed:.4f}s")

        gc.enable()

    return results