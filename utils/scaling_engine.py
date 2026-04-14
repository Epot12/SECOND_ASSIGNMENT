import time
import gc
import math
import stats_engine


# THE SCALING ENGINE - Amdahl, Gustafson and Granularity Profiler


WARMUP_RUNS = 2
MEASURED_RUNS = 5
TOTAL_RUNS = WARMUP_RUNS + MEASURED_RUNS


def compute_statistics(times: list) -> dict:
    """
    Calculate Mean, Median and 95% Confidence Interval (using T-Student for n=5).
    """
    n = len(times)
    mean_val = statistics.mean(times)
    median_val = statistics.median(times)

    if n > 1:
        stdev = statistics.stdev(times)
        sem = stdev / math.sqrt(n)  # Standard Error of the Mean
        # Critical value t for 95% CI with degrees of freedom (df = n-1 = 4)
        t_critical = 2.776
        margin = t_critical * sem
    else:
        margin = 0.0

    return {
        "mean": round(mean_val, 6),
        "median": round(median_val, 6),
        "ci_95_margin": round(margin, 6),
        "ci_95_lower": round(mean_val - margin, 6),
        "ci_95_upper": round(mean_val + margin, 6),
        "raw_runs": [round(t, 6) for t in times]
    }


def run_strong_scaling(BloomClass, max_workers: int, total_items: int, fpr: float = 0.05):
    print(f"\n[STRONG SCALING] Fixed Problem Size: {total_items:,} items. Testing 1 to {max_workers} threads...")
    dataset = [f"STRONG_{i}" for i in range(total_items)]
    results = {}

    for workers in range(1, max_workers + 1):
        run_times = []
        print(f"  -> Workers/Cores: {workers} | Executing {TOTAL_RUNS} runs...")

        for run_idx in range(TOTAL_RUNS):
            gc.disable()
            with BloomClass(initial_capacity=100_000, target_fp_rate=fpr, num_threads=workers) as bf:
                t0 = time.perf_counter()
                bf.add_batch(dataset)
                t1 = time.perf_counter()
            gc.enable()

            if run_idx >= WARMUP_RUNS:
                run_times.append(t1 - t0)

        stats = compute_statistics(run_times)
        results[workers] = stats
        print(
            f"     [=] Mean: {stats['mean']:.4f}s | Median: {stats['median']:.4f}s | 95% CI: ±{stats['ci_95_margin']:.4f}s")

    return results


def run_weak_scaling(BloomClass, max_workers: int, items_per_worker: int, fpr: float = 0.05):
    max_total_items = items_per_worker * max_workers
    print(f"\n[WEAK SCALING] Fixed Load Per Worker: {items_per_worker:,}. Max Total Size: {max_total_items:,}...")
    full_dataset = [f"WEAK_{i}" for i in range(max_total_items)]
    results = {}

    for workers in range(1, max_workers + 1):
        current_workload = items_per_worker * workers
        dataset_slice = full_dataset[:current_workload]
        run_times = []
        print(f"  -> Workers: {workers} | Workload: {current_workload:,} | Executing {TOTAL_RUNS} runs...")

        for run_idx in range(TOTAL_RUNS):
            gc.disable()
            with BloomClass(initial_capacity=100_000, target_fp_rate=fpr, num_threads=workers) as bf:
                t0 = time.perf_counter()
                bf.add_batch(dataset_slice)
                t1 = time.perf_counter()
            gc.enable()

            if run_idx >= WARMUP_RUNS:
                run_times.append(t1 - t0)

        stats = compute_statistics(run_times)
        stats["workload"] = current_workload
        results[workers] = stats
        print(f"     [=] Mean: {stats['mean']:.4f}s | 95% CI: ±{stats['ci_95_margin']:.4f}s")

    return results


def run_chunk_optimization(BloomClass, num_workers: int, total_items: int, chunk_sizes: list, fpr: float = 0.05):
    print(f"\n[CHUNK OPTIMIZATION] Cores: {num_workers} | Items: {total_items:,}. Testing chunk sizes...")
    dataset = [f"CHUNK_{i}" for i in range(total_items)]
    results = {}

    for chunk_size in chunk_sizes:
        run_times = []
        print(f"  -> Chunk Size: {chunk_size:^10} | Executing {TOTAL_RUNS} runs...")

        for run_idx in range(TOTAL_RUNS):
            gc.disable()
            with BloomClass(initial_capacity=100_000, target_fp_rate=fpr, num_threads=num_workers) as bf:
                bf.min_chunk_size = chunk_size
                t0 = time.perf_counter()
                bf.add_batch(dataset)
                t1 = time.perf_counter()
            gc.enable()

            if run_idx >= WARMUP_RUNS:
                run_times.append(t1 - t0)

        stats = compute_statistics(run_times)
        results[chunk_size] = stats
        print(f"     [=] Mean: {stats['mean']:.4f}s | 95% CI: ±{stats['ci_95_margin']:.4f}s")

    return results