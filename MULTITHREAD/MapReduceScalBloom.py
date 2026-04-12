import math
import mmh3
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _thread_add_chunk(args):
    """
    MAP Phase: Zero False Sharing.
    Everything happens in the L1/L2 cache of the single core.
    """
    shadow_buf, m, k, items = args

    for item in items:
        if type(item) is str:
            item_bytes = item.encode('utf-8')
        elif type(item) is bytes:
            item_bytes = item
        elif type(item) is int:
            item_bytes = item.to_bytes(8, 'big', signed=True)
        else:
            item_bytes = str(item).encode('utf-8')


        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)

        for i in range(k):
            index = (h1 + i * h2) % m
            shadow_buf[index] = 1



def _thread_contains_chunk(args):
    layers_info, items = args  # layers_info = [(buf, m, k), ...]
    results = []

    for item in items:
        if type(item) is str:
            item_bytes = item.encode('utf-8')
        elif type(item) is bytes:
            item_bytes = item
        elif type(item) is int:
            item_bytes = item.to_bytes(8, 'big', signed=True)
        else:
            item_bytes = str(item).encode('utf-8')

        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)
        is_present = False

        for buf, m, k in reversed(layers_info):
            found_in_layer = True
            for i in range(k):
                if buf[(h1 + i * h2) % m] == 0:
                    found_in_layer = False
                    break
            if found_in_layer:
                is_present = True
                break
        results.append(is_present)
    return results


class ThreadedScalableBloomFilter:
    """
    Implementation for Free-Threaded (No-GIL) Python.
    Uses concurrent.futures.ThreadPoolExecutor for zero-copy, zero-IPC concurrency.
    """

    def __init__(self, initial_capacity: int, target_fp_rate: float,
                 tightening_ratio: float = 0.9, growth_factor: int = 2,
                 num_threads: int = None):

        self.initial_capacity = initial_capacity
        self.target_fp_rate = target_fp_rate
        self.tightening_ratio = tightening_ratio
        self.growth_factor = growth_factor
        self.num_threads = num_threads or os.cpu_count()
        self.p0 = target_fp_rate * (1 - tightening_ratio)

        # Pure Python Native Memory (No multiprocessing.shared_memory needed)
        self.bitmaps = []  # List of standard Python bytearrays
        self.layers_info = []  # Tuples of (buf_reference, m, k)
        self.capacities = []
        self.elements_counts = []

        self._calibrate_threshold()

        # Persistent Thread Pool
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self.np_bitmaps = []  # New list for NumPy views
        self.shadow_bitmaps = []
        self.np_shadow_bitmaps = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

    def _calibrate_threshold(self):
        """
        Hardware auto-tuning for ThreadPool dispatch overhead.
        Calculates the break-even point for No-GIL threading.
        """
        print("[THREAD-SYSTEM] Hardware calibration for Native Threads...")

        # Measuring pure computing power (T_hash)
        test_data = b"benchmark_string"
        k_test = max(1, round(-math.log(self.p0) / math.log(2)))
        m_test = 100_000

        start_cpu = time.perf_counter()
        for _ in range(50_000):
            h1, h2 = mmh3.hash64(test_data, seed=42, signed=False)
            for i in range(k_test):
                _ = (h1 + i * h2) % m_test
        end_cpu = time.perf_counter()

        t_hash = (end_cpu - start_cpu) / 50_000

        # Measuring Operating System Overhead on Threads (T_overhead)
        def dummy_task(x):
            return x

        # creating a disposable pool just to measure the power-on and dispatching delay
        with ThreadPoolExecutor(max_workers=1) as pool:
            start_thread = time.perf_counter()
            list(pool.map(dummy_task, [[1]]))
            end_thread = time.perf_counter()

        t_overhead = end_thread - start_thread

        # Calculation of the mathematical Break-Even
        target_efficiency = 5

        raw_threshold = int((t_overhead * target_efficiency) / t_hash)

        # Since threads are very light, we put a minimum limit of 10,000
        # to avoid clogging the operating system queues
        self.min_chunk_size = max(10_000, round(raw_threshold, -3))

        print(f"[THREAD-SYSTEM] Auto-Tuning completed:")
        print(f"         - Single hash cost:     {t_hash * 1e6:.4f} µs")
        print(f"         - Thread OS overhead:   {t_overhead * 1000:.4f} ms")
        print(f"         - Calculated threshold: {self.min_chunk_size} elements per chunk")

    def _add_new_layer(self):
        current_depth = len(self.bitmaps)
        new_capacity = self.initial_capacity * (self.growth_factor ** current_depth)
        new_fp_rate = self.p0 * (self.tightening_ratio ** current_depth)

        m = math.ceil(-(new_capacity * math.log(new_fp_rate)) / (math.log(2) ** 2))
        k = max(1, round((m / new_capacity) * math.log(2)))

        # NATIVE MEMORY ALLOCATION: Fast, zero-copy, automatically garbage-collected
        buf = bytearray(m)
        np_buf = np.frombuffer(buf, dtype=np.uint8)
        shadow_bufs = [bytearray(m) for _ in range(self.num_threads)]
        np_shadow_bufs = [np.frombuffer(b, dtype=np.uint8) for b in shadow_bufs]

        self.bitmaps.append(buf)
        self.np_bitmaps.append(np_buf)
        self.shadow_bitmaps.append(shadow_bufs)
        self.np_shadow_bitmaps.append(np_shadow_bufs)
        self.layers_info.append((buf, m, k))

        print(f"[THREAD-SYSTEM] Allocated Layer {current_depth}: Capacity={new_capacity}, m={m} bytes (Zero-Copy Ram)")

    def _chunkify(self, data: list):
        chunk_size = math.ceil(len(data) / self.num_threads)
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def add_batch(self, items: list):
        if not items: return
        total_items = len(items)
        current_idx = 0

        while current_idx < total_items:
            if not self.bitmaps or self.elements_counts[-1] >= self.capacities[-1]:
                self._add_new_layer()

            available_space = self.capacities[-1] - self.elements_counts[-1]
            end_idx = min(current_idx + available_space, total_items)
            items_for_this_layer = items[current_idx:end_idx]

            self.elements_counts[-1] += len(items_for_this_layer)
            buf, m, k = self.layers_info[-1]

            # MAP PHASE (Parallel)
            chunks = self._chunkify(items_for_this_layer)
            jobs = []
            for thread_id, chunk in enumerate(chunks):
                t_shadow_buf = self.shadow_bitmaps[-1][thread_id]
                jobs.append((t_shadow_buf, m, k, chunk))

            # map returns a generator with thread local sets
            list(self.executor.map(_thread_add_chunk, jobs))

            global_np_buf = self.np_bitmaps[-1]
            local_np_bufs = self.np_shadow_bitmaps[-1]

            for thread_id in range(len(chunks)):

                global_np_buf |= local_np_bufs[thread_id]

                local_np_bufs[thread_id].fill(0)

            current_idx = end_idx

    def contains_batch(self, items: list) -> list[bool]:
        if not self.bitmaps: return [False] * len(items)

        if len(items) < self.min_chunk_size:
            return _thread_contains_chunk((self.layers_info, items))
        else:
            chunks = self._chunkify(items)
            jobs = [(self.layers_info, chunk) for chunk in chunks]

            results_2d = list(self.executor.map(_thread_contains_chunk, jobs))
            return [res for sublist in results_2d for res in sublist]

    def total_elements_count(self) -> int:
        return sum(self.elements_counts)