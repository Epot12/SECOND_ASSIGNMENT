import os
import math
import mmh3
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed



# WORKER FUNCTIONS (Must be at the top level for Joblib 'loky' serialization)


def _joblib_add_chunk(m: int, k: int, items: list) -> np.ndarray:
    """
    MAP PHASE: Each worker receives a chunk of items and creates a local
    "shadow" byte array. It computes the hashes and sets the bits locally.
    """
    # Using numpy uint8 for fast bitwise OR operations later and efficient Joblib pickling
    local_buf = np.zeros(m, dtype=np.uint8)

    for item in items:
        # Normalization
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
            local_buf[index] = 1

    return local_buf


def _joblib_contains_chunk(layers_info: list, items: list) -> list[bool]:
    """
    Lookup phase for a chunk of items across all existing Bloom Filter layers.
    layers_info is a list of tuples: (numpy_buffer, m, k)
    """
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

        # Locality Principle: check newest layers first
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



# MASTER CLASS


class JoblibScalableBloomFilter:
    """
    A Map-Reduce implementation of a Scalable Bloom Filter using Joblib.
    Designed specifically to benchmark IPC overhead and serialization bottlenecks.
    """

    def __init__(self, initial_capacity: int, target_fp_rate: float,
                 tightening_ratio: float = 0.9, growth_factor: int = 2,
                 n_jobs: int = -1, backend: str = 'loky', num_threads: int = None, **kwargs):

        self.initial_capacity = initial_capacity
        self.target_fp_rate = target_fp_rate
        self.tightening_ratio = tightening_ratio
        self.growth_factor = growth_factor
        if num_threads is not None and n_jobs != -1:
            raise ValueError(
                "[ERROR] Cannot specify both num_threads and n_jobs. Choose one. They both refer to the number of processes, there is no"
                "multithreading in this class. num_threads has been added just to facilitate adapting")

        # Unifies the interface by accepting both n_jobs and num_threads
        workers = num_threads if num_threads is not None else n_jobs
        self.n_jobs = workers or mp.cpu_count()
        self.backend = backend

        self.p0 = target_fp_rate * (1 - tightening_ratio)

        # Internal state
        self.bitmaps = []  # List of numpy arrays
        self.layers_info = []  # Tuples of (buf, m, k)
        self.capacities = []
        self.elements_counts = []

        # Context manager to reuse the pool
        self.parallel_pool = Parallel(n_jobs=self.n_jobs, backend=self.backend, batch_size='auto')

    def __enter__(self):
        """Activates the Joblib persistent pool to avoid start/stop overhead."""
        self.parallel_pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Terminates the Joblib pool."""
        self.parallel_pool.__exit__(exc_type, exc_val, exc_tb)

    def _add_new_layer(self):
        """Allocates a new scaling layer."""
        current_depth = len(self.bitmaps)
        new_capacity = self.initial_capacity * (self.growth_factor ** current_depth)
        new_fp_rate = self.p0 * (self.tightening_ratio ** current_depth)

        m = math.ceil(-(new_capacity * math.log(new_fp_rate)) / (math.log(2) ** 2))
        k = max(1, round((m / new_capacity) * math.log(2)))

        # Using numpy array for fast bitwise operations during the Reduce phase
        buf = np.zeros(m, dtype=np.uint8)

        self.bitmaps.append(buf)
        self.layers_info.append((buf, m, k))
        self.capacities.append(new_capacity)
        self.elements_counts.append(0)

        print(f"[JOBLIB-SYSTEM] Added new layer. Capacity: {new_capacity}, m: {m} bytes")

    def _chunkify(self, data: list, n_chunks: int) -> list:
        """Helper method to split data into balanced chunks for workers."""
        if not data: return []
        chunk_size = math.ceil(len(data) / n_chunks)
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def add_batch(self, items: list):
        """
        Joblib requires batch processing. Inserting one element at a time
        would spawn a process per element, crashing the system.
        """
        if not items: return
        total_items = len(items)
        current_idx = 0

        # Determine optimal chunking
        actual_jobs = os.cpu_count() if self.n_jobs == -1 else self.n_jobs

        while current_idx < total_items:
            if not self.bitmaps or self.elements_counts[-1] >= self.capacities[-1]:
                self._add_new_layer()

            available_space = self.capacities[-1] - self.elements_counts[-1]
            end_idx = min(current_idx + available_space, total_items)
            items_for_this_layer = items[current_idx:end_idx]

            self.elements_counts[-1] += len(items_for_this_layer)
            buf, m, k = self.layers_info[-1]

            # MAP PHASE: Dispatch chunks to Joblib workers

            chunks = self._chunkify(items_for_this_layer, actual_jobs)

            # Returns a list of local numpy arrays from the workers
            shadow_arrays = self.parallel_pool(
                delayed(_joblib_add_chunk)(m, k, chunk) for chunk in chunks
            )

            # REDUCE PHASE: Merge shadow arrays into the main layer

            # using Numpy's fast bitwise OR to merge the returned buffers
            for shadow_buf in shadow_arrays:
                self.bitmaps[-1] |= shadow_buf

            current_idx = end_idx

    def contains_batch(self, items: list) -> list[bool]:
        """Parallel lookup of elements."""
        if not self.bitmaps:
            return [False] * len(items)

        actual_jobs = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
        chunks = self._chunkify(items, actual_jobs)

        # Dispatch chunks to workers. Pass a copy of layers_info.
        results_2d = self.parallel_pool(
            delayed(_joblib_contains_chunk)(self.layers_info, chunk) for chunk in chunks
        )

        # Flatten the 2D list of results
        return [res for sublist in results_2d for res in sublist]

    def total_elements_count(self) -> int:
        return sum(self.elements_counts)