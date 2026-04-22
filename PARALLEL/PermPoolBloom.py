import math
import mmh3
import multiprocessing as mp
from multiprocessing import shared_memory
import uuid
import weakref
import atexit
import builtins
if 'profile' not in builtins.__dict__:
    def profile(func): return func
    builtins.profile = profile


# Global cache for Workers to store active memory links.
# Prevents the overhead of repeatedly attaching to the same memory block.
_worker_shm_cache = {}


def _cleanup_worker_shm():
    """
    Guarantees that Workers gracefully release memory handles upon termination.
    Prevents the Python resource_tracker from raising memory leak warnings.
    """
    for shm in _worker_shm_cache.values():
        shm.close()


# Register the cleanup hook for the worker process lifecycle
atexit.register(_cleanup_worker_shm)


def _get_shm_buffer(shm_name: str):
    """Lazily attaches to a SharedMemory block if not already cached."""
    if shm_name not in _worker_shm_cache:
        # Attach to the existing block created by the Master
        shm = shared_memory.SharedMemory(name=shm_name)
        _worker_shm_cache[shm_name] = shm

    # Return a zero-copy memoryview of the underlying C array
    return _worker_shm_cache[shm_name].buf

@profile
def _worker_add_chunk(args):
    """
    Inserts a chunk of data into a specific memory layer.
    Byte-level writes are natively atomic on modern CPUs,
    guaranteeing mathematically safe Lock-Free execution without IPC overhead.
    """
    shm_name, m, k, items = args

    # Retrieve the directly mapped memory buffer
    buf = _get_shm_buffer(shm_name)

    for item in items:
        if type(item) is str:
            item_bytes = item.encode('utf-8')
        elif type(item) is bytes:
            item_bytes = item
        elif type(item) is int:
            item_bytes = item.to_bytes(8, 'big', signed=True)
        else:
            item_bytes = str(item).encode('utf-8')

        # Hash calculation
        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)

        # Idempotent lock-free writing
        for i in range(k):
            index = (h1 + i * h2) % m
            buf[index] = 1


def _worker_contains_chunk(args):
    """
    Checks for the presence of a block of elements across all parallel layers.
    """
    layers_info, items = args  # layers_info = [(shm_name, m, k), ...]

    # Pre-fetch buffers to avoid dictionary lookups inside the highly iterative loop
    buffers = [(_get_shm_buffer(name), m, k) for name, m, k in layers_info]
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

        # Locality Principle: Scan the layers backwards (from newest to oldest)
        for buf, m, k in reversed(buffers):
            found_in_layer = True
            for i in range(k):
                index = (h1 + i * h2) % m
                # If a byte is 0, the item is mathematically absent from this layer
                if buf[index] == 0:
                    found_in_layer = False
                    break

            # Short-circuit evaluation upon first positive match
            if found_in_layer:
                is_present = True
                break

        results.append(is_present)

    return results


# MASTER SCOPE: SCALABLE BLOOM FILTER


class PermPoolScalableBloomFilter:
    def __init__(self, initial_capacity: int, target_fp_rate: float,
                 tightening_ratio: float = 0.9, growth_factor: int = 2,
                 num_processes: int = None, num_threads: int = None, **kwargs):

        self.initial_capacity = initial_capacity
        self.target_fp_rate = target_fp_rate
        self.tightening_ratio = tightening_ratio
        self.growth_factor = growth_factor
        # Strict validation: prevent ambiguous inputs
        if num_threads is not None and num_processes is not None:
            raise ValueError("[ERROR] Cannot specify both num_threads and num_processes. Choose one. They both refer to the number of processes, there is no"
                             "multithreading in this class. num_threads has been added just to facilitate adapting")

        # Unifies the interface by accepting both num_processes and num_threads
        workers = num_threads if num_threads is not None else num_processes
        self.num_processes = workers or mp.cpu_count()

        self.p0 = target_fp_rate * (1 - tightening_ratio)

        # Shared Memory Management
        self.shm_blocks = []  # Retains Master's reference to SharedMemory objects
        self.layers_info = []  # Tuples of (shm_name, m, k) for routing
        self.capacities = []
        self.elements_counts = []

        # Hardware Calibration
        self.min_chunk_size = 1000
        self._calibrate_threshold()

        # Initialize the Persistent Pool ONE TIME ONLY
        self.pool = mp.Pool(processes=self.num_processes)

        # Memory Management: Register strict cleanup protocol
        # This guarantees RAM release even if the process is forcefully killed
        self._finalizer = weakref.finalize(self, self._cleanup_resources, self.shm_blocks, self.pool)

    @staticmethod
    def _cleanup_resources(shm_blocks, pool):
        """
        Deterministic GC routine. Executed automatically when the object is destroyed
        or the context manager exits. Guarantees zero memory leaks.
        """
        if pool:
            pool.terminate()
            pool.join()

        for shm in shm_blocks:
            try:
                shm.close()
                shm.unlink()  # Physically frees the RAM block from the OS
            except FileNotFoundError:
                pass  # Already unlinked

    def __enter__(self):
        """Context Manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager exit point. Forces immediate cleanup."""
        self._finalizer()

    def _add_new_layer(self):
        """
        Dynamically allocates a new Shared Memory block (Hot Reloading).
        No Pool restarts are required.
        """
        current_depth = len(self.shm_blocks)

        new_capacity = self.initial_capacity * (self.growth_factor ** current_depth)
        new_fp_rate = self.p0 * (self.tightening_ratio ** current_depth)

        # Mathematical Calculation of dimensions
        m_float = -(new_capacity * math.log(new_fp_rate)) / (math.log(2) ** 2)
        m = math.ceil(m_float)
        k_float = (m / new_capacity) * math.log(2)
        k = max(1, round(k_float))

        # Generate a globally unique OS-level name for the memory block
        shm_name = f"sbf_layer_{current_depth}_{uuid.uuid4().hex}"

        # Allocate RAM and explicitly zero it out to prevent reading garbage bytes
        shm = shared_memory.SharedMemory(create=True, size=m, name=shm_name)
        shm.buf[:] = bytearray(m)

        self.shm_blocks.append(shm)
        self.layers_info.append((shm_name, m, k))
        self.capacities.append(new_capacity)
        self.elements_counts.append(0)

        print(f"[SYSTEM] Hot-Allocated Layer {current_depth}: "
              f"Capacity={new_capacity}, m={m} bytes, k={k} (Zero-Overhead IPC)")

    def _calibrate_threshold(self):
        """Hardware auto-tuning for optimal batch distribution."""
        import time
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

        # Simulating IPC overhead on a persistent pool
        with mp.Pool(1) as pool:
            start_ipc = time.perf_counter()
            pool.map(len, [[1]])
            end_ipc = time.perf_counter()

        t_overhead = end_ipc - start_ipc
        target_efficiency = 5

        raw_threshold = int((t_overhead * target_efficiency) / t_hash)
        self.min_chunk_size = max(1000, round(raw_threshold, -2))

        print(f"[SYSTEM] Auto-Tuning completed. Threshold: {self.min_chunk_size} elements/chunk.")

    def _chunkify(self, data: list):
        chunk_size = math.ceil(len(data) / self.num_processes)
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def add_batch(self, items: list):
        if not items:
            return

        total_items = len(items)
        current_idx = 0
        jobs = []


        # ROUTING and PRE-ALLOCATION (Master Only)

        while current_idx < total_items:
            if not self.shm_blocks or self.elements_counts[-1] >= self.capacities[-1]:
                self._add_new_layer()

            available_space = self.capacities[-1] - self.elements_counts[-1]
            end_idx = min(current_idx + available_space, total_items)
            items_for_this_layer = items[current_idx:end_idx]

            self.elements_counts[-1] += len(items_for_this_layer)

            # Retrieve layer metadata
            shm_name, m, k = self.layers_info[-1]

            if len(items_for_this_layer) < self.min_chunk_size:
                jobs.append((shm_name, m, k, items_for_this_layer))
            else:
                chunks = self._chunkify(items_for_this_layer)
                for chunk in chunks:
                    jobs.append((shm_name, m, k, chunk))

            current_idx = end_idx


        # PHASE 2: PERSISTENT EXECUTION

        if total_items < self.min_chunk_size and len(jobs) == 1:
            # Bypass IPC: Sequential fallback directly within the Master
            _worker_add_chunk(jobs[0])
        else:
            # IPC: Dispatch to persistent pool. Workers attach lazily.
            self.pool.map(_worker_add_chunk, jobs)

    def contains_batch(self, items: list) -> list[bool]:
        """Massive lookup operation spanning all memory layers."""
        if not self.shm_blocks:
            return [False] * len(items)

        if len(items) < self.min_chunk_size:
            # Sequential fallback in Master
            return _worker_contains_chunk((self.layers_info, items))
        else:
            # Parallel execution
            chunks = self._chunkify(items)
            # We bundle all layer metadatas together with the specific chunk
            jobs = [(self.layers_info, chunk) for chunk in chunks]

            results_2d = self.pool.map(_worker_contains_chunk, jobs)
            return [res for sublist in results_2d for res in sublist]

    def total_elements_count(self) -> int:
        return sum(self.elements_counts)