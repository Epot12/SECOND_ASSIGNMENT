import math
import mmh3
import multiprocessing as mp
import ctypes

# GLOBAL WORKER VARIABLES (Pre-initialized by the Master)

shared_bitmaps = []
shared_params = []  # tuple list: [(m0, k0), (m1, k1), ...]


def worker_init(bitmaps, params):
    """
    Inserts shared memory maps and parameters into the global scope
    of the Workers only once when the Pool is started.
    """
    global shared_bitmaps, shared_params
    shared_bitmaps = bitmaps
    shared_params = params


def _worker_add_chunk(args):
    """
    Inserts a chunk of data into a specific filter level.
    TOTALLY LOCK-FREE
    """
    layer_idx, items = args

    # Local Caching
    _bitmap = shared_bitmaps[layer_idx]
    m, k = shared_params[layer_idx]

    for item in items:
        # Casting
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

        # Idempotent writing
        for i in range(k):
            index = (h1 + i * h2) % m
            _bitmap[index] = 1


def _worker_contains_chunk(items):
    """
    Checks for a block of elements in all parallel layers.
    """
    results = []

    # Local Caching
    _bitmaps = shared_bitmaps
    _params = shared_params
    num_layers = len(_bitmaps)

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

        # Locality Principle.
        # scanning the layers backwards (from newest to oldest).
        for layer_idx in range(num_layers - 1, -1, -1):
            m, k = _params[layer_idx]
            bitmap = _bitmaps[layer_idx]

            found_in_layer = True
            for i in range(k):
                index = (h1 + i * h2) % m
                # If a bit is 0, it is not in THIS level
                if bitmap[index] == 0:
                    found_in_layer = False
                    break

            # If we found it in this level, we stop the search
            # for this element and let's move on to the next one.
            if found_in_layer:
                is_present = True
                break

        results.append(is_present)

    return results


# MASTER: PARALLEL SCALABLE BLOOM FILTER (Optimized with Lazy Restart)


class ParallelScalableBloomFilter:
    def __init__(self, initial_capacity: int, target_fp_rate: float,
                 tightening_ratio: float = 0.9, growth_factor: int = 2,
                 num_processes: int = None, num_threads: int = None):

        self.initial_capacity = initial_capacity
        self.target_fp_rate = target_fp_rate
        self.tightening_ratio = tightening_ratio
        self.growth_factor = growth_factor
        if num_threads is not None and num_processes is not None:
            raise ValueError(
                "[ERROR] Cannot specify both num_threads and num_processes. Choose one. They both refer to the number of processes, there is no"
                "multithreading in this class. num_threads has been added just to facilitate adapting")

        # Unifies the interface by accepting both num_processes and num_threads
        workers = num_threads if num_threads is not None else num_processes
        self.num_processes = workers or mp.cpu_count()

        self.p0 = target_fp_rate * (1 - tightening_ratio)

        # Level Management (Parallel Arrays)
        self.bitmaps = []  # List of mp.Arrays
        self.params = []  # List of tuples (m, k)
        self.capacities = []  # List of abilities (n) of each level
        self.elements_counts = []  # Counters purely managed by the Master (no Lock)
        self.min_chunk_size = 1000  # fallback value

        self.pool = None
        self._calibrate_threshold()


    # CONTEXT MANAGER

    def _restart_pool(self):

        if self.pool is not None:
            self.pool.close()
            self.pool.join()

        self.pool = mp.Pool(processes=self.num_processes,
                            initializer=worker_init,
                            initargs=(self.bitmaps, self.params))

    def __enter__(self):

        self._restart_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.pool is not None:
            if exc_type is not None:
                self.pool.terminate()
            else:
                self.pool.close()
            self.pool.join()
            self.pool = None



    def _add_new_layer(self):
        """Allocate memory for a new level (Performed only by the Master)"""
        current_depth = len(self.bitmaps)

        new_capacity = self.initial_capacity * (self.growth_factor ** current_depth)
        new_fp_rate = self.p0 * (self.tightening_ratio ** current_depth)

        # Calculation m, k
        m_float = -(new_capacity * math.log(new_fp_rate)) / (math.log(2) ** 2)
        m = math.ceil(m_float)
        k_float = (m / new_capacity) * math.log(2)
        k = max(1, round(k_float))

        # Shared Memory Allocation
        new_bitmap = mp.Array(ctypes.c_byte, m, lock=False)

        self.bitmaps.append(new_bitmap)
        self.params.append((m, k))
        self.capacities.append(new_capacity)
        self.elements_counts.append(0)

        print(f"[SYSTEM] Allocated Level {current_depth}: "
              f"Capacity={new_capacity}, m={m} bits, k={k}")

        # LAZY RESTART
        if self.pool is not None:
            print("[SYSTEM] Restarting Pool to sync new memory layer with Workers...")
            self._restart_pool()

    def _calibrate_threshold(self):
        """
        Runs a micro-benchmark at startup to calculate the optimal threshold
        based on current hardware
        """
        import time
        print("[SYSTEM] Hardware calibration ")

        # pure computation time of a single element (T_hash)
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

        # measurement of the communication overhead (T_overhead)
        with mp.Pool(1) as pool:
            start_ipc = time.perf_counter()
            pool.map(len, [[1]])
            end_ipc = time.perf_counter()

        t_overhead = end_ipc - start_ipc
        target_efficiency = 5

        raw_threshold = int((t_overhead * target_efficiency) / t_hash)
        self.min_chunk_size = max(1000, round(raw_threshold, -2))

        print(f"[SYSTEM] Calibration completed:")
        print(f"         - Single hash cost: {t_hash * 1e6:.4f} µs")
        print(f"         - Measured overhead: {t_overhead * 1000:.2f} ms")
        print(f"         - Calculated threshold: {self.min_chunk_size} elements per chunk")

    def _chunkify(self, data: list):
        chunk_size = math.ceil(len(data) / self.num_processes)
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def add_batch(self, items: list):
        if not items:
            return

        total_items = len(items)
        current_idx = 0
        jobs = []

        # Job creation
        while current_idx < total_items:
            if not self.bitmaps or self.elements_counts[-1] >= self.capacities[-1]:
                self._add_new_layer()

            active_layer_idx = len(self.bitmaps) - 1
            available_space = self.capacities[-1] - self.elements_counts[-1]

            end_idx = min(current_idx + available_space, total_items)
            items_for_this_layer = items[current_idx:end_idx]

            self.elements_counts[-1] += len(items_for_this_layer)

            if len(items_for_this_layer) < self.min_chunk_size:
                jobs.append((active_layer_idx, items_for_this_layer))
            else:
                chunks = self._chunkify(items_for_this_layer)
                for chunk in chunks:
                    jobs.append((active_layer_idx, chunk))

            current_idx = end_idx

        # EXECUTION
        if total_items < self.min_chunk_size and len(jobs) == 1:
            print(f"[ROUTER] Small Batch ({len(items)} < {self.min_chunk_size}) -> Sequential Execution")
            worker_init(self.bitmaps, self.params)
            _worker_add_chunk(jobs[0])
        else:
            print(f"[ROUTER] Parallel Execution triggered")

            self.pool.map(_worker_add_chunk, jobs)

    def contains_batch(self, items: list) -> list[bool]:
        """Massive search using all levels"""
        if not self.bitmaps:
            return [False] * len(items)

        # check of the threshold
        if len(items) < self.min_chunk_size:
            print(f"[ROUTER] Small Batch ({len(items)} < {self.min_chunk_size}) -> Sequential Execution")
            worker_init(self.bitmaps, self.params)
            return _worker_contains_chunk(items)
        else:
            print(f"[ROUTER] Parallel Execution triggered")
            chunks = self._chunkify(items)


            results_2d = self.pool.map(_worker_contains_chunk, chunks)

            # flattening the list of lists returned by the workers
            return [res for sublist in results_2d for res in sublist]

    def total_elements_count(self) -> int:
        """Returns the mathematical total of the sorted items."""
        return sum(self.elements_counts)