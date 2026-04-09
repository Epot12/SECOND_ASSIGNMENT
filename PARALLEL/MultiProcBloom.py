import math
import mmh3
import multiprocessing as mp
import ctypes

# On Windows, "child" processes must be able to import these functions.
# Here it is used an "initializer" to hook shared memory without
# having to copy it or pass it as an argument every time (avoiding overhead).

shared_bitmap = None
shared_elements_count = None
shared_set_bits = None
shared_lock = None


def worker_init(bitmap, elements_count, set_bits, lock):
    """Initializes global variables for each Worker process."""
    global shared_bitmap, shared_elements_count, shared_set_bits, shared_lock
    shared_bitmap = bitmap
    shared_elements_count = elements_count
    shared_set_bits = set_bits
    shared_lock = lock


def _worker_add_chunk(args):
    """The worker receives a block of data and calculates the hashes."""
    items, k, m = args
    local_set_bits = 0
    local_elements = len(items)

    for item in items:
        # bytes conversion
        item_bytes = item.encode('utf-8') if isinstance(item, str) else str(item).encode('utf-8')
        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)

        for i in range(k):
            index = (h1 + i * h2) % m

            # Writing to the Shared Array.
            # Here the lock is omitted to maximize speed.
            # In the worst case Race Condition, a 1 overrides a 1.
            if shared_bitmap[index] == 0:
                shared_bitmap[index] = 1
                local_set_bits += 1

    # global counters are updated using lock
    # possible race conditons here. TODO: modify to avoid any race condition
    with shared_lock:
        shared_elements_count.value += local_elements
        shared_set_bits.value += local_set_bits


def _worker_contains_chunk(args):
    """Tests a block of elements in parallel."""
    items, k, m = args
    results = []

    for item in items:
        item_bytes = item.encode('utf-8') if isinstance(item, str) else str(item).encode('utf-8')
        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)

        is_present = True
        for i in range(k):
            index = (h1 + i * h2) % m
            if shared_bitmap[index] == 0:
                is_present = False
                break
        results.append(is_present)

    return results


# ==========================================
# PARALLEL BLOOM FILTER
# ==========================================

class ParallelBloomFilter:
    def __init__(self, expected_elements: int, false_positive_rate: float, num_processes: int = None):
        self.n = expected_elements
        self.p = false_positive_rate
        self.m, self.k = self._calculate_params()

        # Shared Memory Array (Type 'b' = boolean/byte, lock=False for performance)
        self.bitmap = mp.Array(ctypes.c_byte, self.m, lock=False)

        # Shared counters protected by Lock
        self.elements_count = mp.Value('i', 0)
        self.set_bits = mp.Value('i', 0)
        self.lock = mp.Lock()

        # Number of Processes: Default = all available cores
        self.num_processes = num_processes or mp.cpu_count()

    def _calculate_params(self) -> tuple[int, int]:
        m_float = -(self.n * math.log(self.p)) / (math.log(2) ** 2)
        m = math.ceil(m_float)
        k_float = (m / self.n) * math.log(2)
        k = max(1, round(k_float))
        return m, k

    def _chunkify(self, data: list):
        """Divides data into evenly distributed blocks for processes."""
        chunk_size = math.ceil(len(data) / self.num_processes)
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def add_batch(self, items: list):
        """Inserts a list of items using all CPU cores."""
        chunks = self._chunkify(items)
        args = [(chunk, self.k, self.m) for chunk in chunks]

        # creating a Pool of processes that share memory
        with mp.Pool(processes=self.num_processes,
                     initializer=worker_init,
                     initargs=(self.bitmap, self.elements_count, self.set_bits, self.lock)) as pool:
            pool.map(_worker_add_chunk, args)

    def contains_batch(self, items: list) -> list[bool]:
        """Tests a list of elements and returns a list of booleans."""
        chunks = self._chunkify(items)
        args = [(chunk, self.k, self.m) for chunk in chunks]

        with mp.Pool(processes=self.num_processes,
                     initializer=worker_init,
                     initargs=(self.bitmap, self.elements_count, self.set_bits, self.lock)) as pool:
            # Prints the partial results of each worker in a flat list
            results_2d = pool.map(_worker_contains_chunk, args)

        return [res for sublist in results_2d for res in sublist]

    def get_fill_ratio(self) -> float:
        with self.lock:
            return self.set_bits.value / self.m

    def get_actual_fp_rate(self) -> float:
        with self.lock:
            return (self.set_bits.value / self.m) ** self.k