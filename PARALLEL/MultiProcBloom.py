import math
import mmh3
import multiprocessing as mp
import ctypes


shared_bitmap = None
shared_elements_count = None
shared_lock = None


def worker_init(bitmap, elements_count, lock):
    """Initializes shared memory references for each Worker process."""
    global shared_bitmap, shared_elements_count, shared_lock
    shared_bitmap = bitmap
    shared_elements_count = elements_count
    shared_lock = lock


def _worker_add_chunk(args):
    """The worker inserts a block of data without intermediate counters."""
    items, k, m = args
    local_elements = len(items)

    # Copying global reference in a local variable

    _bitmap = shared_bitmap

    for item in items:
        if type(item) is str:
            item_bytes = item.encode('utf-8')
        elif type(item) is bytes:
            item_bytes = item
        elif type(item) is int:
            item_bytes = item.to_bytes(8, 'big', signed=True)
        else:
            item_bytes = str(item).encode('utf-8')

        # calculating hash
        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)

        for i in range(k):
            index = (h1 + i * h2) % m

            _bitmap[index] = 1

    with shared_lock:
        shared_elements_count.value += local_elements


def _worker_contains_chunk(args):
    """Tests a block of elements in parallel."""
    items, k, m = args
    results = []
    _bitmap = shared_bitmap
    for item in items:
        if type(item) is str:
            item_bytes = item.encode('utf-8')
        elif type(item) is bytes:
            item_bytes = item
        elif type(item) is int:
            # Converts the integer to 8 native bytes. No string created.
            item_bytes = item.to_bytes(8, 'big', signed=True)
        else:
            # Fallback
            item_bytes = str(item).encode('utf-8')

        # Calculating hash
        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)
        is_present = True
        for i in range(k):
            index = (h1 + i * h2) % m

            # reading memory. If even one bit is 0, the element is NOT there.
            if _bitmap[index] == 0:
                is_present = False
                break

        # saving result
        results.append(is_present)
    return results

# ==========================================
# PARALLEL BLOOM FILTER
# ==========================================

class ParallelBloomFilter:
    def __init__(self, expected_elements: int, false_positive_rate: float, num_processes: int = None, num_threads: int = None):
        self.n = expected_elements
        self.p = false_positive_rate
        self.m, self.k = self._calculate_params()

        # Shared array (ctypes.c_byte occupies 1 byte per cell)
        self.bitmap = mp.Array(ctypes.c_byte, self.m, lock=False)

        # Unique element counter
        self.elements_count = mp.Value('i', 0)
        self.lock = mp.Lock()

        if num_threads is not None and num_processes is not None:
            raise ValueError(
                "[ERROR] Cannot specify both num_threads and num_processes. Choose one. They both refer to the number of processes, there is no"
                "multithreading in this class. num_threads has been added just to facilitate adapting")

        # Unifies the interface by accepting both num_processes and num_threads
        workers = num_threads if num_threads is not None else num_processes
        self.num_processes = workers or mp.cpu_count()

    def _calculate_params(self) -> tuple[int, int]:
        m_float = -(self.n * math.log(self.p)) / (math.log(2) ** 2)
        m = math.ceil(m_float)
        k_float = (m / self.n) * math.log(2)
        k = max(1, round(k_float))
        return m, k

    def _chunkify(self, data: list):
        chunk_size = math.ceil(len(data) / self.num_processes)
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def add_batch(self, items: list):
        """Performs parallel insertion without managing bit counters."""
        chunks = self._chunkify(items)
        args = [(chunk, self.k, self.m) for chunk in chunks]

        with mp.Pool(processes=self.num_processes, # TODO: valutare se lasciare la gestione dei processi così oppure far sì che i processi rimangano aperti in background per tutta la vita dell'istanza della classe
                     initializer=worker_init,
                     initargs=(self.bitmap, self.elements_count, self.lock)) as pool:
            pool.map(_worker_add_chunk, args)

    def contains_batch(self, items: list) -> list[bool]:
        chunks = self._chunkify(items)
        args = [(chunk, self.k, self.m) for chunk in chunks]

        with mp.Pool(processes=self.num_processes,
                     initializer=worker_init,
                     initargs=(self.bitmap, self.elements_count, self.lock)) as pool:
            results_2d = pool.map(_worker_contains_chunk, args)

        return [res for sublist in results_2d for res in sublist]

    def count_set_bits(self) -> int:
        """
        Counts the 1's physically present in RAM.
        The C array is cast into a Python bytearray, using the .count() method
        implemented natively in C.
        """

        return bytearray(self.bitmap).count(1)

    def get_fill_ratio(self) -> float:
        """Returns the exact fill ratio based on memory status."""
        return self.count_set_bits() / self.m

    def get_actual_fp_rate(self) -> float:
        """Calculate the real FP rate without race condition risks."""
        return (self.count_set_bits() / self.m) ** self.k