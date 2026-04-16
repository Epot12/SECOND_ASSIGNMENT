import numpy as np
import mmh3
import math
from concurrent.futures import ThreadPoolExecutor


def _worker_add_striped_soa(args):
    """Test Pattern: Structure of Arrays (SoA) with SIMD"""
    stripe_buf, m_stripe, k, items = args
    if len(items) == 0: return

    num_items = len(items)

    # creation of SoA structures
    h1_arr = np.zeros(num_items, dtype=np.uint64)
    h2_arr = np.zeros(num_items, dtype=np.uint64)

    for idx, item in enumerate(items):
        if type(item) is str:
            item_bytes = item.encode('utf-8')
        elif type(item) is bytes:
            item_bytes = item
        elif type(item) is int or type(item).__name__.startswith('int'):
            item_bytes = int(item).to_bytes(8, 'big', signed=True)
        else:
            item_bytes = str(item).encode('utf-8')

        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)
        h1_arr[idx] = h1
        h2_arr[idx] = h2

    # SIMD vectorization
    for i in range(k):
        # Calculating all indices for the nth hash function in one step
        indices = (h1_arr + i * h2_arr) % m_stripe

        # vectorized writing
        stripe_buf[indices] = 1


def _worker_contains_striped(args):
    """Performs a massive search in the corresponding stripes of all layers."""
    stripes_per_layer, m_stripe_list, k_list, items, original_indices = args
    if len(items) == 0: return [], []

    results = []
    for item in items:
        if type(item) is str:
            item_bytes = item.encode('utf-8')
        elif type(item) is bytes:
            item_bytes = item
        elif type(item) is int or type(item).__name__.startswith('int'):
            item_bytes = int(item).to_bytes(8, 'big', signed=True)
        else:
            item_bytes = str(item).encode('utf-8')

        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)
        is_present = False

        # layer scanning (Locality Principle)
        for layer_idx, stripe_buf in enumerate(stripes_per_layer):
            k = k_list[layer_idx]
            m_stripe = m_stripe_list[layer_idx] # dynamic dimension for layer

            found_in_layer = True
            for i in range(k):
                if stripe_buf[(h1 + i * h2) % m_stripe] == 0:
                    found_in_layer = False
                    break
            if found_in_layer:
                is_present = True
                break
        results.append(is_present)

    return results, original_indices


def _worker_calc_routing_hashes(items_chunk):
    """Computes routing hashes in parallel leveraging No-GIL."""
    return np.array([mmh3.hash(x, seed=0) for x in items_chunk], dtype=np.int32)



class StripedBloomFilterSoA:
    def __init__(self, initial_capacity: int, target_fp_rate: float, num_threads: int = 4):
        self.num_threads = num_threads
        self.initial_capacity = initial_capacity
        self.target_fp_rate = target_fp_rate
        self.p0 = target_fp_rate * 0.1

        self.layers = []
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self._add_new_layer()

    def _add_new_layer(self):
        depth = len(self.layers)
        cap = self.initial_capacity * (2 ** depth)
        m = math.ceil(-(cap * math.log(self.p0)) / (math.log(2) ** 2))
        k = max(1, round((m / cap) * math.log(2)))

        m_stripe = math.ceil(m / self.num_threads)
        stripes = [np.zeros(m_stripe, dtype=np.uint8) for _ in range(self.num_threads)]

        self.layers.append({
            'stripes': stripes, 'm_stripe': m_stripe, 'k': k,
            'capacity': cap, 'count': 0
        })

    def add_batch(self, items: list):
        if not items: return
        total_items = len(items)
        current_idx = 0

        while current_idx < total_items:
            layer = self.layers[-1]
            if layer['count'] >= layer['capacity']:
                self._add_new_layer()
                layer = self.layers[-1]

            available_space = layer['capacity'] - layer['count']
            end_idx = min(current_idx + available_space, total_items)
            items_for_this_layer = items[current_idx:end_idx]

            chunk_size = math.ceil(len(items_for_this_layer) / self.num_threads)
            item_chunks = [items_for_this_layer[i:i + chunk_size] for i in range(0, len(items_for_this_layer), chunk_size)]

            hash_chunks = list(self.executor.map(_worker_calc_routing_hashes, item_chunks))

            routing_hashes = np.concatenate(hash_chunks)
            stripe_ids = np.mod(routing_hashes, self.num_threads)

            jobs = []
            for i in range(self.num_threads):
                stripe_indices = np.nonzero(stripe_ids == i)[0]
                items_for_stripe = [items_for_this_layer[idx] for idx in stripe_indices]
                jobs.append((layer['stripes'][i], layer['m_stripe'], layer['k'], items_for_stripe))

            # SoA worker
            list(self.executor.map(_worker_add_striped_soa, jobs))

            layer['count'] += len(items_for_this_layer)
            current_idx = end_idx

    # this is the same of Row-Major version
    def contains_batch(self, items: list) -> list[bool]:
        if not items: return []
        if not self.layers: return [False] * len(items)

        orig_idx_np = np.arange(len(items))

        chunk_size = math.ceil(len(items) / self.num_threads)
        item_chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        hash_chunks = list(self.executor.map(_worker_calc_routing_hashes, item_chunks))
        routing_hashes = np.concatenate(hash_chunks)
        stripe_ids = np.mod(routing_hashes, self.num_threads)

        jobs = []
        m_stripe_list = [l['m_stripe'] for l in self.layers]
        k_list = [l['k'] for l in self.layers]

        for i in range(self.num_threads):
            stripes_per_id = [l['stripes'][i] for l in self.layers]
            stripe_indices = np.nonzero(stripe_ids == i)[0]
            items_for_stripe = [items[idx] for idx in stripe_indices]
            orig_idx_for_stripe = orig_idx_np[stripe_indices]

            jobs.append((stripes_per_id, m_stripe_list, k_list, items_for_stripe, orig_idx_for_stripe))

        worker_results = list(self.executor.map(_worker_contains_striped, jobs))

        final_results = [None] * len(items)
        for res_block, idx_block in worker_results:
            for r, idx in zip(res_block, idx_block):
                final_results[idx] = r

        return final_results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()