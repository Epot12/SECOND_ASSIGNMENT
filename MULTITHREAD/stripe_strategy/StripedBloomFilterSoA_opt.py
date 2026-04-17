import numpy as np
import mmh3
import math
from concurrent.futures import ThreadPoolExecutor


def _normalize_items(items):
    out = []
    for item in items:
        if isinstance(item, bytes):
            out.append(item)
        elif isinstance(item, str):
            out.append(item.encode('utf-8'))
        elif isinstance(item, int) or type(item).__name__.startswith('int'):
            out.append(int(item).to_bytes(8, 'big', signed=True))
        else:
            out.append(str(item).encode('utf-8'))
    return out

def _worker_add_striped_soa(args):
    stripe_buf, m_stripe, k, h1_arr, h2_arr = args
    if len(h1_arr) == 0:
        return

    for i in range(k):
        indices = (h1_arr + i * h2_arr) % m_stripe
        stripe_buf[indices] = 1


def _worker_contains_striped(args):
    stripes_per_layer, m_stripe_list, k_list, h1_arr, h2_arr, original_indices = args

    if len(h1_arr) == 0:
        return [], []

    n = len(h1_arr)
    results = np.zeros(n, dtype=bool)

    # LOOP INTERCHANGE
    for layer_idx, stripe_buf in enumerate(stripes_per_layer):
        k = k_list[layer_idx]
        m_stripe = m_stripe_list[layer_idx]

        # skip for already found elements
        active = np.where(~results)[0]
        if len(active) == 0:
            break

        h1_active = h1_arr[active]
        h2_active = h2_arr[active]

        # vectorized check
        found_mask = np.ones(len(active), dtype=bool)

        for i in range(k):
            indices = (h1_active + i * h2_active) % m_stripe
            found_mask &= (stripe_buf[indices] == 1)

            # local early exit
            if not found_mask.any():
                break

        # update global results
        results[active[found_mask]] = True

    return results.tolist(), original_indices


def _worker_compute_hashes(items_chunk):
    n = len(items_chunk)
    h1_arr = np.zeros(n, dtype=np.uint64)
    h2_arr = np.zeros(n, dtype=np.uint64)
    routing_arr = np.zeros(n, dtype=np.int32)

    for i, item in enumerate(items_chunk):
        h1, h2 = mmh3.hash64(item, seed=42, signed=False)
        h1_arr[i] = h1
        h2_arr[i] = h2
        routing_arr[i] = h1 % (2**31 - 1)

    return h1_arr, h2_arr, routing_arr



class StripedBloomFilterSoA:
    def __init__(self, initial_capacity: int, target_fp_rate: float, num_threads: int = 4):
        self.num_threads = num_threads
        self.initial_capacity = initial_capacity
        self.target_fp_rate = target_fp_rate
        self.p0 = target_fp_rate * 0.1

        self.layers = []
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self._add_new_layer()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()

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
        if not items:
            return

        items = _normalize_items(items)

        total_items = len(items)
        current_idx = 0

        while current_idx < total_items:
            layer = self.layers[-1]
            if layer['count'] >= layer['capacity']:
                self._add_new_layer()
                layer = self.layers[-1]

            available_space = layer['capacity'] - layer['count']
            end_idx = min(current_idx + available_space, total_items)
            items_chunk = items[current_idx:end_idx]

            # split
            chunk_size = math.ceil(len(items_chunk) / self.num_threads)
            item_chunks = [items_chunk[i:i + chunk_size] for i in range(0, len(items_chunk), chunk_size)]

            # unic hashing
            hash_results = list(self.executor.map(_worker_compute_hashes, item_chunks))

            h1_all = np.concatenate([r[0] for r in hash_results])
            h2_all = np.concatenate([r[1] for r in hash_results])
            routing_all = np.concatenate([r[2] for r in hash_results])

            stripe_ids = np.mod(routing_all, self.num_threads)

            buckets = [[] for _ in range(self.num_threads)]
            for idx, sid in enumerate(stripe_ids):
                buckets[sid].append(idx)

            jobs = []
            for thread_id in range(self.num_threads):
                idx = np.array(buckets[thread_id], dtype=np.int64)

                jobs.append((
                    layer['stripes'][thread_id],
                    layer['m_stripe'],
                    layer['k'],
                    h1_all[idx],
                    h2_all[idx]
                ))

            list(self.executor.map(_worker_add_striped_soa, jobs))

            layer['count'] += len(items_chunk)
            current_idx = end_idx

    def contains_batch(self, items: list) -> list[bool]:
        if not items:
            return []
        if not self.layers:
            return [False] * len(items)

        # normalizing one time
        items = _normalize_items(items)

        orig_idx_np = np.arange(len(items))

        # split
        chunk_size = math.ceil(len(items) / self.num_threads)
        item_chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        # HASH one time
        hash_results = list(self.executor.map(_worker_compute_hashes, item_chunks))

        h1_all = np.concatenate([r[0] for r in hash_results])
        h2_all = np.concatenate([r[1] for r in hash_results])
        routing_all = np.concatenate([r[2] for r in hash_results])

        stripe_ids = np.mod(routing_all, self.num_threads)

        jobs = []
        m_stripe_list = [l['m_stripe'] for l in self.layers]
        k_list = [l['k'] for l in self.layers]

        for i in range(self.num_threads):
            stripes_per_id = [l['stripes'][i] for l in self.layers]

            idx = np.nonzero(stripe_ids == i)[0]

            jobs.append((
                stripes_per_id,
                m_stripe_list,
                k_list,
                h1_all[idx],
                h2_all[idx],
                orig_idx_np[idx]
            ))

        worker_results = list(self.executor.map(_worker_contains_striped, jobs))

        final_results = [None] * len(items)

        for res_block, idx_block in worker_results:
            for r, idx in zip(res_block, idx_block):
                final_results[idx] = r

        return final_results

