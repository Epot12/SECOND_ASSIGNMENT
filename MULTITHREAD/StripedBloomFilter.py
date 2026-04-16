import math
import mmh3
import numpy as np
from concurrent.futures import ThreadPoolExecutor



# VECTORIZED WORKERS


def _worker_add_striped(args):
    """Performs massive writing in the relevant stripe."""
    stripe_buf, m_stripe, k, items = args
    if items.size == 0: return

    indices = []
    for item in items:
        h1, h2 = mmh3.hash64(item, seed=42, signed=False)
        for i in range(k):
            indices.append((h1 + i * h2) % m_stripe)

    # writing using NumPy
    np_indices = np.array(indices, dtype=np.uint64)
    stripe_buf[np_indices] = 1


def _worker_contains_striped(args):
    """Performs a massive search in the corresponding stripes of all layers."""
    stripes_per_layer, m_stripe, k_list, items, original_indices = args
    if items.size == 0: return [], []

    results = []
    for item in items:
        h1, h2 = mmh3.hash64(item, seed=42, signed=False)
        is_present = False

        # layer scanning (Locality Principle)
        for layer_idx, stripe_buf in enumerate(stripes_per_layer):
            k = k_list[layer_idx]
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



# class


class UltimateStripedBloomFilter:
    def __init__(self, initial_capacity: int, target_fp_rate: float, num_threads: int = 4):
        self.num_threads = num_threads  # Ottimizzato per 4 core fisici
        self.initial_capacity = initial_capacity
        self.target_fp_rate = target_fp_rate
        self.p0 = target_fp_rate * 0.1

        self.layers = []
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self._add_new_layer()

    def _add_new_layer(self):
        depth = len(self.layers)
        cap = self.initial_capacity * (2 ** depth)
        # Calcolo matematico dimensioni layer
        m = math.ceil(-(cap * math.log(self.p0)) / (math.log(2) ** 2))
        k = max(1, round((m / cap) * math.log(2)))

        m_stripe = math.ceil(m / self.num_threads)
        # Stripes indipendenti: entrano nella Cache L2!
        stripes = [np.zeros(m_stripe, dtype=np.uint8) for _ in range(self.num_threads)]

        self.layers.append({
            'stripes': stripes, 'm_stripe': m_stripe, 'k': k,
            'capacity': cap, 'count': 0
        })
        print(f"[SYSTEM] Layer {depth} Allocated. Stripe: {m_stripe / 1024:.1f} KB (Cache-Friendly)")

    def add_batch(self, items: list):
        layer = self.layers[-1]
        if layer['count'] >= layer['capacity']:
            self._add_new_layer()
            layer = self.layers[-1]

        # VETTORIZZAZIONE ROUTING
        items_np = np.array(items)
        routing_hashes = np.array([mmh3.hash(x, seed=0) for x in items], dtype=np.int32)
        stripe_ids = np.mod(routing_hashes, self.num_threads)

        jobs = []
        for i in range(self.num_threads):
            jobs.append((layer['stripes'][i], layer['m_stripe'], layer['k'], items_np[stripe_ids == i]))

        list(self.executor.map(_worker_add_striped, jobs))
        layer['count'] += len(items)

    def contains_batch(self, items: list) -> list[bool]:
        items_np = np.array(items)
        # Creiamo un array di indici per ricostruire l'ordine alla fine
        orig_idx_np = np.arange(len(items))

        routing_hashes = np.array([mmh3.hash(x, seed=0) for x in items], dtype=np.int32)
        stripe_ids = np.mod(routing_hashes, self.num_threads)

        jobs = []
        m_stripe = self.layers[0]['m_stripe']
        k_list = [l['k'] for l in self.layers]

        for i in range(self.num_threads):
            # Passiamo al worker la lista di tutte le stripes corrispondenti al suo ID
            stripes_per_id = [l['stripes'][i] for l in self.layers]
            mask = (stripe_ids == i)
            jobs.append((stripes_per_id, m_stripe, k_list, items_np[mask], orig_idx_np[mask]))

        # Fase di MAP
        worker_results = list(self.executor.map(_worker_contains_striped, jobs))

        # Fase di RECONSTRUCTION (Reduce)
        final_results = [None] * len(items)
        for res_block, idx_block in worker_results:
            for r, idx in zip(res_block, idx_block):
                final_results[idx] = r

        return final_results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()