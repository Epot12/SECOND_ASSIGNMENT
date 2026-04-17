import pytest
import math
import os
import sys

# paths setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# importing all classes
from SEQUENTIAL.ScalableBloomFilter import ScalableBloomFilter as Sequential
from MULTITHREAD.MapReduceScalBloom import ThreadedScalableBloomFilter as MapReduceNoGIL
from MULTITHREAD.ThreadedScalBloomFilter import ThreadedScalableBloomFilter as MultiThreading
from MULTITHREAD.stripe_strategy.StripedBloomFilter import StripedBloomFilter
from MULTITHREAD.stripe_strategy.StripedBloomFilterColMajor import StripedBloomFilterColMajor
from MULTITHREAD.stripe_strategy.StripedBloomFilterSoA_opt import StripedBloomFilterSoA as StripedBloomSoA_opt
from PARALLEL.PermPoolBloom import PermPoolScalableBloomFilter as SotaIPC
from PARALLEL.ScalJobLib import JoblibScalableBloomFilter as JoblibBloom
from PARALLEL.ScalMultProcBloomOpt import ParallelScalableBloomFilter as ScalParBloomOpt

# setting parameters

BLOOM_FILTER_IMPLEMENTATIONS = [
    (Sequential, {}),
    (MultiThreading, {'num_threads': 2}),
    (MapReduceNoGIL, {'num_threads': 2}),
    (StripedBloomFilter, {'num_threads': 2}),
    (StripedBloomFilterColMajor, {'num_threads': 2}),
    (StripedBloomSoA_opt, {'num_threads': 2}),
    (SotaIPC, {'num_processes': 2}),
    (JoblibBloom, {'n_jobs': 2}),
    (ScalParBloomOpt, {'num_processes': 2})
]


# Helper to instantiate filters ensuring resource cleanliness (Context Manager Management)
class BloomFilterContext:
    def __init__(self, bf_class, kwargs, init_cap, fpr):
        self.bf = bf_class(initial_capacity=init_cap, target_fp_rate=fpr, **kwargs)
        self.has_context = hasattr(self.bf, '__enter__')

    def __enter__(self):
        if self.has_context:
            return self.bf.__enter__()
        return self.bf

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.has_context:
            self.bf.__exit__(exc_type, exc_val, exc_tb)


@pytest.mark.parametrize("bf_class, kwargs", BLOOM_FILTER_IMPLEMENTATIONS)
class TestScalableBloomFilters:
    """
    This class contains tests. Pytest will create an instance of this class
    and will execute all methods for EVERY implementation defined in the list above.
    """

    def test_no_false_negatives(self, bf_class, kwargs):
        """Rule #1: An inserted item MUST always be found."""
        init_cap = 1000
        fpr = 0.01
        dataset = [f"item_pos_{i}" for i in range(500)]

        with BloomFilterContext(bf_class, kwargs, init_cap, fpr) as bf:
            # Batch insertion (if supported) or single insertion
            if hasattr(bf, 'add_batch'):
                bf.add_batch(dataset)
            else:
                for item in dataset:
                    bf.add(item)

            # check
            if hasattr(bf, 'contains_batch'):
                results = bf.contains_batch(dataset)
                assert all(results), f"[{bf_class.__name__}] False Negative found in batch!"
            else:
                for item in dataset:
                    assert item in bf, f"[{bf_class.__name__}] False Negative found for {item}!"

    def test_empirical_false_positive_rate(self, bf_class, kwargs):
        """Rule #2: The real false positive rate must not explode beyond the theoretical limit."""
        init_cap = 5000
        target_fpr = 0.05

        known_data = [f"known_{i}" for i in range(8000)]  # enforce scaling
        unknown_data = [f"unknown_{i}" for i in range(5000)]

        with BloomFilterContext(bf_class, kwargs, init_cap, target_fpr) as bf:
            # Insertion
            if hasattr(bf, 'add_batch'):
                bf.add_batch(known_data)
            else:
                for item in known_data:
                    bf.add(item)

            # Check false positives
            false_positives = 0
            if hasattr(bf, 'contains_batch'):
                results = bf.contains_batch(unknown_data)
                false_positives = sum(results)
            else:
                for item in unknown_data:
                    if item in bf:
                        false_positives += 1

            empirical_fpr = false_positives / len(unknown_data)

            # Statistical tolerance margin (30% more than theoretical for hash fluctuations)
            tolerance_limit = target_fpr * 1.3

            assert empirical_fpr <= tolerance_limit, \
                f"[{bf_class.__name__}] FPR too high! Target: {target_fpr}, Real: {empirical_fpr}"

    def test_scaling_behavior(self, bf_class, kwargs):
        """Rule #3: The structure must create new layers when it fills."""
        init_cap = 100
        fpr = 0.01

        with BloomFilterContext(bf_class, kwargs, init_cap, fpr) as bf:
            def get_layer_count(bloom_obj):
                if hasattr(bloom_obj, 'filters'): return len(bloom_obj.filters)
                if hasattr(bloom_obj, 'layers'): return len(bloom_obj.layers)
                if hasattr(bloom_obj, 'bitmaps'): return len(bloom_obj.bitmaps)
                if hasattr(bloom_obj, 'shm_blocks'): return len(bloom_obj.shm_blocks)
                return 1  # Fallback

            initial_layers = get_layer_count(bf)

            # entering triple the initial capacity to force the creation of 1 or 2 new layers
            dataset = [f"scale_test_{i}" for i in range(init_cap * 3)]

            if hasattr(bf, 'add_batch'):
                # forcing a low min_chunk_size if it exists to avoid anomalous jumps
                if hasattr(bf, 'min_chunk_size'):
                    bf.min_chunk_size = 10
                bf.add_batch(dataset)
            else:
                for item in dataset:
                    bf.add(item)

            final_layers = get_layer_count(bf)
            assert final_layers > initial_layers, \
                f"[{bf_class.__name__}] The structure did not scale! Initial layers: {initial_layers}, Final: {final_layers}"