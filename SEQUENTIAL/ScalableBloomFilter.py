import os, sys
import builtins
if 'profile' not in builtins.__dict__:
    def profile(func): return func
    builtins.profile = profile

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from SEQUENTIAL.BloomFilter import  *

class ScalableBloomFilter:
    def __init__(self, initial_capacity: int, target_fp_rate: float, tightening_ratio: float = 0.9,
                 growth_factor: int = 2, **kwargs):
        """
        :param initial_capacity: The size n of the first Bloom Filter.
        :param target_fp_rate: The desired MAXIMUM global false positive rate.
        :param tightening_ratio: How much stiffer each new filter becomes (0.9 = 10% stiffer).
        :param growth_factor: How much the memory grows each time (2 = doubles each time).
        """

        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be strictly positive.")

        if not (0 < target_fp_rate < 1):
            raise ValueError("target_fp_rate must be between 0 and 1 (exclusive).")

        if not (0 < tightening_ratio < 1):
            raise ValueError("tightening_ratio must be between 0 and 1 (exclusive).")

        if growth_factor <= 1:
            raise ValueError("growth_factor must be strictly greater than 1 to scale.")
        self.initial_capacity = initial_capacity
        self.target_fp_rate = target_fp_rate
        self.tightening_ratio = tightening_ratio
        self.growth_factor = growth_factor

        # The first filter must have a FP rate calculated so that
        # the infinite sum of all future filters does not exceed the target_fp_rate.
        self.p0 = target_fp_rate * (1 - tightening_ratio)

        # list of filters
        self.filters: list[BloomFilter] = []

        # creating first filter
        self._add_new_filter()

    def _add_new_filter(self):
        """Create and append a new Bloom Filter with scaled parameters."""
        current_depth = len(self.filters)

        # Calculating capacity
        new_capacity = self.initial_capacity * (self.growth_factor ** current_depth)

        # Calculating new FP rate
        new_fp_rate = self.p0 * (self.tightening_ratio ** current_depth)

        new_bf = BloomFilter(new_capacity, new_fp_rate)
        self.filters.append(new_bf)

        print(f"[SYSTEM] Added new layer to Bloom Filter. "
              f"Capacity: {new_capacity}, FP Rate: {new_fp_rate:.5f}")

    @profile
    def add(self, item):
        """Adds an item. If the current filter is full, it creates a new one."""
        active_filter = self.filters[-1]  # The last filter in the list is the active one

        # prevention of exceptions
        if active_filter.elements_count >= active_filter.n:
            self._add_new_filter()
            active_filter = self.filters[-1]  # updating pointer to new filter

        active_filter.add(item)

    @profile
    def add_batch(self, items: list):
        for item in items:
            self.add(item)

    def __contains__(self, item) -> bool:
        """
        Search for the item in all filters.
        Returns True if AT LEAST ONE filter says "Yes".
        """
        # OPTIMIZATION: iterating the list IN THE CONTRARY (reversed).
        # Due to the 'Locality Principle', the most frequently searched data
        # are often the ones inserted most recently (in the last filter added).
        for bf in reversed(self.filters):
            if item in bf:
                return True

        return False

    def total_elements_count(self) -> int:
        """Returns the total element count across all filters."""
        return sum(bf.elements_count for bf in self.filters)

    def contains_batch(self, items: list) -> list[bool]:
        return [item in self for item in items]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False