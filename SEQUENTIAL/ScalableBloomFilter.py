from BloomFilter import *

class ScalableBloomFilter:
    def __init__(self, initial_capacity: int, target_fp_rate: float, tightening_ratio: float = 0.9,
                 growth_factor: int = 2):
        """
        :param initial_capacity: The size n of the first Bloom Filter.
        :param target_fp_rate: The desired MAXIMUM global false positive rate.
        :param tightening_ratio: How much stiffer each new filter becomes (0.9 = 10% stiffer).
        :param growth_factor: How much the memory grows each time (2 = doubles each time).
        """
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

    def add(self, item):
        """Adds an item. If the current filter is full, it creates a new one."""
        active_filter = self.filters[-1]  # The last filter in the list is the active one

        # prevention of exceptions
        if active_filter.elements_count >= active_filter.n:
            self._add_new_filter()
            active_filter = self.filters[-1]  # updating pointer to new filter

        active_filter.add(item)

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