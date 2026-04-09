import math
from bitarray import bitarray
import mmh3

class BloomFilter:
    def __init__(self, expected_elements:int, false_positive_rate:float):
        self.n = expected_elements
        self.p = false_positive_rate
        self.m, self.k = self._calculate_params()
        self.bitmap = bitarray(self.m)
        self.bitmap.setall(0) #to set all 0
        self.elements_count = 0
        self.set_bits = 0

    def _calculate_params(self)-> tuple[int, int]:
        if self.n <= 0:
            raise ValueError(f"The number of expected items (n) must be > 0. Received: {self.n}")
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"The probability of false positives (p) must be between 0 and 1. Received: {self.p}")

        #Calculation of m: Size of the bit array
        m_float = -(self.n * math.log(self.p)) / (math.log(2) ** 2)
        m = math.ceil(m_float)

        #Calculation of k: Number of hash functions
        k_float = (m / self.n) * math.log(2)
        k = round(k_float)

        # Sanity check
        k = max(1, k)

        return m, k

    def _to_bytes(self, item) -> bytes:
        # If it is already bytes, do nothing
        if isinstance(item, bytes):
            return item

        # If it's a string, skip str() and go straight to encode
        if isinstance(item, str):
            return item.encode('utf-8')

        # If it's an integer, convert it directly to bytes without going through the string
        if isinstance(item, int):
            return item.to_bytes(8, 'big', signed=True)

        # Fallback for other types (less frequent)
        return str(item).encode('utf-8')

    def _get_hashes(self, item):
        """Generate k indices within the range [0, m-1] for a given element."""
        # converting the item to string and then to byte for hashing
        item_bytes = self._to_bytes(item)

        # mmh3.hash64 already returns a tuple with two 64-bit integers (h1 and h2)
        # It is used signed=False to have positive numbers, and a seed
        h1, h2 = mmh3.hash64(item_bytes, seed=42, signed=False)

        for i in range(self.k):
            # Double hashing formula to obtain k different indices
            yield (h1 + i * h2) % self.m

    def add(self, item):
        """Inserts an element and controls saturation efficiently."""
        # calculate the indices before writing
        indices = list(self._get_hashes(item))

        # setting bits
        for index in indices:
            # If the bit is 0, it is set to 1 and the global counter is incremented.
            # If it was already at 1 (collision), nothing is done
            if not self.bitmap[index]:
                self.bitmap[index] = True
                self.set_bits += 1

        self.elements_count += 1

        if self.elements_count > self.n and self.elements_count % 100 == 0:
            actual_fp = self.get_actual_fp_rate()
            if actual_fp > self.p:
                raise ValueError(f"Saturated filter! Current FP rate: {actual_fp:.4f}")

    def __contains__(self, item) -> bool:
        """
        Check if an item is (probably) present.
        Returns False if the item is definitely NOT in the set.
        Returns True if the item COULD be in the set.
        """
        # check the k indices generated for this item
        for index in self._get_hashes(item):
            # If even ONE bit is 0 (False), the element was never inserted
            if not self.bitmap[index]:
                return False

        # arriving here means all bits were 1
        return True


    def get_fill_ratio(self) -> float:
        """Returns the percentage of bits set to 1."""
        return self.set_bits / self.m


    def get_actual_fp_rate(self) -> float:
        """Calculates the current probability of false positives based on the bits set."""
        return (self.set_bits / self.m) ** self.k
