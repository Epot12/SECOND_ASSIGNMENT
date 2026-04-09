import math
from bitarray import bitarray

class BloomFilter:
    def __init__(self, expected_elements:int, false_positive_rate:float):
        self.n = expected_elements
        self.p = false_positive_rate

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

