import math
from bitarray import bitarray

class BloomFilter:
    def __init__(self, expected_elements:int, false_positive_rate:float):
        self.n = expected_elements
        self.p = false_positive_rate
        self.m, self.k = self._calculate_params()
        self.bitmap = bitarray(self.m)
        self.bitmap.setall(0) #to set all 0

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

        # We generate two basic hashes (SHA-256 broken in half)
        digest = hashlib.sha256(item_bytes).digest()
        h1 = int.from_bytes(digest[:4], 'big')
        h2 = int.from_bytes(digest[4:8], 'big')

        for i in range(self.k):
            # Double hashing formula to obtain k different indices
            yield (h1 + i * h2) % self.m

    def add(self, item):
        """Inserts an element into the filter by setting the corresponding k bits to 1."""
        # call the utility to get the k indices
        for index in self._get_hashes(item):
            self.bitmap[index] = True

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
