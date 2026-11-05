import struct
from typing import List, BinaryIO

# Bit set. The engine stores up to 64 words (2048 indices); files can
# legally carry fewer, and a corrupt count is rejected rather than read.
_MAX_WORDS = 4096


class TSIntegerSet:
    def __init__(self):
        self.values: List[int] = []

    def copy_from(self, other):
        self.values = other.values.copy()

    def read(self, stream: BinaryIO):
        # Leading count is unused. The second count is the number of words.
        struct.unpack('<L', stream.read(4))
        word_count = struct.unpack('<L', stream.read(4))[0]
        if word_count > _MAX_WORDS:
            raise ValueError(f"TSIntegerSet word count {word_count} is implausible")
        if word_count:
            self.values = list(struct.unpack(f'<{word_count}I', stream.read(4 * word_count)))
        else:
            self.values = []

    def from_list(self, words: List[int]):
        self.values = [int(word) & 0xFFFFFFFF for word in words]

    def test(self, index: int) -> bool:
        word, bit = divmod(index, 32)
        if word >= len(self.values):
            return False
        return ((self.values[word] >> bit) & 1) != 0

    def rank(self, index: int) -> int:
        """Number of set bits with an index strictly below `index`."""
        word, bit = divmod(index, 32)
        count = 0
        for value in self.values[:word]:
            count += value.bit_count()
        if bit and word < len(self.values):
            count += (self.values[word] & ((1 << bit) - 1)).bit_count()
        return count
