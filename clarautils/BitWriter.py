from typing import List

import numpy as np

try:
    from .common import iter_bits
    from .commonEncoding import get_bits, get_number
except ImportError:
    from common import iter_bits
    from commonEncoding import get_bits, get_number


class BitWriter:
    def __init__(self):
        self.bytes: List[np.ubyte] = []
        self.current_byte: np.ubyte = np.ubyte(0)
        self.current_bit = 0

    def _put_bit(self, bit: 1 | 0):
        if self.current_bit < 8:
            self.current_byte |= np.ubyte(int(bit) << (7 - self.current_bit))
            self.current_bit += 1
        else:
            self.bytes.append(self.current_byte)
            self.current_byte = np.ubyte(0)
            self.current_bit = 0
            self._put_bit(bit)

    def put(self, value, length: int | List[int] = 0):
        if isinstance(value, bool):
            self._put_bit(1 if value is True else 0)
        elif isinstance(length, List):
            for l in length:
                self._put_bit(l)
        elif length > 0:
            for b in get_bits(value, length):
                self._put_bit(b)

    def get_bytes(self):
        tail = [self.current_byte] if self.current_bit > 0 else []
        return np.array(self.bytes + tail, dtype=np.uint8)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.iterator = iter_bits(data)

    def get_bool(self):
        value = self.iterator.__next__()
        return bool(value == 1)

    def get_int(self, length: int):
        bits = [self.iterator.__next__() for _ in range(length)]
        return get_number(bits)


if __name__ == "__main__":
    bw = BitWriter()
    bw.put(True)
    bw.put(2, length=5)
    bw.put(False)
    bw.put(12, length=5)

    data = bw.get_bytes()
    print(data)

    br = BitReader(data)
    print(br.get_bool())
    print(br.get_int(5))
    print(br.get_bool())
    print(br.get_int(5))
