from typing import Tuple

import numpy as np
import numpy.typing as npt

from common import get_type_for_bit_count, get_as_unsigned, get_as_signed
from commonEncoding import get_number, get_bits


class Multislice:
    def __init__(self, indices: npt.ArrayLike):
        indices_ary = get_as_unsigned(indices)
        indices_dtype = indices_ary.dtype
        value_dtype = get_type_for_bit_count(np.max(indices_ary))

        breaks = np.array(np.where(np.diff(indices) != 1)[0], dtype=indices_dtype)
        self.total_len = indices_ary.size
        self.start = np.concatenate((indices_ary[:1], indices_ary[breaks + 1]))
        self.stop = np.concatenate((indices_ary[breaks], indices_ary[-1:])) + 1
        self.lengths = self.stop - self.start

        ones = np.ones_like(self.start, dtype=value_dtype)
        self.masks = np.array(((ones << self.lengths) - 1) << self.start)

        target_start = get_as_signed(self.total_len - np.cumsum(self.lengths))
        self.shift = (target_start - self.start)

    def get_slices(self):
        return [slice(start, stop) for start, stop in zip(self.start, self.stop)]

    def select_bits(self, arr: np.ndarray[Tuple[int], np.dtype[np.unsignedinteger]]):
        return np.sum((arr.reshape((-1, 1)) & self.masks) << self.shift, axis=0)


multi = Multislice([1,2,3,5,6,7])
get_bits(multi.masks)

test_bits = np.array([[1,0,1,0,1,0,1,0,1,0,1],
                      [0,1,0,1,0,1,0,1,0,1,0],
                      [1,0,0,1,0,0,1,0,0,1,0],
                      [0,1,0,0,1,0,0,1,0,0,1],
                      [0,0,1,0,0,1,0,0,1,0,0]])
test_n_ary = get_number(test_bits)

result = multi.select_bits(test_n_ary.array)

print(get_bits(value=result, count=test_n_ary.bit_count))