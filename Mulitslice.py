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

        breaks = np.array(np.where(np.diff(indices_ary) != 1)[0], dtype=indices_dtype)
        self.total_len = indices_ary.size
        self.start = np.concatenate((indices_ary[:1], indices_ary[breaks + 1]))
        self.stop = np.concatenate((indices_ary[breaks], indices_ary[-1:])) + 1
        self.lengths = self.stop - self.start

        self.target_start = get_as_signed(self.total_len - np.cumsum(self.lengths))

    def get_slices(self):
        return [slice(start, stop) for start, stop in zip(self.start, self.stop)]

    def select_bits(self, arr: np.ndarray[Tuple[int], np.dtype[np.unsignedinteger]], bit_count: int = None):
        if bit_count is None:
            bit_count = np.iinfo(arr.dtype).bits

        src_start = bit_count - self.stop
        shift = (self.target_start - src_start).astype(np.intp)

        ones = np.ones_like(self.start, dtype=arr.dtype)
        masks = ((ones << self.lengths) - 1) << src_start

        extracted = (arr.reshape((-1, 1)) & masks)
        pos_mask = shift >= 0
        pos_shift = np.where(pos_mask, shift, 0).astype(arr.dtype)
        neg_shift = np.where(~pos_mask, -shift, 0).astype(arr.dtype)
        shifted = np.where(pos_mask, extracted << pos_shift, extracted >> neg_shift)

        return np.sum(shifted, axis=1).astype(arr.dtype)


if __name__ == "__main__":
    multi = Multislice([1, 2, 3, 5, 6, 7])
    print("masks/slices:", multi.get_slices())

    test_bits = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                          [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                          [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                          [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])
    test_n_ary = get_number(test_bits, axis=1)

    result = multi.select_bits(test_n_ary.array, test_n_ary.bit_count)

    print(get_bits(value=result, count=multi.total_len))
