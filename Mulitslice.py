import numpy as np
import numpy.typing as npt

from commonTyping import get_type_for_bit_count, get_type_for_scalar, get_as_unsigned
from commonEncoding import get_number, get_bits, CommonNBitAry


class Multislice:
    def __init__(self, indices: npt.ArrayLike):
        indices_ary = get_as_unsigned(indices).astype(get_type_for_scalar(np.max(indices)))
        indices_dtype = indices_ary.dtype

        breaks = np.array(np.where(np.diff(indices_ary) != 1)[0], dtype=indices_dtype)
        self.total_len = indices_ary.size
        self.target_dtype = get_type_for_bit_count(self.total_len)

        self.src_start = np.concat((indices_ary[:1], indices_ary[breaks + 1]), dtype=indices_dtype)
        self.src_stop = np.concat((indices_ary[breaks], indices_ary[-1:])) + 1

        self.lengths = self.src_stop - self.src_start

        # packed-target positions, MSB-first like src_start/src_stop
        cum = np.cumsum(self.lengths)
        self.target_starts = cum - self.lengths
        self.target_stops = cum

    def get_slices(self):
        return [slice(start, stop) for start, stop in zip(self.src_start, self.src_stop)]

    def select_bits(self, arr: CommonNBitAry):
        """Extract the selected bit runs (MSB-first) into one packed value."""
        src_lsb = arr.bit_count - self.src_stop            # run start, LSB frame of the source word
        tgt_lsb = self.total_len - self.target_stops       # run start, LSB frame of the packed target
        shift = (tgt_lsb - src_lsb).astype(np.intp)        # > 0 left, < 0 right (signed!)

        ones = np.ones_like(self.lengths, dtype=arr.dtype)
        masks = ((ones << self.lengths) - 1) << src_lsb

        extracted = arr.array.reshape((-1, 1)) & masks
        left = np.where(shift >= 0, shift, 0).astype(arr.dtype)
        right = np.where(shift < 0, -shift, 0).astype(arr.dtype)
        shifted = np.where(shift >= 0, extracted << left, extracted >> right)

        return np.bitwise_or.reduce(shifted, axis=1).astype(self.target_dtype)


if __name__ == "__main__":
    multi = Multislice([1, 2, 3, 5, 6, 7])
    print("masks/slices:", multi.get_slices())

    test_bits = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                          [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                          [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                          [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])
    test_n_ary = get_number(test_bits, axis=1)

    result = multi.select_bits(test_n_ary)

    print(get_bits(value=result, count=multi.total_len))
