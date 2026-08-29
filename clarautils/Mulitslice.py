import numpy as np
import numpy.typing as npt

from .commonTyping import get_type_for_bit_count, get_type_for_scalar, get_as_unsigned
from .commonEncoding import get_number, get_bits, CommonNBitAry


class Multislice:
    def __init__(self, indices: npt.ArrayLike):
        indices = get_as_unsigned(indices).astype(get_type_for_scalar(np.max(indices)))

        # segment the selection into runs
        breaks = np.where(np.diff(indices) != 1)[0]                  # end of each run
        self.total_len = indices.size
        self.target_dtype = get_type_for_bit_count(self.total_len)

        self.src_start = np.concat((indices[:1], indices[breaks + 1]))
        self.src_stop = np.concat((indices[breaks], indices[-1:])) + 1
        self.lengths = self.src_stop - self.src_start

    def get_slices(self):
        return [slice(int(start), int(stop)) for start, stop in zip(self.src_start, self.src_stop)]

    def select_bits(self, arr: CommonNBitAry):
        # arrange the runs MSB-first, in selection order
        src_lsb = arr.bit_count - self.src_stop            # run position in the source word
        tgt_lsb = self.total_len - np.cumsum(self.lengths)  # run position in the packed target
        shift = (tgt_lsb - src_lsb).astype(np.intp)         # > 0 left, < 0 right

        masks = ((1 << self.lengths.astype(arr.dtype)) - 1) << src_lsb
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
