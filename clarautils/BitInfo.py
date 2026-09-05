from enum import StrEnum
from typing import NamedTuple, Literal, Iterable

import npt

import numpy as np
import numpy.typing as npt

from clarautils import get_bit_count, CommonNBitSc, get_as_signed


class BitInfo(NamedTuple):
    class Mode(StrEnum):
        BITS = "bits"
        FLAGS = "flags"
        INDICES = "indices"
        B_COUNT = "count"

    type resultMode = BitInfo.Mode | Literal["bits", "flags", "indices", "count"]
    value: npt.ArrayLike | int
    count: int
    mode: resultMode

    #processed like flags, only then processing to actual type
    FLAGS_SUBTYPES = {
        Mode.FLAGS, Mode.INDICES, Mode.B_COUNT
    }

    @staticmethod
    def from_value(value: npt.ArrayLike | int, bit_count: int, mode: resultMode):
        bit_count = value.bit_count if isinstance(value, CommonNBitSc) \
                    else int(bit_count if bit_count is not None and bit_count > 0
                    else get_bit_count(np.max(value)))
        bit_range_bits = np.arange(bit_count - 1, -1, -1)

        bit_indices = np.arange(bit_count)
        flags_range = np.array(1 << bit_indices)
        # flags_to_idx = dict(zip(flags, bit_indices))

        if isinstance(value, CommonNBitSc):
            value = value.value

        if np.isscalar(value):
            if BitInfo.FLAGS_SUBTYPES.issubset(mode):
                res = value & flags_range
                flags = get_as_signed(res[np.nonzero(res)], fit=True)
                if mode == BitInfo.Mode.FLAGS:
                elif mode == BitInfo.Mode.INDICES:
                    indices = np.where(flags_range == flags)[0]
                    return indices
                elif mode == BitInfo.Mode.B_COUNT:
                    max = flags[0]
                    # max = flags[-1]
                    lengths = np.where(flags_range == max)[0]
                    return lengths
            elif mode == BitInfo.Mode.BITS:
                return (get_as_signed(value) >> bit_range_bits) & 1

        value = np.array(value)
        if isinstance(value, np.ndarray):
            if BitInfo.FLAGS_SUBTYPES.issubset(mode):
                res = value.reshape((-1, 1)) & flags_range
                return get_as_signed(res[np.nonzero(res)], fit=True)
            elif mode == BitInfo.Mode.BITS:
                return (get_as_signed(value.reshape((-1, 1))) >> bit_range_bits) & 1
        else:
            raise Exception("invalid Value")

    @classmethod
    def from_string(cls, in_val: str | Iterable[str], mode: resultMode) -> 'BitInfo':
        bit_vals = []
        def bit_from_single_str(s):
            bit_vals.append(np.fromiter((c == "1" for c in s), dtype=np.uint8))

        if isinstance(in_val, str):
            bit_from_single_str(in_val)
        else:
            for item in np.array(in_val):
                bit_from_single_str(item)

        only_bits_allowed = np.max(len(itm) for itm in bit_vals) > 128
        if only_bits_allowed and mode != BitInfo.Mode.BITS:
            raise Exception("Long Strings can only converted to bit. Consider splitting.")

        cls.from_value(bit_vals, mode)