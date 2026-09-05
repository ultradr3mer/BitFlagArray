from enum import StrEnum
from typing import Literal, Iterable

import numpy as np
import numpy.typing as npt

from clarautils import CommonNBitSc, get_bit_count, get_number, get_as_unsigned
from clarautils import get_as_signed


class BitInfo:
    class Mode(StrEnum):
        BITS = "bits"
        FLAGS = "flags"
        INDICES = "indices"
        B_COUNT = "count"

    type resultMode = Mode | Literal["bits", "flags", "indices", "count"]

    #processed like flags, only then processing to actual type
    FLAGS_SUBTYPES = {
        Mode.FLAGS, Mode.INDICES, Mode.B_COUNT
    }


    @staticmethod
    def from_value(value: npt.ArrayLike | int | str | CommonNBitSc,
                   bit_count: int | None = None, mode: resultMode = Mode.BITS,
                   acc_floats: bool = False):
        def get_signed_or_raise(v_sub: npt.ArrayLike | int):
            if not isinstance(v_sub, (float, np.floating)):
                return v_sub
            if acc_floats:
                return get_as_unsigned(v_sub, fit=True, acc_floats=True)
            else:
                raise Exception(
                    "Float not allowed, but provided, use acc_floats=True to let BitInfo get a shot at parsing them")

        mode = BitInfo.Mode(mode)

        if isinstance(value, str):
            return BitInfo.from_string(value, mode=mode)

        if isinstance(value, CommonNBitSc):
            bit_count, value = value.bit_count, value.value

        bit_count = int(bit_count if bit_count is not None and bit_count > 0
                        else get_bit_count(np.max(value)))
        bit_range_bits = np.arange(bit_count - 1, -1, -1)
        flags_range = np.array(1 << np.arange(bit_count))

        if np.isscalar(value):
            value = get_signed_or_raise(value)
            if mode in BitInfo.FLAGS_SUBTYPES:
                res = value & flags_range
                if mode == BitInfo.Mode.B_COUNT:
                    set_positions = np.nonzero(res)[0]
                    return int(set_positions[-1]) + 1 if set_positions.size else 0
                flags = get_as_signed(res[np.nonzero(res)], fit=True)
                if mode == BitInfo.Mode.FLAGS:
                    return flags
                return np.searchsorted(flags_range, flags)
            if mode == BitInfo.Mode.BITS:
                return (get_as_signed(value) >> bit_range_bits) & 1
        else:
            value = get_signed_or_raise(value)
            if mode in BitInfo.FLAGS_SUBTYPES:
                res = value.reshape((-1, 1)) & flags_range
                if mode == BitInfo.Mode.B_COUNT:
                    if bit_count == 0:
                        return np.zeros(res.shape[0], dtype=np.intp)
                    set_rows = (res > 0).any(axis=1)
                    rev_first = np.argmax((res > 0)[:, ::-1], axis=1)
                    return np.where(set_rows, bit_count - rev_first, 0)
                flags = get_as_signed(res[np.nonzero(res)], fit=True)
                if mode == BitInfo.Mode.FLAGS:
                    return flags
                return np.searchsorted(flags_range, flags)
            if mode == BitInfo.Mode.BITS:
                return (get_as_signed(value.reshape((-1, 1))) >> bit_range_bits) & 1

        raise Exception("invalid Value")

    @classmethod
    def from_string(cls, in_val: str | Iterable[str], mode: resultMode = Mode.BITS):
        mode = BitInfo.Mode(mode)
        single = isinstance(in_val, str)

        bit_vals = [np.fromiter((c == "1" for c in s), dtype=np.uint8)
                    for s in ((in_val,) if single else tuple(in_val))]
        only_bits_allowed = max(len(b) for b in bit_vals) > 128
        if only_bits_allowed and mode != BitInfo.Mode.BITS:
            raise Exception("Long Strings can only converted to bit. Consider splitting.")

        if mode == BitInfo.Mode.BITS:
            return [1 if c == "1" else 0 for c in in_val] if single else np.array(bit_vals)

        if single:
            return cls.from_value(get_number(bit_vals[0]), mode=mode)
        ary = get_number(np.array(bit_vals), axis=1)
        return cls.from_value(ary.array, ary.bit_count, mode=mode)
