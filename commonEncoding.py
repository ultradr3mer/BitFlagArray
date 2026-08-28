from dataclasses import dataclass
from typing import Iterable, Any, List, Tuple, overload

import numpy as np
import numpy.typing as npt

from common import get_type_for_array, get_type_for_bit_count, get_as_unsigned, get_as_signed


@dataclass(frozen=True)
class CommonNBitAry:
    array: np.ndarray
    bit_count: int

    @property
    def dtype(self):
        return self.array.dtype

@dataclass(frozen=True)
class CommonNBitSc:
    value: int | np.unsignedinteger
    bit_count: int


def get_max_value(bit_count: int) -> np.unsignedinteger:
    return np.pow(2, bit_count) - 1


def get_bit_count(value: int | np.unsignedinteger):
    return int(value).bit_length()




def get_bitmask(length, start=0):
    return ((1 << int(length)) - 1) << int(start)
    # result = 0
    # for i in range(length):
    #     result |= 1 << i
    # return result


def trim_leading_zero(bits: List | np.array):
    had_zero = 0
    for bit in bits:
        if bit == 0:
            had_zero += 1
        else:
            break
    had_zero -= 1
    if had_zero > 0:
        return bits[had_zero:]
    return bits


@overload
def get_number(value: np.ndarray[Tuple[int, int], np.dtype[np.unsignedinteger]] | List[List[int]], axis: int = 0) -> CommonNBitAry: ...


@overload
def get_number(value: np.ndarray[Tuple[int], np.dtype[np.unsignedinteger]] | List[int]) -> CommonNBitSc: ...


def get_number(value: npt.NDArray[np.unsignedinteger] | List[int] | List[List[int]], axis: int = 0) -> CommonNBitAry | CommonNBitSc:
    if not isinstance(value, np.ndarray):
        value = np.array(value, get_type_for_array(value))

    if not isinstance(value, np.ndarray):
        value = np.array(value, get_type_for_array(value))

    bit_count = int(value.shape[axis])
    bit_range = np.arange(bit_count - 1, -1, -1)

    if value.ndim == 1:
        value = int(np.sum((value << bit_range), axis=0))
        return CommonNBitSc(value, bit_count)
    else:
        if  axis == 0:
            value = value.swapaxes(1, axis)
        value = np.sum((value << bit_range), axis=1)
        return CommonNBitAry(value.astype(get_type_for_array(value)), bit_count)


def get_bits(value: np.ndarray | int | str, count=None):
    if isinstance(value, str):
        return [1 if c == '1' else 0 for c in value]

    bit_count = int(count if count is not None and count > 0
                    else get_bit_count(np.max(value)))
    bit_range = np.arange(bit_count - 1, -1, -1)

    if np.isscalar(value):
        return (value >> bit_range) & 1
    elif isinstance(value, np.ndarray):
        return (get_as_signed(value.reshape((-1, 1))) >> bit_range) & 1
    else:
        raise Exception("invalid Value")


def flip_enc(value):
    def gen(ary):
        last = '0' if isinstance(ary[0], str) else 0
        for v in ary:
            yield (v if v != 0 and v != 1
                        and v != '1' and v != '0'
                   else 1 if v != last else 0)
            last = v

    return list(gen(value))


def trim_trailing_zero(p):
    if sum(p) == 0:
        return [0, 0]
    zeros = 0
    while sum(p[-zeros - 1:]) == 0:
        zeros += 1
    return p if zeros == 0 else p[:-zeros]


def symbol_to_str(sym: Iterable[Any] | npt.NDArray[np.unsignedinteger]) -> str:
    result = ''
    for s in sym:
        result += str(s)
    return result


def get_bit_flag_idx(value):
    return np.log2(value).astype(int) if value > 0 else 0


def bits_to_hex(bits: str) -> str:
    """Konvertiert einen Binärstring in einen Hexadezimalstring."""
    if not bits:
        return "0"

    # Auf ein Vielfaches von 4 auffüllen
    padding = (4 - len(bits) % 4) % 4
    padded_bits = bits + '0' * padding

    # Jeweils 4 Bits in ein Hex-Zeichen umwandeln
    return "".join(
        f"{int(padded_bits[i:i + 4], 2):X}"
        for i in range(0, len(padded_bits), 4)
    )


def hex_to_bits(hex_str: str, original_length: int | None = None) -> str:
    """Konvertiert einen Hexadezimalstring in einen Binärstring."""
    if not hex_str:
        return ""

    # Jedes Hex-Zeichen in genau 4 Bits umwandeln (mit führenden Nullen)
    bits = "".join(f"{int(c, 16):04b}" for c in hex_str)

    # Falls die Originallänge übergeben wurde, überschüssige Padding-Nullen abschneiden
    if original_length is not None and original_length < len(bits):
        return bits[:original_length]

    return bits


# noinspection string-conversion-without-dunder-method
def build_bins_n_print(v, q):
    print(f"Max: {np.max(v)}, Avg: {np.average(v)}")
    bins = np.bincount(v)
    print(f"Distribution: {bins}")
    value_perc = np.array(np.percentile(v, q=q), dtype=np.uint32)
    value_perc_delta = value_perc - np.concat((np.zeros(1, dtype=np.uint32), value_perc[:-1]))
    print(f"Q{q}: {value_perc}->{value_perc_delta}")
    return value_perc_delta


def fmt_k_bits(b: int) -> str:
    size, u = b / 8, 0
    while size >= 1024 and u < 4:
        size /= 1024
        u += 1
    units = ["B", "KB", "MB", "GB", "TB"]
    return f"{b:,} bits ({size:.1f} {units[u]})"


# NOTE: arrange_bits is defined in BitFlagArray.py (single source of truth).


def arrange_bits(values, take_indices):
    result = np.zeros_like(values.get_array())

    shifts = (np.full_like(take_indices, fill_value=values.get_bit_count() - 1)
              - np.flip(take_indices, axis=0))
    from_ary = values.get_array()

    for shift_to, shift_from in enumerate(shifts):
        result = result + (1 << shift_to) * ((from_ary >> shift_from) & 1)

    return result

