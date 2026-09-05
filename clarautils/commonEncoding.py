from dataclasses import dataclass
from typing import Iterable, Any, List, NamedTuple, Tuple, overload

import numpy as np
import numpy.typing as npt
from clarautils import get_as_unsigned

from commonTyping import get_type_for_array, get_type_for_bit_count


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


def get_bits(value: np.ndarray | int | str | CommonNBitSc, count=None,
             allow_integer_floats: bool = False):
    from BitInfo import BitInfo
    return BitInfo.from_value(value, count, BitInfo.Mode.BITS, allow_integer_floats)


def get_bit_flags(value: np.ndarray | int | str | CommonNBitSc, count=None,
                  allow_integer_floats: bool = False) -> np.ndarray:
    from BitInfo import BitInfo
    return BitInfo.from_value(value, count, BitInfo.Mode.FLAGS, allow_integer_floats)

def normalize_flags(idx: npt.ArrayLike, bit: npt.ArrayLike=None) -> Tuple[int,int,int]:
    idx = np.array(idx)
    bit = np.array(bit)

    bit_count = np.max(idx)
    t = get_type_for_bit_count(bit_count)
    if (np.diff(idx) >= 0).all():  # if indices ascending, they are from left
        idx = - idx + bit_count

    flags = get_as_unsigned(1 << idx, fit=True)
    mask = np.bitwise_or.reduce(flags, dtype=t)
    if bit is not None:
        value_flags = bit.astype(t) * flags
        value_mask = np.bitwise_or.reduce(bit.astype(t) << idx, dtype=t)
        return bit_count, mask, value_mask
        # return bit_count, ((mask, flags), (value_mask, value_flags))
    else:
        return bit_count, mask, 0
        # return bit_count, ((mask, flags))


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


def get_bitwise_entropy(a, bit_count=None) -> npt.NDArray[np.float32]:
    from .BitFlagArray import NBitArray

    if isinstance(a, NBitArray):
        v = a.get_array()
        bit_count = a.get_bit_count()
    else:
        v = a
        if bit_count is None:
            raise ValueError("bit_count is required for plain arrays")

    v = np.asarray(v, dtype=np.int64).ravel()
    if v.size == 0:
        return np.zeros(bit_count, dtype=np.float32)

    p = np.zeros(bit_count, dtype=np.float32)
    for i in range(bit_count):
        p[i] = np.mean((v >> (bit_count - 1 - i)) & 1)

    entropy = np.zeros(bit_count, dtype=np.float32)
    mask = (p > 0) & (p < 1)
    pm = p[mask]
    entropy[mask] = -pm * np.log2(pm) - (1 - pm) * np.log2(1 - pm)
    return entropy


def get_bitwise_mean(a, bit_count=None, axis=1) -> npt.NDArray[np.float64]:
    """Mean over bits without unpacking; axis=1 per item (popcount/bc), axis=0 per bit position."""
    from .BitFlagArray import NBitArray

    if isinstance(a, NBitArray):
        v = a.get_array()
        bit_count = a.get_bit_count()
    else:
        v = np.asarray(a)
        if bit_count is None:
            raise ValueError("bit_count is required for plain arrays")

    if v.size == 0:
        return np.zeros(len(v) if axis == 1 else bit_count, dtype=np.float64)

    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    if axis == 1:
        return np.bitwise_count(v).astype(np.float64) / bit_count

    shifts = np.arange(bit_count - 1, -1, -1, dtype=np.intp)
    return np.array([np.count_nonzero((v >> s) & 1) for s in shifts], dtype=np.float64) / v.size


class DefinedBit(NamedTuple):
    idx: int
    bit: int


def get_defined_bits(a, bit_count=None) -> List[DefinedBit]:
    """Bits that are constant over all items, as (idx, bit) pairs; MSB-first, without unpacking."""
    from .BitFlagArray import NBitArray

    if isinstance(a, NBitArray):
        ary = a.get_array()
        bit_count = a.get_bit_count()
    else:
        ary = np.asarray(a)
        if bit_count is None:
            raise ValueError("bit_count is required for plain arrays")

    if len(ary) == 0:
        return []

    shifts = np.arange(bit_count - 1, -1, -1, dtype=np.intp)
    and_bits = (int(np.bitwise_and.reduce(ary)) >> shifts) & 1
    or_bits = (int(np.bitwise_or.reduce(ary)) >> shifts) & 1
    idx = np.flatnonzero(and_bits == or_bits)
    return [DefinedBit(int(i), int(b)) for i, b in zip(idx, and_bits[idx])]

